"""
Feature alignment service for consistent multi-modal data representation.

This module implements the FeatureAlignment class that ensures consistent data
representation across different modalities and handles missing modalities and
partial data scenarios for robust multi-modal fusion.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import numpy as np
import torch
import logging
from enum import Enum
from abc import ABC, abstractmethod

from monai.transforms import Resize, ResizeWithPadOrCrop, Compose
from scipy import interpolate
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from src.models.patient import PatientRecord, ImagingStudy, WearableSession
from src.services.wearable_sensor.temporal_synchronizer import AlignedSensorData, TemporalFeatures
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class AlignmentStrategy(Enum):
    """Strategies for feature alignment."""
    TEMPORAL_INTERPOLATION = "temporal_interpolation"
    SPATIAL_REGISTRATION = "spatial_registration"
    FEATURE_MATCHING = "feature_matching"
    CROSS_MODAL_ALIGNMENT = "cross_modal_alignment"


class MissingDataStrategy(Enum):
    """Strategies for handling missing data."""
    ZERO_FILL = "zero_fill"
    MEAN_IMPUTATION = "mean_imputation"
    INTERPOLATION = "interpolation"
    LEARNED_IMPUTATION = "learned_imputation"
    SKIP_MODALITY = "skip_modality"


class NormalizationMethod(Enum):
    """Normalization methods for feature alignment."""
    STANDARD = "standard"  # Z-score normalization
    MINMAX = "minmax"     # Min-max scaling
    ROBUST = "robust"     # Robust scaling
    UNIT_NORM = "unit_norm"  # Unit vector normalization
    NONE = "none"         # No normalization


@dataclass
class AlignmentConfig:
    """Configuration for feature alignment."""
    temporal_resolution_seconds: float = 1.0
    spatial_resolution_mm: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    missing_data_strategy: MissingDataStrategy = MissingDataStrategy.INTERPOLATION
    normalization_method: NormalizationMethod = NormalizationMethod.STANDARD
    alignment_tolerance_seconds: float = 5.0
    max_interpolation_gap_seconds: float = 300.0  # 5 minutes
    feature_dimension_target: int = 512
    enable_cross_modal_normalization: bool = True


@dataclass
class AlignmentResult:
    """Result of feature alignment operation."""
    aligned_features: Dict[str, torch.Tensor]
    alignment_quality: Dict[str, float]
    missing_data_regions: Dict[str, List[Tuple[int, int]]]
    normalization_parameters: Dict[str, Dict[str, Any]]
    temporal_alignment_info: Dict[str, Any]
    spatial_alignment_info: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class FeatureRepresentation:
    """Standardized feature representation for a modality."""
    features: torch.Tensor
    timestamps: Optional[np.ndarray] = None
    spatial_coordinates: Optional[np.ndarray] = None
    quality_mask: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class FeatureAligner(ABC):
    """Abstract base class for feature alignment strategies."""
    
    @abstractmethod
    def align_features(
        self, 
        source_features: FeatureRepresentation,
        target_features: FeatureRepresentation,
        config: AlignmentConfig
    ) -> Tuple[FeatureRepresentation, float]:
        """Align source features to target feature space."""
        pass


class TemporalAligner(FeatureAligner):
    """Temporal feature alignment using interpolation."""
    
    def align_features(
        self,
        source_features: FeatureRepresentation,
        target_features: FeatureRepresentation,
        config: AlignmentConfig
    ) -> Tuple[FeatureRepresentation, float]:
        """Align features temporally using interpolation."""
        if source_features.timestamps is None or target_features.timestamps is None:
            logger.warning("Missing timestamps for temporal alignment")
            return source_features, 0.5
        
        try:
            # Interpolate source features to target timestamps
            aligned_features = self._interpolate_temporal_features(
                source_features.features,
                source_features.timestamps,
                target_features.timestamps,
                config
            )
            
            # Calculate alignment quality
            quality = self._calculate_temporal_alignment_quality(
                source_features.timestamps,
                target_features.timestamps,
                config
            )
            
            aligned_repr = FeatureRepresentation(
                features=aligned_features,
                timestamps=target_features.timestamps,
                spatial_coordinates=source_features.spatial_coordinates,
                quality_mask=source_features.quality_mask,
                metadata=source_features.metadata.copy()
            )
            
            return aligned_repr, quality
            
        except Exception as e:
            logger.error(f"Temporal alignment failed: {str(e)}")
            return source_features, 0.0
    
    def _interpolate_temporal_features(
        self,
        features: torch.Tensor,
        source_timestamps: np.ndarray,
        target_timestamps: np.ndarray,
        config: AlignmentConfig
    ) -> torch.Tensor:
        """Interpolate features to target timestamps."""
        if features.dim() == 1:
            # 1D feature vector
            interpolator = interpolate.interp1d(
                source_timestamps,
                features.numpy(),
                kind='linear',
                bounds_error=False,
                fill_value=0.0
            )
            interpolated = interpolator(target_timestamps)
            return torch.tensor(interpolated, dtype=features.dtype)
        
        elif features.dim() == 2:
            # 2D feature matrix (time x features)
            interpolated_features = []
            for i in range(features.shape[1]):
                interpolator = interpolate.interp1d(
                    source_timestamps,
                    features[:, i].numpy(),
                    kind='linear',
                    bounds_error=False,
                    fill_value=0.0
                )
                interpolated_features.append(interpolator(target_timestamps))
            
            return torch.tensor(
                np.column_stack(interpolated_features),
                dtype=features.dtype
            )
        
        else:
            # Higher dimensional features - interpolate along first dimension
            interpolated_shape = (len(target_timestamps),) + features.shape[1:]
            interpolated = torch.zeros(interpolated_shape, dtype=features.dtype)
            
            features_flat = features.view(features.shape[0], -1)
            for i in range(features_flat.shape[1]):
                interpolator = interpolate.interp1d(
                    source_timestamps,
                    features_flat[:, i].numpy(),
                    kind='linear',
                    bounds_error=False,
                    fill_value=0.0
                )
                interpolated_flat = interpolator(target_timestamps)
                interpolated.view(len(target_timestamps), -1)[:, i] = torch.tensor(
                    interpolated_flat, dtype=features.dtype
                )
            
            return interpolated
    
    def _calculate_temporal_alignment_quality(
        self,
        source_timestamps: np.ndarray,
        target_timestamps: np.ndarray,
        config: AlignmentConfig
    ) -> float:
        """Calculate quality of temporal alignment."""
        # Calculate overlap between timestamp ranges
        source_range = (source_timestamps.min(), source_timestamps.max())
        target_range = (target_timestamps.min(), target_timestamps.max())
        
        overlap_start = max(source_range[0], target_range[0])
        overlap_end = min(source_range[1], target_range[1])
        
        if overlap_end <= overlap_start:
            return 0.0
        
        overlap_duration = overlap_end - overlap_start
        target_duration = target_range[1] - target_range[0]
        
        overlap_ratio = overlap_duration / target_duration if target_duration > 0 else 0.0
        
        # Penalize for large gaps in source data
        source_gaps = np.diff(source_timestamps)
        large_gaps = source_gaps > config.max_interpolation_gap_seconds
        gap_penalty = np.sum(large_gaps) / len(source_gaps) if len(source_gaps) > 0 else 0.0
        
        quality = overlap_ratio * (1.0 - gap_penalty)
        return max(0.0, min(1.0, quality))


class SpatialAligner(FeatureAligner):
    """Spatial feature alignment using registration."""
    
    def align_features(
        self,
        source_features: FeatureRepresentation,
        target_features: FeatureRepresentation,
        config: AlignmentConfig
    ) -> Tuple[FeatureRepresentation, float]:
        """Align features spatially."""
        # For now, implement basic spatial resizing
        # In a full implementation, this would include registration algorithms
        
        if source_features.features.dim() < 3:
            # Non-spatial features, return as-is
            return source_features, 1.0
        
        try:
            # Resize spatial dimensions to match target
            target_spatial_size = target_features.features.shape[-3:]
            
            if source_features.features.shape[-3:] != target_spatial_size:
                resize_transform = ResizeWithPadOrCrop(spatial_size=target_spatial_size)
                aligned_features = resize_transform(source_features.features)
            else:
                aligned_features = source_features.features
            
            aligned_repr = FeatureRepresentation(
                features=aligned_features,
                timestamps=source_features.timestamps,
                spatial_coordinates=target_features.spatial_coordinates,
                quality_mask=source_features.quality_mask,
                metadata=source_features.metadata.copy()
            )
            
            return aligned_repr, 0.9  # High quality for simple resizing
            
        except Exception as e:
            logger.error(f"Spatial alignment failed: {str(e)}")
            return source_features, 0.0


class FeatureAlignment:
    """
    Feature alignment service for consistent multi-modal data representation.
    
    This class ensures consistent data representation across different modalities
    and handles missing modalities and partial data scenarios.
    """
    
    def __init__(self, config: Optional[AlignmentConfig] = None):
        """
        Initialize the feature alignment service.
        
        Args:
            config: Alignment configuration parameters
        """
        self.config = config or AlignmentConfig()
        self.logger = get_logger(__name__)
        
        # Initialize aligners
        self.temporal_aligner = TemporalAligner()
        self.spatial_aligner = SpatialAligner()
        
        # Normalization scalers
        self.scalers = {}
        
    def align_multi_modal_features(
        self,
        modality_features: Dict[str, FeatureRepresentation],
        reference_modality: Optional[str] = None
    ) -> AlignmentResult:
        """
        Align features across multiple modalities.
        
        Args:
            modality_features: Dictionary of features by modality
            reference_modality: Modality to use as alignment reference
            
        Returns:
            AlignmentResult: Aligned features and metadata
        """
        if not modality_features:
            raise ValueError("No modality features provided")
        
        # Determine reference modality
        if reference_modality is None:
            # Use the modality with the most complete data
            reference_modality = self._select_reference_modality(modality_features)
        
        if reference_modality not in modality_features:
            raise ValueError(f"Reference modality '{reference_modality}' not found")
        
        reference_features = modality_features[reference_modality]
        aligned_features = {}
        alignment_quality = {}
        missing_data_regions = {}
        normalization_parameters = {}
        warnings = []
        errors = []
        
        # Align each modality to the reference
        for modality, features in modality_features.items():
            try:
                if modality == reference_modality:
                    # Reference modality doesn't need alignment
                    aligned_features[modality] = features.features
                    alignment_quality[modality] = 1.0
                    missing_data_regions[modality] = []
                else:
                    # Align to reference
                    aligned_repr, quality = self._align_to_reference(
                        features, reference_features
                    )
                    aligned_features[modality] = aligned_repr.features
                    alignment_quality[modality] = quality
                    
                    # Detect missing data regions
                    missing_regions = self._detect_missing_data_regions(aligned_repr.features)
                    missing_data_regions[modality] = missing_regions
                
                # Handle missing data
                aligned_features[modality] = self._handle_missing_data(
                    aligned_features[modality], missing_data_regions[modality]
                )
                
                # Normalize features
                normalized_features, norm_params = self._normalize_features(
                    aligned_features[modality], modality
                )
                aligned_features[modality] = normalized_features
                normalization_parameters[modality] = norm_params
                
            except Exception as e:
                error_msg = f"Failed to align modality '{modality}': {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)
                
                # Apply fallback processing (requirement 2.4)
                try:
                    fallback_modalities = self.handle_alignment_failure(
                        modality_features, modality, e
                    )
                    
                    # Update with fallback results
                    if modality in fallback_modalities:
                        fallback_repr = fallback_modalities[modality]
                        aligned_features[modality] = fallback_repr.features
                        alignment_quality[modality] = 0.1  # Low quality for fallback
                        missing_data_regions[modality] = []
                        
                        # Normalize fallback features
                        normalized_features, norm_params = self._normalize_features(
                            aligned_features[modality], modality
                        )
                        aligned_features[modality] = normalized_features
                        normalization_parameters[modality] = norm_params
                        
                        warnings.append(f"Applied fallback processing for modality '{modality}'")
                    else:
                        warnings.append(f"Modality '{modality}' removed due to alignment failure")
                        
                except Exception as fallback_error:
                    fallback_error_msg = f"Fallback processing failed for '{modality}': {str(fallback_error)}"
                    errors.append(fallback_error_msg)
                    logger.error(fallback_error_msg)
        
        # Cross-modal normalization if enabled
        if self.config.enable_cross_modal_normalization and len(aligned_features) > 1:
            aligned_features = self._cross_modal_normalization(aligned_features)
        
        return AlignmentResult(
            aligned_features=aligned_features,
            alignment_quality=alignment_quality,
            missing_data_regions=missing_data_regions,
            normalization_parameters=normalization_parameters,
            temporal_alignment_info={"reference_modality": reference_modality},
            spatial_alignment_info={"reference_modality": reference_modality},
            warnings=warnings,
            errors=errors
        )
    
    def create_unified_tensor(
        self,
        aligned_features: Dict[str, torch.Tensor],
        target_shape: Optional[Tuple[int, ...]] = None
    ) -> torch.Tensor:
        """
        Create a unified tensor from aligned multi-modal features.
        
        Args:
            aligned_features: Dictionary of aligned feature tensors
            target_shape: Target shape for the unified tensor
            
        Returns:
            Unified tensor combining all modalities
        """
        if not aligned_features:
            raise ValueError("No aligned features provided")
        
        # Standardize feature dimensions
        standardized_features = {}
        for modality, features in aligned_features.items():
            standardized_features[modality] = self._standardize_feature_dimensions(
                features, target_shape
            )
        
        # Concatenate features along channel dimension
        feature_list = list(standardized_features.values())
        
        # Ensure all tensors have the same spatial dimensions
        reference_shape = feature_list[0].shape
        for i, tensor in enumerate(feature_list[1:], 1):
            if tensor.shape != reference_shape:
                # Resize to match reference using torch interpolation
                feature_list[i] = self._resize_tensor_to_shape(tensor, reference_shape)
        
        # Concatenate along appropriate dimension
        if len(reference_shape) >= 4:  # (B, C, H, W, D) or (C, H, W, D)
            unified_tensor = torch.cat(feature_list, dim=-4)  # Channel dimension
        elif len(reference_shape) == 3:  # (C, H, W)
            unified_tensor = torch.cat(feature_list, dim=0)  # Channel dimension
        else:  # 1D or 2D features
            unified_tensor = torch.cat(feature_list, dim=-1)  # Feature dimension
        
        return unified_tensor
    
    def handle_partial_data(
        self,
        available_modalities: Dict[str, FeatureRepresentation],
        expected_modalities: List[str]
    ) -> Dict[str, FeatureRepresentation]:
        """
        Handle scenarios with missing modalities by imputation or substitution.
        
        Args:
            available_modalities: Available modality features
            expected_modalities: List of expected modality names
            
        Returns:
            Complete set of modality features with imputed missing ones
        """
        complete_modalities = available_modalities.copy()
        
        for modality in expected_modalities:
            if modality not in available_modalities:
                # Create placeholder features for missing modality
                imputed_features = self._impute_missing_modality(
                    modality, available_modalities
                )
                complete_modalities[modality] = imputed_features
                
                logger.warning(f"Imputed missing modality: {modality}")
        
        return complete_modalities
    
    def _select_reference_modality(
        self, 
        modality_features: Dict[str, FeatureRepresentation]
    ) -> str:
        """Select the best modality to use as alignment reference."""
        # Score each modality based on data completeness and quality
        scores = {}
        
        for modality, features in modality_features.items():
            score = 0.0
            
            # Prefer modalities with temporal information
            if features.timestamps is not None:
                score += 0.3
            
            # Prefer modalities with spatial information
            if features.spatial_coordinates is not None:
                score += 0.3
            
            # Prefer modalities with quality masks
            if features.quality_mask is not None:
                score += 0.2
            
            # Prefer modalities with more data points
            if features.features.numel() > 0:
                score += 0.2 * min(1.0, features.features.numel() / 10000)
            
            scores[modality] = score
        
        # Return modality with highest score
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def _align_to_reference(
        self,
        source_features: FeatureRepresentation,
        reference_features: FeatureRepresentation
    ) -> Tuple[FeatureRepresentation, float]:
        """Align source features to reference modality."""
        # Temporal alignment if both have timestamps
        if (source_features.timestamps is not None and 
            reference_features.timestamps is not None):
            aligned_features, temporal_quality = self.temporal_aligner.align_features(
                source_features, reference_features, self.config
            )
        else:
            aligned_features = source_features
            temporal_quality = 0.5
        
        # Spatial alignment if both have spatial dimensions
        if (aligned_features.features.dim() >= 3 and 
            reference_features.features.dim() >= 3):
            aligned_features, spatial_quality = self.spatial_aligner.align_features(
                aligned_features, reference_features, self.config
            )
        else:
            spatial_quality = 1.0
        
        # Combined quality score
        overall_quality = (temporal_quality + spatial_quality) / 2.0
        
        return aligned_features, overall_quality
    
    def _detect_missing_data_regions(
        self, 
        features: torch.Tensor
    ) -> List[Tuple[int, int]]:
        """Detect regions of missing or invalid data."""
        missing_regions = []
        
        # Check for NaN or zero regions
        if features.dim() == 1:
            # 1D feature vector
            invalid_mask = torch.isnan(features) | (features == 0)
        else:
            # Multi-dimensional features - check along first dimension
            invalid_mask = torch.isnan(features).any(dim=tuple(range(1, features.dim())))
        
        # Find continuous regions of invalid data
        invalid_indices = torch.where(invalid_mask)[0]
        if len(invalid_indices) > 0:
            # Group consecutive indices
            regions = []
            start_idx = invalid_indices[0].item()
            prev_idx = start_idx
            
            for idx in invalid_indices[1:]:
                if idx.item() != prev_idx + 1:
                    # End of current region
                    regions.append((start_idx, prev_idx + 1))
                    start_idx = idx.item()
                prev_idx = idx.item()
            
            # Add final region
            regions.append((start_idx, prev_idx + 1))
            missing_regions = regions
        
        return missing_regions
    
    def _handle_missing_data(
        self,
        features: torch.Tensor,
        missing_regions: List[Tuple[int, int]]
    ) -> torch.Tensor:
        """Handle missing data regions based on configured strategy."""
        if not missing_regions:
            return features
        
        if self.config.missing_data_strategy == MissingDataStrategy.ZERO_FILL:
            # Replace NaN values with zeros
            features = torch.where(torch.isnan(features), torch.zeros_like(features), features)
            return features
        
        elif self.config.missing_data_strategy == MissingDataStrategy.MEAN_IMPUTATION:
            # Replace missing values with mean of valid data
            valid_mask = ~torch.isnan(features)
            if torch.any(valid_mask):
                mean_value = torch.mean(features[valid_mask])
                features = torch.where(torch.isnan(features), mean_value, features)
        
        elif self.config.missing_data_strategy == MissingDataStrategy.INTERPOLATION:
            # Linear interpolation for missing regions
            features = self._interpolate_missing_regions(features, missing_regions)
        
        return features
    
    def _interpolate_missing_regions(
        self,
        features: torch.Tensor,
        missing_regions: List[Tuple[int, int]]
    ) -> torch.Tensor:
        """Interpolate missing data regions."""
        interpolated_features = features.clone()
        
        for start_idx, end_idx in missing_regions:
            if start_idx == 0 or end_idx >= len(features):
                # Can't interpolate at boundaries - use forward/backward fill
                if start_idx == 0 and end_idx < len(features):
                    # Forward fill
                    interpolated_features[start_idx:end_idx] = features[end_idx]
                elif end_idx >= len(features) and start_idx > 0:
                    # Backward fill
                    interpolated_features[start_idx:end_idx] = features[start_idx - 1]
            else:
                # Linear interpolation
                start_val = features[start_idx - 1]
                end_val = features[end_idx]
                
                # Create interpolation values
                num_points = end_idx - start_idx
                interp_values = torch.linspace(0, 1, num_points + 2)[1:-1]
                
                for i, alpha in enumerate(interp_values):
                    interpolated_features[start_idx + i] = (1 - alpha) * start_val + alpha * end_val
        
        return interpolated_features
    
    def _normalize_features(
        self,
        features: torch.Tensor,
        modality: str
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Normalize features based on configured method."""
        if self.config.normalization_method == NormalizationMethod.NONE:
            return features, {}
        
        # Flatten features for normalization
        original_shape = features.shape
        features_flat = features.view(-1, features.shape[-1] if features.dim() > 1 else 1)
        
        # Apply normalization
        if self.config.normalization_method == NormalizationMethod.STANDARD:
            if modality not in self.scalers:
                self.scalers[modality] = StandardScaler()
            scaler = self.scalers[modality]
            
        elif self.config.normalization_method == NormalizationMethod.MINMAX:
            if modality not in self.scalers:
                self.scalers[modality] = MinMaxScaler()
            scaler = self.scalers[modality]
            
        elif self.config.normalization_method == NormalizationMethod.ROBUST:
            if modality not in self.scalers:
                self.scalers[modality] = RobustScaler()
            scaler = self.scalers[modality]
            
        else:  # UNIT_NORM
            # Unit normalization
            norm = torch.norm(features_flat, dim=1, keepdim=True)
            normalized_flat = features_flat / (norm + 1e-8)
            normalized_features = normalized_flat.view(original_shape)
            return normalized_features, {"method": "unit_norm"}
        
        # Fit and transform using sklearn scaler
        try:
            normalized_flat = torch.tensor(
                scaler.fit_transform(features_flat.numpy()),
                dtype=features.dtype
            )
            normalized_features = normalized_flat.view(original_shape)
            
            # Store normalization parameters
            norm_params = {
                "method": self.config.normalization_method.value,
                "mean": getattr(scaler, 'mean_', None),
                "scale": getattr(scaler, 'scale_', None),
                "data_min": getattr(scaler, 'data_min_', None),
                "data_max": getattr(scaler, 'data_max_', None)
            }
            
            return normalized_features, norm_params
            
        except Exception as e:
            logger.warning(f"Normalization failed for {modality}: {str(e)}")
            return features, {}
    
    def _cross_modal_normalization(
        self,
        aligned_features: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Apply cross-modal normalization to balance feature magnitudes."""
        # Calculate global statistics across all modalities
        all_features = torch.cat([f.flatten() for f in aligned_features.values()])
        global_mean = torch.mean(all_features)
        global_std = torch.std(all_features)
        
        # Normalize each modality to have similar scale
        normalized_features = {}
        for modality, features in aligned_features.items():
            modality_mean = torch.mean(features)
            modality_std = torch.std(features)
            
            # Scale to global statistics
            if modality_std > 1e-8:
                normalized = (features - modality_mean) / modality_std
                normalized = normalized * global_std + global_mean
            else:
                normalized = features
            
            normalized_features[modality] = normalized
        
        return normalized_features
    
    def _standardize_feature_dimensions(
        self,
        features: torch.Tensor,
        target_shape: Optional[Tuple[int, ...]]
    ) -> torch.Tensor:
        """Standardize feature tensor dimensions."""
        if target_shape is None:
            return features
        
        # Resize or reshape to target shape
        if features.shape == target_shape:
            return features
        
        # Handle different dimensionalities
        if len(target_shape) >= 3:  # Spatial data
            if features.dim() < len(target_shape):
                # Add missing dimensions
                while features.dim() < len(target_shape):
                    features = features.unsqueeze(0)
            
            # Resize spatial dimensions
            spatial_size = target_shape[-3:]
            resize_transform = ResizeWithPadOrCrop(spatial_size=spatial_size)
            return resize_transform(features)
        
        else:  # Non-spatial data
            return self._resize_feature_vector(features, target_shape)
    
    def _resize_feature_vector(
        self,
        features: torch.Tensor,
        target_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """Resize feature vector to target shape."""
        target_size = np.prod(target_shape)
        current_size = features.numel()
        
        if current_size == target_size:
            return features.view(target_shape)
        elif current_size > target_size:
            # Truncate
            return features.flatten()[:target_size].view(target_shape)
        else:
            # Pad with zeros
            padding = torch.zeros(target_size - current_size, dtype=features.dtype)
            padded = torch.cat([features.flatten(), padding])
            return padded.view(target_shape)
    
    def _impute_missing_modality(
        self,
        missing_modality: str,
        available_modalities: Dict[str, FeatureRepresentation]
    ) -> FeatureRepresentation:
        """Impute features for a missing modality."""
        if not available_modalities:
            # Create zero features
            features = torch.zeros(self.config.feature_dimension_target)
        else:
            # Use mean of available modalities as imputation
            feature_list = [repr.features.flatten() for repr in available_modalities.values()]
            
            # Ensure all features have the same size
            max_size = max(f.numel() for f in feature_list)
            padded_features = []
            
            for f in feature_list:
                if f.numel() < max_size:
                    padding = torch.zeros(max_size - f.numel(), dtype=f.dtype)
                    padded = torch.cat([f, padding])
                else:
                    padded = f[:max_size]
                padded_features.append(padded)
            
            # Compute mean
            stacked_features = torch.stack(padded_features)
            mean_features = torch.mean(stacked_features, dim=0)
            
            # Resize to target dimension
            if mean_features.numel() > self.config.feature_dimension_target:
                features = mean_features[:self.config.feature_dimension_target]
            else:
                padding = torch.zeros(
                    self.config.feature_dimension_target - mean_features.numel(),
                    dtype=mean_features.dtype
                )
                features = torch.cat([mean_features, padding])
        
        return FeatureRepresentation(
            features=features,
            metadata={"imputed": True, "source_modalities": list(available_modalities.keys())}
        )
    
    def _resize_tensor_to_shape(self, tensor: torch.Tensor, target_shape: Tuple[int, ...]) -> torch.Tensor:
        """Resize tensor to target shape using appropriate method based on dimensions."""
        if tensor.shape == target_shape:
            return tensor
        
        # If dimensions don't match, use padding/truncation approach
        if len(tensor.shape) != len(target_shape):
            return self._resize_feature_vector(tensor, target_shape)
        
        import torch.nn.functional as F
        
        if len(target_shape) == 3 and len(tensor.shape) == 3:  # 3D tensor
            # Add batch and channel dimensions for interpolation
            tensor_with_dims = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W, D)
            resized = F.interpolate(
                tensor_with_dims,
                size=target_shape,
                mode='trilinear',
                align_corners=False
            )
            return resized.squeeze(0).squeeze(0)
        
        elif len(target_shape) == 2 and len(tensor.shape) == 2:  # 2D tensor
            tensor_with_dims = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            resized = F.interpolate(
                tensor_with_dims,
                size=target_shape,
                mode='bilinear',
                align_corners=False
            )
            return resized.squeeze(0).squeeze(0)
        
        else:  # Use padding/truncation for other cases
            return self._resize_feature_vector(tensor, target_shape)

    def _resize_feature_vector(
        self,
        tensor: torch.Tensor,
        target_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """Resize feature vector to target shape."""
        if tensor.shape == target_shape:
            return tensor
        
        # Flatten and resize
        flattened = tensor.flatten()
        target_size = np.prod(target_shape)
        
        if len(flattened) > target_size:
            # Truncate
            resized = flattened[:target_size]
        else:
            # Pad with zeros
            padding = torch.zeros(target_size - len(flattened), dtype=tensor.dtype)
            resized = torch.cat([flattened, padding])
        
        return resized.reshape(target_shape)
    
    def _impute_missing_modality(
        self,
        missing_modality: str,
        available_modalities: Dict[str, FeatureRepresentation]
    ) -> FeatureRepresentation:
        """Impute features for a missing modality."""
        if not available_modalities:
            # Create zero features if no modalities available
            return FeatureRepresentation(
                features=torch.zeros(self.config.feature_dimension_target),
                metadata={"imputed": True, "method": "zero_fill"}
            )
        
        # Use the first available modality as template
        template_modality = next(iter(available_modalities.values()))
        
        if self.config.missing_data_strategy == MissingDataStrategy.ZERO_FILL:
            imputed_features = torch.zeros_like(template_modality.features)
        elif self.config.missing_data_strategy == MissingDataStrategy.MEAN_IMPUTATION:
            # Use mean of available modalities
            all_features = [repr.features for repr in available_modalities.values()]
            stacked_features = torch.stack(all_features)
            imputed_features = torch.mean(stacked_features, dim=0)
        else:
            # Default to zero fill
            imputed_features = torch.zeros_like(template_modality.features)
        
        return FeatureRepresentation(
            features=imputed_features,
            timestamps=template_modality.timestamps,
            spatial_coordinates=template_modality.spatial_coordinates,
            metadata={
                "imputed": True,
                "method": self.config.missing_data_strategy.value,
                "missing_modality": missing_modality
            }
        )
    
    def handle_alignment_failure(
        self,
        modality_features: Dict[str, FeatureRepresentation],
        failed_modality: str,
        error: Exception
    ) -> Dict[str, FeatureRepresentation]:
        """
        Handle alignment failure with fallback processing options.
        
        This method implements requirement 2.4: "IF timestamp alignment fails 
        THEN the system SHALL log the error and provide fallback processing options"
        
        Args:
            modality_features: Original modality features
            failed_modality: Name of the modality that failed alignment
            error: The alignment error that occurred
            
        Returns:
            Updated modality features with fallback processing applied
        """
        logger.error(f"Alignment failed for modality '{failed_modality}': {str(error)}")
        
        # Fallback strategy 1: Use zero-padding alignment
        try:
            logger.info(f"Attempting zero-padding fallback for '{failed_modality}'")
            fallback_features = modality_features.copy()
            
            if failed_modality in fallback_features:
                failed_repr = fallback_features[failed_modality]
                
                # Create a simple aligned version using zero-padding
                reference_modality = self._select_reference_modality(
                    {k: v for k, v in modality_features.items() if k != failed_modality}
                )
                
                if reference_modality and reference_modality in modality_features:
                    reference_repr = modality_features[reference_modality]
                    
                    # Align using simple resizing/padding
                    aligned_features = self._simple_alignment_fallback(
                        failed_repr, reference_repr
                    )
                    
                    fallback_features[failed_modality] = aligned_features
                    logger.info(f"Successfully applied zero-padding fallback for '{failed_modality}'")
                    return fallback_features
                    
        except Exception as fallback_error:
            logger.error(f"Zero-padding fallback failed: {str(fallback_error)}")
        
        # Fallback strategy 2: Remove the problematic modality
        try:
            logger.warning(f"Removing problematic modality '{failed_modality}' from alignment")
            fallback_features = {k: v for k, v in modality_features.items() if k != failed_modality}
            
            if fallback_features:
                logger.info(f"Successfully removed '{failed_modality}', continuing with {len(fallback_features)} modalities")
                return fallback_features
            else:
                logger.error("No modalities remaining after removal")
                
        except Exception as removal_error:
            logger.error(f"Modality removal fallback failed: {str(removal_error)}")
        
        # Fallback strategy 3: Create minimal representation
        logger.warning("Using minimal representation fallback")
        minimal_features = {}
        
        for modality, features in modality_features.items():
            try:
                if modality == failed_modality:
                    # Create minimal zero features
                    minimal_features[modality] = FeatureRepresentation(
                        features=torch.zeros(self.config.feature_dimension_target),
                        metadata={
                            "fallback": True,
                            "method": "minimal_zero_fill",
                            "original_error": str(error)
                        }
                    )
                else:
                    minimal_features[modality] = features
                    
            except Exception as minimal_error:
                logger.error(f"Minimal fallback failed for '{modality}': {str(minimal_error)}")
        
        return minimal_features
    
    def _simple_alignment_fallback(
        self,
        failed_repr: FeatureRepresentation,
        reference_repr: FeatureRepresentation
    ) -> FeatureRepresentation:
        """
        Simple alignment fallback using basic resizing and padding.
        
        Args:
            failed_repr: The feature representation that failed alignment
            reference_repr: Reference feature representation to align to
            
        Returns:
            Aligned feature representation using simple methods
        """
        try:
            # Get target shape from reference
            target_shape = reference_repr.features.shape
            
            # Simple resize using padding/truncation
            aligned_features = self._resize_tensor_to_shape(
                failed_repr.features, target_shape
            )
            
            # Create aligned representation
            aligned_repr = FeatureRepresentation(
                features=aligned_features,
                timestamps=reference_repr.timestamps,  # Use reference timestamps
                spatial_coordinates=reference_repr.spatial_coordinates,
                quality_mask=None,  # No quality mask for fallback
                metadata={
                    **failed_repr.metadata,
                    "alignment_fallback": True,
                    "fallback_method": "simple_resize"
                }
            )
            
            return aligned_repr
            
        except Exception as e:
            logger.error(f"Simple alignment fallback failed: {str(e)}")
            # Return zero-filled features as last resort
            return FeatureRepresentation(
                features=torch.zeros_like(reference_repr.features),
                timestamps=reference_repr.timestamps,
                spatial_coordinates=reference_repr.spatial_coordinates,
                metadata={
                    "alignment_fallback": True,
                    "fallback_method": "zero_fill",
                    "fallback_error": str(e)
                }
            )


class InputTensorBuilder:
    """
    Builder class for creating MONAI-compatible input tensors from aligned features.
    
    This class handles the final step of converting aligned multi-modal features
    into tensors that can be directly used with MONAI models.
    """
    
    def __init__(self, target_shape: Tuple[int, ...] = (96, 96, 96)):
        """
        Initialize the tensor builder.
        
        Args:
            target_shape: Target spatial shape for MONAI models
        """
        self.target_shape = target_shape
        self.logger = get_logger(__name__)
    
    def build_monai_tensor(
        self,
        aligned_features: Dict[str, torch.Tensor],
        metadata: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """
        Build a MONAI-compatible tensor from aligned features.
        
        This method implements requirement 2.3: "WHEN data fusion occurs THEN the system 
        SHALL combine imaging and wearable data into a unified input for the AI model"
        
        Args:
            aligned_features: Dictionary of aligned feature tensors
            metadata: Optional metadata for the tensor
            
        Returns:
            MONAI-compatible tensor with proper dimensions and metadata
        """
        if not aligned_features:
            raise ValueError("No aligned features provided")
        
        # Separate imaging and wearable modalities for specialized processing
        imaging_modalities = {}
        wearable_modalities = {}
        
        for modality, features in aligned_features.items():
            if modality.lower() in ['mri', 'ct', 'ultrasound', 'imaging']:
                imaging_modalities[modality] = features
            else:
                wearable_modalities[modality] = features
        
        # Process imaging modalities
        imaging_tensor = None
        if imaging_modalities:
            imaging_tensor = self._process_imaging_modalities(imaging_modalities)
        
        # Process wearable modalities
        wearable_tensor = None
        if wearable_modalities:
            wearable_tensor = self._process_wearable_modalities(wearable_modalities)
        
        # Combine imaging and wearable data
        if imaging_tensor is not None and wearable_tensor is not None:
            # Both imaging and wearable data available
            unified_tensor = self._combine_imaging_wearable(imaging_tensor, wearable_tensor)
        elif imaging_tensor is not None:
            # Only imaging data available
            unified_tensor = imaging_tensor
        elif wearable_tensor is not None:
            # Only wearable data available
            unified_tensor = wearable_tensor
        else:
            # Fallback: standardize all features to target shape
            standardized_features = {}
            for modality, features in aligned_features.items():
                standardized_features[modality] = self._standardize_to_target_shape(
                    features, modality
                )
            
            feature_tensors = list(standardized_features.values())
            reference_shape = feature_tensors[0].shape
            
            for i, tensor in enumerate(feature_tensors):
                if tensor.shape != reference_shape:
                    feature_tensors[i] = self._resize_to_reference(tensor, reference_shape)
            
            # Concatenate along appropriate dimension
            if len(reference_shape) >= 4:  # Spatial data
                unified_tensor = torch.cat(feature_tensors, dim=0)  # Channel dimension
            else:
                unified_tensor = torch.cat(feature_tensors, dim=-1)  # Feature dimension
        
        # Create MetaTensor with metadata if available
        if metadata:
            try:
                from monai.data import MetaTensor
                unified_tensor = MetaTensor(unified_tensor, meta=metadata)
            except ImportError:
                logger.warning("MONAI MetaTensor not available, returning regular tensor")
        
        return unified_tensor
    
    def _process_imaging_modalities(self, imaging_modalities: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Process imaging modalities into a unified tensor."""
        processed_tensors = []
        
        for modality, features in imaging_modalities.items():
            # Ensure imaging features are 3D spatial volumes
            if features.dim() < 3:
                # Expand lower dimensional features to 3D
                processed = self._expand_to_3d_volume(features)
            elif features.dim() == 3:
                # Already 3D, resize to target shape
                processed = self._resize_3d_volume(features)
            elif features.dim() == 4:
                # 4D tensor (C, H, W, D) - process each channel
                processed_channels = []
                for c in range(features.shape[0]):
                    channel_volume = self._resize_3d_volume(features[c])
                    processed_channels.append(channel_volume)
                processed = torch.stack(processed_channels, dim=0)
            else:
                # Higher dimensions - flatten and expand
                flattened = features.flatten()
                processed = self._expand_to_3d_volume(flattened)
            
            processed_tensors.append(processed)
        
        # Concatenate imaging modalities along channel dimension
        if len(processed_tensors) == 1:
            # Single modality - add channel dimension if needed
            result = processed_tensors[0]
            if result.dim() == 3:
                result = result.unsqueeze(0)  # Add channel dimension
        else:
            # Multiple modalities - stack along channel dimension
            # Ensure all tensors have same spatial dimensions
            reference_shape = processed_tensors[0].shape[-3:]
            aligned_tensors = []
            
            for tensor in processed_tensors:
                if tensor.dim() == 3:
                    tensor = tensor.unsqueeze(0)  # Add channel dimension
                
                # Resize spatial dimensions if needed
                if tensor.shape[-3:] != reference_shape:
                    resized_channels = []
                    for c in range(tensor.shape[0]):
                        resized_channel = self._resize_3d_volume(tensor[c])
                        resized_channels.append(resized_channel)
                    tensor = torch.stack(resized_channels, dim=0)
                
                aligned_tensors.append(tensor)
            
            result = torch.cat(aligned_tensors, dim=0)  # Concatenate along channel dimension
        
        return result
    
    def _process_wearable_modalities(self, wearable_modalities: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Process wearable modalities into a unified tensor."""
        processed_tensors = []
        
        for modality, features in wearable_modalities.items():
            # Wearable data is typically temporal - expand to spatial volume
            if features.dim() == 1:
                # 1D time series or feature vector
                processed = self._expand_to_3d_volume(features)
            elif features.dim() == 2:
                # 2D time series (time x channels) - flatten and expand
                flattened = features.flatten()
                processed = self._expand_to_3d_volume(flattened)
            else:
                # Higher dimensions - flatten and expand
                flattened = features.flatten()
                processed = self._expand_to_3d_volume(flattened)
            
            processed_tensors.append(processed)
        
        # Combine wearable modalities
        if len(processed_tensors) == 1:
            result = processed_tensors[0]
            if result.dim() == 3:
                result = result.unsqueeze(0)  # Add channel dimension
        else:
            # Stack wearable modalities along channel dimension
            aligned_tensors = []
            reference_shape = processed_tensors[0].shape
            
            for tensor in processed_tensors:
                if tensor.shape != reference_shape:
                    tensor = self._resize_3d_volume(tensor)
                aligned_tensors.append(tensor.unsqueeze(0))  # Add channel dimension
            
            result = torch.cat(aligned_tensors, dim=0)
        
        return result
    
    def _combine_imaging_wearable(
        self, 
        imaging_tensor: torch.Tensor, 
        wearable_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Combine imaging and wearable tensors into unified multi-modal input.
        
        This implements the core requirement 2.3 functionality.
        """
        # Ensure both tensors have channel dimensions
        if imaging_tensor.dim() == 3:
            imaging_tensor = imaging_tensor.unsqueeze(0)
        if wearable_tensor.dim() == 3:
            wearable_tensor = wearable_tensor.unsqueeze(0)
        
        # Ensure spatial dimensions match
        imaging_spatial = imaging_tensor.shape[-3:]
        wearable_spatial = wearable_tensor.shape[-3:]
        
        if imaging_spatial != wearable_spatial:
            # Resize wearable to match imaging spatial dimensions
            target_shape = imaging_spatial
            resized_wearable_channels = []
            
            for c in range(wearable_tensor.shape[0]):
                resized_channel = self._resize_3d_volume_to_shape(
                    wearable_tensor[c], target_shape
                )
                resized_wearable_channels.append(resized_channel)
            
            wearable_tensor = torch.stack(resized_wearable_channels, dim=0)
        
        # Concatenate along channel dimension
        combined_tensor = torch.cat([imaging_tensor, wearable_tensor], dim=0)
        
        return combined_tensor
    
    def _expand_to_3d_volume(self, features: torch.Tensor) -> torch.Tensor:
        """Expand any dimensional features to 3D volume."""
        if features.dim() == 3 and features.shape == self.target_shape:
            return features
        
        # Flatten and expand to target volume
        flattened = features.flatten()
        target_volume = np.prod(self.target_shape)
        
        if len(flattened) >= target_volume:
            # Truncate and reshape
            truncated = flattened[:target_volume]
            return truncated.reshape(self.target_shape)
        else:
            # Repeat features to fill volume
            repeat_factor = int(np.ceil(target_volume / len(flattened)))
            repeated = flattened.repeat(repeat_factor)[:target_volume]
            return repeated.reshape(self.target_shape)
    
    def _resize_3d_volume_to_shape(
        self, 
        volume: torch.Tensor, 
        target_shape: Tuple[int, int, int]
    ) -> torch.Tensor:
        """Resize 3D volume to specific target shape."""
        if volume.shape == target_shape:
            return volume
        
        import torch.nn.functional as F
        
        # Add batch and channel dimensions for interpolation
        volume_with_dims = volume.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W, D)
        
        # Resize using trilinear interpolation
        resized = F.interpolate(
            volume_with_dims,
            size=target_shape,
            mode='trilinear',
            align_corners=False
        )
        
        # Remove batch and channel dimensions
        return resized.squeeze(0).squeeze(0)
    
    def build_batch_tensor(
        self,
        batch_features: List[Dict[str, torch.Tensor]],
        metadata_list: Optional[List[Dict[str, Any]]] = None
    ) -> torch.Tensor:
        """
        Build a batch tensor from multiple aligned feature sets.
        
        Args:
            batch_features: List of aligned feature dictionaries
            metadata_list: Optional list of metadata dictionaries
            
        Returns:
            Batched MONAI-compatible tensor
        """
        if not batch_features:
            raise ValueError("No batch features provided")
        
        # Build individual tensors
        individual_tensors = []
        for i, features in enumerate(batch_features):
            metadata = metadata_list[i] if metadata_list else None
            tensor = self.build_monai_tensor(features, metadata)
            
            # Ensure tensor has batch dimension
            if tensor.dim() == 4:  # (C, H, W, D)
                tensor = tensor.unsqueeze(0)  # (1, C, H, W, D)
            
            individual_tensors.append(tensor)
        
        # Stack into batch
        batch_tensor = torch.cat(individual_tensors, dim=0)
        
        return batch_tensor
    
    def _standardize_to_target_shape(
        self,
        features: torch.Tensor,
        modality: str
    ) -> torch.Tensor:
        """Standardize feature tensor to target shape."""
        # Handle different input dimensionalities
        if features.dim() == 1:
            # 1D feature vector - expand to 3D volume
            return self._expand_1d_to_3d(features)
        elif features.dim() == 2:
            # 2D features - could be time series or 2D image
            return self._expand_2d_to_3d(features)
        elif features.dim() == 3:
            # 3D volume - resize to target shape
            return self._resize_3d_volume(features)
        elif features.dim() == 4:
            # 4D tensor (C, H, W, D) - flatten channels and resize spatial dimensions
            c, h, w, d = features.shape
            # Flatten to 3D by combining channels into the first spatial dimension
            flattened_3d = features.view(c * h, w, d)
            return self._resize_3d_volume(flattened_3d)
        else:
            # Higher dimensions - flatten and expand
            flattened = features.flatten()
            return self._expand_1d_to_3d(flattened)
    
    def _expand_1d_to_3d(self, features: torch.Tensor) -> torch.Tensor:
        """Expand 1D feature vector to 3D volume."""
        target_volume = np.prod(self.target_shape)
        
        if len(features) >= target_volume:
            # Truncate and reshape
            truncated = features[:target_volume]
            return truncated.reshape(self.target_shape)
        else:
            # Repeat features to fill volume
            repeat_factor = int(np.ceil(target_volume / len(features)))
            repeated = features.repeat(repeat_factor)[:target_volume]
            return repeated.reshape(self.target_shape)
    
    def _expand_2d_to_3d(self, features: torch.Tensor) -> torch.Tensor:
        """Expand 2D features to 3D volume."""
        h, w = features.shape
        target_h, target_w, target_d = self.target_shape
        
        # Resize 2D to target H, W using interpolation
        if (h, w) != (target_h, target_w):
            # Use torch.nn.functional.interpolate for 2D resizing
            import torch.nn.functional as F
            features_resized = F.interpolate(
                features.unsqueeze(0).unsqueeze(0),  # Add batch and channel dims
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)  # Remove batch and channel dims
        else:
            features_resized = features
        
        # Repeat along depth dimension
        features_3d = features_resized.unsqueeze(-1).repeat(1, 1, target_d)
        
        return features_3d
    
    def _resize_3d_volume(self, features: torch.Tensor) -> torch.Tensor:
        """Resize 3D volume to target shape."""
        if features.shape == self.target_shape:
            return features
        
        # Use torch.nn.functional.interpolate for 3D resizing
        import torch.nn.functional as F
        
        # Add batch and channel dimensions for interpolation
        features_with_dims = features.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W, D)
        
        # Resize using trilinear interpolation
        resized = F.interpolate(
            features_with_dims,
            size=self.target_shape,
            mode='trilinear',
            align_corners=False
        )
        
        # Remove batch and channel dimensions
        return resized.squeeze(0).squeeze(0)
    
    def _resize_4d_tensor(self, features: torch.Tensor) -> torch.Tensor:
        """Resize 4D tensor (C, H, W, D) to target shape."""
        c = features.shape[0]
        spatial_shape = features.shape[1:]
        
        if spatial_shape == self.target_shape:
            return features
        
        # Use torch.nn.functional.interpolate for 4D tensor resizing
        import torch.nn.functional as F
        
        # Add batch dimension for interpolation
        features_with_batch = features.unsqueeze(0)  # (1, C, H, W, D)
        
        # Resize using trilinear interpolation
        resized = F.interpolate(
            features_with_batch,
            size=self.target_shape,
            mode='trilinear',
            align_corners=False
        )
        
        # Remove batch dimension
        return resized.squeeze(0)
    
    def _resize_to_reference(
        self,
        tensor: torch.Tensor,
        reference_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """Resize tensor to match reference shape."""
        if tensor.shape == reference_shape:
            return tensor
        
        # Handle different dimensionalities
        if len(reference_shape) >= 3:
            # Spatial data - use MONAI transforms
            target_spatial = reference_shape[-3:]
            resize_transform = ResizeWithPadOrCrop(spatial_size=target_spatial)
            
            if tensor.dim() == len(reference_shape):
                return resize_transform(tensor)
            else:
                # Add or remove dimensions as needed
                while tensor.dim() < len(reference_shape):
                    tensor = tensor.unsqueeze(0)
                while tensor.dim() > len(reference_shape):
                    tensor = tensor.squeeze(0)
                return resize_transform(tensor)
        else:
            # Non-spatial data - pad or truncate
            target_size = np.prod(reference_shape)
            flattened = tensor.flatten()
            
            if len(flattened) > target_size:
                resized = flattened[:target_size]
            else:
                padding = torch.zeros(target_size - len(flattened), dtype=tensor.dtype)
                resized = torch.cat([flattened, padding])
            
            return resized.reshape(reference_shape)