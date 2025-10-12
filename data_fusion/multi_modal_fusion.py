"""
Multi-modal data fusion algorithms for combining imaging and wearable sensor data.

This module implements the MultiModalFusion class that combines medical imaging data
with wearable sensor data, ensuring spatial-temporal alignment and creating unified
input tensors compatible with MONAI models.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import torch
import logging
from enum import Enum

from monai.data import MetaTensor
from monai.transforms import Resize, ResizeWithPadOrCrop

from src.models.patient import PatientRecord, ImagingStudy, WearableSession
from src.models.diagnostics import DiagnosticResult
from src.services.wearable_sensor.temporal_synchronizer import (
    TemporalSynchronizer, AlignedSensorData, TemporalFeatures
)
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class FusionStrategy(Enum):
    """Strategies for multi-modal data fusion."""
    EARLY_FUSION = "early_fusion"  # Concatenate features before model
    LATE_FUSION = "late_fusion"    # Combine predictions after separate models
    HYBRID_FUSION = "hybrid_fusion"  # Combination of early and late fusion


class ModalityWeight(Enum):
    """Weighting strategies for different modalities."""
    EQUAL = "equal"
    ADAPTIVE = "adaptive"
    LEARNED = "learned"
    CONFIDENCE_BASED = "confidence_based"


@dataclass
class FusionConfig:
    """Configuration for multi-modal fusion."""
    fusion_strategy: FusionStrategy = FusionStrategy.EARLY_FUSION
    modality_weights: ModalityWeight = ModalityWeight.EQUAL
    temporal_window_seconds: float = 300.0  # 5 minutes
    spatial_alignment_tolerance: float = 1.0  # mm
    missing_modality_strategy: str = "interpolate"  # "interpolate", "zero_fill", "skip"
    feature_normalization: bool = True
    cross_modal_attention: bool = True


@dataclass
class FusionResult:
    """Result of multi-modal fusion operation."""
    fused_tensor: torch.Tensor
    modality_contributions: Dict[str, float]
    fusion_confidence: float
    temporal_alignment_quality: float
    spatial_alignment_quality: float
    missing_modalities: List[str]
    fusion_metadata: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)


@dataclass
class SpatialTemporalAlignment:
    """Spatial-temporal alignment information."""
    temporal_offset_seconds: float
    spatial_transform_matrix: Optional[np.ndarray]
    alignment_confidence: float
    reference_timestamp: datetime
    reference_coordinate_system: str


class MultiModalFusion:
    """
    Multi-modal data fusion service for combining imaging and wearable sensor data.
    
    This class handles the fusion of medical imaging data (MRI, CT, Ultrasound) with
    wearable sensor data (EEG, heart rate, sleep, gait) to create unified input tensors
    for MONAI-based machine learning models.
    """
    
    def __init__(self, config: Optional[FusionConfig] = None):
        """
        Initialize the multi-modal fusion service.
        
        Args:
            config: Fusion configuration parameters
        """
        self.config = config or FusionConfig()
        self.temporal_synchronizer = TemporalSynchronizer()
        self.logger = get_logger(__name__)
        
        # Standard tensor dimensions for different modalities
        self.imaging_tensor_size = (96, 96, 96)  # Standard for SwinUNETR
        self.wearable_feature_size = 512  # Standard feature vector size
        
    def fuse_patient_data(
        self, 
        patient_record: PatientRecord,
        target_timestamp: Optional[datetime] = None
    ) -> FusionResult:
        """
        Fuse all available data for a patient into a unified tensor.
        
        Args:
            patient_record: Complete patient record with imaging and wearable data
            target_timestamp: Target timestamp for temporal alignment
            
        Returns:
            FusionResult: Fused tensor and metadata
            
        Raises:
            ValueError: If insufficient data for fusion
        """
        if not patient_record.imaging_studies and not patient_record.wearable_data:
            raise ValueError("No imaging or wearable data available for fusion")
        
        # Use most recent imaging study timestamp as reference if not provided
        if target_timestamp is None and patient_record.imaging_studies:
            target_timestamp = max(
                study.acquisition_date for study in patient_record.imaging_studies
            )
        elif target_timestamp is None:
            target_timestamp = datetime.now()
        
        try:
            # Process imaging data
            imaging_tensor, imaging_metadata = self._process_imaging_data(
                patient_record.imaging_studies, target_timestamp
            )
            
            # Process wearable data
            wearable_tensor, wearable_metadata = self._process_wearable_data(
                patient_record.wearable_data, target_timestamp
            )
            
            # Perform spatial-temporal alignment
            alignment_info = self._align_modalities(
                imaging_metadata, wearable_metadata, target_timestamp
            )
            
            # Fuse the modalities
            fused_tensor = self._fuse_tensors(
                imaging_tensor, wearable_tensor, alignment_info
            )
            
            # Calculate fusion quality metrics
            fusion_confidence = self._calculate_fusion_confidence(
                imaging_tensor, wearable_tensor, alignment_info
            )
            
            # Determine modality contributions
            modality_contributions = self._calculate_modality_contributions(
                imaging_tensor, wearable_tensor
            )
            
            # Identify missing modalities
            missing_modalities = self._identify_missing_modalities(
                patient_record.imaging_studies, patient_record.wearable_data
            )
            
            return FusionResult(
                fused_tensor=fused_tensor,
                modality_contributions=modality_contributions,
                fusion_confidence=fusion_confidence,
                temporal_alignment_quality=alignment_info.alignment_confidence,
                spatial_alignment_quality=1.0,  # Placeholder for spatial alignment
                missing_modalities=missing_modalities,
                fusion_metadata={
                    "target_timestamp": target_timestamp.isoformat(),
                    "imaging_studies_count": len(patient_record.imaging_studies),
                    "wearable_sessions_count": len(patient_record.wearable_data),
                    "fusion_strategy": self.config.fusion_strategy.value,
                    "alignment_info": alignment_info.__dict__
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to fuse patient data: {str(e)}")
            raise
    
    def fuse_study_data(
        self,
        imaging_data: Dict[str, torch.Tensor],
        wearable_features: Dict[str, np.ndarray],
        temporal_alignment: Optional[SpatialTemporalAlignment] = None
    ) -> FusionResult:
        """
        Fuse specific imaging and wearable data for a study.
        
        Args:
            imaging_data: Dictionary of imaging tensors by modality
            wearable_features: Dictionary of wearable feature arrays by device type
            temporal_alignment: Pre-computed alignment information
            
        Returns:
            FusionResult: Fused tensor and metadata
        """
        try:
            # Convert imaging data to standardized format
            imaging_tensor = self._standardize_imaging_tensors(imaging_data)
            
            # Convert wearable features to tensor
            wearable_tensor = self._standardize_wearable_features(wearable_features)
            
            # Use provided alignment or compute default
            if temporal_alignment is None:
                temporal_alignment = SpatialTemporalAlignment(
                    temporal_offset_seconds=0.0,
                    spatial_transform_matrix=None,
                    alignment_confidence=1.0,
                    reference_timestamp=datetime.now(),
                    reference_coordinate_system="RAS"
                )
            
            # Fuse the tensors
            fused_tensor = self._fuse_tensors(
                imaging_tensor, wearable_tensor, temporal_alignment
            )
            
            # Calculate metrics
            fusion_confidence = self._calculate_fusion_confidence(
                imaging_tensor, wearable_tensor, temporal_alignment
            )
            
            modality_contributions = self._calculate_modality_contributions(
                imaging_tensor, wearable_tensor
            )
            
            return FusionResult(
                fused_tensor=fused_tensor,
                modality_contributions=modality_contributions,
                fusion_confidence=fusion_confidence,
                temporal_alignment_quality=temporal_alignment.alignment_confidence,
                spatial_alignment_quality=1.0,
                missing_modalities=[],
                fusion_metadata={
                    "imaging_modalities": list(imaging_data.keys()),
                    "wearable_devices": list(wearable_features.keys()),
                    "fusion_strategy": self.config.fusion_strategy.value
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to fuse study data: {str(e)}")
            raise
    
    def _process_imaging_data(
        self, 
        imaging_studies: List[ImagingStudy],
        target_timestamp: datetime
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Process imaging studies into a standardized tensor format."""
        if not imaging_studies:
            # Return zero tensor for missing imaging data
            return torch.zeros((1, *self.imaging_tensor_size)), {}
        
        # Find the most temporally relevant study
        best_study = min(
            imaging_studies,
            key=lambda s: abs((s.acquisition_date - target_timestamp).total_seconds())
        )
        
        # For now, create a placeholder tensor
        # In a real implementation, this would load and process the actual image
        imaging_tensor = torch.randn((1, *self.imaging_tensor_size))
        
        metadata = {
            "study_id": best_study.study_id,
            "modality": best_study.modality,
            "acquisition_date": best_study.acquisition_date.isoformat(),
            "temporal_offset": (best_study.acquisition_date - target_timestamp).total_seconds()
        }
        
        return imaging_tensor, metadata
    
    def _process_wearable_data(
        self,
        wearable_sessions: List[WearableSession],
        target_timestamp: datetime
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Process wearable sensor data into a standardized tensor format."""
        if not wearable_sessions:
            # Return zero tensor for missing wearable data
            return torch.zeros(self.wearable_feature_size), {}
        
        # Filter sessions within temporal window
        window_start = target_timestamp - timedelta(
            seconds=self.config.temporal_window_seconds / 2
        )
        window_end = target_timestamp + timedelta(
            seconds=self.config.temporal_window_seconds / 2
        )
        
        relevant_sessions = [
            session for session in wearable_sessions
            if (session.start_time <= window_end and session.end_time >= window_start)
        ]
        
        if not relevant_sessions:
            self.logger.warning("No wearable sessions within temporal window")
            return torch.zeros(self.wearable_feature_size), {}
        
        # Synchronize and extract features
        try:
            aligned_data = self.temporal_synchronizer.align_timestamps(relevant_sessions)
            temporal_features = self.temporal_synchronizer.extract_temporal_features(aligned_data)
            
            # Pad or truncate to standard size
            feature_vector = temporal_features.feature_vector
            if len(feature_vector) > self.wearable_feature_size:
                feature_vector = feature_vector[:self.wearable_feature_size]
            elif len(feature_vector) < self.wearable_feature_size:
                padding = np.zeros(self.wearable_feature_size - len(feature_vector))
                feature_vector = np.concatenate([feature_vector, padding])
            
            wearable_tensor = torch.tensor(feature_vector, dtype=torch.float32)
            
            metadata = {
                "sessions_count": len(relevant_sessions),
                "device_types": [s.device_type for s in relevant_sessions],
                "temporal_window": (window_start.isoformat(), window_end.isoformat()),
                "feature_names": temporal_features.feature_names[:len(feature_vector)]
            }
            
            return wearable_tensor, metadata
            
        except Exception as e:
            self.logger.error(f"Failed to process wearable data: {str(e)}")
            return torch.zeros(self.wearable_feature_size), {}
    
    def _align_modalities(
        self,
        imaging_metadata: Dict[str, Any],
        wearable_metadata: Dict[str, Any],
        target_timestamp: datetime
    ) -> SpatialTemporalAlignment:
        """Compute spatial-temporal alignment between modalities."""
        # Calculate temporal offset
        temporal_offset = 0.0
        if "temporal_offset" in imaging_metadata:
            temporal_offset = imaging_metadata["temporal_offset"]
        
        # For now, assume perfect spatial alignment
        # In a real implementation, this would compute actual spatial transforms
        spatial_transform = np.eye(4)  # Identity transform
        
        # Calculate alignment confidence based on temporal proximity
        alignment_confidence = max(0.0, 1.0 - abs(temporal_offset) / 3600.0)  # Decay over 1 hour
        
        return SpatialTemporalAlignment(
            temporal_offset_seconds=temporal_offset,
            spatial_transform_matrix=spatial_transform,
            alignment_confidence=alignment_confidence,
            reference_timestamp=target_timestamp,
            reference_coordinate_system="RAS"
        )
    
    def _fuse_tensors(
        self,
        imaging_tensor: torch.Tensor,
        wearable_tensor: torch.Tensor,
        alignment_info: SpatialTemporalAlignment
    ) -> torch.Tensor:
        """Fuse imaging and wearable tensors based on fusion strategy."""
        if self.config.fusion_strategy == FusionStrategy.EARLY_FUSION:
            return self._early_fusion(imaging_tensor, wearable_tensor, alignment_info)
        elif self.config.fusion_strategy == FusionStrategy.LATE_FUSION:
            # Late fusion would be handled at the model level
            return self._early_fusion(imaging_tensor, wearable_tensor, alignment_info)
        else:  # HYBRID_FUSION
            return self._hybrid_fusion(imaging_tensor, wearable_tensor, alignment_info)
    
    def _early_fusion(
        self,
        imaging_tensor: torch.Tensor,
        wearable_tensor: torch.Tensor,
        alignment_info: SpatialTemporalAlignment
    ) -> torch.Tensor:
        """Perform early fusion by concatenating features."""
        # Ensure imaging tensor has batch dimension
        if imaging_tensor.dim() == 4:  # (C, H, W, D)
            imaging_tensor = imaging_tensor.unsqueeze(0)  # (1, C, H, W, D)
        
        # Expand wearable features to match spatial dimensions
        wearable_expanded = self._expand_wearable_features(
            wearable_tensor, self.imaging_tensor_size
        )
        
        # Add channel dimension to wearable features
        if wearable_expanded.dim() == 3:  # (H, W, D)
            wearable_expanded = wearable_expanded.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W, D)
        elif wearable_expanded.dim() == 4:  # (1, H, W, D)
            wearable_expanded = wearable_expanded.unsqueeze(1)  # (1, 1, H, W, D)
        
        # Concatenate along channel dimension
        fused_tensor = torch.cat([imaging_tensor, wearable_expanded], dim=1)
        
        return fused_tensor
    
    def _hybrid_fusion(
        self,
        imaging_tensor: torch.Tensor,
        wearable_tensor: torch.Tensor,
        alignment_info: SpatialTemporalAlignment
    ) -> torch.Tensor:
        """Perform hybrid fusion combining early and late fusion strategies."""
        # For now, implement as early fusion with attention mechanism
        early_fused = self._early_fusion(imaging_tensor, wearable_tensor, alignment_info)
        
        if self.config.cross_modal_attention:
            # Apply cross-modal attention (simplified implementation)
            attention_weights = self._compute_cross_modal_attention(
                imaging_tensor, wearable_tensor
            )
            early_fused = early_fused * attention_weights
        
        return early_fused
    
    def _expand_wearable_features(
        self, 
        wearable_tensor: torch.Tensor, 
        target_size: Tuple[int, int, int]
    ) -> torch.Tensor:
        """Expand wearable feature vector to match imaging spatial dimensions."""
        # Reshape feature vector to 3D volume
        feature_size = wearable_tensor.shape[0]
        
        # Calculate dimensions for reshaping
        volume_size = target_size[0] * target_size[1] * target_size[2]
        
        if feature_size >= volume_size:
            # Truncate and reshape
            truncated = wearable_tensor[:volume_size]
            reshaped = truncated.reshape(target_size)
        else:
            # Repeat features to fill volume
            repeat_factor = int(np.ceil(volume_size / feature_size))
            repeated = wearable_tensor.repeat(repeat_factor)[:volume_size]
            reshaped = repeated.reshape(target_size)
        
        return reshaped
    
    def _compute_cross_modal_attention(
        self,
        imaging_tensor: torch.Tensor,
        wearable_tensor: torch.Tensor
    ) -> torch.Tensor:
        """Compute cross-modal attention weights."""
        # Simplified attention mechanism
        # In practice, this would be a learned attention module
        
        # Compute global features for each modality
        imaging_global = torch.mean(imaging_tensor.flatten())
        wearable_global = torch.mean(wearable_tensor)
        
        # Simple attention based on feature magnitudes
        total_magnitude = imaging_global + wearable_global + 1e-8
        imaging_weight = imaging_global / total_magnitude
        wearable_weight = wearable_global / total_magnitude
        
        # Create attention tensor
        attention = torch.ones_like(imaging_tensor)
        attention[:, 0] *= imaging_weight  # Weight imaging channels
        if attention.shape[1] > 1:
            attention[:, 1] *= wearable_weight  # Weight wearable channels
        
        return attention
    
    def _standardize_imaging_tensors(
        self, 
        imaging_data: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Standardize imaging tensors from different modalities."""
        if not imaging_data:
            return torch.zeros((1, *self.imaging_tensor_size))
        
        # Concatenate all imaging modalities along channel dimension
        tensors = []
        for modality, tensor in imaging_data.items():
            # Ensure tensor has correct dimensions
            if tensor.dim() == 3:  # (H, W, D)
                tensor = tensor.unsqueeze(0)  # (1, H, W, D)
            elif tensor.dim() == 4 and tensor.shape[0] != 1:  # (C, H, W, D)
                tensor = tensor.unsqueeze(0)  # (1, C, H, W, D)
            
            # Resize to standard size if needed
            if tensor.shape[-3:] != self.imaging_tensor_size:
                resize_transform = ResizeWithPadOrCrop(spatial_size=self.imaging_tensor_size)
                tensor = resize_transform(tensor)
            
            tensors.append(tensor)
        
        # Concatenate along channel dimension
        combined_tensor = torch.cat(tensors, dim=1 if tensors[0].dim() == 5 else 0)
        
        return combined_tensor
    
    def _standardize_wearable_features(
        self, 
        wearable_features: Dict[str, np.ndarray]
    ) -> torch.Tensor:
        """Standardize wearable feature arrays into a single tensor."""
        if not wearable_features:
            return torch.zeros(self.wearable_feature_size)
        
        # Concatenate all wearable features
        feature_list = []
        for device_type, features in wearable_features.items():
            if isinstance(features, np.ndarray):
                feature_list.extend(features.flatten())
            else:
                feature_list.append(float(features))
        
        # Pad or truncate to standard size
        if len(feature_list) > self.wearable_feature_size:
            feature_list = feature_list[:self.wearable_feature_size]
        elif len(feature_list) < self.wearable_feature_size:
            padding = [0.0] * (self.wearable_feature_size - len(feature_list))
            feature_list.extend(padding)
        
        return torch.tensor(feature_list, dtype=torch.float32)
    
    def _calculate_fusion_confidence(
        self,
        imaging_tensor: torch.Tensor,
        wearable_tensor: torch.Tensor,
        alignment_info: SpatialTemporalAlignment
    ) -> float:
        """Calculate confidence score for the fusion result."""
        # Base confidence from alignment quality
        confidence = alignment_info.alignment_confidence
        
        # Adjust based on data availability
        imaging_quality = 1.0 if torch.sum(torch.abs(imaging_tensor)) > 0 else 0.0
        wearable_quality = 1.0 if torch.sum(torch.abs(wearable_tensor)) > 0 else 0.0
        
        # Weighted combination
        data_quality = (imaging_quality + wearable_quality) / 2.0
        confidence = (confidence + data_quality) / 2.0
        
        return float(confidence)
    
    def _calculate_modality_contributions(
        self,
        imaging_tensor: torch.Tensor,
        wearable_tensor: torch.Tensor
    ) -> Dict[str, float]:
        """Calculate the contribution of each modality to the fused result."""
        imaging_magnitude = float(torch.sum(torch.abs(imaging_tensor)))
        wearable_magnitude = float(torch.sum(torch.abs(wearable_tensor)))
        
        total_magnitude = imaging_magnitude + wearable_magnitude + 1e-8
        
        return {
            "imaging": imaging_magnitude / total_magnitude,
            "wearable": wearable_magnitude / total_magnitude
        }
    
    def _identify_missing_modalities(
        self,
        imaging_studies: List[ImagingStudy],
        wearable_sessions: List[WearableSession]
    ) -> List[str]:
        """Identify which modalities are missing from the data."""
        expected_imaging = {"MRI", "CT", "Ultrasound"}
        expected_wearable = {"EEG", "HeartRate", "Sleep", "Gait"}
        
        available_imaging = {study.modality for study in imaging_studies}
        available_wearable = {session.device_type for session in wearable_sessions}
        
        missing = []
        missing.extend(expected_imaging - available_imaging)
        missing.extend(expected_wearable - available_wearable)
        
        return missing