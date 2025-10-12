"""
Diagnostic result and metrics data models for NeuroDx-MultiModal system.

This module defines data structures for diagnostic results, model metrics,
and performance measurements used in neurodegenerative disease detection.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Literal, Union
import numpy as np
from enum import Enum


class DiagnosticConfidence(Enum):
    """Confidence levels for diagnostic results."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"


class DiseaseStage(Enum):
    """Neurodegenerative disease stages."""
    HEALTHY = "healthy"
    MILD_COGNITIVE_IMPAIRMENT = "mci"
    EARLY_STAGE = "early"
    MODERATE_STAGE = "moderate"
    SEVERE_STAGE = "severe"


@dataclass
class ModelMetrics:
    """
    Model performance metrics structure for Dice, Hausdorff, and AUC scores.
    
    This class stores various metrics used to evaluate the performance of
    the MONAI SwinUNETR model for segmentation and classification tasks.
    """
    # Required metrics
    dice_score: float
    hausdorff_distance: float
    auc_score: float
    
    # Optional segmentation metrics
    jaccard_index: Optional[float] = None
    surface_distance_mean: Optional[float] = None
    surface_distance_std: Optional[float] = None
    
    # Optional classification metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    specificity: Optional[float] = None
    
    # Multi-class metrics
    macro_auc: Optional[float] = None
    weighted_auc: Optional[float] = None
    
    # Confidence and uncertainty metrics
    prediction_entropy: Optional[float] = None
    epistemic_uncertainty: Optional[float] = None
    aleatoric_uncertainty: Optional[float] = None
    
    # Computation metadata
    computation_time_ms: Optional[float] = None
    model_version: Optional[str] = None
    
    def __post_init__(self):
        """Validate metrics after initialization."""
        self._validate_metrics()
    
    def _validate_metrics(self):
        """Validate metric values are within expected ranges."""
        # Validate Dice score (0 to 1)
        if not (0.0 <= self.dice_score <= 1.0):
            raise ValueError(f"Dice score must be between 0 and 1, got {self.dice_score}")
        
        # Validate Hausdorff distance (non-negative)
        if self.hausdorff_distance < 0:
            raise ValueError(f"Hausdorff distance must be non-negative, got {self.hausdorff_distance}")
        
        # Validate AUC score (0 to 1)
        if not (0.0 <= self.auc_score <= 1.0):
            raise ValueError(f"AUC score must be between 0 and 1, got {self.auc_score}")
        
        # Validate optional metrics if present
        percentage_metrics = [
            self.accuracy, self.precision, self.recall, 
            self.f1_score, self.specificity, self.macro_auc, self.weighted_auc
        ]
        
        for metric in percentage_metrics:
            if metric is not None and not (0.0 <= metric <= 1.0):
                raise ValueError(f"Metric must be between 0 and 1, got {metric}")
        
        # Validate Jaccard index if present
        if self.jaccard_index is not None and not (0.0 <= self.jaccard_index <= 1.0):
            raise ValueError(f"Jaccard index must be between 0 and 1, got {self.jaccard_index}")
        
        # Validate surface distances if present
        if self.surface_distance_mean is not None and self.surface_distance_mean < 0:
            raise ValueError("Surface distance mean must be non-negative")
        
        if self.surface_distance_std is not None and self.surface_distance_std < 0:
            raise ValueError("Surface distance standard deviation must be non-negative")
    
    def get_overall_performance_score(self) -> float:
        """
        Calculate an overall performance score combining multiple metrics.
        
        Returns:
            float: Overall performance score between 0 and 1
        """
        # Weighted combination of key metrics
        weights = {
            'dice': 0.3,
            'auc': 0.3,
            'accuracy': 0.2,
            'f1': 0.2
        }
        
        score = weights['dice'] * self.dice_score + weights['auc'] * self.auc_score
        
        if self.accuracy is not None:
            score += weights['accuracy'] * self.accuracy
        else:
            # Redistribute weight if accuracy is not available
            score += weights['accuracy'] * self.auc_score
        
        if self.f1_score is not None:
            score += weights['f1'] * self.f1_score
        else:
            # Redistribute weight if F1 is not available
            score += weights['f1'] * self.dice_score
        
        return min(1.0, max(0.0, score))


@dataclass
class SegmentationResult:
    """Segmentation output from the MONAI SwinUNETR model."""
    segmentation_mask: np.ndarray
    class_probabilities: Dict[str, np.ndarray]
    confidence_map: Optional[np.ndarray] = None
    uncertainty_map: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Validate segmentation result after initialization."""
        self._validate_segmentation_data()
    
    def _validate_segmentation_data(self):
        """Validate segmentation mask and probability maps."""
        if not isinstance(self.segmentation_mask, np.ndarray):
            raise ValueError("Segmentation mask must be a numpy array")
        
        if self.segmentation_mask.ndim not in [2, 3]:
            raise ValueError("Segmentation mask must be 2D or 3D")
        
        # Validate class probabilities
        if not self.class_probabilities:
            raise ValueError("Class probabilities cannot be empty")
        
        mask_shape = self.segmentation_mask.shape
        for class_name, prob_map in self.class_probabilities.items():
            if not isinstance(prob_map, np.ndarray):
                raise ValueError(f"Probability map for {class_name} must be numpy array")
            
            if prob_map.shape != mask_shape:
                raise ValueError(
                    f"Probability map shape {prob_map.shape} doesn't match "
                    f"segmentation mask shape {mask_shape}"
                )
            
            if not (0.0 <= prob_map.min() and prob_map.max() <= 1.0):
                raise ValueError(f"Probabilities for {class_name} must be between 0 and 1")
        
        # Validate optional maps
        for map_name, map_data in [("confidence", self.confidence_map), ("uncertainty", self.uncertainty_map)]:
            if map_data is not None:
                if not isinstance(map_data, np.ndarray):
                    raise ValueError(f"{map_name} map must be numpy array")
                
                if map_data.shape != mask_shape:
                    raise ValueError(
                        f"{map_name} map shape {map_data.shape} doesn't match "
                        f"segmentation mask shape {mask_shape}"
                    )


@dataclass
class ClassificationResult:
    """Classification output from the MONAI SwinUNETR model."""
    predicted_class: str
    class_probabilities: Dict[str, float]
    confidence_score: float
    disease_stage: Optional[DiseaseStage] = None
    risk_factors: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate classification result after initialization."""
        self._validate_classification_data()
    
    def _validate_classification_data(self):
        """Validate classification probabilities and confidence."""
        if not self.predicted_class:
            raise ValueError("Predicted class cannot be empty")
        
        if not self.class_probabilities:
            raise ValueError("Class probabilities cannot be empty")
        
        # Check if predicted class is in probabilities
        if self.predicted_class not in self.class_probabilities:
            raise ValueError(f"Predicted class '{self.predicted_class}' not in probabilities")
        
        # Validate probability values
        for class_name, prob in self.class_probabilities.items():
            if not (0.0 <= prob <= 1.0):
                raise ValueError(f"Probability for {class_name} must be between 0 and 1")
        
        # Check if probabilities sum to approximately 1
        prob_sum = sum(self.class_probabilities.values())
        if not (0.95 <= prob_sum <= 1.05):  # Allow small numerical errors
            raise ValueError(f"Class probabilities should sum to 1, got {prob_sum}")
        
        # Validate confidence score
        if not (0.0 <= self.confidence_score <= 1.0):
            raise ValueError(f"Confidence score must be between 0 and 1, got {self.confidence_score}")
        
        # Validate risk factors
        for factor, score in self.risk_factors.items():
            if not (0.0 <= score <= 1.0):
                raise ValueError(f"Risk factor '{factor}' score must be between 0 and 1")


@dataclass
class ExplainabilityResult:
    """Explainability visualization results from Grad-CAM and Integrated Gradients."""
    grad_cam_maps: Dict[str, np.ndarray]
    integrated_gradients: Optional[np.ndarray] = None
    attention_maps: Dict[str, np.ndarray] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate explainability data after initialization."""
        self._validate_explainability_data()
    
    def _validate_explainability_data(self):
        """Validate explainability maps and importance scores."""
        if not self.grad_cam_maps:
            raise ValueError("Grad-CAM maps cannot be empty")
        
        # Validate Grad-CAM maps
        for layer_name, cam_map in self.grad_cam_maps.items():
            if not isinstance(cam_map, np.ndarray):
                raise ValueError(f"Grad-CAM map for {layer_name} must be numpy array")
            
            if cam_map.ndim not in [2, 3]:
                raise ValueError(f"Grad-CAM map for {layer_name} must be 2D or 3D")
        
        # Validate integrated gradients if present
        if self.integrated_gradients is not None:
            if not isinstance(self.integrated_gradients, np.ndarray):
                raise ValueError("Integrated gradients must be numpy array")
        
        # Validate feature importance scores
        for feature, importance in self.feature_importance.items():
            if not isinstance(importance, (int, float)):
                raise ValueError(f"Feature importance for {feature} must be numeric")


@dataclass
class DiagnosticResult:
    """
    Complete diagnostic result with segmentation and classification outputs.
    
    This is the main result structure returned by the NeuroDx-MultiModal system
    containing all diagnostic information, metrics, and explainability data.
    """
    patient_id: str
    study_ids: List[str]
    timestamp: datetime
    
    # Core results
    segmentation_result: SegmentationResult
    classification_result: ClassificationResult
    metrics: ModelMetrics
    
    # Explainability and visualization
    explainability_maps: Optional[ExplainabilityResult] = None
    
    # Multi-modal fusion information
    modalities_used: List[str] = field(default_factory=list)
    wearable_data_included: bool = False
    fusion_confidence: Optional[float] = None
    
    # Clinical context
    diagnostic_confidence: DiagnosticConfidence = DiagnosticConfidence.MEDIUM
    clinical_recommendations: List[str] = field(default_factory=list)
    follow_up_required: bool = False
    
    # Processing metadata
    model_version: str = "1.0.0"
    processing_time_seconds: Optional[float] = None
    gpu_memory_used_mb: Optional[float] = None
    
    def __post_init__(self):
        """Validate diagnostic result after initialization."""
        self._validate_diagnostic_result()
    
    def _validate_diagnostic_result(self):
        """Validate complete diagnostic result."""
        # Validate patient ID format
        if not self.patient_id:
            raise ValueError("Patient ID cannot be empty")
        
        # Validate study IDs
        if not self.study_ids:
            raise ValueError("At least one study ID must be provided")
        
        for study_id in self.study_ids:
            if not study_id:
                raise ValueError("Study IDs cannot be empty")
        
        # Validate timestamp
        if self.timestamp > datetime.now():
            raise ValueError("Diagnostic timestamp cannot be in the future")
        
        # Validate modalities
        valid_modalities = ["MRI", "CT", "Ultrasound", "EEG", "HeartRate", "Sleep", "Gait"]
        for modality in self.modalities_used:
            if modality not in valid_modalities:
                raise ValueError(f"Invalid modality: {modality}")
        
        # Validate fusion confidence if present
        if self.fusion_confidence is not None:
            if not (0.0 <= self.fusion_confidence <= 1.0):
                raise ValueError("Fusion confidence must be between 0 and 1")
        
        # Validate processing time if present
        if self.processing_time_seconds is not None:
            if self.processing_time_seconds < 0:
                raise ValueError("Processing time cannot be negative")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the diagnostic result.
        
        Returns:
            Dict containing key diagnostic information
        """
        return {
            "patient_id": self.patient_id,
            "timestamp": self.timestamp.isoformat(),
            "predicted_class": self.classification_result.predicted_class,
            "confidence": self.classification_result.confidence_score,
            "diagnostic_confidence": self.diagnostic_confidence.value,
            "dice_score": self.metrics.dice_score,
            "auc_score": self.metrics.auc_score,
            "overall_performance": self.metrics.get_overall_performance_score(),
            "modalities_used": self.modalities_used,
            "follow_up_required": self.follow_up_required,
            "processing_time": self.processing_time_seconds
        }
    
    def requires_manual_review(self) -> bool:
        """
        Determine if the diagnostic result requires manual review.
        
        Returns:
            bool: True if manual review is recommended
        """
        # Low confidence predictions need review
        if self.classification_result.confidence_score < 0.7:
            return True
        
        # Poor model performance needs review
        if self.metrics.dice_score < 0.8 or self.metrics.auc_score < 0.85:
            return True
        
        # Uncertain diagnostic confidence needs review
        if self.diagnostic_confidence in [DiagnosticConfidence.LOW, DiagnosticConfidence.UNCERTAIN]:
            return True
        
        # High uncertainty in predictions needs review
        if (self.metrics.epistemic_uncertainty is not None and 
            self.metrics.epistemic_uncertainty > 0.3):
            return True
        
        return False


@dataclass
class BatchDiagnosticResult:
    """Results from batch processing multiple patients."""
    batch_id: str
    processing_timestamp: datetime
    results: List[DiagnosticResult]
    batch_metrics: Dict[str, float] = field(default_factory=dict)
    failed_cases: List[Dict[str, str]] = field(default_factory=list)
    
    def __post_init__(self):
        """Calculate batch statistics after initialization."""
        self._calculate_batch_metrics()
    
    def _calculate_batch_metrics(self):
        """Calculate aggregate metrics for the batch."""
        if not self.results:
            return
        
        # Calculate mean metrics
        dice_scores = [r.metrics.dice_score for r in self.results]
        auc_scores = [r.metrics.auc_score for r in self.results]
        confidence_scores = [r.classification_result.confidence_score for r in self.results]
        
        self.batch_metrics.update({
            "mean_dice_score": np.mean(dice_scores),
            "std_dice_score": np.std(dice_scores),
            "mean_auc_score": np.mean(auc_scores),
            "std_auc_score": np.std(auc_scores),
            "mean_confidence": np.mean(confidence_scores),
            "std_confidence": np.std(confidence_scores),
            "total_processed": len(self.results),
            "total_failed": len(self.failed_cases),
            "success_rate": len(self.results) / (len(self.results) + len(self.failed_cases))
        })


# Utility functions for creating and validating diagnostic results
def create_diagnostic_result(
    patient_id: str,
    study_ids: List[str],
    segmentation_mask: np.ndarray,
    class_probabilities: Dict[str, float],
    metrics: ModelMetrics,
    modalities_used: List[str],
    **kwargs
) -> DiagnosticResult:
    """
    Create a diagnostic result with validation.
    
    Args:
        patient_id: Patient identifier
        study_ids: List of study identifiers
        segmentation_mask: Segmentation output array
        class_probabilities: Classification probabilities
        metrics: Model performance metrics
        modalities_used: List of imaging modalities used
        **kwargs: Additional optional parameters
        
    Returns:
        DiagnosticResult: Validated diagnostic result
    """
    # Create segmentation result
    seg_class_probs = {}
    for class_name, prob in class_probabilities.items():
        # Create probability map with same shape as mask
        prob_map = np.full_like(segmentation_mask, prob, dtype=np.float32)
        seg_class_probs[class_name] = prob_map
    
    segmentation_result = SegmentationResult(
        segmentation_mask=segmentation_mask,
        class_probabilities=seg_class_probs
    )
    
    # Create classification result
    predicted_class = max(class_probabilities.items(), key=lambda x: x[1])[0]
    confidence_score = max(class_probabilities.values())
    
    classification_result = ClassificationResult(
        predicted_class=predicted_class,
        class_probabilities=class_probabilities,
        confidence_score=confidence_score
    )
    
    # Create diagnostic result
    return DiagnosticResult(
        patient_id=patient_id,
        study_ids=study_ids,
        timestamp=datetime.now(),
        segmentation_result=segmentation_result,
        classification_result=classification_result,
        metrics=metrics,
        modalities_used=modalities_used,
        **kwargs
    )


def validate_metrics_consistency(metrics: ModelMetrics) -> bool:
    """
    Validate consistency between different metrics.
    
    Args:
        metrics: ModelMetrics instance to validate
        
    Returns:
        bool: True if metrics are consistent
        
    Raises:
        ValueError: If metrics are inconsistent
    """
    # Check if Dice and Jaccard are consistent (if both present)
    if metrics.jaccard_index is not None:
        # Jaccard = Dice / (2 - Dice)
        expected_jaccard = metrics.dice_score / (2 - metrics.dice_score)
        if abs(metrics.jaccard_index - expected_jaccard) > 0.05:
            raise ValueError(
                f"Inconsistent Dice ({metrics.dice_score}) and "
                f"Jaccard ({metrics.jaccard_index}) scores"
            )
    
    # Check if precision, recall, and F1 are consistent (if all present)
    if all(x is not None for x in [metrics.precision, metrics.recall, metrics.f1_score]):
        expected_f1 = 2 * (metrics.precision * metrics.recall) / (metrics.precision + metrics.recall)
        if abs(metrics.f1_score - expected_f1) > 0.05:
            raise ValueError(
                f"Inconsistent precision ({metrics.precision}), "
                f"recall ({metrics.recall}), and F1 ({metrics.f1_score}) scores"
            )
    
    return True