"""
Unit tests for diagnostic result and metrics data models.

This module tests diagnostic result creation and validation with various input scenarios
and validates error handling for invalid data formats.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from src.models.diagnostics import (
    ModelMetrics, SegmentationResult, ClassificationResult, ExplainabilityResult,
    DiagnosticResult, BatchDiagnosticResult, DiagnosticConfidence, DiseaseStage,
    create_diagnostic_result, validate_metrics_consistency
)


class TestModelMetrics:
    """Test cases for ModelMetrics data model."""
    
    def test_valid_model_metrics(self):
        """Test creating valid model metrics."""
        metrics = ModelMetrics(
            dice_score=0.85,
            hausdorff_distance=2.5,
            auc_score=0.92,
            accuracy=0.88,
            precision=0.90,
            recall=0.86,
            f1_score=0.88,
            jaccard_index=0.74
        )
        
        assert metrics.dice_score == 0.85
        assert metrics.hausdorff_distance == 2.5
        assert metrics.auc_score == 0.92
        assert metrics.accuracy == 0.88
        assert metrics.precision == 0.90
        assert metrics.recall == 0.86
        assert metrics.f1_score == 0.88
        assert metrics.jaccard_index == 0.74
    
    def test_minimal_model_metrics(self):
        """Test model metrics with only required fields."""
        metrics = ModelMetrics(
            dice_score=0.80,
            hausdorff_distance=3.0,
            auc_score=0.85
        )
        
        assert metrics.dice_score == 0.80
        assert metrics.hausdorff_distance == 3.0
        assert metrics.auc_score == 0.85
        assert metrics.accuracy is None
        assert metrics.precision is None
    
    def test_invalid_dice_score(self):
        """Test invalid Dice score raises error."""
        with pytest.raises(ValueError, match="Dice score must be between 0 and 1"):
            ModelMetrics(
                dice_score=1.5,  # Invalid: > 1
                hausdorff_distance=2.0,
                auc_score=0.85
            )
    
    def test_negative_hausdorff_distance(self):
        """Test negative Hausdorff distance raises error."""
        with pytest.raises(ValueError, match="Hausdorff distance must be non-negative"):
            ModelMetrics(
                dice_score=0.85,
                hausdorff_distance=-1.0,  # Invalid: negative
                auc_score=0.85
            )
    
    def test_invalid_auc_score(self):
        """Test invalid AUC score raises error."""
        with pytest.raises(ValueError, match="AUC score must be between 0 and 1"):
            ModelMetrics(
                dice_score=0.85,
                hausdorff_distance=2.0,
                auc_score=1.2  # Invalid: > 1
            )


class TestSegmentationResult:
    """Test cases for SegmentationResult data model."""
    
    def test_valid_segmentation_result_2d(self):
        """Test creating valid 2D segmentation result."""
        mask = np.array([[0, 1, 1], [1, 1, 0], [0, 0, 1]], dtype=np.int32)
        class_probs = {
            "background": np.array([[0.8, 0.2, 0.1], [0.1, 0.2, 0.9], [0.7, 0.8, 0.2]], dtype=np.float32),
            "lesion": np.array([[0.2, 0.8, 0.9], [0.9, 0.8, 0.1], [0.3, 0.2, 0.8]], dtype=np.float32)
        }
        
        result = SegmentationResult(
            segmentation_mask=mask,
            class_probabilities=class_probs
        )
        
        assert result.segmentation_mask.shape == (3, 3)
        assert len(result.class_probabilities) == 2
        assert "background" in result.class_probabilities
        assert "lesion" in result.class_probabilities
    
    def test_invalid_segmentation_mask_type(self):
        """Test invalid segmentation mask type raises error."""
        with pytest.raises(ValueError, match="Segmentation mask must be a numpy array"):
            SegmentationResult(
                segmentation_mask=[[0, 1], [1, 0]],  # List instead of numpy array
                class_probabilities={"bg": np.array([[0.5, 0.5], [0.5, 0.5]])}
            )
    
    def test_empty_class_probabilities(self):
        """Test empty class probabilities raises error."""
        mask = np.array([[0, 1], [1, 0]])
        
        with pytest.raises(ValueError, match="Class probabilities cannot be empty"):
            SegmentationResult(
                segmentation_mask=mask,
                class_probabilities={}  # Empty dict
            )


class TestClassificationResult:
    """Test cases for ClassificationResult data model."""
    
    def test_valid_classification_result(self):
        """Test creating valid classification result."""
        class_probs = {
            "healthy": 0.1,
            "mci": 0.3,
            "alzheimer": 0.6
        }
        
        result = ClassificationResult(
            predicted_class="alzheimer",
            class_probabilities=class_probs,
            confidence_score=0.85,
            disease_stage=DiseaseStage.MODERATE_STAGE
        )
        
        assert result.predicted_class == "alzheimer"
        assert result.class_probabilities["alzheimer"] == 0.6
        assert result.confidence_score == 0.85
        assert result.disease_stage == DiseaseStage.MODERATE_STAGE
    
    def test_empty_predicted_class(self):
        """Test empty predicted class raises error."""
        with pytest.raises(ValueError, match="Predicted class cannot be empty"):
            ClassificationResult(
                predicted_class="",  # Empty string
                class_probabilities={"healthy": 1.0},
                confidence_score=0.9
            )
    
    def test_invalid_probability_values(self):
        """Test invalid probability values raise error."""
        with pytest.raises(ValueError, match="Probability for.*must be between 0 and 1"):
            ClassificationResult(
                predicted_class="disease",
                class_probabilities={"healthy": 0.3, "disease": 1.2},  # > 1
                confidence_score=0.8
            )


class TestDiagnosticResult:
    """Test cases for DiagnosticResult data model."""
    
    def test_valid_diagnostic_result(self):
        """Test creating valid diagnostic result."""
        # Create segmentation result
        mask = np.random.randint(0, 3, size=(32, 32, 16))
        class_probs_seg = {
            "background": np.random.rand(32, 32, 16),
            "lesion": np.random.rand(32, 32, 16)
        }
        seg_result = SegmentationResult(mask, class_probs_seg)
        
        # Create classification result
        class_probs_cls = {"healthy": 0.3, "mci": 0.7}
        cls_result = ClassificationResult("mci", class_probs_cls, 0.85)
        
        # Create metrics
        metrics = ModelMetrics(0.82, 2.1, 0.88)
        
        # Create diagnostic result
        result = DiagnosticResult(
            patient_id="PAT_20241010_00001",
            study_ids=["STUDY_20241010_143000_001"],
            timestamp=datetime.now(),
            segmentation_result=seg_result,
            classification_result=cls_result,
            metrics=metrics,
            modalities_used=["MRI", "CT"]
        )
        
        assert result.patient_id == "PAT_20241010_00001"
        assert len(result.study_ids) == 1
        assert len(result.modalities_used) == 2
        assert result.segmentation_result is not None
        assert result.classification_result is not None
    
    def test_empty_patient_id(self):
        """Test empty patient ID raises error."""
        mask = np.random.randint(0, 2, size=(16, 16))
        seg_result = SegmentationResult(mask, {"bg": np.random.rand(16, 16)})
        cls_result = ClassificationResult("healthy", {"healthy": 1.0}, 0.9)
        metrics = ModelMetrics(0.9, 1.5, 0.95)
        
        with pytest.raises(ValueError, match="Patient ID cannot be empty"):
            DiagnosticResult(
                patient_id="",  # Empty
                study_ids=["STUDY_20241010_143000_001"],
                timestamp=datetime.now(),
                segmentation_result=seg_result,
                classification_result=cls_result,
                metrics=metrics
            )
    
    def test_invalid_modality(self):
        """Test invalid modality raises error."""
        mask = np.random.randint(0, 2, size=(16, 16))
        seg_result = SegmentationResult(mask, {"bg": np.random.rand(16, 16)})
        cls_result = ClassificationResult("healthy", {"healthy": 1.0}, 0.9)
        metrics = ModelMetrics(0.9, 1.5, 0.95)
        
        with pytest.raises(ValueError, match="Invalid modality"):
            DiagnosticResult(
                patient_id="PAT_20241010_00001",
                study_ids=["STUDY_20241010_143000_001"],
                timestamp=datetime.now(),
                segmentation_result=seg_result,
                classification_result=cls_result,
                metrics=metrics,
                modalities_used=["INVALID_MODALITY"]
            )


class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    def test_create_diagnostic_result_valid(self):
        """Test creating diagnostic result with utility function."""
        mask = np.random.randint(0, 3, size=(32, 32, 16))
        class_probs = {"healthy": 0.3, "mci": 0.4, "alzheimer": 0.3}
        metrics = ModelMetrics(0.85, 2.1, 0.88)
        
        result = create_diagnostic_result(
            patient_id="PAT_20241010_00001",
            study_ids=["STUDY_20241010_143000_001"],
            segmentation_mask=mask,
            class_probabilities=class_probs,
            metrics=metrics,
            modalities_used=["MRI", "CT"]
        )
        
        assert result.patient_id == "PAT_20241010_00001"
        assert result.classification_result.predicted_class == "mci"  # Highest probability
        assert result.segmentation_result.segmentation_mask.shape == (32, 32, 16)
        assert len(result.modalities_used) == 2
    
    def test_validate_metrics_consistency_valid(self):
        """Test metrics consistency validation with valid metrics."""
        metrics = ModelMetrics(
            dice_score=0.8,
            hausdorff_distance=2.0,
            auc_score=0.85,
            jaccard_index=0.667,  # Consistent with Dice: 0.8/(2-0.8) = 0.667
            precision=0.9,
            recall=0.8,
            f1_score=0.842  # Consistent: 2*(0.9*0.8)/(0.9+0.8) = 0.842
        )
        
        result = validate_metrics_consistency(metrics)
        assert result is True


if __name__ == "__main__":
    pytest.main([__file__])