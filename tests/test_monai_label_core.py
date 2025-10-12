"""
Core tests for MONAI Label integration functionality.

Tests the basic functionality without requiring external dependencies.
"""

import pytest
import tempfile
import shutil
import numpy as np
import torch
from datetime import datetime
from pathlib import Path

from src.services.monai_label.annotation_manager import (
    AnnotationManager, Annotation, AnnotationSession
)
from src.services.monai_label.active_learning_engine import (
    ActiveLearningEngine, SampleCandidate, UncertaintyCalculator
)
from src.models.patient import ImagingStudy


class TestAnnotationManager:
    """Test annotation storage and retrieval workflows."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def annotation_manager(self, temp_dir):
        """Create annotation manager instance."""
        return AnnotationManager(temp_dir)
    
    def test_annotation_manager_initialization(self, annotation_manager, temp_dir):
        """Test annotation manager initialization."""
        assert annotation_manager.storage.storage_path == Path(temp_dir)
        assert annotation_manager.storage.annotations_dir.exists()
        assert annotation_manager.storage.sessions_dir.exists()
        assert annotation_manager.storage.masks_dir.exists()
    
    def test_create_segmentation_annotation(self, annotation_manager):
        """Test creating segmentation annotation."""
        annotation_data = {
            "mask": np.random.randint(0, 3, size=(32, 32, 32), dtype=np.uint8),
            "confidence": 0.85,
            "processing_time": 2.3
        }
        
        annotation = annotation_manager.create_annotation(
            patient_id="PAT_20241010_001",
            study_id="STUDY_20241010_120000_001",
            task_name="brain_segmentation",
            annotator_id="annotator_001",
            annotation_type="segmentation",
            annotation_data=annotation_data,
            metadata={"tool_version": "1.0", "review_required": True}
        )
        
        assert annotation is not None
        assert annotation.patient_id == "PAT_20241010_001"
        assert annotation.study_id == "STUDY_20241010_120000_001"
        assert annotation.task_name == "brain_segmentation"
        assert annotation.annotation_type == "segmentation"
        assert annotation.quality_score is not None
        assert 0.0 <= annotation.quality_score <= 1.0
    
    def test_create_classification_annotation(self, annotation_manager):
        """Test creating classification annotation."""
        classification_data = {
            "class_label": "alzheimer",
            "confidence": 0.92,
            "probabilities": {"healthy": 0.08, "alzheimer": 0.92}
        }
        
        annotation = annotation_manager.create_annotation(
            patient_id="PAT_20241010_002",
            study_id="STUDY_20241010_130000_001",
            task_name="disease_classification",
            annotator_id="annotator_002",
            annotation_type="classification",
            annotation_data=classification_data
        )
        
        assert annotation is not None
        assert annotation.annotation_type == "classification"
        assert annotation.data["class_label"] == "alzheimer"
    
    def test_annotation_retrieval(self, annotation_manager):
        """Test annotation retrieval by ID."""
        annotation_data = {
            "mask": np.random.randint(0, 3, size=(16, 16, 16), dtype=np.uint8),
            "confidence": 0.75
        }
        
        # Create annotation
        created_annotation = annotation_manager.create_annotation(
            patient_id="PAT_20241010_003",
            study_id="STUDY_20241010_140000_001",
            task_name="brain_segmentation",
            annotator_id="annotator_001",
            annotation_type="segmentation",
            annotation_data=annotation_data
        )
        
        # Retrieve annotation
        retrieved_annotation = annotation_manager.get_annotation(created_annotation.annotation_id)
        
        assert retrieved_annotation is not None
        assert retrieved_annotation.annotation_id == created_annotation.annotation_id
        assert retrieved_annotation.patient_id == created_annotation.patient_id
    
    def test_annotation_session_workflow(self, annotation_manager):
        """Test annotation session management."""
        # Start session
        session_id = annotation_manager.start_annotation_session(
            annotator_id="annotator_001",
            task_name="brain_segmentation",
            session_metadata={"batch_id": "batch_001"}
        )
        
        assert session_id in annotation_manager.active_sessions
        
        # Create annotation in session
        annotation_data = {
            "mask": np.random.randint(0, 3, size=(16, 16, 16), dtype=np.uint8),
            "confidence": 0.8
        }
        
        annotation = annotation_manager.create_annotation(
            patient_id="PAT_20241010_005",
            study_id="STUDY_20241010_160000_001",
            task_name="brain_segmentation",
            annotator_id="annotator_001",
            annotation_type="segmentation",
            annotation_data=annotation_data
        )
        
        # Add annotation to session
        annotation_manager.add_annotation_to_session(session_id, annotation.annotation_id)
        
        # End session
        completed_session = annotation_manager.end_annotation_session(session_id)
        
        assert completed_session is not None
        assert len(completed_session.annotations) == 1
        assert session_id not in annotation_manager.active_sessions


class TestUncertaintyCalculator:
    """Test uncertainty calculation methods."""
    
    @pytest.fixture
    def uncertainty_calculator(self):
        """Create uncertainty calculator instance."""
        return UncertaintyCalculator()
    
    def test_entropy_uncertainty(self, uncertainty_calculator):
        """Test entropy-based uncertainty calculation."""
        # Create sample predictions with varying confidence
        predictions = torch.tensor([
            [0.9, 0.05, 0.05],  # High confidence
            [0.4, 0.3, 0.3],    # High uncertainty
            [0.7, 0.15, 0.15],  # Medium confidence
            [0.33, 0.33, 0.34], # Very high uncertainty
        ], dtype=torch.float32)
        
        uncertainty_scores = uncertainty_calculator.entropy_uncertainty(predictions)
        
        assert len(uncertainty_scores) == 4
        assert all(0.0 <= score <= 1.0 for score in uncertainty_scores)
        
        # Sample with [0.33, 0.33, 0.34] should have highest uncertainty
        assert uncertainty_scores[3] > uncertainty_scores[0]  # High uncertainty > low uncertainty
    
    def test_margin_uncertainty(self, uncertainty_calculator):
        """Test margin-based uncertainty calculation."""
        predictions = torch.tensor([
            [0.9, 0.05, 0.05],  # Large margin
            [0.4, 0.35, 0.25],  # Small margin
        ], dtype=torch.float32)
        
        uncertainty_scores = uncertainty_calculator.margin_uncertainty(predictions)
        
        assert len(uncertainty_scores) == 2
        assert all(0.0 <= score <= 1.0 for score in uncertainty_scores)
        
        # Smaller margin should have higher uncertainty
        assert uncertainty_scores[1] > uncertainty_scores[0]
    
    def test_least_confidence_uncertainty(self, uncertainty_calculator):
        """Test least confidence uncertainty calculation."""
        predictions = torch.tensor([
            [0.9, 0.05, 0.05],  # High max confidence
            [0.4, 0.35, 0.25],  # Low max confidence
        ], dtype=torch.float32)
        
        uncertainty_scores = uncertainty_calculator.least_confidence_uncertainty(predictions)
        
        assert len(uncertainty_scores) == 2
        assert all(0.0 <= score <= 1.0 for score in uncertainty_scores)
        
        # Lower max confidence should have higher uncertainty
        assert uncertainty_scores[1] > uncertainty_scores[0]


class TestActiveLearningEngine:
    """Test active learning sample selection workflows."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def annotation_manager(self, temp_dir):
        """Create annotation manager instance."""
        return AnnotationManager(temp_dir)
    
    @pytest.fixture
    def active_learning_engine(self, annotation_manager):
        """Create active learning engine instance."""
        return ActiveLearningEngine(annotation_manager)
    
    @pytest.fixture
    def sample_predictions(self):
        """Create sample model predictions."""
        return torch.tensor([
            [0.8, 0.1, 0.1],    # High confidence
            [0.4, 0.3, 0.3],    # High uncertainty
            [0.6, 0.2, 0.2],    # Medium confidence
            [0.33, 0.33, 0.34], # Very high uncertainty
            [0.9, 0.05, 0.05]   # Very high confidence
        ], dtype=torch.float32)
    
    @pytest.fixture
    def sample_studies(self):
        """Create sample imaging studies."""
        studies = []
        for i in range(5):
            study = ImagingStudy(
                study_id=f"STUDY_20241010_{100000 + i}_001",
                modality="MRI",
                acquisition_date=datetime.now(),
                file_path=f"/data/images/study_{i}.nii.gz",
                preprocessing_metadata=None
            )
            # Add patient_id as an attribute for testing
            study.patient_id = f"PAT_20241010_{100 + i:03d}"
            studies.append(study)
        return studies
    
    def test_uncertainty_calculation(self, active_learning_engine, sample_predictions):
        """Test uncertainty score calculation."""
        uncertainty_scores = active_learning_engine.calculate_uncertainty_scores(
            sample_predictions, method="entropy"
        )
        
        assert len(uncertainty_scores) == 5
        assert all(0.0 <= score <= 1.0 for score in uncertainty_scores)
        
        # Sample with [0.33, 0.33, 0.34] should have highest uncertainty
        assert uncertainty_scores[3] > uncertainty_scores[0]
    
    def test_sample_candidate_creation(self, active_learning_engine, sample_studies, sample_predictions):
        """Test creation of sample candidates."""
        candidates = active_learning_engine.create_sample_candidates(
            sample_studies, sample_predictions
        )
        
        assert len(candidates) == 5
        
        for candidate in candidates:
            assert isinstance(candidate, SampleCandidate)
            assert candidate.patient_id.startswith("PAT_")
            assert candidate.study_id.startswith("STUDY_")
            assert 0.0 <= candidate.uncertainty_score <= 1.0
            assert 0.0 <= candidate.diversity_score <= 1.0
    
    def test_uncertainty_strategy_selection(self, active_learning_engine, sample_studies, sample_predictions):
        """Test uncertainty-based sample selection."""
        selected_samples = active_learning_engine.select_samples_for_annotation(
            task_name="brain_segmentation",
            unlabeled_studies=sample_studies,
            predictions=sample_predictions,
            num_samples=3,
            strategy_name="uncertainty"
        )
        
        assert len(selected_samples) == 3
        
        # Check that samples are sorted by uncertainty (descending)
        for i in range(len(selected_samples) - 1):
            assert selected_samples[i].uncertainty_score >= selected_samples[i + 1].uncertainty_score
    
    def test_active_learning_round_management(self, active_learning_engine):
        """Test active learning round lifecycle."""
        # Start round
        round_id = active_learning_engine.start_active_learning_round(
            task_name="brain_segmentation",
            strategy_name="uncertainty"
        )
        
        assert round_id in active_learning_engine.current_rounds
        
        # Complete round
        selected_samples = ["STUDY_001", "STUDY_002", "STUDY_003"]
        performance_metrics = {"accuracy": 0.85, "dice_score": 0.78}
        
        success = active_learning_engine.complete_active_learning_round(
            round_id, selected_samples, performance_metrics
        )
        
        assert success
        assert round_id not in active_learning_engine.current_rounds
    
    def test_active_learning_statistics(self, active_learning_engine):
        """Test active learning statistics calculation."""
        stats = active_learning_engine.get_active_learning_statistics("brain_segmentation")
        
        assert "task_name" in stats
        assert "total_annotations" in stats
        assert "quality_statistics" in stats
        assert "validation_status_counts" in stats
        assert "active_rounds_count" in stats
        assert "available_strategies" in stats
        
        # Check available strategies
        expected_strategies = ["uncertainty", "diversity", "hybrid", "random"]
        for strategy in expected_strategies:
            assert strategy in stats["available_strategies"]


class TestSampleCandidate:
    """Test sample candidate data structure."""
    
    def test_valid_sample_candidate(self):
        """Test creating valid sample candidate."""
        candidate = SampleCandidate(
            patient_id="PAT_001",
            study_id="STUDY_001",
            image_path="/data/image.nii.gz",
            uncertainty_score=0.75,
            diversity_score=0.60,
            combined_score=0.68
        )
        
        assert candidate.patient_id == "PAT_001"
        assert candidate.study_id == "STUDY_001"
        assert candidate.uncertainty_score == 0.75
        assert candidate.diversity_score == 0.60
    
    def test_invalid_uncertainty_score(self):
        """Test validation of uncertainty score."""
        with pytest.raises(ValueError, match="Uncertainty score must be between"):
            SampleCandidate(
                patient_id="PAT_001",
                study_id="STUDY_001",
                image_path="/data/image.nii.gz",
                uncertainty_score=1.5,  # Invalid
                diversity_score=0.60,
                combined_score=0.68
            )
    
    def test_invalid_diversity_score(self):
        """Test validation of diversity score."""
        with pytest.raises(ValueError, match="Diversity score must be between"):
            SampleCandidate(
                patient_id="PAT_001",
                study_id="STUDY_001",
                image_path="/data/image.nii.gz",
                uncertainty_score=0.75,
                diversity_score=-0.1,  # Invalid
                combined_score=0.68
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])