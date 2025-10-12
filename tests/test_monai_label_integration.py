"""
Integration tests for MONAI Label workflows.

Tests annotation creation, retrieval, active learning sample selection,
and end-to-end MONAI Label integration workflows.
"""

import pytest
import tempfile
import shutil
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

# Check if MONAI Label is available
try:
    import monailabel
    MONAI_LABEL_AVAILABLE = True
except ImportError:
    MONAI_LABEL_AVAILABLE = False

from src.services.monai_label.monai_label_server import (
    MONAILabelServer, MONAILabelConfig, TaskDefinition, NeuroDxMONAILabelApp
)
from src.services.monai_label.annotation_manager import (
    AnnotationManager, Annotation, AnnotationSession
)
from src.services.monai_label.active_learning_engine import (
    ActiveLearningEngine, SampleCandidate, UncertaintyCalculator
)
from src.config.monai_label_config import MONAILabelIntegrationConfig
from src.models.patient import PatientRecord, ImagingStudy, Demographics
from src.models.diagnostics import DiagnosticResult


class TestMONAILabelServer:
    """Test MONAI Label server configuration and management."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def server_config(self, temp_dir):
        """Create test server configuration."""
        return MONAILabelConfig(
            server_host="127.0.0.1",
            server_port=8001,
            studies_path=str(Path(temp_dir) / "studies"),
            models_path=str(Path(temp_dir) / "models"),
            app_dir=str(Path(temp_dir) / "app"),
            auto_update_scoring=True,
            scoring_enabled=True
        )
    
    @pytest.fixture
    def monai_server(self, server_config):
        """Create MONAI Label server instance."""
        return MONAILabelServer(server_config)
    
    def test_server_initialization(self, monai_server, server_config):
        """Test server initialization with configuration."""
        assert monai_server.config == server_config
        assert monai_server.app is None
        assert not monai_server.is_running
    
    def test_configuration_validation(self, monai_server):
        """Test server configuration validation."""
        issues = monai_server.validate_configuration()
        assert isinstance(issues, list)
        # Should have no issues with valid configuration
        assert len(issues) == 0
    
    def test_invalid_port_configuration(self, temp_dir):
        """Test configuration validation with invalid port."""
        config = MONAILabelConfig(
            studies_path=str(Path(temp_dir) / "studies"),
            server_port=99999  # Invalid port
        )
        
        with pytest.raises(ValueError, match="Server port must be between"):
            MONAILabelServer(config)
    
    def test_app_directory_setup(self, monai_server):
        """Test MONAI Label app directory setup."""
        monai_server.setup_app_directory()
        
        app_path = Path(monai_server.config.app_dir)
        assert app_path.exists()
        assert (app_path / "lib").exists()
        assert (app_path / "model").exists()
        assert (app_path / "logs").exists()
        assert (app_path / "main.py").exists()
        assert (app_path / "app.json").exists()
    
    def test_app_initialization(self, monai_server):
        """Test MONAI Label application initialization."""
        app = monai_server.initialize_app()
        
        assert isinstance(app, NeuroDxMONAILabelApp)
        assert monai_server.app is not None
        assert len(app.task_definitions) > 0
    
    def test_task_definitions_creation(self, monai_server):
        """Test creation of default task definitions."""
        app = monai_server.initialize_app()
        task_definitions = app.task_definitions
        
        # Check that we have expected tasks
        task_names = [task.name for task in task_definitions]
        assert "brain_segmentation" in task_names
        assert "disease_classification" in task_names
        assert "lesion_detection" in task_names
        
        # Validate task structure
        for task_def in task_definitions:
            assert isinstance(task_def, TaskDefinition)
            assert task_def.name
            assert task_def.type in ["segmentation", "classification"]
            assert task_def.labels
            assert isinstance(task_def.labels, dict)
    
    def test_server_info(self, monai_server):
        """Test server information retrieval."""
        info = monai_server.get_server_info()
        
        assert "config" in info
        assert "app_initialized" in info
        assert "is_running" in info
        assert "gpu_info" in info
        assert "task_definitions" in info
        
        assert info["app_initialized"] is False
        assert info["is_running"] is False
    
    def test_add_task_definition(self, monai_server):
        """Test adding custom task definition."""
        monai_server.initialize_app()
        
        custom_task = TaskDefinition(
            name="custom_segmentation",
            type="segmentation",
            description="Custom segmentation task",
            labels={"background": 0, "roi": 1}
        )
        
        success = monai_server.add_task_definition(custom_task)
        assert success
        
        task_definitions = monai_server.get_task_definitions()
        task_names = [task.name for task in task_definitions]
        assert "custom_segmentation" in task_names


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
    
    @pytest.fixture
    def sample_annotation_data(self):
        """Create sample annotation data."""
        return {
            "mask": np.random.randint(0, 3, size=(64, 64, 64), dtype=np.uint8),
            "confidence": 0.85,
            "processing_time": 2.3
        }
    
    def test_annotation_manager_initialization(self, annotation_manager, temp_dir):
        """Test annotation manager initialization."""
        assert annotation_manager.storage.storage_path == Path(temp_dir)
        assert annotation_manager.storage.annotations_dir.exists()
        assert annotation_manager.storage.sessions_dir.exists()
        assert annotation_manager.storage.masks_dir.exists()
    
    def test_create_segmentation_annotation(self, annotation_manager, sample_annotation_data):
        """Test creating segmentation annotation."""
        annotation = annotation_manager.create_annotation(
            patient_id="PAT_20241010_001",
            study_id="STUDY_20241010_120000_001",
            task_name="brain_segmentation",
            annotator_id="annotator_001",
            annotation_type="segmentation",
            annotation_data=sample_annotation_data,
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
    
    def test_annotation_retrieval(self, annotation_manager, sample_annotation_data):
        """Test annotation retrieval by ID."""
        # Create annotation
        created_annotation = annotation_manager.create_annotation(
            patient_id="PAT_20241010_003",
            study_id="STUDY_20241010_140000_001",
            task_name="brain_segmentation",
            annotator_id="annotator_001",
            annotation_type="segmentation",
            annotation_data=sample_annotation_data
        )
        
        # Retrieve annotation
        retrieved_annotation = annotation_manager.get_annotation(created_annotation.annotation_id)
        
        assert retrieved_annotation is not None
        assert retrieved_annotation.annotation_id == created_annotation.annotation_id
        assert retrieved_annotation.patient_id == created_annotation.patient_id
        assert np.array_equal(retrieved_annotation.data["mask"], sample_annotation_data["mask"])
    
    def test_annotation_update(self, annotation_manager, sample_annotation_data):
        """Test annotation update functionality."""
        # Create annotation
        annotation = annotation_manager.create_annotation(
            patient_id="PAT_20241010_004",
            study_id="STUDY_20241010_150000_001",
            task_name="brain_segmentation",
            annotator_id="annotator_001",
            annotation_type="segmentation",
            annotation_data=sample_annotation_data
        )
        
        # Update annotation
        updates = {
            "validation_status": "approved",
            "metadata": {"reviewer": "expert_001", "review_date": "2024-10-10"}
        }
        
        updated_annotation = annotation_manager.update_annotation(annotation.annotation_id, updates)
        
        assert updated_annotation is not None
        assert updated_annotation.validation_status == "approved"
        assert updated_annotation.metadata["reviewer"] == "expert_001"
    
    def test_annotation_session_workflow(self, annotation_manager, sample_annotation_data):
        """Test annotation session management."""
        # Start session
        session_id = annotation_manager.start_annotation_session(
            annotator_id="annotator_001",
            task_name="brain_segmentation",
            session_metadata={"batch_id": "batch_001"}
        )
        
        assert session_id in annotation_manager.active_sessions
        
        # Create annotations in session
        annotation1 = annotation_manager.create_annotation(
            patient_id="PAT_20241010_005",
            study_id="STUDY_20241010_160000_001",
            task_name="brain_segmentation",
            annotator_id="annotator_001",
            annotation_type="segmentation",
            annotation_data=sample_annotation_data
        )
        
        annotation2 = annotation_manager.create_annotation(
            patient_id="PAT_20241010_006",
            study_id="STUDY_20241010_170000_001",
            task_name="brain_segmentation",
            annotator_id="annotator_001",
            annotation_type="segmentation",
            annotation_data=sample_annotation_data
        )
        
        # Add annotations to session
        annotation_manager.add_annotation_to_session(session_id, annotation1.annotation_id)
        annotation_manager.add_annotation_to_session(session_id, annotation2.annotation_id)
        
        # End session
        completed_session = annotation_manager.end_annotation_session(session_id)
        
        assert completed_session is not None
        assert len(completed_session.annotations) == 2
        assert session_id not in annotation_manager.active_sessions
    
    def test_get_annotations_for_patient(self, annotation_manager, sample_annotation_data):
        """Test retrieving annotations for specific patient."""
        patient_id = "PAT_20241010_007"
        
        # Create multiple annotations for same patient
        for i in range(3):
            annotation_manager.create_annotation(
                patient_id=patient_id,
                study_id=f"STUDY_20241010_{180000 + i}_001",
                task_name="brain_segmentation",
                annotator_id="annotator_001",
                annotation_type="segmentation",
                annotation_data=sample_annotation_data
            )
        
        # Retrieve annotations
        patient_annotations = annotation_manager.get_annotations_for_patient(patient_id)
        
        assert len(patient_annotations) == 3
        for annotation in patient_annotations:
            assert annotation.patient_id == patient_id
    
    def test_annotation_validation_workflow(self, annotation_manager, sample_annotation_data):
        """Test annotation validation workflow."""
        # Create annotation
        annotation = annotation_manager.create_annotation(
            patient_id="PAT_20241010_008",
            study_id="STUDY_20241010_190000_001",
            task_name="brain_segmentation",
            annotator_id="annotator_001",
            annotation_type="segmentation",
            annotation_data=sample_annotation_data
        )
        
        # Validate annotation
        success = annotation_manager.validate_annotation(
            annotation.annotation_id,
            validator_id="validator_001",
            validation_result="approved",
            comments="High quality segmentation"
        )
        
        assert success
        
        # Check validation status
        validated_annotation = annotation_manager.get_annotation(annotation.annotation_id)
        assert validated_annotation.validation_status == "approved"
        assert validated_annotation.metadata["validator_id"] == "validator_001"
    
    def test_annotation_statistics(self, annotation_manager, sample_annotation_data):
        """Test annotation statistics calculation."""
        task_name = "brain_segmentation"
        
        # Create multiple annotations with different statuses
        for i, status in enumerate(["pending", "approved", "rejected"]):
            annotation = annotation_manager.create_annotation(
                patient_id=f"PAT_20241010_{200 + i:03d}",
                study_id=f"STUDY_20241010_{200000 + i}_001",
                task_name=task_name,
                annotator_id="annotator_001",
                annotation_type="segmentation",
                annotation_data=sample_annotation_data
            )
            
            # Update validation status
            annotation_manager.validate_annotation(
                annotation.annotation_id,
                validator_id="validator_001",
                validation_result=status
            )
        
        # Get statistics
        stats = annotation_manager.get_annotation_statistics(task_name)
        
        assert stats["task_name"] == task_name
        assert stats["total_annotations"] == 3
        assert "validation_status_counts" in stats
        assert stats["validation_status_counts"]["approved"] == 1
        assert stats["validation_status_counts"]["rejected"] == 1


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
        # Multi-class predictions (batch_size=5, num_classes=3)
        return torch.tensor([
            [0.8, 0.1, 0.1],  # High confidence
            [0.4, 0.3, 0.3],  # High uncertainty
            [0.6, 0.2, 0.2],  # Medium confidence
            [0.33, 0.33, 0.34],  # Very high uncertainty
            [0.9, 0.05, 0.05]  # Very high confidence
        ], dtype=torch.float32)
    
    @pytest.fixture
    def sample_studies(self):
        """Create sample imaging studies."""
        studies = []
        for i in range(5):
            study = ImagingStudy(
                study_id=f"STUDY_20241010_{100000 + i}_001",
                patient_id=f"PAT_20241010_{100 + i:03d}",
                modality="MRI",
                acquisition_date=datetime.now(),
                file_path=f"/data/images/study_{i}.nii.gz",
                preprocessing_metadata={}
            )
            studies.append(study)
        return studies
    
    def test_uncertainty_calculation_entropy(self, active_learning_engine, sample_predictions):
        """Test entropy-based uncertainty calculation."""
        uncertainty_scores = active_learning_engine.calculate_uncertainty_scores(
            sample_predictions, method="entropy"
        )
        
        assert len(uncertainty_scores) == 5
        assert all(0.0 <= score <= 1.0 for score in uncertainty_scores)
        
        # Sample with [0.33, 0.33, 0.34] should have highest uncertainty
        assert uncertainty_scores[3] > uncertainty_scores[0]  # High uncertainty > low uncertainty
        assert uncertainty_scores[3] > uncertainty_scores[4]  # High uncertainty > very low uncertainty
    
    def test_uncertainty_calculation_margin(self, active_learning_engine, sample_predictions):
        """Test margin-based uncertainty calculation."""
        uncertainty_scores = active_learning_engine.calculate_uncertainty_scores(
            sample_predictions, method="margin"
        )
        
        assert len(uncertainty_scores) == 5
        assert all(0.0 <= score <= 1.0 for score in uncertainty_scores)
        
        # Samples with smaller margins should have higher uncertainty
        assert uncertainty_scores[3] > uncertainty_scores[0]
    
    def test_diversity_score_calculation(self, active_learning_engine):
        """Test diversity score calculation."""
        # Create sample features
        features = np.random.rand(5, 10)  # 5 samples, 10 features each
        labeled_features = np.random.rand(2, 10)  # 2 labeled samples
        
        diversity_scores = active_learning_engine.calculate_diversity_scores(
            features, labeled_features
        )
        
        assert len(diversity_scores) == 5
        assert all(0.0 <= score <= 1.0 for score in diversity_scores)
    
    def test_sample_candidate_creation(self, active_learning_engine, sample_studies, sample_predictions):
        """Test creation of sample candidates."""
        features = np.random.rand(5, 10)
        
        candidates = active_learning_engine.create_sample_candidates(
            sample_studies, sample_predictions, features
        )
        
        assert len(candidates) == 5
        
        for candidate in candidates:
            assert isinstance(candidate, SampleCandidate)
            assert candidate.patient_id.startswith("PAT_")
            assert candidate.study_id.startswith("STUDY_")
            assert 0.0 <= candidate.uncertainty_score <= 1.0
            assert 0.0 <= candidate.diversity_score <= 1.0
            assert candidate.features is not None
    
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
    
    def test_diversity_strategy_selection(self, active_learning_engine, sample_studies, sample_predictions):
        """Test diversity-based sample selection."""
        features = np.random.rand(5, 10)
        
        selected_samples = active_learning_engine.select_samples_for_annotation(
            task_name="brain_segmentation",
            unlabeled_studies=sample_studies,
            predictions=sample_predictions,
            num_samples=3,
            strategy_name="diversity",
            features=features
        )
        
        assert len(selected_samples) == 3
        
        # All selected samples should have features
        for sample in selected_samples:
            assert sample.features is not None
    
    def test_hybrid_strategy_selection(self, active_learning_engine, sample_studies, sample_predictions):
        """Test hybrid strategy sample selection."""
        features = np.random.rand(5, 10)
        
        selected_samples = active_learning_engine.select_samples_for_annotation(
            task_name="brain_segmentation",
            unlabeled_studies=sample_studies,
            predictions=sample_predictions,
            num_samples=3,
            strategy_name="hybrid",
            features=features
        )
        
        assert len(selected_samples) == 3
        
        # Check that combined scores are calculated
        for sample in selected_samples:
            assert sample.combined_score > 0.0
    
    def test_active_learning_round_management(self, active_learning_engine):
        """Test active learning round lifecycle."""
        # Start round
        round_id = active_learning_engine.start_active_learning_round(
            task_name="brain_segmentation",
            strategy_name="uncertainty"
        )
        
        assert round_id in active_learning_engine.current_rounds
        
        round_obj = active_learning_engine.current_rounds[round_id]
        assert round_obj.task_name == "brain_segmentation"
        assert round_obj.strategy_name == "uncertainty"
        assert round_obj.completed_at is None
        
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
    
    def test_model_feedback_loop(self, active_learning_engine, temp_dir):
        """Test model feedback loop with new annotations."""
        # Create sample annotations
        annotation_manager = AnnotationManager(temp_dir)
        
        annotations = []
        for i in range(3):
            annotation_data = {
                "mask": np.random.randint(0, 3, size=(32, 32, 32), dtype=np.uint8),
                "confidence": 0.8 + i * 0.05
            }
            
            annotation = annotation_manager.create_annotation(
                patient_id=f"PAT_20241010_{300 + i:03d}",
                study_id=f"STUDY_20241010_{300000 + i}_001",
                task_name="brain_segmentation",
                annotator_id="annotator_001",
                annotation_type="segmentation",
                annotation_data=annotation_data
            )
            annotations.append(annotation)
        
        # Process feedback
        feedback_results = active_learning_engine.update_model_with_feedback(
            "brain_segmentation", annotations
        )
        
        assert feedback_results["task_name"] == "brain_segmentation"
        assert feedback_results["new_annotations_count"] == 3
        assert "feedback_processed_at" in feedback_results


class TestMONAILabelIntegration:
    """Test end-to-end MONAI Label integration workflows."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def integration_config(self, temp_dir):
        """Create integration configuration."""
        config = MONAILabelIntegrationConfig()
        config.server.studies_path = str(Path(temp_dir) / "studies")
        config.server.models_path = str(Path(temp_dir) / "models")
        config.server.app_dir = str(Path(temp_dir) / "app")
        config.server.annotations_path = str(Path(temp_dir) / "annotations")
        return config
    
    def test_configuration_validation(self, integration_config):
        """Test integration configuration validation."""
        issues = integration_config.validate_configuration()
        assert isinstance(issues, list)
        # Should have no issues with valid configuration
        assert len(issues) == 0
    
    def test_default_task_configurations(self, integration_config):
        """Test default task configurations."""
        assert "brain_segmentation" in integration_config.tasks
        assert "disease_classification" in integration_config.tasks
        assert "lesion_detection" in integration_config.tasks
        
        # Validate brain segmentation task
        brain_seg_task = integration_config.get_task_config("brain_segmentation")
        assert brain_seg_task.type == "segmentation"
        assert "hippocampus" in brain_seg_task.labels
        assert "amygdala" in brain_seg_task.labels
        
        # Validate disease classification task
        disease_class_task = integration_config.get_task_config("disease_classification")
        assert disease_class_task.type == "classification"
        assert disease_class_task.num_classes == 5
        assert "alzheimer" in disease_class_task.labels
    
    def test_end_to_end_annotation_workflow(self, integration_config, temp_dir):
        """Test complete annotation workflow from creation to validation."""
        # Initialize components
        server_config = MONAILabelConfig(
            studies_path=integration_config.server.studies_path,
            models_path=integration_config.server.models_path,
            app_dir=integration_config.server.app_dir
        )
        
        server = MONAILabelServer(server_config)
        annotation_manager = AnnotationManager(integration_config.server.annotations_path)
        active_learning_engine = ActiveLearningEngine(annotation_manager)
        
        # Initialize server app
        app = server.initialize_app()
        assert app is not None
        
        # Start annotation session
        session_id = annotation_manager.start_annotation_session(
            annotator_id="test_annotator",
            task_name="brain_segmentation"
        )
        
        # Create annotation
        annotation_data = {
            "mask": np.random.randint(0, 7, size=(64, 64, 64), dtype=np.uint8),
            "confidence": 0.87,
            "processing_time": 1.5
        }
        
        annotation = annotation_manager.create_annotation(
            patient_id="PAT_20241010_TEST",
            study_id="STUDY_20241010_TEST_001",
            task_name="brain_segmentation",
            annotator_id="test_annotator",
            annotation_type="segmentation",
            annotation_data=annotation_data
        )
        
        assert annotation is not None
        
        # Add to session
        annotation_manager.add_annotation_to_session(session_id, annotation.annotation_id)
        
        # Validate annotation
        validation_success = annotation_manager.validate_annotation(
            annotation.annotation_id,
            validator_id="test_validator",
            validation_result="approved"
        )
        
        assert validation_success
        
        # End session
        completed_session = annotation_manager.end_annotation_session(session_id)
        assert completed_session is not None
        assert len(completed_session.annotations) == 1
        
        # Get statistics
        stats = annotation_manager.get_annotation_statistics("brain_segmentation")
        assert stats["total_annotations"] >= 1
    
    def test_active_learning_integration_workflow(self, integration_config, temp_dir):
        """Test active learning integration with annotation workflow."""
        # Initialize components
        annotation_manager = AnnotationManager(integration_config.server.annotations_path)
        active_learning_engine = ActiveLearningEngine(annotation_manager)
        
        # Create sample studies and predictions
        studies = []
        for i in range(10):
            study = ImagingStudy(
                study_id=f"STUDY_AL_TEST_{i:03d}",
                patient_id=f"PAT_AL_TEST_{i:03d}",
                modality="MRI",
                acquisition_date=datetime.now(),
                file_path=f"/data/test/study_{i}.nii.gz",
                preprocessing_metadata={}
            )
            studies.append(study)
        
        # Create predictions with varying uncertainty
        predictions = torch.tensor([
            [0.9, 0.05, 0.05],  # Low uncertainty
            [0.4, 0.3, 0.3],    # High uncertainty
            [0.7, 0.15, 0.15],  # Medium uncertainty
            [0.33, 0.33, 0.34], # Very high uncertainty
            [0.85, 0.1, 0.05],  # Low uncertainty
            [0.5, 0.25, 0.25],  # High uncertainty
            [0.6, 0.2, 0.2],    # Medium uncertainty
            [0.95, 0.025, 0.025], # Very low uncertainty
            [0.45, 0.35, 0.2],  # High uncertainty
            [0.8, 0.1, 0.1]     # Low uncertainty
        ], dtype=torch.float32)
        
        # Start active learning round
        round_id = active_learning_engine.start_active_learning_round(
            task_name="brain_segmentation",
            strategy_name="uncertainty"
        )
        
        # Select samples for annotation
        selected_samples = active_learning_engine.select_samples_for_annotation(
            task_name="brain_segmentation",
            unlabeled_studies=studies,
            predictions=predictions,
            num_samples=3,
            strategy_name="uncertainty"
        )
        
        assert len(selected_samples) == 3
        
        # Create annotations for selected samples
        created_annotations = []
        for sample in selected_samples:
            annotation_data = {
                "mask": np.random.randint(0, 7, size=(32, 32, 32), dtype=np.uint8),
                "confidence": sample.uncertainty_score,
                "selection_score": sample.combined_score
            }
            
            annotation = annotation_manager.create_annotation(
                patient_id=sample.patient_id,
                study_id=sample.study_id,
                task_name="brain_segmentation",
                annotator_id="active_learning_annotator",
                annotation_type="segmentation",
                annotation_data=annotation_data
            )
            
            created_annotations.append(annotation)
        
        # Complete active learning round
        selected_study_ids = [sample.study_id for sample in selected_samples]
        performance_metrics = {"model_accuracy": 0.82, "annotation_quality": 0.78}
        
        success = active_learning_engine.complete_active_learning_round(
            round_id, selected_study_ids, performance_metrics
        )
        
        assert success
        
        # Process feedback
        feedback_results = active_learning_engine.update_model_with_feedback(
            "brain_segmentation", created_annotations
        )
        
        assert feedback_results["new_annotations_count"] == 3
        
        # Get final statistics
        al_stats = active_learning_engine.get_active_learning_statistics("brain_segmentation")
        assert al_stats["total_annotations"] >= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])