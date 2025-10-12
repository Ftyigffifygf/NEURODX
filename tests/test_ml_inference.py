"""
Unit tests for MONAI ML inference service components.

Tests model initialization, forward pass, inference engine, training orchestrator,
loss functions, and metric calculations with synthetic data.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import os

# Mock environment variables before importing modules
os.environ.setdefault("NVIDIA_PALMYRA_API_KEY", "test_key")
os.environ.setdefault("DATABASE_URL", "sqlite:///test.db")
os.environ.setdefault("INFLUXDB_TOKEN", "test_token")
os.environ.setdefault("SECRET_KEY", "test_secret")
os.environ.setdefault("ENCRYPTION_KEY", "test_encryption_key")
os.environ.setdefault("JWT_SECRET_KEY", "test_jwt_secret")
os.environ.setdefault("FEDERATED_ENCRYPTION_KEY", "test_federated_key")

from src.services.ml_inference import (
    SwinUNETRConfig, MultiTaskSwinUNETR, ModelManager, ModelCheckpoint,
    InferenceEngine, InferenceRequest, InferenceResult,
    TrainingOrchestrator, TrainingConfig, TrainingMetrics,
    LossFunction, MetricsCalculator,
    create_default_model_manager, create_inference_engine, create_training_orchestrator
)
from src.models.diagnostics import ModelMetrics, DiagnosticConfidence


class TestSwinUNETRConfig:
    """Test SwinUNETR configuration class."""
    
    def test_valid_config_creation(self):
        """Test creating a valid configuration."""
        config = SwinUNETRConfig(
            img_size=(96, 96, 96),
            in_channels=4,
            out_channels=3,
            feature_size=48
        )
        
        assert config.img_size == (96, 96, 96)
        assert config.in_channels == 4
        assert config.out_channels == 3
        assert config.feature_size == 48
        assert config.spatial_dims == 3
        assert config.use_checkpoint is True
    
    def test_invalid_spatial_dims(self):
        """Test validation of spatial dimensions."""
        with pytest.raises(ValueError, match="Only 3D spatial dimensions are supported"):
            SwinUNETRConfig(spatial_dims=2)
    
    def test_invalid_input_channels(self):
        """Test validation of input channels."""
        with pytest.raises(ValueError, match="Input channels must be at least 1"):
            SwinUNETRConfig(in_channels=0)
    
    def test_invalid_output_channels(self):
        """Test validation of output channels."""
        with pytest.raises(ValueError, match="Output channels must be at least 2"):
            SwinUNETRConfig(out_channels=1)
    
    def test_invalid_image_size(self):
        """Test validation of image size."""
        with pytest.raises(ValueError, match="Image size must be at least 32"):
            SwinUNETRConfig(img_size=(16, 16, 16))
    
    def test_invalid_drop_rate(self):
        """Test validation of drop rate."""
        with pytest.raises(ValueError, match="Drop rate must be between 0 and 1"):
            SwinUNETRConfig(drop_rate=1.5)


class TestMultiTaskSwinUNETR:
    """Test MultiTaskSwinUNETR model class."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return SwinUNETRConfig(
            img_size=(96, 96, 96),
            in_channels=4,
            out_channels=3,
            feature_size=48
        )
    
    @pytest.fixture
    def model(self, config):
        """Create test model."""
        return MultiTaskSwinUNETR(config, num_classes=4)
    
    def test_model_creation(self, model, config):
        """Test model creation and architecture."""
        assert isinstance(model, MultiTaskSwinUNETR)
        assert model.config == config
        assert model.num_classes == 4
        
        # Check if model has required components
        assert hasattr(model, 'swin_unetr')
        assert hasattr(model, 'classification_head')
        assert hasattr(model, 'global_avg_pool')
    
    def test_forward_pass(self, model):
        """Test model forward pass with synthetic data."""
        # Create synthetic input
        batch_size = 2
        input_tensor = torch.randn(batch_size, 4, 96, 96, 96)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(input_tensor)
        
        # Check outputs
        assert isinstance(outputs, dict)
        assert "segmentation" in outputs
        assert "classification" in outputs
        
        # Check output shapes
        seg_output = outputs["segmentation"]
        cls_output = outputs["classification"]
        
        assert seg_output.shape == (batch_size, 3, 96, 96, 96)
        assert cls_output.shape == (batch_size, 4)
    
    def test_model_info(self, model):
        """Test model information retrieval."""
        info = model.get_model_info()
        
        assert isinstance(info, dict)
        assert "model_type" in info
        assert "config" in info
        assert "parameters" in info
        
        assert info["model_type"] == "MultiTaskSwinUNETR"
        assert info["parameters"]["total"] > 0
        assert info["parameters"]["trainable"] > 0
    
    def test_model_device_placement(self, config):
        """Test model device placement."""
        device = torch.device("cpu")
        model = MultiTaskSwinUNETR(config, num_classes=4)
        model.to(device)
        
        # Check if model parameters are on correct device
        for param in model.parameters():
            assert param.device == device


class TestModelManager:
    """Test ModelManager class."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return SwinUNETRConfig(
            img_size=(96, 96, 96),
            in_channels=4,
            out_channels=3,
            feature_size=48
        )
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def model_manager(self, config, temp_dir):
        """Create test model manager."""
        with patch('src.services.ml_inference.swin_unetr_model.get_settings') as mock_settings:
            mock_settings.return_value.monai.model_cache = temp_dir
            return ModelManager(config, device=torch.device("cpu"))
    
    def test_model_manager_creation(self, model_manager, config):
        """Test model manager creation."""
        assert model_manager.config == config
        assert model_manager.device == torch.device("cpu")
        assert model_manager.model is None
    
    def test_create_model(self, model_manager):
        """Test model creation."""
        model = model_manager.create_model(num_classes=4)
        
        assert isinstance(model, MultiTaskSwinUNETR)
        assert model_manager.model is model
        assert model.num_classes == 4
    
    def test_save_and_load_checkpoint(self, model_manager, temp_dir):
        """Test checkpoint saving and loading."""
        # Create model
        model = model_manager.create_model(num_classes=4)
        
        # Create test metrics
        metrics = ModelMetrics(
            dice_score=0.85,
            hausdorff_distance=2.5,
            auc_score=0.92
        )
        
        # Save checkpoint
        checkpoint_path = temp_dir / "test_checkpoint.pth"
        checkpoint = model_manager.save_checkpoint(
            checkpoint_path=checkpoint_path,
            model_version="test_v1.0",
            epoch=10,
            metrics=metrics
        )
        
        assert isinstance(checkpoint, ModelCheckpoint)
        assert checkpoint.checkpoint_path == checkpoint_path
        assert checkpoint.model_version == "test_v1.0"
        assert checkpoint.epoch == 10
        assert checkpoint_path.exists()
        
        # Create new model manager and load checkpoint
        new_manager = ModelManager(model_manager.config, device=torch.device("cpu"))
        metadata = new_manager.load_checkpoint(checkpoint_path)
        
        assert metadata["model_version"] == "test_v1.0"
        assert metadata["epoch"] == 10
        assert new_manager.model is not None
    
    def test_list_checkpoints(self, model_manager, temp_dir):
        """Test listing checkpoints."""
        # Initially no checkpoints
        checkpoints = model_manager.list_checkpoints()
        assert len(checkpoints) == 0
        
        # Create model and save checkpoint
        model_manager.create_model(num_classes=4)
        checkpoint_path = temp_dir / "test_checkpoint.pth"
        model_manager.save_checkpoint(
            checkpoint_path=checkpoint_path,
            model_version="test_v1.0"
        )
        
        # List checkpoints
        checkpoints = model_manager.list_checkpoints()
        assert len(checkpoints) == 1
        assert checkpoints[0].model_version == "test_v1.0"
    
    def test_get_latest_checkpoint(self, model_manager, temp_dir):
        """Test getting latest checkpoint."""
        # No checkpoints initially
        latest = model_manager.get_latest_checkpoint()
        assert latest is None
        
        # Create model and save checkpoints
        model_manager.create_model(num_classes=4)
        
        # Save first checkpoint
        checkpoint1_path = temp_dir / "checkpoint1.pth"
        model_manager.save_checkpoint(
            checkpoint_path=checkpoint1_path,
            model_version="v1.0"
        )
        
        # Save second checkpoint (newer)
        checkpoint2_path = temp_dir / "checkpoint2.pth"
        model_manager.save_checkpoint(
            checkpoint_path=checkpoint2_path,
            model_version="v2.0"
        )
        
        # Get latest checkpoint
        latest = model_manager.get_latest_checkpoint()
        assert latest is not None
        assert latest.model_version == "v2.0"


class TestLossFunction:
    """Test LossFunction class."""
    
    @pytest.fixture
    def loss_function(self):
        """Create test loss function."""
        return LossFunction(
            num_classes=3,
            segmentation_weight=0.7,
            classification_weight=0.3
        )
    
    def test_loss_calculation(self, loss_function):
        """Test loss calculation with synthetic data."""
        batch_size = 2
        
        # Create synthetic outputs
        seg_output = torch.randn(batch_size, 3, 96, 96, 96)
        cls_output = torch.randn(batch_size, 4)
        
        outputs = {
            "segmentation": seg_output,
            "classification": cls_output
        }
        
        # Create synthetic targets
        seg_target = torch.randint(0, 3, (batch_size, 1, 96, 96, 96))
        cls_target = torch.randint(0, 4, (batch_size,))
        
        targets = {
            "segmentation": seg_target,
            "classification": cls_target
        }
        
        # Calculate losses
        losses = loss_function(outputs, targets)
        
        # Check loss components
        assert isinstance(losses, dict)
        assert "total_loss" in losses
        assert "segmentation_loss" in losses
        assert "classification_loss" in losses
        assert "dice_loss" in losses
        assert "ce_seg_loss" in losses
        
        # Check that losses are tensors with gradients
        for loss_name, loss_value in losses.items():
            assert isinstance(loss_value, torch.Tensor)
            assert loss_value.requires_grad


class TestMetricsCalculator:
    """Test MetricsCalculator class."""
    
    @pytest.fixture
    def metrics_calculator(self):
        """Create test metrics calculator."""
        return MetricsCalculator(num_classes=3, num_cls_classes=4)
    
    def test_segmentation_metrics(self, metrics_calculator):
        """Test segmentation metrics calculation."""
        batch_size = 2
        
        # Create synthetic predictions and targets
        predictions = torch.randn(batch_size, 3, 64, 64, 64)
        targets = torch.randint(0, 3, (batch_size, 1, 64, 64, 64))
        
        # Calculate metrics
        metrics = metrics_calculator.calculate_segmentation_metrics(predictions, targets)
        
        # Check metrics
        assert isinstance(metrics, dict)
        assert "dice_score" in metrics
        assert "hausdorff_distance" in metrics
        assert "mean_iou" in metrics
        
        # Check metric ranges
        assert 0.0 <= metrics["dice_score"] <= 1.0
        assert metrics["hausdorff_distance"] >= 0.0
        assert 0.0 <= metrics["mean_iou"] <= 1.0
    
    def test_classification_metrics(self, metrics_calculator):
        """Test classification metrics calculation."""
        batch_size = 10
        
        # Create synthetic predictions and targets
        predictions = torch.randn(batch_size, 4)
        targets = torch.randint(0, 4, (batch_size,))
        
        # Calculate metrics
        metrics = metrics_calculator.calculate_classification_metrics(predictions, targets)
        
        # Check metrics
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert "auc_score" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        
        # Check metric ranges
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["auc_score"] <= 1.0
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0
        assert 0.0 <= metrics["f1_score"] <= 1.0


class TestInferenceEngine:
    """Test InferenceEngine class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_model_manager(self, temp_dir):
        """Create mock model manager."""
        config = SwinUNETRConfig()
        
        with patch('src.services.ml_inference.swin_unetr_model.get_settings') as mock_settings:
            mock_settings.return_value.monai.model_cache = temp_dir
            manager = ModelManager(config, device=torch.device("cpu"))
            manager.create_model(num_classes=4)
            return manager
    
    @pytest.fixture
    def inference_engine(self, mock_model_manager):
        """Create test inference engine."""
        return InferenceEngine(
            model_manager=mock_model_manager,
            batch_size=1,
            roi_size=(96, 96, 96)
        )
    
    def test_inference_engine_creation(self, inference_engine):
        """Test inference engine creation."""
        assert isinstance(inference_engine, InferenceEngine)
        assert inference_engine.batch_size == 1
        assert inference_engine.roi_size == (96, 96, 96)
    
    def test_load_model(self, inference_engine):
        """Test model loading."""
        success = inference_engine.load_model()
        assert success is True
        assert inference_engine.model_manager.model is not None
    
    def test_predict_single(self, inference_engine):
        """Test single prediction."""
        # Load model
        inference_engine.load_model()
        
        # Create test request
        input_data = {
            "image": torch.randn(4, 96, 96, 96)
        }
        
        request = InferenceRequest(
            request_id="test_001",
            patient_id="PAT_20240101_00001",
            study_ids=["STUDY_20240101_120000_001"],
            input_data=input_data,
            modalities_used=["MRI", "CT"]
        )
        
        # Perform inference
        result = inference_engine.predict_single(request)
        
        # Check result
        assert isinstance(result, InferenceResult)
        assert result.request_id == "test_001"
        assert result.patient_id == "PAT_20240101_00001"
        assert isinstance(result.segmentation_output, torch.Tensor)
        assert isinstance(result.classification_output, torch.Tensor)
        assert isinstance(result.confidence_scores, dict)
        assert result.processing_time_ms > 0
    
    def test_predict_batch(self, inference_engine):
        """Test batch prediction."""
        # Load model
        inference_engine.load_model()
        
        # Create test requests
        requests = []
        for i in range(3):
            input_data = {
                "image": torch.randn(4, 96, 96, 96)
            }
            
            request = InferenceRequest(
                request_id=f"test_{i:03d}",
                patient_id=f"PAT_20240101_0000{i}",
                study_ids=[f"STUDY_20240101_120000_00{i}"],
                input_data=input_data,
                modalities_used=["MRI"]
            )
            requests.append(request)
        
        # Perform batch inference
        results = inference_engine.predict_batch(requests)
        
        # Check results
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.request_id == f"test_{i:03d}"
            assert isinstance(result.segmentation_output, torch.Tensor)
            assert isinstance(result.classification_output, torch.Tensor)
    
    def test_create_diagnostic_result(self, inference_engine):
        """Test diagnostic result creation."""
        # Create mock inference result
        inference_result = InferenceResult(
            request_id="test_001",
            patient_id="PAT_20240101_00001",
            segmentation_output=torch.randn(1, 3, 96, 96, 96),
            classification_output=torch.randn(1, 4),
            confidence_scores={
                "segmentation": {"overall_confidence": 0.85},
                "classification": {"max_probability": 0.92},
                "overall": 0.88
            },
            processing_time_ms=150.0
        )
        
        # Create diagnostic result
        diagnostic_result = inference_engine.create_diagnostic_result(
            inference_result=inference_result,
            study_ids=["STUDY_20240101_120000_001"],
            modalities_used=["MRI", "CT"]
        )
        
        # Check diagnostic result
        assert diagnostic_result.patient_id == "PAT_20240101_00001"
        assert diagnostic_result.study_ids == ["STUDY_20240101_120000_001"]
        assert diagnostic_result.modalities_used == ["MRI", "CT"]
        assert isinstance(diagnostic_result.segmentation_result.segmentation_mask, np.ndarray)
        assert isinstance(diagnostic_result.classification_result.class_probabilities, dict)
        assert diagnostic_result.processing_time_seconds == 0.15


class TestTrainingConfig:
    """Test TrainingConfig class."""
    
    def test_default_config(self):
        """Test default training configuration."""
        config = TrainingConfig()
        
        assert config.max_epochs == 100
        assert config.batch_size == 2
        assert config.learning_rate == 1e-4
        assert config.segmentation_weight == 0.7
        assert config.classification_weight == 0.3
        assert config.use_augmentation is True
        assert config.optimizer_type == "AdamW"
        assert config.scheduler_type == "CosineAnnealingLR"
    
    def test_custom_config(self):
        """Test custom training configuration."""
        config = TrainingConfig(
            max_epochs=50,
            batch_size=4,
            learning_rate=2e-4,
            optimizer_type="SGD"
        )
        
        assert config.max_epochs == 50
        assert config.batch_size == 4
        assert config.learning_rate == 2e-4
        assert config.optimizer_type == "SGD"


class TestTrainingMetrics:
    """Test TrainingMetrics class."""
    
    def test_metrics_creation(self):
        """Test training metrics creation."""
        metrics = TrainingMetrics(
            epoch=10,
            train_loss=0.25,
            val_loss=0.30,
            segmentation_loss=0.15,
            classification_loss=0.10,
            dice_score=0.85,
            hausdorff_distance=2.5,
            mean_iou=0.80,
            accuracy=0.92,
            auc_score=0.88,
            precision=0.90,
            recall=0.87,
            f1_score=0.89,
            learning_rate=1e-4,
            epoch_time_seconds=120.5,
            gpu_memory_mb=2048.0
        )
        
        assert metrics.epoch == 10
        assert metrics.train_loss == 0.25
        assert metrics.dice_score == 0.85
        assert metrics.accuracy == 0.92
        assert isinstance(metrics.timestamp, datetime)


class TestIntegration:
    """Integration tests for ML inference components."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_end_to_end_inference_workflow(self, temp_dir):
        """Test complete inference workflow."""
        with patch('src.services.ml_inference.swin_unetr_model.get_settings') as mock_settings:
            mock_settings.return_value.monai.model_cache = temp_dir
            
            # Create model manager and model
            config = SwinUNETRConfig(
                img_size=(96, 96, 96),
                in_channels=4,
                out_channels=3,
                feature_size=48
            )
            
            model_manager = ModelManager(config, device=torch.device("cpu"))
            model = model_manager.create_model(num_classes=4)
            
            # Create inference engine
            engine = InferenceEngine(
                model_manager=model_manager,
                batch_size=1,
                roi_size=(96, 96, 96)
            )
            
            # Load model
            success = engine.load_model()
            assert success is True
            
            # Create test request
            input_data = {
                "image": torch.randn(4, 96, 96, 96)
            }
            
            request = InferenceRequest(
                request_id="integration_test",
                patient_id="PAT_20240101_00001",
                study_ids=["STUDY_20240101_120000_001"],
                input_data=input_data,
                modalities_used=["MRI", "CT"]
            )
            
            # Perform inference
            result = engine.predict_single(request)
            
            # Create diagnostic result
            diagnostic_result = engine.create_diagnostic_result(
                inference_result=result,
                study_ids=request.study_ids,
                modalities_used=request.modalities_used
            )
            
            # Verify complete workflow
            assert diagnostic_result.patient_id == request.patient_id
            assert len(diagnostic_result.study_ids) == 1
            assert len(diagnostic_result.modalities_used) == 2
            assert diagnostic_result.processing_time_seconds > 0
    
    def test_model_save_load_inference_consistency(self, temp_dir):
        """Test that saved and loaded models produce consistent results."""
        with patch('src.services.ml_inference.swin_unetr_model.get_settings') as mock_settings:
            mock_settings.return_value.monai.model_cache = temp_dir
            
            # Create original model
            config = SwinUNETRConfig()
            model_manager1 = ModelManager(config, device=torch.device("cpu"))
            model1 = model_manager1.create_model(num_classes=4)
            
            # Create test input
            test_input = torch.randn(1, 4, 96, 96, 96)
            
            # Get original output
            with torch.no_grad():
                original_output = model1(test_input)
            
            # Save checkpoint
            checkpoint_path = temp_dir / "consistency_test.pth"
            model_manager1.save_checkpoint(
                checkpoint_path=checkpoint_path,
                model_version="consistency_test"
            )
            
            # Create new model manager and load checkpoint
            model_manager2 = ModelManager(config, device=torch.device("cpu"))
            model_manager2.load_checkpoint(checkpoint_path)
            model2 = model_manager2.model
            
            # Get loaded model output
            with torch.no_grad():
                loaded_output = model2(test_input)
            
            # Compare outputs (should be identical)
            seg_diff = torch.abs(original_output["segmentation"] - loaded_output["segmentation"]).max()
            cls_diff = torch.abs(original_output["classification"] - loaded_output["classification"]).max()
            
            assert seg_diff < 1e-6, "Segmentation outputs should be identical"
            assert cls_diff < 1e-6, "Classification outputs should be identical"


# Test fixtures and utilities
@pytest.fixture
def synthetic_medical_data():
    """Create synthetic medical imaging data for testing."""
    batch_size = 2
    channels = 4
    height, width, depth = 96, 96, 96
    
    # Create synthetic multi-modal imaging data
    images = torch.randn(batch_size, channels, height, width, depth)
    
    # Create synthetic segmentation masks
    segmentation_masks = torch.randint(0, 3, (batch_size, 1, height, width, depth))
    
    # Create synthetic classification labels
    classification_labels = torch.randint(0, 4, (batch_size,))
    
    return {
        "images": images,
        "segmentation_masks": segmentation_masks,
        "classification_labels": classification_labels
    }


def test_validate_model_setup():
    """Test model setup validation function."""
    with patch('src.services.ml_inference.swin_unetr_model.create_default_model_manager') as mock_create:
        # Mock successful setup
        mock_manager = Mock()
        mock_model = Mock()
        mock_manager.create_model.return_value = mock_model
        mock_manager.device = torch.device("cpu")
        
        # Mock successful forward pass
        mock_outputs = {
            "segmentation": torch.randn(1, 3, 96, 96, 96),
            "classification": torch.randn(1, 4)
        }
        mock_model.return_value = mock_outputs
        
        mock_create.return_value = mock_manager
        
        from src.services.ml_inference.swin_unetr_model import validate_model_setup
        
        # Test successful validation
        result = validate_model_setup()
        assert result is True
        
        # Test failed validation
        mock_model.side_effect = Exception("Test error")
        result = validate_model_setup()
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__])