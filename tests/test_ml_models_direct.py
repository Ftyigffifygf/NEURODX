"""
Direct unit tests for MONAI ML model components.

Tests core functionality by importing modules directly to avoid configuration issues.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil
from datetime import datetime
from unittest.mock import Mock, patch


def test_swin_unetr_config_direct():
    """Test SwinUNETR configuration directly."""
    # Import the specific module directly
    import sys
    sys.path.append('src')
    
    from services.ml_inference.swin_unetr_model import SwinUNETRConfig
    
    # Test valid configuration
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
    
    # Test validation
    with pytest.raises(ValueError, match="Only 3D spatial dimensions are supported"):
        SwinUNETRConfig(spatial_dims=2)
    
    with pytest.raises(ValueError, match="Input channels must be at least 1"):
        SwinUNETRConfig(in_channels=0)
    
    with pytest.raises(ValueError, match="Output channels must be at least 2"):
        SwinUNETRConfig(out_channels=1)


def test_multi_task_model_direct():
    """Test MultiTaskSwinUNETR model directly."""
    import sys
    sys.path.append('src')
    
    from services.ml_inference.swin_unetr_model import SwinUNETRConfig, MultiTaskSwinUNETR
    
    # Create configuration
    config = SwinUNETRConfig(
        img_size=(96, 96, 96),
        in_channels=4,
        out_channels=3,
        feature_size=48
    )
    
    # Create model
    model = MultiTaskSwinUNETR(config, num_classes=4)
    
    # Test model properties
    assert isinstance(model, MultiTaskSwinUNETR)
    assert model.config == config
    assert model.num_classes == 4
    
    # Test model info
    info = model.get_model_info()
    assert isinstance(info, dict)
    assert "model_type" in info
    assert info["model_type"] == "MultiTaskSwinUNETR"
    assert info["parameters"]["total"] > 0
    
    # Test forward pass with synthetic data
    batch_size = 1
    input_tensor = torch.randn(batch_size, 4, 96, 96, 96)
    
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


def test_loss_function_direct():
    """Test LossFunction directly."""
    import sys
    sys.path.append('src')
    
    from services.ml_inference.training_orchestrator import LossFunction
    
    # Create loss function
    loss_function = LossFunction(
        num_classes=3,
        segmentation_weight=0.7,
        classification_weight=0.3
    )
    
    batch_size = 2
    
    # Create synthetic outputs
    seg_output = torch.randn(batch_size, 3, 32, 32, 32, requires_grad=True)
    cls_output = torch.randn(batch_size, 4, requires_grad=True)
    
    outputs = {
        "segmentation": seg_output,
        "classification": cls_output
    }
    
    # Create synthetic targets
    seg_target = torch.randint(0, 3, (batch_size, 1, 32, 32, 32))
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
        
    # Test that total loss is reasonable
    total_loss = losses["total_loss"]
    assert total_loss.item() > 0
    assert torch.isfinite(total_loss)


def test_metrics_calculator_direct():
    """Test MetricsCalculator directly."""
    import sys
    sys.path.append('src')
    
    from services.ml_inference.training_orchestrator import MetricsCalculator
    
    # Create metrics calculator
    metrics_calculator = MetricsCalculator(num_classes=3, num_cls_classes=4)
    
    batch_size = 2
    
    # Test segmentation metrics
    seg_predictions = torch.randn(batch_size, 3, 32, 32, 32)
    seg_targets = torch.randint(0, 3, (batch_size, 1, 32, 32, 32))
    
    seg_metrics = metrics_calculator.calculate_segmentation_metrics(seg_predictions, seg_targets)
    
    assert isinstance(seg_metrics, dict)
    assert "dice_score" in seg_metrics
    assert "hausdorff_distance" in seg_metrics
    assert "mean_iou" in seg_metrics
    
    # Check metric ranges
    assert 0.0 <= seg_metrics["dice_score"] <= 1.0
    assert seg_metrics["hausdorff_distance"] >= 0.0
    assert 0.0 <= seg_metrics["mean_iou"] <= 1.0
    
    # Test classification metrics
    cls_predictions = torch.randn(batch_size, 4)
    cls_targets = torch.randint(0, 4, (batch_size,))
    
    cls_metrics = metrics_calculator.calculate_classification_metrics(cls_predictions, cls_targets)
    
    assert isinstance(cls_metrics, dict)
    assert "accuracy" in cls_metrics
    assert "auc_score" in cls_metrics
    assert "precision" in cls_metrics
    assert "recall" in cls_metrics
    assert "f1_score" in cls_metrics
    
    # Check metric ranges
    assert 0.0 <= cls_metrics["accuracy"] <= 1.0
    assert 0.0 <= cls_metrics["auc_score"] <= 1.0
    assert 0.0 <= cls_metrics["precision"] <= 1.0
    assert 0.0 <= cls_metrics["recall"] <= 1.0
    assert 0.0 <= cls_metrics["f1_score"] <= 1.0


def test_training_config_direct():
    """Test TrainingConfig directly."""
    import sys
    sys.path.append('src')
    
    from services.ml_inference.training_orchestrator import TrainingConfig
    
    # Test default configuration
    config = TrainingConfig()
    
    assert config.max_epochs == 100
    assert config.batch_size == 2
    assert config.learning_rate == 1e-4
    assert config.segmentation_weight == 0.7
    assert config.classification_weight == 0.3
    assert config.use_augmentation is True
    assert config.optimizer_type == "AdamW"
    assert config.scheduler_type == "CosineAnnealingLR"
    
    # Test custom configuration
    custom_config = TrainingConfig(
        max_epochs=50,
        batch_size=4,
        learning_rate=2e-4,
        optimizer_type="SGD"
    )
    
    assert custom_config.max_epochs == 50
    assert custom_config.batch_size == 4
    assert custom_config.learning_rate == 2e-4
    assert custom_config.optimizer_type == "SGD"


def test_inference_classes_direct():
    """Test inference classes directly."""
    import sys
    sys.path.append('src')
    
    from services.ml_inference.inference_engine import (
        InferenceRequest, InferenceResult, ConfidenceCalculator
    )
    
    # Test InferenceRequest
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
    
    assert request.request_id == "test_001"
    assert request.patient_id == "PAT_20240101_00001"
    assert len(request.study_ids) == 1
    assert len(request.modalities_used) == 2
    assert request.priority == 1  # default
    
    # Test InferenceResult
    result = InferenceResult(
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
    
    assert result.request_id == "test_001"
    assert result.patient_id == "PAT_20240101_00001"
    assert isinstance(result.segmentation_output, torch.Tensor)
    assert isinstance(result.classification_output, torch.Tensor)
    assert result.processing_time_ms == 150.0
    
    # Test ConfidenceCalculator
    # Test segmentation confidence
    segmentation_probs = torch.softmax(torch.randn(3, 32, 32, 32), dim=0)
    segmentation_mask = torch.argmax(segmentation_probs, dim=0)
    
    seg_confidence = ConfidenceCalculator.calculate_segmentation_confidence(
        segmentation_probs, segmentation_mask
    )
    
    assert isinstance(seg_confidence, dict)
    assert "overall_confidence" in seg_confidence
    assert "mean_entropy" in seg_confidence
    assert "mean_margin" in seg_confidence
    assert 0.0 <= seg_confidence["overall_confidence"] <= 1.0
    
    # Test classification confidence
    classification_probs = torch.softmax(torch.randn(4), dim=0)
    
    cls_confidence = ConfidenceCalculator.calculate_classification_confidence(
        classification_probs
    )
    
    assert isinstance(cls_confidence, dict)
    assert "max_probability" in cls_confidence
    assert "entropy" in cls_confidence
    assert "margin" in cls_confidence
    assert 0.0 <= cls_confidence["max_probability"] <= 1.0


def test_model_manager_basic_direct():
    """Test ModelManager basic functionality directly."""
    import sys
    sys.path.append('src')
    
    from services.ml_inference.swin_unetr_model import SwinUNETRConfig, ModelManager, ModelCheckpoint
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Create configuration
        config = SwinUNETRConfig(
            img_size=(96, 96, 96),
            in_channels=4,
            out_channels=3,
            feature_size=48
        )
        
        # Mock settings to avoid configuration issues
        with patch('services.ml_inference.swin_unetr_model.get_settings') as mock_settings:
            mock_settings.return_value.monai.model_cache = temp_dir
            
            # Create model manager
            model_manager = ModelManager(config, device=torch.device("cpu"))
            
            # Test initial state
            assert model_manager.config == config
            assert model_manager.device == torch.device("cpu")
            assert model_manager.model is None
            
            # Test model creation
            model = model_manager.create_model(num_classes=4)
            assert model is not None
            assert model_manager.model is model
            assert isinstance(model_manager.model.get_model_info(), dict)
            
            # Test checkpoint saving
            checkpoint_path = temp_dir / "test_checkpoint.pth"
            checkpoint = model_manager.save_checkpoint(
                checkpoint_path=checkpoint_path,
                model_version="test_v1.0"
            )
            
            assert isinstance(checkpoint, ModelCheckpoint)
            assert checkpoint_path.exists()
            assert checkpoint.model_version == "test_v1.0"
            
            # Test checkpoint listing
            checkpoints = model_manager.list_checkpoints()
            assert len(checkpoints) == 1
            assert checkpoints[0].model_version == "test_v1.0"
            
            # Test latest checkpoint
            latest = model_manager.get_latest_checkpoint()
            assert latest is not None
            assert latest.model_version == "test_v1.0"
    
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__])