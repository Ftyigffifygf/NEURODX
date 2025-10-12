"""
Validation script for ML inference components.

This script validates that the core ML inference functionality works correctly
without requiring full configuration setup.
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append('src')

def validate_swin_unetr_config():
    """Validate SwinUNETR configuration."""
    print("Testing SwinUNETR configuration...")
    
    try:
        # Import directly to avoid __init__.py issues
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "swin_unetr_model", 
            "src/services/ml_inference/swin_unetr_model.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        SwinUNETRConfig = module.SwinUNETRConfig
        
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
        
        print("✓ SwinUNETR configuration validation passed")
        return True
        
    except Exception as e:
        print(f"✗ SwinUNETR configuration validation failed: {e}")
        return False


def validate_model_creation():
    """Validate model creation and forward pass."""
    print("Testing model creation and forward pass...")
    
    try:
        # Import directly
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "swin_unetr_model", 
            "src/services/ml_inference/swin_unetr_model.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        SwinUNETRConfig = module.SwinUNETRConfig
        MultiTaskSwinUNETR = module.MultiTaskSwinUNETR
        
        # Create configuration
        config = SwinUNETRConfig(
            img_size=(96, 96, 96),
            in_channels=4,
            out_channels=3,
            feature_size=48
        )
        
        # Create model
        model = MultiTaskSwinUNETR(config, num_classes=4)
        
        # Test forward pass
        batch_size = 1
        input_tensor = torch.randn(batch_size, 4, 96, 96, 96)
        
        with torch.no_grad():
            outputs = model(input_tensor)
        
        # Validate outputs
        assert isinstance(outputs, dict)
        assert "segmentation" in outputs
        assert "classification" in outputs
        
        seg_output = outputs["segmentation"]
        cls_output = outputs["classification"]
        
        assert seg_output.shape == (batch_size, 3, 96, 96, 96)
        assert cls_output.shape == (batch_size, 4)
        
        print("✓ Model creation and forward pass validation passed")
        return True
        
    except Exception as e:
        print(f"✗ Model creation validation failed: {e}")
        return False


def validate_loss_function():
    """Validate loss function calculation."""
    print("Testing loss function...")
    
    try:
        # Import directly
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "training_orchestrator", 
            "src/services/ml_inference/training_orchestrator.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        LossFunction = module.LossFunction
        
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
        
        # Validate losses
        assert isinstance(losses, dict)
        assert "total_loss" in losses
        assert "segmentation_loss" in losses
        assert "classification_loss" in losses
        
        # Check that losses are valid tensors
        for loss_name, loss_value in losses.items():
            assert isinstance(loss_value, torch.Tensor)
            assert torch.isfinite(loss_value)
            assert loss_value.item() >= 0
        
        print("✓ Loss function validation passed")
        return True
        
    except Exception as e:
        print(f"✗ Loss function validation failed: {e}")
        return False


def validate_metrics_calculator():
    """Validate metrics calculator."""
    print("Testing metrics calculator...")
    
    try:
        # Import directly
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "training_orchestrator", 
            "src/services/ml_inference/training_orchestrator.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        MetricsCalculator = module.MetricsCalculator
        
        # Create metrics calculator
        metrics_calculator = MetricsCalculator(num_classes=3, num_cls_classes=4)
        
        batch_size = 2
        
        # Test segmentation metrics
        seg_predictions = torch.randn(batch_size, 3, 32, 32, 32)
        seg_targets = torch.randint(0, 3, (batch_size, 1, 32, 32, 32))
        
        seg_metrics = metrics_calculator.calculate_segmentation_metrics(seg_predictions, seg_targets)
        
        # Validate segmentation metrics
        assert isinstance(seg_metrics, dict)
        assert "dice_score" in seg_metrics
        assert "hausdorff_distance" in seg_metrics
        assert "mean_iou" in seg_metrics
        
        assert 0.0 <= seg_metrics["dice_score"] <= 1.0
        assert seg_metrics["hausdorff_distance"] >= 0.0
        assert 0.0 <= seg_metrics["mean_iou"] <= 1.0
        
        # Test classification metrics
        cls_predictions = torch.randn(batch_size, 4)
        cls_targets = torch.randint(0, 4, (batch_size,))
        
        cls_metrics = metrics_calculator.calculate_classification_metrics(cls_predictions, cls_targets)
        
        # Validate classification metrics
        assert isinstance(cls_metrics, dict)
        assert "accuracy" in cls_metrics
        assert "auc_score" in cls_metrics
        assert "precision" in cls_metrics
        assert "recall" in cls_metrics
        assert "f1_score" in cls_metrics
        
        assert 0.0 <= cls_metrics["accuracy"] <= 1.0
        assert 0.0 <= cls_metrics["auc_score"] <= 1.0
        
        print("✓ Metrics calculator validation passed")
        return True
        
    except Exception as e:
        print(f"✗ Metrics calculator validation failed: {e}")
        return False


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("ML Inference Components Validation")
    print("=" * 60)
    
    tests = [
        validate_swin_unetr_config,
        validate_model_creation,
        validate_loss_function,
        validate_metrics_calculator
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"Validation Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All ML inference components validated successfully!")
        return True
    else:
        print("✗ Some validations failed. Check the output above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)