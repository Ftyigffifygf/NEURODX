"""
MONAI SwinUNETR model configuration and initialization for NeuroDx-MultiModal system.

This module provides the SwinUNETR model setup with multi-modal input channels
for both segmentation and classification outputs, including checkpoint management.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any, List
from pathlib import Path
import logging
from dataclasses import dataclass
from datetime import datetime

from monai.networks.nets import SwinUNETR
from monai.networks.layers import Norm
from monai.utils import ensure_tuple_rep
from monai.data import MetaTensor

try:
    from src.config.settings import get_settings
except ImportError:
    # For testing purposes, create a mock settings function
    def get_settings():
        from pathlib import Path
        class MockSettings:
            class MonaiSettings:
                model_cache = Path("./models")
                swin_unetr_img_size = (96, 96, 96)
                swin_unetr_in_channels = 4
                swin_unetr_out_channels = 3
                swin_unetr_feature_size = 48
            monai = MonaiSettings()
        return MockSettings()
from src.models.diagnostics import ModelMetrics

logger = logging.getLogger(__name__)


@dataclass
class ModelCheckpoint:
    """Model checkpoint metadata."""
    checkpoint_path: Path
    model_version: str
    creation_date: datetime
    metrics: Optional[ModelMetrics] = None
    epoch: Optional[int] = None
    optimizer_state: bool = False


class SwinUNETRConfig:
    """Configuration class for SwinUNETR model."""
    
    def __init__(
        self,
        img_size: Tuple[int, int, int] = (96, 96, 96),
        in_channels: int = 4,
        out_channels: int = 3,
        feature_size: int = 48,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        use_checkpoint: bool = True,
        spatial_dims: int = 3,
        norm_name: str = "instance",
        use_v2: bool = False
    ):
        """
        Initialize SwinUNETR configuration.
        
        Args:
            img_size: Input image size (H, W, D)
            in_channels: Number of input channels (multi-modal fusion)
            out_channels: Number of output segmentation classes
            feature_size: Feature dimension for the model
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
            dropout_path_rate: Stochastic depth rate
            use_checkpoint: Use gradient checkpointing to save memory
            spatial_dims: Number of spatial dimensions
            norm_name: Normalization layer type
            use_v2: Use SwinUNETR v2 architecture
        """
        self.img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_size = feature_size
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.dropout_path_rate = dropout_path_rate
        self.use_checkpoint = use_checkpoint
        self.spatial_dims = spatial_dims
        self.norm_name = norm_name
        self.use_v2 = use_v2
        
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        if self.spatial_dims != 3:
            raise ValueError("Only 3D spatial dimensions are supported")
        
        if self.in_channels < 1:
            raise ValueError("Input channels must be at least 1")
        
        if self.out_channels < 2:
            raise ValueError("Output channels must be at least 2 for segmentation")
        
        if any(size < 32 for size in self.img_size):
            raise ValueError("Image size must be at least 32 in each dimension")
        
        if not (0.0 <= self.drop_rate <= 1.0):
            raise ValueError("Drop rate must be between 0 and 1")
        
        if not (0.0 <= self.attn_drop_rate <= 1.0):
            raise ValueError("Attention drop rate must be between 0 and 1")
        
        if not (0.0 <= self.dropout_path_rate <= 1.0):
            raise ValueError("Dropout path rate must be between 0 and 1")


class MultiTaskSwinUNETR(nn.Module):
    """
    Multi-task SwinUNETR model for both segmentation and classification.
    
    This model extends the standard SwinUNETR to provide both segmentation
    masks and classification probabilities for neurodegenerative disease detection.
    """
    
    def __init__(self, config: SwinUNETRConfig, num_classes: int = 4):
        """
        Initialize multi-task SwinUNETR model.
        
        Args:
            config: SwinUNETR configuration
            num_classes: Number of classification classes
        """
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        
        # Main SwinUNETR backbone for segmentation
        self.swin_unetr = SwinUNETR(
            img_size=config.img_size,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            feature_size=config.feature_size,
            drop_rate=config.drop_rate,
            attn_drop_rate=config.attn_drop_rate,
            dropout_path_rate=config.dropout_path_rate,
            use_checkpoint=config.use_checkpoint,
            spatial_dims=config.spatial_dims,
            norm_name=config.norm_name,
            use_v2=config.use_v2
        )
        
        # Classification head using encoder features
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.classification_head = nn.Sequential(
            nn.Linear(config.feature_size * 16, config.feature_size * 4),  # Encoder output features
            nn.ReLU(inplace=True),
            nn.Dropout(config.drop_rate),
            nn.Linear(config.feature_size * 4, config.feature_size),
            nn.ReLU(inplace=True),
            nn.Dropout(config.drop_rate),
            nn.Linear(config.feature_size, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize classification head weights."""
        for module in self.classification_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for multi-task learning.
        
        Args:
            x: Input tensor [B, C, H, W, D]
            
        Returns:
            Dictionary containing segmentation and classification outputs
        """
        # Get segmentation output
        segmentation_logits = self.swin_unetr(x)
        
        # Extract encoder features for classification
        # Access the encoder features from the SwinUNETR model
        encoder_features = self.swin_unetr.swinViT(x, normalize=True)
        
        # Global average pooling and classification
        pooled_features = self.global_avg_pool(encoder_features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        classification_logits = self.classification_head(pooled_features)
        
        return {
            "segmentation": segmentation_logits,
            "classification": classification_logits
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_type": "MultiTaskSwinUNETR",
            "config": {
                "img_size": self.config.img_size,
                "in_channels": self.config.in_channels,
                "out_channels": self.config.out_channels,
                "feature_size": self.config.feature_size,
                "num_classes": self.num_classes
            },
            "parameters": {
                "total": total_params,
                "trainable": trainable_params,
                "non_trainable": total_params - trainable_params
            }
        }


class ModelManager:
    """
    Model manager for SwinUNETR checkpoint loading and saving.
    
    Handles model initialization, checkpoint management, and device placement.
    """
    
    def __init__(self, config: SwinUNETRConfig, device: Optional[torch.device] = None):
        """
        Initialize model manager.
        
        Args:
            config: SwinUNETR configuration
            device: Target device for model
        """
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[MultiTaskSwinUNETR] = None
        self.settings = get_settings()
        
        # Ensure model cache directory exists
        self.checkpoint_dir = self.settings.monai.model_cache / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ModelManager initialized with device: {self.device}")
    
    def create_model(self, num_classes: int = 4) -> MultiTaskSwinUNETR:
        """
        Create a new SwinUNETR model instance.
        
        Args:
            num_classes: Number of classification classes
            
        Returns:
            MultiTaskSwinUNETR model instance
        """
        self.model = MultiTaskSwinUNETR(self.config, num_classes)
        self.model.to(self.device)
        
        logger.info(f"Created SwinUNETR model with {num_classes} classes")
        logger.info(f"Model info: {self.model.get_model_info()}")
        
        return self.model
    
    def load_checkpoint(
        self, 
        checkpoint_path: Path, 
        load_optimizer: bool = False,
        strict: bool = True
    ) -> Dict[str, Any]:
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            load_optimizer: Whether to load optimizer state
            strict: Whether to strictly enforce state dict keys
            
        Returns:
            Checkpoint metadata
        """
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Create model if not exists
        if self.model is None:
            num_classes = checkpoint.get("num_classes", 4)
            self.create_model(num_classes)
        
        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        
        # Prepare return metadata
        metadata = {
            "model_version": checkpoint.get("model_version", "unknown"),
            "epoch": checkpoint.get("epoch", 0),
            "metrics": checkpoint.get("metrics"),
            "creation_date": checkpoint.get("creation_date"),
            "optimizer_loaded": False
        }
        
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            metadata["optimizer_state_dict"] = checkpoint["optimizer_state_dict"]
            metadata["optimizer_loaded"] = True
        
        logger.info(f"Checkpoint loaded successfully: {metadata}")
        return metadata
    
    def save_checkpoint(
        self,
        checkpoint_path: Path,
        model_version: str,
        epoch: Optional[int] = None,
        metrics: Optional[ModelMetrics] = None,
        optimizer_state: Optional[Dict] = None,
        additional_info: Optional[Dict] = None
    ) -> ModelCheckpoint:
        """
        Save model checkpoint.
        
        Args:
            checkpoint_path: Path to save checkpoint
            model_version: Model version string
            epoch: Training epoch number
            metrics: Model performance metrics
            optimizer_state: Optimizer state dict
            additional_info: Additional metadata
            
        Returns:
            ModelCheckpoint metadata
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        # Prepare checkpoint data
        checkpoint_data = {
            "model_state_dict": self.model.state_dict(),
            "model_version": model_version,
            "creation_date": datetime.now(),
            "config": {
                "img_size": self.config.img_size,
                "in_channels": self.config.in_channels,
                "out_channels": self.config.out_channels,
                "feature_size": self.config.feature_size,
                "num_classes": self.model.num_classes
            }
        }
        
        if epoch is not None:
            checkpoint_data["epoch"] = epoch
        
        if metrics is not None:
            checkpoint_data["metrics"] = metrics
        
        if optimizer_state is not None:
            checkpoint_data["optimizer_state_dict"] = optimizer_state
        
        if additional_info is not None:
            checkpoint_data.update(additional_info)
        
        # Ensure directory exists
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        
        logger.info(f"Checkpoint saved to: {checkpoint_path}")
        
        return ModelCheckpoint(
            checkpoint_path=checkpoint_path,
            model_version=model_version,
            creation_date=checkpoint_data["creation_date"],
            metrics=metrics,
            epoch=epoch,
            optimizer_state=optimizer_state is not None
        )
    
    def list_checkpoints(self) -> List[ModelCheckpoint]:
        """
        List available model checkpoints.
        
        Returns:
            List of available checkpoints
        """
        checkpoints = []
        
        for checkpoint_file in self.checkpoint_dir.glob("*.pth"):
            try:
                # Load checkpoint metadata without loading the full model
                checkpoint_data = torch.load(checkpoint_file, map_location="cpu")
                
                checkpoint = ModelCheckpoint(
                    checkpoint_path=checkpoint_file,
                    model_version=checkpoint_data.get("model_version", "unknown"),
                    creation_date=checkpoint_data.get("creation_date", datetime.fromtimestamp(checkpoint_file.stat().st_mtime)),
                    metrics=checkpoint_data.get("metrics"),
                    epoch=checkpoint_data.get("epoch"),
                    optimizer_state="optimizer_state_dict" in checkpoint_data
                )
                
                checkpoints.append(checkpoint)
                
            except Exception as e:
                logger.warning(f"Failed to load checkpoint metadata from {checkpoint_file}: {e}")
        
        # Sort by creation date (newest first)
        checkpoints.sort(key=lambda x: x.creation_date, reverse=True)
        
        return checkpoints
    
    def get_latest_checkpoint(self) -> Optional[ModelCheckpoint]:
        """
        Get the latest checkpoint.
        
        Returns:
            Latest checkpoint or None if no checkpoints exist
        """
        checkpoints = self.list_checkpoints()
        return checkpoints[0] if checkpoints else None
    
    def cleanup_old_checkpoints(self, keep_count: int = 5):
        """
        Clean up old checkpoints, keeping only the most recent ones.
        
        Args:
            keep_count: Number of checkpoints to keep
        """
        checkpoints = self.list_checkpoints()
        
        if len(checkpoints) <= keep_count:
            return
        
        # Remove old checkpoints
        for checkpoint in checkpoints[keep_count:]:
            try:
                checkpoint.checkpoint_path.unlink()
                logger.info(f"Removed old checkpoint: {checkpoint.checkpoint_path}")
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {checkpoint.checkpoint_path}: {e}")


def create_default_model_manager() -> ModelManager:
    """
    Create a default model manager with settings from configuration.
    
    Returns:
        Configured ModelManager instance
    """
    settings = get_settings()
    
    config = SwinUNETRConfig(
        img_size=settings.monai.swin_unetr_img_size,
        in_channels=settings.monai.swin_unetr_in_channels,
        out_channels=settings.monai.swin_unetr_out_channels,
        feature_size=settings.monai.swin_unetr_feature_size,
        use_checkpoint=True
    )
    
    return ModelManager(config)


def validate_model_setup() -> bool:
    """
    Validate that the model can be created and basic operations work.
    
    Returns:
        True if model setup is valid
    """
    try:
        # Create model manager
        manager = create_default_model_manager()
        
        # Create model
        model = manager.create_model()
        
        # Test forward pass with dummy data
        dummy_input = torch.randn(1, 4, 96, 96, 96).to(manager.device)
        
        with torch.no_grad():
            outputs = model(dummy_input)
        
        # Validate outputs
        assert "segmentation" in outputs
        assert "classification" in outputs
        assert outputs["segmentation"].shape == (1, 3, 96, 96, 96)
        assert outputs["classification"].shape == (1, 4)
        
        logger.info("Model setup validation successful")
        return True
        
    except Exception as e:
        logger.error(f"Model setup validation failed: {e}")
        return False