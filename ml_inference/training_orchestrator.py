"""
Training orchestrator with MONAI components for NeuroDx-MultiModal system.

This module provides comprehensive training capabilities using MONAI loss functions,
metrics, and training loops with validation and checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging
from pathlib import Path
import time
import json
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm

from monai.losses import DiceLoss, FocalLoss, TverskyLoss
from monai.metrics import (
    DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric,
    ConfusionMatrixMetric, ROCAUCMetric, MeanIoU
)
from monai.transforms import (
    Compose, RandFlipd, RandRotate90d, RandScaleIntensityd,
    RandShiftIntensityd, RandSpatialCropd
)
from monai.data import CacheDataset, ThreadDataLoader, decollate_batch
from monai.utils import set_determinism
from monai.optimizers import Novograd
from monai.networks.utils import one_hot

from src.services.ml_inference.swin_unetr_model import ModelManager, MultiTaskSwinUNETR
from src.models.diagnostics import ModelMetrics
from src.config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training orchestrator."""
    
    # Training parameters
    max_epochs: int = 100
    batch_size: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # Loss function weights
    segmentation_weight: float = 0.7
    classification_weight: float = 0.3
    dice_weight: float = 0.5
    ce_weight: float = 0.5
    
    # Validation parameters
    validation_interval: int = 1
    validation_metric: str = "dice_score"
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    
    # Checkpointing
    checkpoint_interval: int = 5
    save_best_only: bool = True
    max_checkpoints: int = 5
    
    # Data augmentation
    use_augmentation: bool = True
    augmentation_probability: float = 0.5
    
    # Optimizer settings
    optimizer_type: str = "AdamW"  # AdamW, SGD, Novograd
    scheduler_type: str = "CosineAnnealingLR"  # CosineAnnealingLR, StepLR, ReduceLROnPlateau
    
    # Mixed precision training
    use_amp: bool = True
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True


@dataclass
class TrainingMetrics:
    """Training metrics tracking."""
    epoch: int
    train_loss: float
    val_loss: float
    segmentation_loss: float
    classification_loss: float
    
    # Segmentation metrics
    dice_score: float
    hausdorff_distance: float
    mean_iou: float
    
    # Classification metrics
    accuracy: float
    auc_score: float
    precision: float
    recall: float
    f1_score: float
    
    # Training metadata
    learning_rate: float
    epoch_time_seconds: float
    gpu_memory_mb: float
    timestamp: datetime = field(default_factory=datetime.now)


class LossFunction:
    """Combined loss function for multi-task learning."""
    
    def __init__(
        self,
        num_classes: int = 3,
        segmentation_weight: float = 0.7,
        classification_weight: float = 0.3,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        focal_gamma: float = 2.0,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7
    ):
        """
        Initialize combined loss function.
        
        Args:
            num_classes: Number of segmentation classes
            segmentation_weight: Weight for segmentation loss
            classification_weight: Weight for classification loss
            dice_weight: Weight for Dice loss component
            ce_weight: Weight for CrossEntropy loss component
            focal_gamma: Gamma parameter for Focal loss
            tversky_alpha: Alpha parameter for Tversky loss
            tversky_beta: Beta parameter for Tversky loss
        """
        self.num_classes = num_classes
        self.segmentation_weight = segmentation_weight
        self.classification_weight = classification_weight
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        
        # Segmentation losses
        self.dice_loss = DiceLoss(
            to_onehot_y=True,
            softmax=True,
            squared_pred=True,
            jaccard=False
        )
        
        self.ce_loss = nn.CrossEntropyLoss()
        
        self.focal_loss = FocalLoss(
            gamma=focal_gamma,
            to_onehot_y=True,
            use_softmax=True
        )
        
        self.tversky_loss = TverskyLoss(
            alpha=tversky_alpha,
            beta=tversky_beta,
            to_onehot_y=True,
            softmax=True
        )
        
        # Classification loss
        self.classification_ce = nn.CrossEntropyLoss()
        self.classification_focal = nn.CrossEntropyLoss()  # Can be replaced with focal loss
    
    def __call__(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate combined loss.
        
        Args:
            outputs: Model outputs containing 'segmentation' and 'classification'
            targets: Ground truth targets
            
        Returns:
            Dictionary of loss components
        """
        seg_pred = outputs["segmentation"]
        cls_pred = outputs["classification"]
        seg_target = targets["segmentation"]
        cls_target = targets["classification"]
        
        # Segmentation losses
        dice_loss = self.dice_loss(seg_pred, seg_target)
        ce_seg_loss = self.ce_loss(seg_pred, seg_target)
        
        # Combined segmentation loss
        seg_loss = self.dice_weight * dice_loss + self.ce_weight * ce_seg_loss
        
        # Classification loss
        cls_loss = self.classification_ce(cls_pred, cls_target)
        
        # Total loss
        total_loss = (
            self.segmentation_weight * seg_loss +
            self.classification_weight * cls_loss
        )
        
        return {
            "total_loss": total_loss,
            "segmentation_loss": seg_loss,
            "classification_loss": cls_loss,
            "dice_loss": dice_loss,
            "ce_seg_loss": ce_seg_loss
        }


class MetricsCalculator:
    """Calculate training and validation metrics."""
    
    def __init__(self, num_classes: int = 3, num_cls_classes: int = 4):
        """
        Initialize metrics calculator.
        
        Args:
            num_classes: Number of segmentation classes
            num_cls_classes: Number of classification classes
        """
        self.num_classes = num_classes
        self.num_cls_classes = num_cls_classes
        
        # Segmentation metrics
        self.dice_metric = DiceMetric(
            include_background=False,
            reduction="mean",
            get_not_nans=False
        )
        
        self.hausdorff_metric = HausdorffDistanceMetric(
            include_background=False,
            reduction="mean",
            get_not_nans=False,
            percentile=95
        )
        
        self.surface_distance_metric = SurfaceDistanceMetric(
            include_background=False,
            reduction="mean",
            get_not_nans=False
        )
        
        self.iou_metric = MeanIoU(
            include_background=False,
            reduction="mean",
            get_not_nans=False
        )
        
        # Classification metrics
        self.confusion_matrix = ConfusionMatrixMetric(
            metric_name="accuracy",
            compute_sample=False
        )
        
        self.auc_metric = ROCAUCMetric(average="macro")
    
    def calculate_segmentation_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Calculate segmentation metrics.
        
        Args:
            predictions: Predicted segmentation masks
            targets: Ground truth segmentation masks
            
        Returns:
            Dictionary of segmentation metrics
        """
        # Convert to one-hot if needed
        if predictions.shape[1] > 1:  # Multi-class predictions
            pred_onehot = torch.argmax(predictions, dim=1, keepdim=True)
        else:
            pred_onehot = predictions
        
        if targets.shape[1] == 1:  # Single channel targets
            target_onehot = one_hot(targets, num_classes=self.num_classes)
            pred_onehot = one_hot(pred_onehot, num_classes=self.num_classes)
        else:
            target_onehot = targets
        
        # Calculate metrics
        dice_score = self.dice_metric(pred_onehot, target_onehot)
        hausdorff_dist = self.hausdorff_metric(pred_onehot, target_onehot)
        iou_score = self.iou_metric(pred_onehot, target_onehot)
        
        return {
            "dice_score": dice_score.mean().item(),
            "hausdorff_distance": hausdorff_dist.mean().item(),
            "mean_iou": iou_score.mean().item()
        }
    
    def calculate_classification_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Calculate classification metrics.
        
        Args:
            predictions: Predicted class probabilities
            targets: Ground truth class labels
            
        Returns:
            Dictionary of classification metrics
        """
        # Convert predictions to probabilities
        probs = torch.softmax(predictions, dim=1)
        pred_labels = torch.argmax(probs, dim=1)
        
        # Calculate accuracy
        accuracy = (pred_labels == targets).float().mean().item()
        
        # Calculate AUC (requires one-hot targets)
        targets_onehot = one_hot(targets.unsqueeze(1), num_classes=self.num_cls_classes)
        auc_score = self.auc_metric(probs, targets_onehot).item()
        
        # Calculate precision, recall, F1
        # Simple implementation for multi-class
        tp = torch.zeros(self.num_cls_classes)
        fp = torch.zeros(self.num_cls_classes)
        fn = torch.zeros(self.num_cls_classes)
        
        for cls in range(self.num_cls_classes):
            tp[cls] = ((pred_labels == cls) & (targets == cls)).sum().item()
            fp[cls] = ((pred_labels == cls) & (targets != cls)).sum().item()
            fn[cls] = ((pred_labels != cls) & (targets == cls)).sum().item()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        return {
            "accuracy": accuracy,
            "auc_score": auc_score,
            "precision": precision.mean().item(),
            "recall": recall.mean().item(),
            "f1_score": f1.mean().item()
        }


class TrainingOrchestrator:
    """
    Main training orchestrator for MONAI SwinUNETR model.
    
    Handles complete training workflow including data loading, training loops,
    validation, checkpointing, and metrics tracking.
    """
    
    def __init__(
        self,
        model_manager: ModelManager,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_dir: Path
    ):
        """
        Initialize training orchestrator.
        
        Args:
            model_manager: Model manager instance
            config: Training configuration
            train_loader: Training data loader
            val_loader: Validation data loader
            output_dir: Output directory for checkpoints and logs
        """
        self.model_manager = model_manager
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.logs_dir = self.output_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        
        # Set deterministic behavior
        if config.deterministic:
            set_determinism(seed=config.seed)
        
        # Initialize components
        self.loss_function = LossFunction(
            segmentation_weight=config.segmentation_weight,
            classification_weight=config.classification_weight,
            dice_weight=config.dice_weight,
            ce_weight=config.ce_weight
        )
        
        self.metrics_calculator = MetricsCalculator()
        
        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        self.early_stopping_counter = 0
        self.training_history: List[TrainingMetrics] = []
        
        # Initialize model, optimizer, scheduler
        self._setup_training()
        
        logger.info("TrainingOrchestrator initialized")
    
    def _setup_training(self):
        """Setup model, optimizer, and scheduler."""
        # Create model if not exists
        if self.model_manager.model is None:
            self.model_manager.create_model()
        
        self.model = self.model_manager.model
        
        # Setup optimizer
        if self.config.optimizer_type == "AdamW":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type == "SGD":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type == "Novograd":
            self.optimizer = Novograd(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer_type}")
        
        # Setup scheduler
        if self.config.scheduler_type == "CosineAnnealingLR":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.max_epochs
            )
        elif self.config.scheduler_type == "StepLR":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif self.config.scheduler_type == "ReduceLROnPlateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=0.5,
                patience=5
            )
        else:
            raise ValueError(f"Unsupported scheduler: {self.config.scheduler_type}")
        
        # Setup mixed precision training
        if self.config.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        epoch_losses = defaultdict(list)
        epoch_seg_metrics = defaultdict(list)
        epoch_cls_metrics = defaultdict(list)
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch_data in enumerate(progress_bar):
            # Move data to device
            inputs = batch_data["image"].to(self.model_manager.device)
            seg_targets = batch_data["segmentation"].to(self.model_manager.device)
            cls_targets = batch_data["classification"].to(self.model_manager.device)
            
            targets = {
                "segmentation": seg_targets,
                "classification": cls_targets
            }
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.config.use_amp and self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    losses = self.loss_function(outputs, targets)
                
                # Backward pass
                self.scaler.scale(losses["total_loss"]).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                losses = self.loss_function(outputs, targets)
                
                # Backward pass
                losses["total_loss"].backward()
                self.optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                seg_metrics = self.metrics_calculator.calculate_segmentation_metrics(
                    outputs["segmentation"], seg_targets
                )
                cls_metrics = self.metrics_calculator.calculate_classification_metrics(
                    outputs["classification"], cls_targets
                )
            
            # Accumulate metrics
            for key, value in losses.items():
                epoch_losses[key].append(value.item())
            
            for key, value in seg_metrics.items():
                epoch_seg_metrics[key].append(value)
            
            for key, value in cls_metrics.items():
                epoch_cls_metrics[key].append(value)
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": losses["total_loss"].item(),
                "dice": seg_metrics["dice_score"],
                "acc": cls_metrics["accuracy"]
            })
        
        # Calculate epoch averages
        epoch_metrics = {}
        for key, values in epoch_losses.items():
            epoch_metrics[key] = np.mean(values)
        
        for key, values in epoch_seg_metrics.items():
            epoch_metrics[key] = np.mean(values)
        
        for key, values in epoch_cls_metrics.items():
            epoch_metrics[key] = np.mean(values)
        
        return epoch_metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        epoch_losses = defaultdict(list)
        epoch_seg_metrics = defaultdict(list)
        epoch_cls_metrics = defaultdict(list)
        
        with torch.no_grad():
            for batch_data in tqdm(self.val_loader, desc="Validation"):
                # Move data to device
                inputs = batch_data["image"].to(self.model_manager.device)
                seg_targets = batch_data["segmentation"].to(self.model_manager.device)
                cls_targets = batch_data["classification"].to(self.model_manager.device)
                
                targets = {
                    "segmentation": seg_targets,
                    "classification": cls_targets
                }
                
                # Forward pass
                outputs = self.model(inputs)
                losses = self.loss_function(outputs, targets)
                
                # Calculate metrics
                seg_metrics = self.metrics_calculator.calculate_segmentation_metrics(
                    outputs["segmentation"], seg_targets
                )
                cls_metrics = self.metrics_calculator.calculate_classification_metrics(
                    outputs["classification"], cls_targets
                )
                
                # Accumulate metrics
                for key, value in losses.items():
                    epoch_losses[key].append(value.item())
                
                for key, value in seg_metrics.items():
                    epoch_seg_metrics[key].append(value)
                
                for key, value in cls_metrics.items():
                    epoch_cls_metrics[key].append(value)
        
        # Calculate epoch averages
        epoch_metrics = {}
        for key, values in epoch_losses.items():
            epoch_metrics[key] = np.mean(values)
        
        for key, values in epoch_seg_metrics.items():
            epoch_metrics[key] = np.mean(values)
        
        for key, values in epoch_cls_metrics.items():
            epoch_metrics[key] = np.mean(values)
        
        return epoch_metrics
    
    def train(self) -> List[TrainingMetrics]:
        """
        Run complete training loop.
        
        Returns:
            List of training metrics for each epoch
        """
        logger.info(f"Starting training for {self.config.max_epochs} epochs")
        
        for epoch in range(self.config.max_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Training phase
            train_metrics = self.train_epoch()
            
            # Validation phase
            val_metrics = {}
            if epoch % self.config.validation_interval == 0:
                val_metrics = self.validate_epoch()
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Get GPU memory usage
            gpu_memory_mb = 0.0
            if torch.cuda.is_available():
                gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            
            # Create training metrics
            training_metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=train_metrics.get("total_loss", 0.0),
                val_loss=val_metrics.get("total_loss", 0.0),
                segmentation_loss=train_metrics.get("segmentation_loss", 0.0),
                classification_loss=train_metrics.get("classification_loss", 0.0),
                dice_score=val_metrics.get("dice_score", train_metrics.get("dice_score", 0.0)),
                hausdorff_distance=val_metrics.get("hausdorff_distance", train_metrics.get("hausdorff_distance", 0.0)),
                mean_iou=val_metrics.get("mean_iou", train_metrics.get("mean_iou", 0.0)),
                accuracy=val_metrics.get("accuracy", train_metrics.get("accuracy", 0.0)),
                auc_score=val_metrics.get("auc_score", train_metrics.get("auc_score", 0.0)),
                precision=val_metrics.get("precision", train_metrics.get("precision", 0.0)),
                recall=val_metrics.get("recall", train_metrics.get("recall", 0.0)),
                f1_score=val_metrics.get("f1_score", train_metrics.get("f1_score", 0.0)),
                learning_rate=self.optimizer.param_groups[0]["lr"],
                epoch_time_seconds=epoch_time,
                gpu_memory_mb=gpu_memory_mb
            )
            
            self.training_history.append(training_metrics)
            
            # Log metrics
            logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {training_metrics.train_loss:.4f}, "
                f"Val Loss: {training_metrics.val_loss:.4f}, "
                f"Dice: {training_metrics.dice_score:.4f}, "
                f"Accuracy: {training_metrics.accuracy:.4f}"
            )
            
            # Update scheduler
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                current_metric = getattr(training_metrics, self.config.validation_metric)
                self.scheduler.step(current_metric)
            else:
                self.scheduler.step()
            
            # Check for improvement
            current_metric = getattr(training_metrics, self.config.validation_metric)
            if current_metric > self.best_metric + self.config.early_stopping_min_delta:
                self.best_metric = current_metric
                self.early_stopping_counter = 0
                
                # Save best model
                if self.config.save_best_only:
                    self._save_checkpoint(epoch, training_metrics, is_best=True)
            else:
                self.early_stopping_counter += 1
            
            # Regular checkpointing
            if epoch % self.config.checkpoint_interval == 0:
                self._save_checkpoint(epoch, training_metrics, is_best=False)
            
            # Early stopping
            if self.early_stopping_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break
            
            # Save training history
            self._save_training_history()
        
        logger.info("Training completed")
        return self.training_history
    
    def _save_checkpoint(self, epoch: int, metrics: TrainingMetrics, is_best: bool = False):
        """Save model checkpoint."""
        suffix = "best" if is_best else f"epoch_{epoch}"
        checkpoint_path = self.checkpoint_dir / f"model_{suffix}.pth"
        
        # Convert TrainingMetrics to ModelMetrics
        model_metrics = ModelMetrics(
            dice_score=metrics.dice_score,
            hausdorff_distance=metrics.hausdorff_distance,
            auc_score=metrics.auc_score,
            accuracy=metrics.accuracy,
            precision=metrics.precision,
            recall=metrics.recall,
            f1_score=metrics.f1_score,
            computation_time_ms=metrics.epoch_time_seconds * 1000
        )
        
        self.model_manager.save_checkpoint(
            checkpoint_path=checkpoint_path,
            model_version=f"training_epoch_{epoch}",
            epoch=epoch,
            metrics=model_metrics,
            optimizer_state=self.optimizer.state_dict(),
            additional_info={
                "scheduler_state": self.scheduler.state_dict(),
                "training_config": self.config.__dict__,
                "best_metric": self.best_metric
            }
        )
        
        # Cleanup old checkpoints
        if not is_best:
            self.model_manager.cleanup_old_checkpoints(self.config.max_checkpoints)
    
    def _save_training_history(self):
        """Save training history to JSON file."""
        history_file = self.logs_dir / "training_history.json"
        
        # Convert to serializable format
        history_data = []
        for metrics in self.training_history:
            history_data.append({
                "epoch": metrics.epoch,
                "train_loss": metrics.train_loss,
                "val_loss": metrics.val_loss,
                "segmentation_loss": metrics.segmentation_loss,
                "classification_loss": metrics.classification_loss,
                "dice_score": metrics.dice_score,
                "hausdorff_distance": metrics.hausdorff_distance,
                "mean_iou": metrics.mean_iou,
                "accuracy": metrics.accuracy,
                "auc_score": metrics.auc_score,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1_score": metrics.f1_score,
                "learning_rate": metrics.learning_rate,
                "epoch_time_seconds": metrics.epoch_time_seconds,
                "gpu_memory_mb": metrics.gpu_memory_mb,
                "timestamp": metrics.timestamp.isoformat()
            })
        
        with open(history_file, "w") as f:
            json.dump(history_data, f, indent=2)
    
    def plot_training_curves(self, save_path: Optional[Path] = None):
        """Plot training curves."""
        if not self.training_history:
            logger.warning("No training history to plot")
            return
        
        epochs = [m.epoch for m in self.training_history]
        train_losses = [m.train_loss for m in self.training_history]
        val_losses = [m.val_loss for m in self.training_history]
        dice_scores = [m.dice_score for m in self.training_history]
        accuracies = [m.accuracy for m in self.training_history]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss curves
        axes[0, 0].plot(epochs, train_losses, label="Train Loss")
        axes[0, 0].plot(epochs, val_losses, label="Val Loss")
        axes[0, 0].set_title("Loss Curves")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Dice score
        axes[0, 1].plot(epochs, dice_scores, label="Dice Score", color="green")
        axes[0, 1].set_title("Dice Score")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Dice Score")
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Accuracy
        axes[1, 0].plot(epochs, accuracies, label="Accuracy", color="orange")
        axes[1, 0].set_title("Classification Accuracy")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Accuracy")
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate
        learning_rates = [m.learning_rate for m in self.training_history]
        axes[1, 1].plot(epochs, learning_rates, label="Learning Rate", color="red")
        axes[1, 1].set_title("Learning Rate Schedule")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Learning Rate")
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 1].set_yscale("log")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.savefig(self.logs_dir / "training_curves.png", dpi=300, bbox_inches="tight")
        
        plt.close()


def create_training_orchestrator(
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: Path,
    config: Optional[TrainingConfig] = None
) -> TrainingOrchestrator:
    """
    Create a training orchestrator with default configuration.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        output_dir: Output directory for checkpoints and logs
        config: Optional training configuration
        
    Returns:
        Configured TrainingOrchestrator instance
    """
    from src.services.ml_inference import create_default_model_manager
    
    if config is None:
        settings = get_settings()
        config = TrainingConfig(
            max_epochs=settings.monai.max_epochs,
            batch_size=settings.monai.batch_size,
            learning_rate=settings.monai.learning_rate
        )
    
    model_manager = create_default_model_manager()
    
    return TrainingOrchestrator(
        model_manager=model_manager,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=output_dir
    )