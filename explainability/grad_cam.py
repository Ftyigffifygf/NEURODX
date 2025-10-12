"""
Grad-CAM visualization implementation for SwinUNETR model.

This module provides Grad-CAM (Gradient-weighted Class Activation Mapping) 
visualization compatible with MONAI SwinUNETR architecture for both 
segmentation and classification tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import cv2

from monai.transforms import Resize, ScaleIntensity
from monai.utils import ensure_tuple_rep

from src.services.ml_inference.swin_unetr_model import MultiTaskSwinUNETR

logger = logging.getLogger(__name__)


@dataclass
class GradCAMResult:
    """Result from Grad-CAM analysis."""
    attention_maps: Dict[str, np.ndarray]  # Layer name -> attention map
    overlay_images: Dict[str, np.ndarray]  # Layer name -> overlay visualization
    target_class: Union[str, int]
    confidence_score: float
    layer_names: List[str]
    input_shape: Tuple[int, ...]
    metadata: Dict[str, Any]


class GradCAMHook:
    """Hook for capturing gradients and activations."""
    
    def __init__(self):
        self.gradients = None
        self.activations = None
    
    def save_gradient(self, grad):
        """Save gradients during backward pass."""
        self.gradients = grad
    
    def save_activation(self, module, input, output):
        """Save activations during forward pass."""
        self.activations = output


class GradCAMVisualizer:
    """
    Grad-CAM visualizer for SwinUNETR model.
    
    Generates attention maps showing which regions of the input image
    are most important for the model's predictions.
    """
    
    def __init__(
        self,
        model: MultiTaskSwinUNETR,
        target_layers: Optional[List[str]] = None,
        use_guided_gradcam: bool = False
    ):
        """
        Initialize Grad-CAM visualizer.
        
        Args:
            model: SwinUNETR model instance
            target_layers: List of layer names to analyze (if None, uses default layers)
            use_guided_gradcam: Whether to use guided Grad-CAM
        """
        self.model = model
        self.model.eval()
        self.use_guided_gradcam = use_guided_gradcam
        
        # Default target layers for SwinUNETR
        if target_layers is None:
            self.target_layers = [
                "swin_unetr.swinViT.layers1.0.blocks.1.norm1",  # Early stage
                "swin_unetr.swinViT.layers2.0.blocks.1.norm1",  # Mid stage
                "swin_unetr.swinViT.layers3.0.blocks.1.norm1",  # Late stage
                "swin_unetr.encoder4"  # Final encoder layer
            ]
        else:
            self.target_layers = target_layers
        
        # Initialize hooks
        self.hooks = {}
        self.hook_handles = []
        self._register_hooks()
        
        logger.info(f"GradCAM initialized with target layers: {self.target_layers}")
    
    def _register_hooks(self):
        """Register forward and backward hooks on target layers."""
        for layer_name in self.target_layers:
            try:
                # Navigate to the target layer
                layer = self._get_layer_by_name(layer_name)
                if layer is not None:
                    hook = GradCAMHook()
                    self.hooks[layer_name] = hook
                    
                    # Register forward hook
                    handle = layer.register_forward_hook(hook.save_activation)
                    self.hook_handles.append(handle)
                    
                    logger.debug(f"Registered hook for layer: {layer_name}")
                else:
                    logger.warning(f"Layer not found: {layer_name}")
                    
            except Exception as e:
                logger.error(f"Failed to register hook for {layer_name}: {e}")
    
    def _get_layer_by_name(self, layer_name: str) -> Optional[nn.Module]:
        """Get layer by name from the model."""
        try:
            # Split the layer name by dots and navigate through the model
            parts = layer_name.split('.')
            layer = self.model
            
            for part in parts:
                if hasattr(layer, part):
                    layer = getattr(layer, part)
                else:
                    return None
            
            return layer
            
        except Exception as e:
            logger.error(f"Error getting layer {layer_name}: {e}")
            return None
    
    def _register_backward_hooks(self, target_class_idx: int, task_type: str = "classification"):
        """Register backward hooks for gradient computation."""
        for layer_name, hook in self.hooks.items():
            if hook.activations is not None:
                # Register backward hook on activations
                hook.activations.register_hook(hook.save_gradient)
    
    def generate_gradcam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        task_type: str = "classification",
        layer_name: Optional[str] = None
    ) -> GradCAMResult:
        """
        Generate Grad-CAM visualization.
        
        Args:
            input_tensor: Input tensor [1, C, H, W, D]
            target_class: Target class index (if None, uses predicted class)
            task_type: "classification" or "segmentation"
            layer_name: Specific layer to analyze (if None, analyzes all target layers)
            
        Returns:
            GradCAM result with attention maps and visualizations
        """
        if input_tensor.dim() != 5:
            raise ValueError("Input tensor must be 5D: [batch, channels, height, width, depth]")
        
        # Ensure model is in eval mode and gradients are enabled
        self.model.eval()
        input_tensor.requires_grad_(True)
        
        # Forward pass
        with torch.enable_grad():
            outputs = self.model(input_tensor)
        
        # Determine target class
        if task_type == "classification":
            class_output = outputs["classification"]
            if target_class is None:
                target_class = torch.argmax(class_output, dim=1).item()
            
            # Get the score for target class
            target_score = class_output[0, target_class]
            
        elif task_type == "segmentation":
            seg_output = outputs["segmentation"]
            if target_class is None:
                # Use the most frequent predicted class
                pred_mask = torch.argmax(seg_output, dim=1)
                target_class = torch.mode(pred_mask.flatten()).values.item()
            
            # Sum of target class probabilities across spatial dimensions
            target_score = torch.sum(seg_output[0, target_class])
        
        else:
            raise ValueError("task_type must be 'classification' or 'segmentation'")
        
        # Backward pass to compute gradients
        self.model.zero_grad()
        target_score.backward(retain_graph=True)
        
        # Generate attention maps
        attention_maps = {}
        overlay_images = {}
        
        layers_to_process = [layer_name] if layer_name else self.target_layers
        
        for layer_name in layers_to_process:
            if layer_name in self.hooks:
                hook = self.hooks[layer_name]
                
                if hook.activations is not None and hook.gradients is not None:
                    # Compute Grad-CAM
                    attention_map = self._compute_gradcam(
                        hook.activations, hook.gradients
                    )
                    
                    # Resize attention map to input size
                    resized_attention = self._resize_attention_map(
                        attention_map, input_tensor.shape[2:]
                    )
                    
                    attention_maps[layer_name] = resized_attention
                    
                    # Create overlay visualization
                    overlay = self._create_overlay_visualization(
                        input_tensor, resized_attention
                    )
                    overlay_images[layer_name] = overlay
        
        # Calculate confidence score
        if task_type == "classification":
            confidence_score = F.softmax(class_output, dim=1)[0, target_class].item()
        else:
            # For segmentation, use mean probability of target class
            seg_probs = F.softmax(seg_output, dim=1)
            confidence_score = torch.mean(seg_probs[0, target_class]).item()
        
        return GradCAMResult(
            attention_maps=attention_maps,
            overlay_images=overlay_images,
            target_class=target_class,
            confidence_score=confidence_score,
            layer_names=list(attention_maps.keys()),
            input_shape=input_tensor.shape,
            metadata={
                "task_type": task_type,
                "model_type": "SwinUNETR",
                "use_guided_gradcam": self.use_guided_gradcam
            }
        )
    
    def _compute_gradcam(
        self,
        activations: torch.Tensor,
        gradients: torch.Tensor
    ) -> np.ndarray:
        """
        Compute Grad-CAM attention map from activations and gradients.
        
        Args:
            activations: Feature activations [B, C, H, W, D]
            gradients: Gradients w.r.t. activations [B, C, H, W, D]
            
        Returns:
            Attention map as numpy array
        """
        # Global average pooling of gradients to get importance weights
        weights = torch.mean(gradients, dim=(2, 3, 4), keepdim=True)  # [B, C, 1, 1, 1]
        
        # Weighted combination of activations
        weighted_activations = weights * activations  # [B, C, H, W, D]
        attention_map = torch.sum(weighted_activations, dim=1, keepdim=True)  # [B, 1, H, W, D]
        
        # Apply ReLU to focus on positive contributions
        attention_map = F.relu(attention_map)
        
        # Normalize to [0, 1]
        attention_map = attention_map.squeeze()  # Remove batch and channel dims
        
        if attention_map.numel() > 0:
            min_val = torch.min(attention_map)
            max_val = torch.max(attention_map)
            if max_val > min_val:
                attention_map = (attention_map - min_val) / (max_val - min_val)
        
        return attention_map.detach().cpu().numpy()
    
    def _resize_attention_map(
        self,
        attention_map: np.ndarray,
        target_shape: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        Resize attention map to match input tensor spatial dimensions.
        
        Args:
            attention_map: Attention map to resize
            target_shape: Target spatial shape (H, W, D)
            
        Returns:
            Resized attention map
        """
        if attention_map.shape == target_shape:
            return attention_map
        
        # Use MONAI's Resize transform for 3D medical images
        resize_transform = Resize(spatial_size=target_shape, mode="trilinear")
        
        # Add batch and channel dimensions for transform
        attention_tensor = torch.from_numpy(attention_map).unsqueeze(0).unsqueeze(0)
        resized_tensor = resize_transform(attention_tensor)
        
        return resized_tensor.squeeze().numpy()
    
    def _create_overlay_visualization(
        self,
        input_tensor: torch.Tensor,
        attention_map: np.ndarray,
        alpha: float = 0.4
    ) -> np.ndarray:
        """
        Create overlay visualization of attention map on input image.
        
        Args:
            input_tensor: Original input tensor [1, C, H, W, D]
            attention_map: Attention map [H, W, D]
            alpha: Blending factor for overlay
            
        Returns:
            Overlay visualization as numpy array
        """
        # Convert input tensor to numpy and normalize
        input_np = input_tensor[0, 0].detach().cpu().numpy()  # Use first channel
        
        # Normalize input to [0, 1]
        input_normalized = self._normalize_image(input_np)
        
        # Create heatmap from attention map
        heatmap = self._create_heatmap(attention_map)
        
        # Blend input image with heatmap
        overlay = alpha * heatmap + (1 - alpha) * input_normalized
        
        return np.clip(overlay, 0, 1)
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range."""
        min_val = np.min(image)
        max_val = np.max(image)
        
        if max_val > min_val:
            return (image - min_val) / (max_val - min_val)
        else:
            return np.zeros_like(image)
    
    def _create_heatmap(self, attention_map: np.ndarray) -> np.ndarray:
        """
        Create colored heatmap from attention map.
        
        Args:
            attention_map: Attention map values [0, 1]
            
        Returns:
            Colored heatmap
        """
        # For 3D data, we'll create a simple intensity-based heatmap
        # In practice, you might want to use a colormap like jet or viridis
        
        # Simple red-based heatmap
        heatmap = np.zeros_like(attention_map)
        heatmap = attention_map  # Use attention values directly
        
        return heatmap
    
    def generate_multi_layer_gradcam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        task_type: str = "classification"
    ) -> Dict[str, GradCAMResult]:
        """
        Generate Grad-CAM for multiple layers.
        
        Args:
            input_tensor: Input tensor [1, C, H, W, D]
            target_class: Target class index
            task_type: "classification" or "segmentation"
            
        Returns:
            Dictionary mapping layer names to GradCAM results
        """
        results = {}
        
        for layer_name in self.target_layers:
            try:
                result = self.generate_gradcam(
                    input_tensor=input_tensor,
                    target_class=target_class,
                    task_type=task_type,
                    layer_name=layer_name
                )
                results[layer_name] = result
                
            except Exception as e:
                logger.error(f"Failed to generate Grad-CAM for layer {layer_name}: {e}")
        
        return results
    
    def save_visualization(
        self,
        result: GradCAMResult,
        output_dir: Path,
        slice_indices: Optional[List[int]] = None
    ):
        """
        Save Grad-CAM visualizations to files.
        
        Args:
            result: GradCAM result to save
            output_dir: Output directory
            slice_indices: Specific slice indices to save (if None, saves middle slices)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine slice indices to save
        if slice_indices is None:
            depth = result.input_shape[-1]
            slice_indices = [depth // 4, depth // 2, 3 * depth // 4]
        
        for layer_name, attention_map in result.attention_maps.items():
            layer_dir = output_dir / layer_name.replace('.', '_')
            layer_dir.mkdir(parents=True, exist_ok=True)
            
            # Save attention maps for specific slices
            for slice_idx in slice_indices:
                if slice_idx < attention_map.shape[-1]:
                    # Save attention map slice
                    attention_slice = attention_map[:, :, slice_idx]
                    attention_path = layer_dir / f"attention_slice_{slice_idx}.npy"
                    np.save(attention_path, attention_slice)
                    
                    # Save overlay if available
                    if layer_name in result.overlay_images:
                        overlay_slice = result.overlay_images[layer_name][:, :, slice_idx]
                        overlay_path = layer_dir / f"overlay_slice_{slice_idx}.npy"
                        np.save(overlay_path, overlay_slice)
        
        # Save metadata
        metadata_path = output_dir / "gradcam_metadata.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump({
                "target_class": result.target_class,
                "confidence_score": result.confidence_score,
                "layer_names": result.layer_names,
                "input_shape": result.input_shape,
                "metadata": result.metadata
            }, f, indent=2)
        
        logger.info(f"Grad-CAM visualizations saved to {output_dir}")
    
    def cleanup(self):
        """Remove all registered hooks."""
        for handle in self.hook_handles:
            handle.remove()
        
        self.hooks.clear()
        self.hook_handles.clear()
        
        logger.info("Grad-CAM hooks cleaned up")
    
    def __del__(self):
        """Cleanup hooks when object is destroyed."""
        self.cleanup()


def create_gradcam_visualizer(
    model: MultiTaskSwinUNETR,
    target_layers: Optional[List[str]] = None
) -> GradCAMVisualizer:
    """
    Create a Grad-CAM visualizer for the given model.
    
    Args:
        model: SwinUNETR model instance
        target_layers: Target layers for analysis
        
    Returns:
        Configured GradCAMVisualizer
    """
    return GradCAMVisualizer(
        model=model,
        target_layers=target_layers,
        use_guided_gradcam=False
    )


def validate_gradcam_setup(model: MultiTaskSwinUNETR) -> bool:
    """
    Validate that Grad-CAM can be applied to the model.
    
    Args:
        model: SwinUNETR model to validate
        
    Returns:
        True if Grad-CAM setup is valid
    """
    try:
        # Create visualizer
        visualizer = create_gradcam_visualizer(model)
        
        # Test with dummy input
        dummy_input = torch.randn(1, 4, 96, 96, 96)
        
        # Generate Grad-CAM
        result = visualizer.generate_gradcam(
            input_tensor=dummy_input,
            task_type="classification"
        )
        
        # Validate result
        assert len(result.attention_maps) > 0
        assert result.confidence_score >= 0.0
        assert isinstance(result.target_class, int)
        
        # Cleanup
        visualizer.cleanup()
        
        logger.info("Grad-CAM setup validation successful")
        return True
        
    except Exception as e:
        logger.error(f"Grad-CAM setup validation failed: {e}")
        return False