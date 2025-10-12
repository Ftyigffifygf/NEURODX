"""
Integrated Gradients analysis implementation for SwinUNETR model.

This module provides Integrated Gradients feature attribution analysis
for understanding which input features contribute most to model predictions.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass
import logging
from pathlib import Path
from tqdm import tqdm

from src.services.ml_inference.swin_unetr_model import MultiTaskSwinUNETR

logger = logging.getLogger(__name__)


@dataclass
class IntegratedGradientsResult:
    """Result from Integrated Gradients analysis."""
    attribution_maps: Dict[str, np.ndarray]  # Modality -> attribution map
    integrated_gradients: np.ndarray  # Combined attribution map
    baseline_input: np.ndarray  # Baseline used for integration
    target_class: Union[str, int]
    confidence_score: float
    convergence_delta: float  # Measure of path integration accuracy
    num_steps: int
    input_shape: Tuple[int, ...]
    metadata: Dict[str, Any]


class BaselineGenerator:
    """Generate baseline inputs for Integrated Gradients."""
    
    @staticmethod
    def zero_baseline(input_tensor: torch.Tensor) -> torch.Tensor:
        """Generate zero baseline."""
        return torch.zeros_like(input_tensor)
    
    @staticmethod
    def gaussian_noise_baseline(
        input_tensor: torch.Tensor,
        noise_scale: float = 0.1
    ) -> torch.Tensor:
        """Generate Gaussian noise baseline."""
        return torch.randn_like(input_tensor) * noise_scale
    
    @staticmethod
    def uniform_noise_baseline(
        input_tensor: torch.Tensor,
        low: float = -0.1,
        high: float = 0.1
    ) -> torch.Tensor:
        """Generate uniform noise baseline."""
        return torch.empty_like(input_tensor).uniform_(low, high)
    
    @staticmethod
    def mean_baseline(input_tensor: torch.Tensor) -> torch.Tensor:
        """Generate baseline using input mean values."""
        # Calculate mean across spatial dimensions for each channel
        means = torch.mean(input_tensor, dim=(2, 3, 4), keepdim=True)
        return means.expand_as(input_tensor)
    
    @staticmethod
    def blurred_baseline(
        input_tensor: torch.Tensor,
        kernel_size: int = 15,
        sigma: float = 5.0
    ) -> torch.Tensor:
        """Generate blurred version of input as baseline."""
        # Simple approximation of Gaussian blur using average pooling
        # In practice, you might want to use proper Gaussian filtering
        
        # Apply average pooling to blur the image
        pooled = F.avg_pool3d(
            input_tensor,
            kernel_size=min(kernel_size, min(input_tensor.shape[2:])),
            stride=1,
            padding=kernel_size // 2
        )
        
        return pooled
    
    @staticmethod
    def get_baseline(
        input_tensor: torch.Tensor,
        baseline_type: str = "zero",
        **kwargs
    ) -> torch.Tensor:
        """
        Get baseline input based on specified type.
        
        Args:
            input_tensor: Original input tensor
            baseline_type: Type of baseline ("zero", "gaussian", "uniform", "mean", "blurred")
            **kwargs: Additional arguments for baseline generation
            
        Returns:
            Baseline tensor
        """
        if baseline_type == "zero":
            return BaselineGenerator.zero_baseline(input_tensor)
        elif baseline_type == "gaussian":
            return BaselineGenerator.gaussian_noise_baseline(input_tensor, **kwargs)
        elif baseline_type == "uniform":
            return BaselineGenerator.uniform_noise_baseline(input_tensor, **kwargs)
        elif baseline_type == "mean":
            return BaselineGenerator.mean_baseline(input_tensor)
        elif baseline_type == "blurred":
            return BaselineGenerator.blurred_baseline(input_tensor, **kwargs)
        else:
            raise ValueError(f"Unknown baseline type: {baseline_type}")


class IntegratedGradientsAnalyzer:
    """
    Integrated Gradients analyzer for SwinUNETR model.
    
    Computes feature attributions by integrating gradients along a path
    from a baseline input to the actual input.
    """
    
    def __init__(
        self,
        model: MultiTaskSwinUNETR,
        baseline_type: str = "zero",
        num_steps: int = 50,
        batch_size: int = 4
    ):
        """
        Initialize Integrated Gradients analyzer.
        
        Args:
            model: SwinUNETR model instance
            baseline_type: Type of baseline to use
            num_steps: Number of integration steps
            batch_size: Batch size for processing integration steps
        """
        self.model = model
        self.model.eval()
        self.baseline_type = baseline_type
        self.num_steps = num_steps
        self.batch_size = batch_size
        
        logger.info(f"IntegratedGradients initialized with {num_steps} steps, baseline: {baseline_type}")
    
    def compute_integrated_gradients(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        task_type: str = "classification",
        baseline_tensor: Optional[torch.Tensor] = None,
        return_convergence_delta: bool = True
    ) -> IntegratedGradientsResult:
        """
        Compute Integrated Gradients attribution.
        
        Args:
            input_tensor: Input tensor [1, C, H, W, D]
            target_class: Target class index (if None, uses predicted class)
            task_type: "classification" or "segmentation"
            baseline_tensor: Custom baseline (if None, generates based on baseline_type)
            return_convergence_delta: Whether to compute convergence delta
            
        Returns:
            Integrated Gradients result
        """
        if input_tensor.dim() != 5:
            raise ValueError("Input tensor must be 5D: [batch, channels, height, width, depth]")
        
        device = input_tensor.device
        
        # Generate baseline if not provided
        if baseline_tensor is None:
            baseline_tensor = BaselineGenerator.get_baseline(
                input_tensor, self.baseline_type
            ).to(device)
        
        # Get target class and initial prediction
        with torch.no_grad():
            outputs = self.model(input_tensor)
            
            if task_type == "classification":
                class_output = outputs["classification"]
                if target_class is None:
                    target_class = torch.argmax(class_output, dim=1).item()
                confidence_score = F.softmax(class_output, dim=1)[0, target_class].item()
                
            elif task_type == "segmentation":
                seg_output = outputs["segmentation"]
                if target_class is None:
                    pred_mask = torch.argmax(seg_output, dim=1)
                    target_class = torch.mode(pred_mask.flatten()).values.item()
                
                seg_probs = F.softmax(seg_output, dim=1)
                confidence_score = torch.mean(seg_probs[0, target_class]).item()
            else:
                raise ValueError("task_type must be 'classification' or 'segmentation'")
        
        # Compute integrated gradients
        integrated_gradients = self._integrate_gradients(
            input_tensor=input_tensor,
            baseline_tensor=baseline_tensor,
            target_class=target_class,
            task_type=task_type
        )
        
        # Compute convergence delta if requested
        convergence_delta = 0.0
        if return_convergence_delta:
            convergence_delta = self._compute_convergence_delta(
                input_tensor=input_tensor,
                baseline_tensor=baseline_tensor,
                integrated_gradients=integrated_gradients,
                target_class=target_class,
                task_type=task_type
            )
        
        # Create attribution maps per modality (assuming multi-modal input)
        attribution_maps = self._create_modality_attributions(
            integrated_gradients, input_tensor.shape[1]
        )
        
        return IntegratedGradientsResult(
            attribution_maps=attribution_maps,
            integrated_gradients=integrated_gradients.cpu().numpy(),
            baseline_input=baseline_tensor.cpu().numpy(),
            target_class=target_class,
            confidence_score=confidence_score,
            convergence_delta=convergence_delta,
            num_steps=self.num_steps,
            input_shape=input_tensor.shape,
            metadata={
                "task_type": task_type,
                "baseline_type": self.baseline_type,
                "model_type": "SwinUNETR"
            }
        )
    
    def _integrate_gradients(
        self,
        input_tensor: torch.Tensor,
        baseline_tensor: torch.Tensor,
        target_class: int,
        task_type: str
    ) -> torch.Tensor:
        """
        Integrate gradients along the path from baseline to input.
        
        Args:
            input_tensor: Original input tensor
            baseline_tensor: Baseline tensor
            target_class: Target class for attribution
            task_type: Task type ("classification" or "segmentation")
            
        Returns:
            Integrated gradients tensor
        """
        # Initialize integrated gradients
        integrated_gradients = torch.zeros_like(input_tensor)
        
        # Create interpolation steps
        alphas = torch.linspace(0, 1, self.num_steps + 1, device=input_tensor.device)
        
        # Process in batches to manage memory
        for i in tqdm(range(0, len(alphas), self.batch_size), desc="Computing Integrated Gradients"):
            batch_alphas = alphas[i:i + self.batch_size]
            
            # Create interpolated inputs
            batch_inputs = []
            for alpha in batch_alphas:
                interpolated = baseline_tensor + alpha * (input_tensor - baseline_tensor)
                batch_inputs.append(interpolated)
            
            if batch_inputs:
                batch_tensor = torch.cat(batch_inputs, dim=0)
                batch_tensor.requires_grad_(True)
                
                # Forward pass
                batch_outputs = self.model(batch_tensor)
                
                # Compute target scores
                if task_type == "classification":
                    target_scores = batch_outputs["classification"][:, target_class]
                    target_score = torch.sum(target_scores)
                else:  # segmentation
                    seg_outputs = batch_outputs["segmentation"][:, target_class]
                    target_score = torch.sum(seg_outputs)
                
                # Backward pass
                self.model.zero_grad()
                target_score.backward()
                
                # Accumulate gradients
                if batch_tensor.grad is not None:
                    # Average gradients across the batch
                    batch_gradients = batch_tensor.grad
                    for j, grad in enumerate(batch_gradients):
                        integrated_gradients += grad.unsqueeze(0) / self.num_steps
        
        # Multiply by (input - baseline) to get final attributions
        integrated_gradients = integrated_gradients * (input_tensor - baseline_tensor)
        
        return integrated_gradients
    
    def _compute_convergence_delta(
        self,
        input_tensor: torch.Tensor,
        baseline_tensor: torch.Tensor,
        integrated_gradients: torch.Tensor,
        target_class: int,
        task_type: str
    ) -> float:
        """
        Compute convergence delta to validate integration accuracy.
        
        The convergence delta measures how well the integrated gradients
        approximate the difference in model output between input and baseline.
        
        Args:
            input_tensor: Original input tensor
            baseline_tensor: Baseline tensor
            integrated_gradients: Computed integrated gradients
            target_class: Target class
            task_type: Task type
            
        Returns:
            Convergence delta value
        """
        with torch.no_grad():
            # Get model outputs for input and baseline
            input_output = self.model(input_tensor)
            baseline_output = self.model(baseline_tensor)
            
            if task_type == "classification":
                input_score = input_output["classification"][0, target_class]
                baseline_score = baseline_output["classification"][0, target_class]
            else:  # segmentation
                input_score = torch.sum(input_output["segmentation"][0, target_class])
                baseline_score = torch.sum(baseline_output["segmentation"][0, target_class])
            
            # Actual difference in model outputs
            actual_diff = input_score - baseline_score
            
            # Approximated difference using integrated gradients
            approximated_diff = torch.sum(integrated_gradients)
            
            # Convergence delta (smaller is better)
            delta = torch.abs(actual_diff - approximated_diff).item()
            
            return delta
    
    def _create_modality_attributions(
        self,
        integrated_gradients: torch.Tensor,
        num_channels: int
    ) -> Dict[str, np.ndarray]:
        """
        Create per-modality attribution maps.
        
        Args:
            integrated_gradients: Integrated gradients tensor [1, C, H, W, D]
            num_channels: Number of input channels
            
        Returns:
            Dictionary mapping modality names to attribution maps
        """
        attribution_maps = {}
        
        # Assuming standard multi-modal setup
        modality_names = ["MRI", "CT", "Ultrasound", "Fused"]
        
        for i in range(min(num_channels, len(modality_names))):
            modality_name = modality_names[i] if i < len(modality_names) else f"Channel_{i}"
            attribution_map = integrated_gradients[0, i].cpu().numpy()
            attribution_maps[modality_name] = attribution_map
        
        # Create combined attribution (sum across channels)
        if num_channels > 1:
            combined_attribution = torch.sum(integrated_gradients[0], dim=0).cpu().numpy()
            attribution_maps["Combined"] = combined_attribution
        
        return attribution_maps
    
    def compute_multi_baseline_analysis(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        task_type: str = "classification",
        baseline_types: List[str] = None
    ) -> Dict[str, IntegratedGradientsResult]:
        """
        Compute Integrated Gradients with multiple baseline types.
        
        Args:
            input_tensor: Input tensor
            target_class: Target class index
            task_type: Task type
            baseline_types: List of baseline types to compare
            
        Returns:
            Dictionary mapping baseline types to results
        """
        if baseline_types is None:
            baseline_types = ["zero", "gaussian", "mean", "blurred"]
        
        results = {}
        
        for baseline_type in baseline_types:
            try:
                # Temporarily change baseline type
                original_baseline_type = self.baseline_type
                self.baseline_type = baseline_type
                
                result = self.compute_integrated_gradients(
                    input_tensor=input_tensor,
                    target_class=target_class,
                    task_type=task_type
                )
                
                results[baseline_type] = result
                
                # Restore original baseline type
                self.baseline_type = original_baseline_type
                
            except Exception as e:
                logger.error(f"Failed to compute IG with baseline {baseline_type}: {e}")
        
        return results
    
    def compute_feature_importance_ranking(
        self,
        result: IntegratedGradientsResult,
        top_k: int = 100
    ) -> Dict[str, List[Tuple[Tuple[int, ...], float]]]:
        """
        Rank features by importance based on attribution values.
        
        Args:
            result: Integrated Gradients result
            top_k: Number of top features to return
            
        Returns:
            Dictionary mapping modalities to ranked feature lists
        """
        feature_rankings = {}
        
        for modality, attribution_map in result.attribution_maps.items():
            # Get absolute attribution values
            abs_attributions = np.abs(attribution_map)
            
            # Get flattened indices and values
            flat_indices = np.unravel_index(
                np.argsort(abs_attributions.ravel())[-top_k:],
                abs_attributions.shape
            )
            
            # Create list of (coordinates, attribution_value) tuples
            ranked_features = []
            for i in range(len(flat_indices[0])):
                coords = tuple(idx[i] for idx in flat_indices)
                value = attribution_map[coords]
                ranked_features.append((coords, value))
            
            # Sort by absolute attribution value (descending)
            ranked_features.sort(key=lambda x: abs(x[1]), reverse=True)
            
            feature_rankings[modality] = ranked_features
        
        return feature_rankings
    
    def save_attribution_maps(
        self,
        result: IntegratedGradientsResult,
        output_dir: Path,
        slice_indices: Optional[List[int]] = None,
        save_nifti: bool = True
    ):
        """
        Save attribution maps to files.
        
        Args:
            result: Integrated Gradients result
            output_dir: Output directory
            slice_indices: Specific slice indices to save
            save_nifti: Whether to save as NIfTI files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine slice indices
        if slice_indices is None:
            depth = result.input_shape[-1]
            slice_indices = [depth // 4, depth // 2, 3 * depth // 4]
        
        for modality, attribution_map in result.attribution_maps.items():
            modality_dir = output_dir / modality.lower()
            modality_dir.mkdir(parents=True, exist_ok=True)
            
            # Save full attribution map
            full_path = modality_dir / "attribution_map.npy"
            np.save(full_path, attribution_map)
            
            # Save specific slices
            for slice_idx in slice_indices:
                if slice_idx < attribution_map.shape[-1]:
                    slice_data = attribution_map[:, :, slice_idx]
                    slice_path = modality_dir / f"attribution_slice_{slice_idx}.npy"
                    np.save(slice_path, slice_data)
            
            # Save as NIfTI if requested
            if save_nifti:
                try:
                    import nibabel as nib
                    nifti_path = modality_dir / "attribution_map.nii.gz"
                    nifti_img = nib.Nifti1Image(attribution_map, affine=np.eye(4))
                    nib.save(nifti_img, nifti_path)
                except ImportError:
                    logger.warning("nibabel not available, skipping NIfTI export")
        
        # Save metadata
        metadata_path = output_dir / "integrated_gradients_metadata.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump({
                "target_class": result.target_class,
                "confidence_score": result.confidence_score,
                "convergence_delta": result.convergence_delta,
                "num_steps": result.num_steps,
                "input_shape": result.input_shape,
                "metadata": result.metadata
            }, f, indent=2)
        
        logger.info(f"Attribution maps saved to {output_dir}")
    
    def visualize_attributions(
        self,
        result: IntegratedGradientsResult,
        input_tensor: torch.Tensor,
        slice_index: Optional[int] = None,
        modality: str = "Combined"
    ) -> np.ndarray:
        """
        Create visualization of attributions overlaid on input.
        
        Args:
            result: Integrated Gradients result
            input_tensor: Original input tensor
            slice_index: Slice to visualize (if None, uses middle slice)
            modality: Modality to visualize
            
        Returns:
            Visualization as numpy array
        """
        if modality not in result.attribution_maps:
            raise ValueError(f"Modality {modality} not found in attribution maps")
        
        attribution_map = result.attribution_maps[modality]
        
        # Get slice index
        if slice_index is None:
            slice_index = attribution_map.shape[-1] // 2
        
        # Extract slices
        attribution_slice = attribution_map[:, :, slice_index]
        input_slice = input_tensor[0, 0, :, :, slice_index].cpu().numpy()
        
        # Normalize input slice
        input_normalized = (input_slice - input_slice.min()) / (input_slice.max() - input_slice.min())
        
        # Normalize attribution slice
        abs_attribution = np.abs(attribution_slice)
        if abs_attribution.max() > 0:
            attribution_normalized = abs_attribution / abs_attribution.max()
        else:
            attribution_normalized = abs_attribution
        
        # Create overlay (simple red overlay for positive attributions)
        overlay = np.zeros((*input_slice.shape, 3))
        overlay[:, :, 0] = input_normalized  # Red channel
        overlay[:, :, 1] = input_normalized  # Green channel  
        overlay[:, :, 2] = input_normalized  # Blue channel
        
        # Add attribution as red overlay
        overlay[:, :, 0] = np.clip(overlay[:, :, 0] + 0.5 * attribution_normalized, 0, 1)
        
        return overlay


def create_integrated_gradients_analyzer(
    model: MultiTaskSwinUNETR,
    baseline_type: str = "zero",
    num_steps: int = 50
) -> IntegratedGradientsAnalyzer:
    """
    Create an Integrated Gradients analyzer for the given model.
    
    Args:
        model: SwinUNETR model instance
        baseline_type: Type of baseline to use
        num_steps: Number of integration steps
        
    Returns:
        Configured IntegratedGradientsAnalyzer
    """
    return IntegratedGradientsAnalyzer(
        model=model,
        baseline_type=baseline_type,
        num_steps=num_steps
    )


def validate_integrated_gradients_setup(model: MultiTaskSwinUNETR) -> bool:
    """
    Validate that Integrated Gradients can be applied to the model.
    
    Args:
        model: SwinUNETR model to validate
        
    Returns:
        True if Integrated Gradients setup is valid
    """
    try:
        # Create analyzer
        analyzer = create_integrated_gradients_analyzer(model, num_steps=5)  # Use fewer steps for validation
        
        # Test with dummy input
        dummy_input = torch.randn(1, 4, 96, 96, 96)
        
        # Compute Integrated Gradients
        result = analyzer.compute_integrated_gradients(
            input_tensor=dummy_input,
            task_type="classification"
        )
        
        # Validate result
        assert len(result.attribution_maps) > 0
        assert result.confidence_score >= 0.0
        assert isinstance(result.target_class, int)
        assert result.convergence_delta >= 0.0
        
        logger.info("Integrated Gradients setup validation successful")
        return True
        
    except Exception as e:
        logger.error(f"Integrated Gradients setup validation failed: {e}")
        return False