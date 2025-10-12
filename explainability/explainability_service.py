"""
Main explainability service for NeuroDx-MultiModal system.

This module provides a unified interface for all explainability features,
combining Grad-CAM and Integrated Gradients analysis for comprehensive
model interpretation.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging
from pathlib import Path
import json

from src.services.explainability.grad_cam import GradCAMVisualizer, GradCAMResult
from src.services.explainability.integrated_gradients import (
    IntegratedGradientsAnalyzer, IntegratedGradientsResult
)
from src.services.ml_inference.swin_unetr_model import MultiTaskSwinUNETR
from src.models.diagnostics import DiagnosticResult

logger = logging.getLogger(__name__)


@dataclass
class ExplainabilityReport:
    """Comprehensive explainability report combining multiple methods."""
    patient_id: str
    study_ids: List[str]
    target_class: Union[str, int]
    confidence_score: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Grad-CAM results
    gradcam_results: Optional[Dict[str, GradCAMResult]] = None
    
    # Integrated Gradients results
    integrated_gradients_result: Optional[IntegratedGradientsResult] = None
    
    # Summary statistics
    summary_stats: Dict[str, Any] = field(default_factory=dict)
    
    # Visualization paths
    visualization_paths: Dict[str, Path] = field(default_factory=dict)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExplainabilityConfig:
    """Configuration for explainability analysis."""
    
    # Grad-CAM settings
    gradcam_enabled: bool = True
    gradcam_target_layers: Optional[List[str]] = None
    gradcam_guided: bool = False
    
    # Integrated Gradients settings
    integrated_gradients_enabled: bool = True
    ig_baseline_type: str = "zero"
    ig_num_steps: int = 50
    ig_batch_size: int = 4
    
    # Analysis settings
    task_type: str = "classification"  # "classification" or "segmentation"
    target_class: Optional[int] = None
    
    # Output settings
    save_visualizations: bool = True
    output_directory: Optional[Path] = None
    slice_indices: Optional[List[int]] = None
    
    # Performance settings
    use_gpu: bool = True
    max_memory_gb: float = 8.0


class ExplainabilityService:
    """
    Main service for model explainability analysis.
    
    Provides a unified interface for Grad-CAM and Integrated Gradients
    analysis, with options for comprehensive reporting and visualization.
    """
    
    def __init__(
        self,
        model: MultiTaskSwinUNETR,
        config: Optional[ExplainabilityConfig] = None
    ):
        """
        Initialize explainability service.
        
        Args:
            model: SwinUNETR model instance
            config: Explainability configuration
        """
        self.model = model
        self.config = config or ExplainabilityConfig()
        
        # Initialize analyzers
        self.gradcam_visualizer = None
        self.ig_analyzer = None
        
        if self.config.gradcam_enabled:
            self.gradcam_visualizer = GradCAMVisualizer(
                model=self.model,
                target_layers=self.config.gradcam_target_layers,
                use_guided_gradcam=self.config.gradcam_guided
            )
        
        if self.config.integrated_gradients_enabled:
            self.ig_analyzer = IntegratedGradientsAnalyzer(
                model=self.model,
                baseline_type=self.config.ig_baseline_type,
                num_steps=self.config.ig_num_steps,
                batch_size=self.config.ig_batch_size
            )
        
        logger.info("ExplainabilityService initialized")
    
    def analyze_prediction(
        self,
        input_tensor: torch.Tensor,
        patient_id: str,
        study_ids: List[str],
        target_class: Optional[int] = None,
        custom_config: Optional[ExplainabilityConfig] = None
    ) -> ExplainabilityReport:
        """
        Perform comprehensive explainability analysis.
        
        Args:
            input_tensor: Input tensor [1, C, H, W, D]
            patient_id: Patient identifier
            study_ids: List of study identifiers
            target_class: Target class for analysis
            custom_config: Custom configuration for this analysis
            
        Returns:
            Comprehensive explainability report
        """
        config = custom_config or self.config
        
        # Use provided target class or determine from model prediction
        if target_class is None:
            target_class = self._get_predicted_class(input_tensor, config.task_type)
        
        # Get confidence score
        confidence_score = self._get_confidence_score(input_tensor, target_class, config.task_type)
        
        # Initialize report
        report = ExplainabilityReport(
            patient_id=patient_id,
            study_ids=study_ids,
            target_class=target_class,
            confidence_score=confidence_score
        )
        
        # Perform Grad-CAM analysis
        if config.gradcam_enabled and self.gradcam_visualizer:
            try:
                gradcam_results = self._perform_gradcam_analysis(
                    input_tensor, target_class, config
                )
                report.gradcam_results = gradcam_results
                logger.info("Grad-CAM analysis completed")
                
            except Exception as e:
                logger.error(f"Grad-CAM analysis failed: {e}")
        
        # Perform Integrated Gradients analysis
        if config.integrated_gradients_enabled and self.ig_analyzer:
            try:
                ig_result = self._perform_integrated_gradients_analysis(
                    input_tensor, target_class, config
                )
                report.integrated_gradients_result = ig_result
                logger.info("Integrated Gradients analysis completed")
                
            except Exception as e:
                logger.error(f"Integrated Gradients analysis failed: {e}")
        
        # Generate summary statistics
        report.summary_stats = self._generate_summary_stats(report)
        
        # Save visualizations if requested
        if config.save_visualizations and config.output_directory:
            visualization_paths = self._save_visualizations(report, config)
            report.visualization_paths = visualization_paths
        
        # Add metadata
        report.metadata = {
            "model_type": "SwinUNETR",
            "task_type": config.task_type,
            "input_shape": input_tensor.shape,
            "analysis_config": {
                "gradcam_enabled": config.gradcam_enabled,
                "ig_enabled": config.integrated_gradients_enabled,
                "ig_baseline_type": config.ig_baseline_type,
                "ig_num_steps": config.ig_num_steps
            }
        }
        
        logger.info(f"Explainability analysis completed for patient {patient_id}")
        return report
    
    def _get_predicted_class(self, input_tensor: torch.Tensor, task_type: str) -> int:
        """Get predicted class from model output."""
        with torch.no_grad():
            outputs = self.model(input_tensor)
            
            if task_type == "classification":
                return torch.argmax(outputs["classification"], dim=1).item()
            else:  # segmentation
                pred_mask = torch.argmax(outputs["segmentation"], dim=1)
                return torch.mode(pred_mask.flatten()).values.item()
    
    def _get_confidence_score(
        self, 
        input_tensor: torch.Tensor, 
        target_class: int, 
        task_type: str
    ) -> float:
        """Get confidence score for target class."""
        with torch.no_grad():
            outputs = self.model(input_tensor)
            
            if task_type == "classification":
                probs = torch.softmax(outputs["classification"], dim=1)
                return probs[0, target_class].item()
            else:  # segmentation
                probs = torch.softmax(outputs["segmentation"], dim=1)
                return torch.mean(probs[0, target_class]).item()
    
    def _perform_gradcam_analysis(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        config: ExplainabilityConfig
    ) -> Dict[str, GradCAMResult]:
        """Perform Grad-CAM analysis."""
        return self.gradcam_visualizer.generate_multi_layer_gradcam(
            input_tensor=input_tensor,
            target_class=target_class,
            task_type=config.task_type
        )
    
    def _perform_integrated_gradients_analysis(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        config: ExplainabilityConfig
    ) -> IntegratedGradientsResult:
        """Perform Integrated Gradients analysis."""
        return self.ig_analyzer.compute_integrated_gradients(
            input_tensor=input_tensor,
            target_class=target_class,
            task_type=config.task_type
        )
    
    def _generate_summary_stats(self, report: ExplainabilityReport) -> Dict[str, Any]:
        """Generate summary statistics from analysis results."""
        stats = {
            "target_class": report.target_class,
            "confidence_score": report.confidence_score,
            "analysis_timestamp": report.timestamp.isoformat()
        }
        
        # Grad-CAM statistics
        if report.gradcam_results:
            gradcam_stats = {
                "num_layers_analyzed": len(report.gradcam_results),
                "layer_names": list(report.gradcam_results.keys())
            }
            
            # Average attention statistics across layers
            attention_values = []
            for result in report.gradcam_results.values():
                for attention_map in result.attention_maps.values():
                    attention_values.extend(attention_map.flatten())
            
            if attention_values:
                gradcam_stats.update({
                    "mean_attention": float(np.mean(attention_values)),
                    "std_attention": float(np.std(attention_values)),
                    "max_attention": float(np.max(attention_values)),
                    "min_attention": float(np.min(attention_values))
                })
            
            stats["gradcam"] = gradcam_stats
        
        # Integrated Gradients statistics
        if report.integrated_gradients_result:
            ig_result = report.integrated_gradients_result
            
            # Attribution statistics
            all_attributions = []
            for attribution_map in ig_result.attribution_maps.values():
                all_attributions.extend(attribution_map.flatten())
            
            ig_stats = {
                "convergence_delta": ig_result.convergence_delta,
                "num_steps": ig_result.num_steps,
                "baseline_type": ig_result.metadata.get("baseline_type", "unknown"),
                "num_modalities": len(ig_result.attribution_maps)
            }
            
            if all_attributions:
                ig_stats.update({
                    "mean_attribution": float(np.mean(all_attributions)),
                    "std_attribution": float(np.std(all_attributions)),
                    "max_attribution": float(np.max(all_attributions)),
                    "min_attribution": float(np.min(all_attributions)),
                    "positive_attribution_ratio": float(np.mean(np.array(all_attributions) > 0))
                })
            
            stats["integrated_gradients"] = ig_stats
        
        return stats
    
    def _save_visualizations(
        self,
        report: ExplainabilityReport,
        config: ExplainabilityConfig
    ) -> Dict[str, Path]:
        """Save visualization files."""
        output_dir = config.output_directory
        patient_dir = output_dir / f"patient_{report.patient_id}" / f"analysis_{report.timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        visualization_paths = {}
        
        # Save Grad-CAM visualizations
        if report.gradcam_results:
            gradcam_dir = patient_dir / "gradcam"
            
            for layer_name, result in report.gradcam_results.items():
                layer_dir = gradcam_dir / layer_name.replace('.', '_')
                self.gradcam_visualizer.save_visualization(
                    result, layer_dir, config.slice_indices
                )
                visualization_paths[f"gradcam_{layer_name}"] = layer_dir
        
        # Save Integrated Gradients visualizations
        if report.integrated_gradients_result:
            ig_dir = patient_dir / "integrated_gradients"
            self.ig_analyzer.save_attribution_maps(
                report.integrated_gradients_result, ig_dir, config.slice_indices
            )
            visualization_paths["integrated_gradients"] = ig_dir
        
        # Save summary report
        report_path = patient_dir / "explainability_report.json"
        self._save_report_json(report, report_path)
        visualization_paths["report"] = report_path
        
        return visualization_paths
    
    def _save_report_json(self, report: ExplainabilityReport, output_path: Path):
        """Save explainability report as JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare serializable data
        report_data = {
            "patient_id": report.patient_id,
            "study_ids": report.study_ids,
            "target_class": report.target_class,
            "confidence_score": report.confidence_score,
            "timestamp": report.timestamp.isoformat(),
            "summary_stats": report.summary_stats,
            "metadata": report.metadata
        }
        
        # Add Grad-CAM summary
        if report.gradcam_results:
            gradcam_summary = {}
            for layer_name, result in report.gradcam_results.items():
                gradcam_summary[layer_name] = {
                    "target_class": result.target_class,
                    "confidence_score": result.confidence_score,
                    "layer_names": result.layer_names,
                    "input_shape": result.input_shape
                }
            report_data["gradcam_summary"] = gradcam_summary
        
        # Add Integrated Gradients summary
        if report.integrated_gradients_result:
            ig_result = report.integrated_gradients_result
            report_data["integrated_gradients_summary"] = {
                "target_class": ig_result.target_class,
                "confidence_score": ig_result.confidence_score,
                "convergence_delta": ig_result.convergence_delta,
                "num_steps": ig_result.num_steps,
                "modalities": list(ig_result.attribution_maps.keys()),
                "input_shape": ig_result.input_shape
            }
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Explainability report saved to {output_path}")
    
    def compare_explanations(
        self,
        input_tensor: torch.Tensor,
        patient_id: str,
        study_ids: List[str],
        baseline_types: List[str] = None,
        target_layers: List[str] = None
    ) -> Dict[str, ExplainabilityReport]:
        """
        Compare explanations across different configurations.
        
        Args:
            input_tensor: Input tensor
            patient_id: Patient identifier
            study_ids: Study identifiers
            baseline_types: Different baseline types for IG comparison
            target_layers: Different target layers for Grad-CAM comparison
            
        Returns:
            Dictionary mapping configuration names to reports
        """
        if baseline_types is None:
            baseline_types = ["zero", "gaussian", "mean"]
        
        comparison_results = {}
        
        # Compare different IG baseline types
        for baseline_type in baseline_types:
            config = ExplainabilityConfig(
                gradcam_enabled=False,
                integrated_gradients_enabled=True,
                ig_baseline_type=baseline_type,
                save_visualizations=False
            )
            
            try:
                report = self.analyze_prediction(
                    input_tensor, patient_id, study_ids, custom_config=config
                )
                comparison_results[f"ig_baseline_{baseline_type}"] = report
                
            except Exception as e:
                logger.error(f"Comparison failed for baseline {baseline_type}: {e}")
        
        # Compare different Grad-CAM layers if specified
        if target_layers:
            for layer in target_layers:
                config = ExplainabilityConfig(
                    gradcam_enabled=True,
                    gradcam_target_layers=[layer],
                    integrated_gradients_enabled=False,
                    save_visualizations=False
                )
                
                try:
                    report = self.analyze_prediction(
                        input_tensor, patient_id, study_ids, custom_config=config
                    )
                    comparison_results[f"gradcam_layer_{layer.replace('.', '_')}"] = report
                    
                except Exception as e:
                    logger.error(f"Comparison failed for layer {layer}: {e}")
        
        return comparison_results
    
    def get_feature_importance_summary(
        self,
        report: ExplainabilityReport,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Get summary of most important features from explainability analysis.
        
        Args:
            report: Explainability report
            top_k: Number of top features to include
            
        Returns:
            Feature importance summary
        """
        summary = {
            "patient_id": report.patient_id,
            "target_class": report.target_class,
            "confidence_score": report.confidence_score
        }
        
        # Get top features from Integrated Gradients
        if report.integrated_gradients_result and self.ig_analyzer:
            feature_rankings = self.ig_analyzer.compute_feature_importance_ranking(
                report.integrated_gradients_result, top_k
            )
            
            summary["top_features_by_modality"] = {}
            for modality, features in feature_rankings.items():
                summary["top_features_by_modality"][modality] = [
                    {
                        "coordinates": coords,
                        "attribution_value": float(value),
                        "rank": i + 1
                    }
                    for i, (coords, value) in enumerate(features[:top_k])
                ]
        
        # Get attention statistics from Grad-CAM
        if report.gradcam_results:
            gradcam_summary = {}
            for layer_name, result in report.gradcam_results.items():
                attention_stats = {}
                for map_name, attention_map in result.attention_maps.items():
                    # Find peak attention regions
                    flat_attention = attention_map.flatten()
                    top_indices = np.argsort(flat_attention)[-top_k:]
                    top_coords = np.unravel_index(top_indices, attention_map.shape)
                    
                    attention_stats[map_name] = [
                        {
                            "coordinates": tuple(coord[i] for coord in top_coords),
                            "attention_value": float(flat_attention[idx]),
                            "rank": i + 1
                        }
                        for i, idx in enumerate(reversed(top_indices))
                    ]
                
                gradcam_summary[layer_name] = attention_stats
            
            summary["top_attention_regions"] = gradcam_summary
        
        return summary
    
    def cleanup(self):
        """Clean up resources."""
        if self.gradcam_visualizer:
            self.gradcam_visualizer.cleanup()
        
        logger.info("ExplainabilityService cleaned up")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.cleanup()


def create_explainability_service(
    model: MultiTaskSwinUNETR,
    config: Optional[ExplainabilityConfig] = None
) -> ExplainabilityService:
    """
    Create an explainability service for the given model.
    
    Args:
        model: SwinUNETR model instance
        config: Explainability configuration
        
    Returns:
        Configured ExplainabilityService
    """
    return ExplainabilityService(model=model, config=config)


def validate_explainability_service(model: MultiTaskSwinUNETR) -> bool:
    """
    Validate that the explainability service works correctly.
    
    Args:
        model: SwinUNETR model to validate
        
    Returns:
        True if validation successful
    """
    try:
        # Create service with minimal configuration
        config = ExplainabilityConfig(
            gradcam_enabled=True,
            integrated_gradients_enabled=True,
            ig_num_steps=5,  # Use fewer steps for validation
            save_visualizations=False
        )
        
        service = create_explainability_service(model, config)
        
        # Test with dummy input
        dummy_input = torch.randn(1, 4, 96, 96, 96)
        
        # Perform analysis
        report = service.analyze_prediction(
            input_tensor=dummy_input,
            patient_id="TEST_001",
            study_ids=["STUDY_001"]
        )
        
        # Validate report
        assert report.patient_id == "TEST_001"
        assert report.gradcam_results is not None
        assert report.integrated_gradients_result is not None
        assert len(report.summary_stats) > 0
        
        # Cleanup
        service.cleanup()
        
        logger.info("Explainability service validation successful")
        return True
        
    except Exception as e:
        logger.error(f"Explainability service validation failed: {e}")
        return False