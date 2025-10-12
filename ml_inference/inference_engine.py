"""
Inference engine for real-time predictions using MONAI SwinUNETR model.

This module provides the InferenceEngine for model prediction requests,
batch processing capabilities, and confidence score calculation.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging
import time
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

from monai.transforms import (
    Compose, EnsureChannelFirstd, Spacingd, NormalizeIntensityd,
    ToTensord, EnsureTyped, Activationsd, AsDiscreted
)
from monai.data import MetaTensor, decollate_batch
from monai.inferers import sliding_window_inference
from monai.utils import BlendMode, PytorchPadMode

from src.services.ml_inference.swin_unetr_model import ModelManager, MultiTaskSwinUNETR
from src.models.diagnostics import (
    DiagnosticResult, SegmentationResult, ClassificationResult, 
    ModelMetrics, DiagnosticConfidence, DiseaseStage,
    create_diagnostic_result
)
from src.models.patient import PatientRecord, ImagingStudy
from src.config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    """Request for model inference."""
    request_id: str
    patient_id: str
    study_ids: List[str]
    input_data: Dict[str, torch.Tensor]  # Multi-modal input tensors
    modalities_used: List[str]
    priority: int = 1  # 1=high, 2=medium, 3=low
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceResult:
    """Result from model inference."""
    request_id: str
    patient_id: str
    segmentation_output: torch.Tensor
    classification_output: torch.Tensor
    confidence_scores: Dict[str, float]
    processing_time_ms: float
    gpu_memory_used_mb: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class InferencePreprocessor:
    """Preprocessing pipeline for inference data."""
    
    def __init__(self, target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        """
        Initialize preprocessing pipeline.
        
        Args:
            target_spacing: Target voxel spacing for resampling
        """
        self.target_spacing = target_spacing
        
        # Define preprocessing transforms
        self.transforms = Compose([
            EnsureChannelFirstd(keys=["image"]),
            Spacingd(keys=["image"], pixdim=target_spacing, mode="bilinear"),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            ToTensord(keys=["image"]),
            EnsureTyped(keys=["image"], device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        ])
        
        # Post-processing transforms for outputs
        self.post_transforms = Compose([
            EnsureTyped(keys=["pred_seg", "pred_cls"]),
            Activationsd(keys=["pred_seg"], softmax=True),
            Activationsd(keys=["pred_cls"], softmax=True),
            AsDiscreted(keys=["pred_seg"], argmax=True)
        ])
    
    def preprocess(self, input_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Preprocess input data for inference.
        
        Args:
            input_data: Raw input data dictionary
            
        Returns:
            Preprocessed tensor data
        """
        try:
            # Apply preprocessing transforms
            processed_data = self.transforms(input_data)
            return processed_data
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise
    
    def postprocess(self, model_outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Postprocess model outputs.
        
        Args:
            model_outputs: Raw model outputs
            
        Returns:
            Postprocessed outputs
        """
        try:
            # Prepare data for post-processing
            post_data = {
                "pred_seg": model_outputs["segmentation"],
                "pred_cls": model_outputs["classification"]
            }
            
            # Apply post-processing transforms
            processed_outputs = self.post_transforms(post_data)
            return processed_outputs
            
        except Exception as e:
            logger.error(f"Postprocessing failed: {e}")
            raise


class ConfidenceCalculator:
    """Calculate confidence scores for model predictions."""
    
    @staticmethod
    def calculate_segmentation_confidence(
        segmentation_probs: torch.Tensor,
        segmentation_mask: torch.Tensor
    ) -> Dict[str, float]:
        """
        Calculate confidence scores for segmentation results.
        
        Args:
            segmentation_probs: Probability maps [C, H, W, D]
            segmentation_mask: Discrete segmentation mask [H, W, D]
            
        Returns:
            Dictionary of confidence metrics
        """
        with torch.no_grad():
            # Overall confidence (mean of max probabilities)
            max_probs = torch.max(segmentation_probs, dim=0)[0]
            overall_confidence = torch.mean(max_probs).item()
            
            # Per-class confidence
            class_confidences = {}
            for class_idx in range(segmentation_probs.shape[0]):
                class_mask = (segmentation_mask == class_idx)
                if torch.sum(class_mask) > 0:
                    class_probs = segmentation_probs[class_idx][class_mask]
                    class_confidences[f"class_{class_idx}"] = torch.mean(class_probs).item()
                else:
                    class_confidences[f"class_{class_idx}"] = 0.0
            
            # Uncertainty measures
            entropy = -torch.sum(segmentation_probs * torch.log(segmentation_probs + 1e-8), dim=0)
            mean_entropy = torch.mean(entropy).item()
            
            # Prediction margin (difference between top two probabilities)
            sorted_probs = torch.sort(segmentation_probs, dim=0, descending=True)[0]
            margin = sorted_probs[0] - sorted_probs[1]
            mean_margin = torch.mean(margin).item()
            
            return {
                "overall_confidence": overall_confidence,
                "mean_entropy": mean_entropy,
                "mean_margin": mean_margin,
                **class_confidences
            }
    
    @staticmethod
    def calculate_classification_confidence(
        classification_probs: torch.Tensor
    ) -> Dict[str, float]:
        """
        Calculate confidence scores for classification results.
        
        Args:
            classification_probs: Classification probabilities [num_classes]
            
        Returns:
            Dictionary of confidence metrics
        """
        with torch.no_grad():
            # Max probability (standard confidence)
            max_prob = torch.max(classification_probs).item()
            
            # Entropy-based uncertainty
            entropy = -torch.sum(classification_probs * torch.log(classification_probs + 1e-8)).item()
            
            # Prediction margin
            sorted_probs = torch.sort(classification_probs, descending=True)[0]
            margin = (sorted_probs[0] - sorted_probs[1]).item() if len(sorted_probs) > 1 else 1.0
            
            # Temperature-scaled confidence (assuming temperature=1)
            temperature_confidence = max_prob
            
            return {
                "max_probability": max_prob,
                "entropy": entropy,
                "margin": margin,
                "temperature_confidence": temperature_confidence
            }


class InferenceEngine:
    """
    Main inference engine for real-time predictions.
    
    Handles model loading, batch processing, and result formatting.
    """
    
    def __init__(
        self,
        model_manager: ModelManager,
        batch_size: int = 1,
        roi_size: Tuple[int, int, int] = (96, 96, 96),
        sw_batch_size: int = 4,
        overlap: float = 0.25,
        mode: BlendMode = BlendMode.GAUSSIAN,
        padding_mode: PytorchPadMode = PytorchPadMode.CONSTANT,
        max_workers: int = 4
    ):
        """
        Initialize inference engine.
        
        Args:
            model_manager: Model manager instance
            batch_size: Batch size for inference
            roi_size: ROI size for sliding window inference
            sw_batch_size: Sliding window batch size
            overlap: Overlap ratio for sliding window
            mode: Blending mode for sliding window
            padding_mode: Padding mode for sliding window
            max_workers: Maximum number of worker threads
        """
        self.model_manager = model_manager
        self.batch_size = batch_size
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap
        self.mode = mode
        self.padding_mode = padding_mode
        
        # Initialize components
        self.preprocessor = InferencePreprocessor()
        self.confidence_calculator = ConfidenceCalculator()
        
        # Thread pool for async processing
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Request queue and processing state
        self.request_queue: List[InferenceRequest] = []
        self.processing_lock = threading.Lock()
        self.is_processing = False
        
        # Performance tracking
        self.inference_stats = {
            "total_requests": 0,
            "successful_inferences": 0,
            "failed_inferences": 0,
            "average_processing_time_ms": 0.0,
            "total_processing_time_ms": 0.0
        }
        
        logger.info("InferenceEngine initialized")
    
    def load_model(self, checkpoint_path: Optional[Path] = None) -> bool:
        """
        Load model for inference.
        
        Args:
            checkpoint_path: Path to model checkpoint (optional)
            
        Returns:
            True if model loaded successfully
        """
        try:
            if checkpoint_path:
                self.model_manager.load_checkpoint(checkpoint_path)
            else:
                # Try to load latest checkpoint or create new model
                latest_checkpoint = self.model_manager.get_latest_checkpoint()
                if latest_checkpoint:
                    self.model_manager.load_checkpoint(latest_checkpoint.checkpoint_path)
                else:
                    self.model_manager.create_model()
            
            # Set model to evaluation mode
            if self.model_manager.model:
                self.model_manager.model.eval()
                logger.info("Model loaded and set to evaluation mode")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def predict_single(self, request: InferenceRequest) -> InferenceResult:
        """
        Perform inference on a single request.
        
        Args:
            request: Inference request
            
        Returns:
            Inference result
        """
        start_time = time.time()
        
        try:
            if self.model_manager.model is None:
                raise ValueError("Model not loaded")
            
            # Preprocess input data
            processed_data = self.preprocessor.preprocess(request.input_data)
            input_tensor = processed_data["image"]
            
            # Ensure batch dimension
            if input_tensor.dim() == 4:  # [C, H, W, D]
                input_tensor = input_tensor.unsqueeze(0)  # [1, C, H, W, D]
            
            # GPU memory tracking
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                initial_memory = torch.cuda.memory_allocated()
            
            # Perform inference with sliding window
            with torch.no_grad():
                outputs = sliding_window_inference(
                    inputs=input_tensor,
                    roi_size=self.roi_size,
                    sw_batch_size=self.sw_batch_size,
                    predictor=self.model_manager.model,
                    overlap=self.overlap,
                    mode=self.mode,
                    padding_mode=self.padding_mode,
                    device=self.model_manager.device
                )
            
            # Extract segmentation and classification outputs
            if isinstance(outputs, dict):
                segmentation_output = outputs["segmentation"]
                classification_output = outputs["classification"]
            else:
                # Handle case where model returns tuple or single tensor
                segmentation_output = outputs
                classification_output = torch.zeros(1, 4)  # Default classification
            
            # Calculate confidence scores
            seg_probs = F.softmax(segmentation_output, dim=1)
            cls_probs = F.softmax(classification_output, dim=1)
            
            seg_confidence = self.confidence_calculator.calculate_segmentation_confidence(
                seg_probs[0], torch.argmax(seg_probs[0], dim=0)
            )
            cls_confidence = self.confidence_calculator.calculate_classification_confidence(
                cls_probs[0]
            )
            
            # Combine confidence scores
            confidence_scores = {
                "segmentation": seg_confidence,
                "classification": cls_confidence,
                "overall": (seg_confidence["overall_confidence"] + cls_confidence["max_probability"]) / 2
            }
            
            # Calculate processing time and memory usage
            processing_time_ms = (time.time() - start_time) * 1000
            
            gpu_memory_used_mb = None
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated()
                gpu_memory_used_mb = (peak_memory - initial_memory) / (1024 * 1024)
            
            # Create result
            result = InferenceResult(
                request_id=request.request_id,
                patient_id=request.patient_id,
                segmentation_output=segmentation_output,
                classification_output=classification_output,
                confidence_scores=confidence_scores,
                processing_time_ms=processing_time_ms,
                gpu_memory_used_mb=gpu_memory_used_mb,
                metadata={
                    "modalities_used": request.modalities_used,
                    "roi_size": self.roi_size,
                    "sw_batch_size": self.sw_batch_size
                }
            )
            
            # Update statistics
            self._update_stats(processing_time_ms, success=True)
            
            logger.info(f"Inference completed for request {request.request_id} in {processing_time_ms:.2f}ms")
            return result
            
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            self._update_stats(processing_time_ms, success=False)
            logger.error(f"Inference failed for request {request.request_id}: {e}")
            raise
    
    def predict_batch(self, requests: List[InferenceRequest]) -> List[InferenceResult]:
        """
        Perform batch inference on multiple requests.
        
        Args:
            requests: List of inference requests
            
        Returns:
            List of inference results
        """
        results = []
        
        # Process requests in batches
        for i in range(0, len(requests), self.batch_size):
            batch_requests = requests[i:i + self.batch_size]
            
            # Process batch sequentially (can be parallelized further if needed)
            for request in batch_requests:
                try:
                    result = self.predict_single(request)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Batch inference failed for request {request.request_id}: {e}")
                    # Continue with other requests in batch
        
        return results
    
    async def predict_async(self, request: InferenceRequest) -> InferenceResult:
        """
        Perform asynchronous inference.
        
        Args:
            request: Inference request
            
        Returns:
            Inference result
        """
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self.executor, self.predict_single, request)
        return result
    
    def add_to_queue(self, request: InferenceRequest):
        """
        Add request to processing queue.
        
        Args:
            request: Inference request to queue
        """
        with self.processing_lock:
            self.request_queue.append(request)
            # Sort by priority (lower number = higher priority)
            self.request_queue.sort(key=lambda x: x.priority)
    
    def process_queue(self) -> List[InferenceResult]:
        """
        Process all requests in the queue.
        
        Returns:
            List of inference results
        """
        with self.processing_lock:
            if self.is_processing or not self.request_queue:
                return []
            
            self.is_processing = True
            requests_to_process = self.request_queue.copy()
            self.request_queue.clear()
        
        try:
            results = self.predict_batch(requests_to_process)
            return results
        finally:
            with self.processing_lock:
                self.is_processing = False
    
    def create_diagnostic_result(
        self,
        inference_result: InferenceResult,
        study_ids: List[str],
        modalities_used: List[str]
    ) -> DiagnosticResult:
        """
        Convert inference result to diagnostic result.
        
        Args:
            inference_result: Raw inference result
            study_ids: List of study IDs
            modalities_used: List of modalities used
            
        Returns:
            Formatted diagnostic result
        """
        # Extract segmentation data
        seg_output = inference_result.segmentation_output
        seg_probs = F.softmax(seg_output, dim=1)
        seg_mask = torch.argmax(seg_probs, dim=1).squeeze().cpu().numpy()
        
        # Create segmentation result
        class_prob_maps = {}
        for i in range(seg_probs.shape[1]):
            class_prob_maps[f"class_{i}"] = seg_probs[0, i].cpu().numpy()
        
        segmentation_result = SegmentationResult(
            segmentation_mask=seg_mask,
            class_probabilities=class_prob_maps
        )
        
        # Extract classification data
        cls_output = inference_result.classification_output
        cls_probs = F.softmax(cls_output, dim=1)
        cls_probs_dict = {f"class_{i}": prob.item() for i, prob in enumerate(cls_probs[0])}
        predicted_class = f"class_{torch.argmax(cls_probs).item()}"
        
        classification_result = ClassificationResult(
            predicted_class=predicted_class,
            class_probabilities=cls_probs_dict,
            confidence_score=inference_result.confidence_scores["classification"]["max_probability"]
        )
        
        # Create mock metrics (would be calculated from validation data in practice)
        metrics = ModelMetrics(
            dice_score=0.85,  # Would be calculated from ground truth
            hausdorff_distance=2.5,
            auc_score=0.92,
            computation_time_ms=inference_result.processing_time_ms
        )
        
        # Determine diagnostic confidence
        overall_confidence = inference_result.confidence_scores["overall"]
        if overall_confidence > 0.8:
            diagnostic_confidence = DiagnosticConfidence.HIGH
        elif overall_confidence > 0.6:
            diagnostic_confidence = DiagnosticConfidence.MEDIUM
        else:
            diagnostic_confidence = DiagnosticConfidence.LOW
        
        # Create diagnostic result
        diagnostic_result = DiagnosticResult(
            patient_id=inference_result.patient_id,
            study_ids=study_ids,
            timestamp=inference_result.timestamp,
            segmentation_result=segmentation_result,
            classification_result=classification_result,
            metrics=metrics,
            modalities_used=modalities_used,
            diagnostic_confidence=diagnostic_confidence,
            processing_time_seconds=inference_result.processing_time_ms / 1000,
            gpu_memory_used_mb=inference_result.gpu_memory_used_mb
        )
        
        return diagnostic_result
    
    def _update_stats(self, processing_time_ms: float, success: bool):
        """Update inference statistics."""
        self.inference_stats["total_requests"] += 1
        self.inference_stats["total_processing_time_ms"] += processing_time_ms
        
        if success:
            self.inference_stats["successful_inferences"] += 1
        else:
            self.inference_stats["failed_inferences"] += 1
        
        # Update average processing time
        if self.inference_stats["successful_inferences"] > 0:
            self.inference_stats["average_processing_time_ms"] = (
                self.inference_stats["total_processing_time_ms"] / 
                self.inference_stats["successful_inferences"]
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get inference engine statistics."""
        return self.inference_stats.copy()
    
    def reset_stats(self):
        """Reset inference statistics."""
        self.inference_stats = {
            "total_requests": 0,
            "successful_inferences": 0,
            "failed_inferences": 0,
            "average_processing_time_ms": 0.0,
            "total_processing_time_ms": 0.0
        }


def create_inference_engine(checkpoint_path: Optional[Path] = None) -> InferenceEngine:
    """
    Create and initialize an inference engine.
    
    Args:
        checkpoint_path: Optional path to model checkpoint
        
    Returns:
        Configured InferenceEngine instance
    """
    from src.services.ml_inference import create_default_model_manager
    
    # Create model manager
    model_manager = create_default_model_manager()
    
    # Create inference engine
    settings = get_settings()
    engine = InferenceEngine(
        model_manager=model_manager,
        batch_size=settings.monai.batch_size
    )
    
    # Load model
    if not engine.load_model(checkpoint_path):
        logger.warning("Failed to load model, creating new model instance")
    
    return engine