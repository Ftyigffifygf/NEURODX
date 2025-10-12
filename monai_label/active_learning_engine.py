"""
Active learning engine for uncertainty-based sample selection in MONAI Label integration.

This module implements various active learning strategies for selecting the most
informative samples for annotation in neurodegenerative disease detection workflows.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from datetime import datetime
import torch
import torch.nn.functional as F

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Mock sklearn for testing
    class MockKMeans:
        def __init__(self, n_clusters, random_state=None):
            self.n_clusters = n_clusters
            self.cluster_centers_ = None
        
        def fit_predict(self, X):
            # Simple mock clustering - just assign samples to clusters sequentially
            n_samples = len(X)
            labels = np.arange(n_samples) % self.n_clusters
            # Create mock centroids
            self.cluster_centers_ = np.random.rand(self.n_clusters, X.shape[1])
            return labels
    
    def cosine_similarity(X, Y=None):
        # Mock cosine similarity
        if Y is None:
            Y = X
        return np.random.rand(len(X), len(Y))
    
    KMeans = MockKMeans

from src.models.patient import PatientRecord, ImagingStudy
from src.models.diagnostics import DiagnosticResult
from src.services.monai_label.annotation_manager import Annotation, AnnotationManager

logger = logging.getLogger(__name__)


@dataclass
class SampleCandidate:
    """Represents a candidate sample for active learning selection."""
    patient_id: str
    study_id: str
    image_path: str
    uncertainty_score: float
    diversity_score: float
    combined_score: float
    features: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate sample candidate data."""
        if not (0.0 <= self.uncertainty_score <= 1.0):
            raise ValueError("Uncertainty score must be between 0.0 and 1.0")
        
        if not (0.0 <= self.diversity_score <= 1.0):
            raise ValueError("Diversity score must be between 0.0 and 1.0")


@dataclass
class ActiveLearningRound:
    """Represents one round of active learning."""
    round_id: str
    task_name: str
    strategy_name: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    selected_samples: List[str] = None  # List of study IDs
    performance_metrics: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        """Initialize selected samples list if None."""
        if self.selected_samples is None:
            self.selected_samples = []


class UncertaintyCalculator:
    """Calculates uncertainty scores for model predictions."""
    
    @staticmethod
    def entropy_uncertainty(predictions: torch.Tensor) -> torch.Tensor:
        """Calculate entropy-based uncertainty."""
        # Convert to probabilities if logits
        if predictions.dim() > 1 and predictions.size(-1) > 1:
            probs = F.softmax(predictions, dim=-1)
        else:
            probs = torch.sigmoid(predictions)
        
        # Calculate entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        
        # Normalize by maximum possible entropy
        max_entropy = np.log(probs.size(-1)) if probs.dim() > 1 else np.log(2)
        normalized_entropy = entropy / max_entropy
        
        return normalized_entropy
    
    @staticmethod
    def variance_uncertainty(predictions: torch.Tensor) -> torch.Tensor:
        """Calculate variance-based uncertainty for ensemble predictions."""
        if predictions.dim() < 3:
            raise ValueError("Variance uncertainty requires ensemble predictions (batch, ensemble, classes)")
        
        # Calculate variance across ensemble dimension
        variance = torch.var(predictions, dim=1)
        
        # Take mean variance across classes for multi-class
        if variance.dim() > 1:
            variance = torch.mean(variance, dim=-1)
        
        return variance
    
    @staticmethod
    def margin_uncertainty(predictions: torch.Tensor) -> torch.Tensor:
        """Calculate margin-based uncertainty (difference between top 2 predictions)."""
        if predictions.dim() > 1 and predictions.size(-1) > 1:
            probs = F.softmax(predictions, dim=-1)
            sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
            margin = sorted_probs[:, 0] - sorted_probs[:, 1]
            uncertainty = 1.0 - margin  # Higher margin = lower uncertainty
        else:
            probs = torch.sigmoid(predictions)
            uncertainty = 1.0 - torch.abs(probs - 0.5) * 2  # Distance from decision boundary
        
        return uncertainty
    
    @staticmethod
    def least_confidence_uncertainty(predictions: torch.Tensor) -> torch.Tensor:
        """Calculate least confidence uncertainty."""
        if predictions.dim() > 1 and predictions.size(-1) > 1:
            probs = F.softmax(predictions, dim=-1)
            max_probs, _ = torch.max(probs, dim=-1)
            uncertainty = 1.0 - max_probs
        else:
            probs = torch.sigmoid(predictions)
            uncertainty = 1.0 - torch.max(torch.stack([probs, 1.0 - probs], dim=-1), dim=-1)[0]
        
        return uncertainty


class ActiveLearningStrategy(ABC):
    """Abstract base class for active learning strategies."""
    
    def __init__(self, name: str):
        """Initialize strategy with name."""
        self.name = name
    
    @abstractmethod
    def select_samples(self, candidates: List[SampleCandidate], 
                      num_samples: int, **kwargs) -> List[SampleCandidate]:
        """Select samples for annotation."""
        pass


class UncertaintyStrategy(ActiveLearningStrategy):
    """Uncertainty-based sample selection strategy."""
    
    def __init__(self, uncertainty_method: str = "entropy"):
        """Initialize uncertainty strategy."""
        super().__init__(f"uncertainty_{uncertainty_method}")
        self.uncertainty_method = uncertainty_method
    
    def select_samples(self, candidates: List[SampleCandidate], 
                      num_samples: int, **kwargs) -> List[SampleCandidate]:
        """Select samples with highest uncertainty scores."""
        # Sort by uncertainty score (descending)
        sorted_candidates = sorted(candidates, key=lambda x: x.uncertainty_score, reverse=True)
        
        # Select top num_samples
        selected = sorted_candidates[:num_samples]
        
        logger.info(f"Selected {len(selected)} samples using uncertainty strategy")
        return selected


class DiversityStrategy(ActiveLearningStrategy):
    """Diversity-based sample selection strategy."""
    
    def __init__(self, clustering_method: str = "kmeans"):
        """Initialize diversity strategy."""
        super().__init__(f"diversity_{clustering_method}")
        self.clustering_method = clustering_method
    
    def select_samples(self, candidates: List[SampleCandidate], 
                      num_samples: int, **kwargs) -> List[SampleCandidate]:
        """Select diverse samples using clustering."""
        if not candidates:
            return []
        
        # Extract features for clustering
        features = []
        valid_candidates = []
        
        for candidate in candidates:
            if candidate.features is not None:
                features.append(candidate.features.flatten())
                valid_candidates.append(candidate)
        
        if not features:
            logger.warning("No features available for diversity selection, falling back to random")
            return np.random.choice(candidates, min(num_samples, len(candidates)), replace=False).tolist()
        
        features_array = np.array(features)
        
        # Perform clustering
        n_clusters = min(num_samples, len(valid_candidates))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features_array)
        
        # Select one sample from each cluster (closest to centroid)
        selected = []
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_features = features_array[cluster_indices]
            centroid = kmeans.cluster_centers_[cluster_id]
            
            # Find closest sample to centroid
            distances = np.linalg.norm(cluster_features - centroid, axis=1)
            closest_idx = cluster_indices[np.argmin(distances)]
            selected.append(valid_candidates[closest_idx])
        
        logger.info(f"Selected {len(selected)} diverse samples using clustering")
        return selected


class HybridStrategy(ActiveLearningStrategy):
    """Hybrid strategy combining uncertainty and diversity."""
    
    def __init__(self, uncertainty_weight: float = 0.7, diversity_weight: float = 0.3):
        """Initialize hybrid strategy."""
        super().__init__("hybrid")
        self.uncertainty_weight = uncertainty_weight
        self.diversity_weight = diversity_weight
        
        if abs(uncertainty_weight + diversity_weight - 1.0) > 1e-6:
            raise ValueError("Uncertainty and diversity weights must sum to 1.0")
    
    def select_samples(self, candidates: List[SampleCandidate], 
                      num_samples: int, **kwargs) -> List[SampleCandidate]:
        """Select samples using combined uncertainty and diversity scores."""
        # Calculate combined scores
        for candidate in candidates:
            candidate.combined_score = (
                self.uncertainty_weight * candidate.uncertainty_score +
                self.diversity_weight * candidate.diversity_score
            )
        
        # Sort by combined score (descending)
        sorted_candidates = sorted(candidates, key=lambda x: x.combined_score, reverse=True)
        
        # Select top num_samples
        selected = sorted_candidates[:num_samples]
        
        logger.info(f"Selected {len(selected)} samples using hybrid strategy")
        return selected


class ActiveLearningEngine:
    """
    Main active learning engine for MONAI Label integration.
    
    Manages active learning workflows including uncertainty calculation,
    sample selection, and feedback loop coordination.
    """
    
    def __init__(self, annotation_manager: AnnotationManager):
        """Initialize active learning engine."""
        self.annotation_manager = annotation_manager
        self.uncertainty_calculator = UncertaintyCalculator()
        
        # Initialize strategies
        self.strategies = {
            "uncertainty": UncertaintyStrategy(),
            "diversity": DiversityStrategy(),
            "hybrid": HybridStrategy(),
            "random": self._create_random_strategy()
        }
        
        # Active learning state
        self.current_rounds: Dict[str, ActiveLearningRound] = {}
        
        logger.info("Initialized active learning engine")
    
    def _create_random_strategy(self) -> ActiveLearningStrategy:
        """Create random selection strategy."""
        class RandomStrategy(ActiveLearningStrategy):
            def __init__(self):
                super().__init__("random")
            
            def select_samples(self, candidates: List[SampleCandidate], 
                             num_samples: int, **kwargs) -> List[SampleCandidate]:
                return np.random.choice(candidates, min(num_samples, len(candidates)), replace=False).tolist()
        
        return RandomStrategy()
    
    def calculate_uncertainty_scores(self, predictions: torch.Tensor, 
                                   method: str = "entropy") -> np.ndarray:
        """Calculate uncertainty scores for model predictions."""
        try:
            if method == "entropy":
                uncertainty = self.uncertainty_calculator.entropy_uncertainty(predictions)
            elif method == "variance":
                uncertainty = self.uncertainty_calculator.variance_uncertainty(predictions)
            elif method == "margin":
                uncertainty = self.uncertainty_calculator.margin_uncertainty(predictions)
            elif method == "least_confidence":
                uncertainty = self.uncertainty_calculator.least_confidence_uncertainty(predictions)
            else:
                raise ValueError(f"Unknown uncertainty method: {method}")
            
            return uncertainty.cpu().numpy()
            
        except Exception as e:
            logger.error(f"Failed to calculate uncertainty scores: {e}")
            return np.zeros(predictions.size(0))
    
    def calculate_diversity_scores(self, features: np.ndarray, 
                                 labeled_features: Optional[np.ndarray] = None) -> np.ndarray:
        """Calculate diversity scores based on feature similarity."""
        try:
            if labeled_features is None or len(labeled_features) == 0:
                # If no labeled data, all samples are equally diverse
                return np.ones(len(features))
            
            # Calculate similarity to labeled samples
            similarities = cosine_similarity(features, labeled_features)
            
            # Diversity is inverse of maximum similarity to any labeled sample
            max_similarities = np.max(similarities, axis=1)
            diversity_scores = 1.0 - max_similarities
            
            # Normalize to [0, 1]
            if diversity_scores.max() > diversity_scores.min():
                diversity_scores = (diversity_scores - diversity_scores.min()) / (diversity_scores.max() - diversity_scores.min())
            
            return diversity_scores
            
        except Exception as e:
            logger.error(f"Failed to calculate diversity scores: {e}")
            return np.ones(len(features))
    
    def create_sample_candidates(self, unlabeled_studies: List[ImagingStudy],
                               predictions: torch.Tensor,
                               features: Optional[np.ndarray] = None,
                               labeled_features: Optional[np.ndarray] = None,
                               uncertainty_method: str = "entropy") -> List[SampleCandidate]:
        """Create sample candidates with uncertainty and diversity scores."""
        try:
            # Calculate uncertainty scores
            uncertainty_scores = self.calculate_uncertainty_scores(predictions, uncertainty_method)
            
            # Calculate diversity scores if features are provided
            diversity_scores = np.ones(len(unlabeled_studies))
            if features is not None:
                diversity_scores = self.calculate_diversity_scores(features, labeled_features)
            
            # Create candidates
            candidates = []
            for i, study in enumerate(unlabeled_studies):
                candidate = SampleCandidate(
                    patient_id=study.patient_id,
                    study_id=study.study_id,
                    image_path=study.file_path,
                    uncertainty_score=float(uncertainty_scores[i]),
                    diversity_score=float(diversity_scores[i]),
                    combined_score=0.0,  # Will be calculated by strategy
                    features=features[i] if features is not None else None,
                    metadata={
                        "modality": study.modality,
                        "acquisition_date": study.acquisition_date.isoformat()
                    }
                )
                candidates.append(candidate)
            
            logger.info(f"Created {len(candidates)} sample candidates")
            return candidates
            
        except Exception as e:
            logger.error(f"Failed to create sample candidates: {e}")
            return []
    
    def select_samples_for_annotation(self, task_name: str, 
                                    unlabeled_studies: List[ImagingStudy],
                                    predictions: torch.Tensor,
                                    num_samples: int,
                                    strategy_name: str = "uncertainty",
                                    features: Optional[np.ndarray] = None,
                                    labeled_features: Optional[np.ndarray] = None,
                                    **strategy_kwargs) -> List[SampleCandidate]:
        """Select samples for annotation using specified strategy."""
        try:
            # Create sample candidates
            candidates = self.create_sample_candidates(
                unlabeled_studies, predictions, features, labeled_features
            )
            
            if not candidates:
                logger.warning("No candidates available for selection")
                return []
            
            # Get strategy
            if strategy_name not in self.strategies:
                logger.warning(f"Unknown strategy '{strategy_name}', using uncertainty")
                strategy_name = "uncertainty"
            
            strategy = self.strategies[strategy_name]
            
            # Select samples
            selected_samples = strategy.select_samples(candidates, num_samples, **strategy_kwargs)
            
            logger.info(f"Selected {len(selected_samples)} samples using {strategy_name} strategy")
            return selected_samples
            
        except Exception as e:
            logger.error(f"Failed to select samples for annotation: {e}")
            return []
    
    def start_active_learning_round(self, task_name: str, strategy_name: str) -> str:
        """Start a new active learning round."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        round_id = f"AL_ROUND_{task_name}_{strategy_name}_{timestamp}"
        
        round_obj = ActiveLearningRound(
            round_id=round_id,
            task_name=task_name,
            strategy_name=strategy_name,
            started_at=datetime.now()
        )
        
        self.current_rounds[round_id] = round_obj
        logger.info(f"Started active learning round: {round_id}")
        
        return round_id
    
    def complete_active_learning_round(self, round_id: str, 
                                     selected_samples: List[str],
                                     performance_metrics: Optional[Dict[str, float]] = None) -> bool:
        """Complete an active learning round."""
        try:
            if round_id not in self.current_rounds:
                logger.error(f"Active learning round not found: {round_id}")
                return False
            
            round_obj = self.current_rounds[round_id]
            round_obj.completed_at = datetime.now()
            round_obj.selected_samples = selected_samples
            round_obj.performance_metrics = performance_metrics or {}
            
            # Remove from current rounds
            del self.current_rounds[round_id]
            
            logger.info(f"Completed active learning round: {round_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to complete active learning round {round_id}: {e}")
            return False
    
    def get_active_learning_statistics(self, task_name: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics about active learning performance."""
        try:
            # Get annotations for the task
            annotations = []
            if task_name:
                annotations = self.annotation_manager.get_annotations_for_task(task_name)
            else:
                # Get all annotations
                all_annotation_ids = self.annotation_manager.storage.list_annotations()
                for annotation_id in all_annotation_ids:
                    annotation = self.annotation_manager.get_annotation(annotation_id)
                    if annotation:
                        annotations.append(annotation)
            
            # Calculate statistics
            total_annotations = len(annotations)
            
            # Quality statistics
            quality_scores = [ann.quality_score for ann in annotations if ann.quality_score is not None]
            quality_stats = {}
            if quality_scores:
                quality_stats = {
                    "mean": np.mean(quality_scores),
                    "std": np.std(quality_scores),
                    "min": np.min(quality_scores),
                    "max": np.max(quality_scores)
                }
            
            # Validation status counts
            status_counts = {}
            for annotation in annotations:
                status = annotation.validation_status
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # Active rounds
            active_rounds_count = len(self.current_rounds)
            
            return {
                "task_name": task_name,
                "total_annotations": total_annotations,
                "quality_statistics": quality_stats,
                "validation_status_counts": status_counts,
                "active_rounds_count": active_rounds_count,
                "available_strategies": list(self.strategies.keys())
            }
            
        except Exception as e:
            logger.error(f"Failed to get active learning statistics: {e}")
            return {}
    
    def update_model_with_feedback(self, task_name: str, 
                                 new_annotations: List[Annotation]) -> Dict[str, Any]:
        """Update model performance based on new annotations (feedback loop)."""
        try:
            # This is a placeholder for model retraining logic
            # In a real implementation, this would trigger model retraining
            # with the new annotations and return updated performance metrics
            
            feedback_results = {
                "task_name": task_name,
                "new_annotations_count": len(new_annotations),
                "feedback_processed_at": datetime.now().isoformat(),
                "model_updated": False,  # Would be True after actual retraining
                "performance_improvement": 0.0  # Would contain actual metrics
            }
            
            logger.info(f"Processed feedback for task {task_name} with {len(new_annotations)} new annotations")
            return feedback_results
            
        except Exception as e:
            logger.error(f"Failed to update model with feedback: {e}")
            return {}
    
    def add_custom_strategy(self, strategy: ActiveLearningStrategy) -> bool:
        """Add a custom active learning strategy."""
        try:
            self.strategies[strategy.name] = strategy
            logger.info(f"Added custom strategy: {strategy.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add custom strategy: {e}")
            return False