"""
MONAI Label application for NeuroDx-MultiModal system.
"""

import logging
from typing import Dict, Any

from monailabel import MONAILabelApp
from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer import InferTask
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.tasks.infer.basic_infer import BasicInferTask
from monailabel.tasks.train.basic_train import BasicTrainTask

logger = logging.getLogger(__name__)


class NeuroDxApp(MONAILabelApp):
    """
    MONAI Label application for neurodegenerative disease detection.
    """

    def __init__(self, app_dir: str, studies: str, conf: Dict[str, Any] = None):
        """Initialize the NeuroDx MONAI Label application."""
        
        self.model_dir = f"{app_dir}/models"
        
        configs = {
            "use_cache": True,
            "cache_transforms": True,
            "skip_trainers": False,
            "skip_strategies": False,
        }
        
        if conf:
            configs.update(conf)
            
        super().__init__(
            app_dir=app_dir,
            studies=studies,
            conf=configs,
            name="NeuroDx-MultiModal",
            description="Neurodegenerative disease detection with multi-modal imaging"
        )

    def init_infers(self) -> Dict[str, InferTask]:
        """Initialize inference tasks."""
        
        infers = {
            "segmentation": BasicInferTask(
                path=f"{self.model_dir}/segmentation.pt",
                network="swin_unetr",
                type="segmentation",
                labels={
                    "background": 0,
                    "brain_lesion": 1,
                    "atrophy_region": 2,
                },
                dimension=3,
                description="Brain lesion and atrophy segmentation"
            ),
            
            "classification": BasicInferTask(
                path=f"{self.model_dir}/classification.pt",
                network="swin_unetr",
                type="classification",
                labels={
                    "healthy": 0,
                    "mild_cognitive_impairment": 1,
                    "alzheimers": 2,
                    "parkinsons": 3,
                },
                dimension=3,
                description="Neurodegenerative disease classification"
            )
        }
        
        return infers

    def init_trainers(self) -> Dict[str, TrainTask]:
        """Initialize training tasks."""
        
        trainers = {
            "segmentation_trainer": BasicTrainTask(
                model_dir=self.model_dir,
                network="swin_unetr",
                type="segmentation",
                description="Train segmentation model for brain lesions"
            ),
            
            "classification_trainer": BasicTrainTask(
                model_dir=self.model_dir,
                network="swin_unetr", 
                type="classification",
                description="Train classification model for disease detection"
            )
        }
        
        return trainers

    def init_strategies(self) -> Dict[str, Any]:
        """Initialize active learning strategies."""
        
        strategies = {
            "random": "Random sampling strategy",
            "entropy": "Entropy-based uncertainty sampling",
            "margin": "Margin-based uncertainty sampling",
        }
        
        return strategies


# Create the app instance
def create_app(app_dir: str, studies: str, conf: Dict[str, Any] = None) -> MONAILabelApp:
    """Create and return the NeuroDx MONAI Label application."""
    return NeuroDxApp(app_dir, studies, conf)