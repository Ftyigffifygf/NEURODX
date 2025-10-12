"""
MONAI Label server configuration and management for neurodegenerative disease annotation.

This module provides server setup, task definitions, and configuration management
for MONAI Label active learning workflows.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict
import torch

try:
    from monailabel.config import settings
    from monailabel.interfaces.app import MONAILabelApp
    from monailabel.interfaces.tasks.infer import InferTask
    from monailabel.interfaces.tasks.train import TrainTask
    from monailabel.interfaces.tasks.strategy import Strategy
    from monailabel.tasks.infer.basic_infer import BasicInferTask
    from monailabel.tasks.train.basic_train import BasicTrainTask
    from monailabel.tasks.strategy.random import Random
    from monailabel.utils.others.generic import device_list, gpu_memory_map
    MONAI_LABEL_AVAILABLE = True
except ImportError:
    # Mock classes for testing when MONAI Label is not installed
    MONAI_LABEL_AVAILABLE = False
    
    class MONAILabelApp:
        def __init__(self, app_dir, studies, conf, name=None, description=None):
            self.app_dir = app_dir
            self.studies = studies
            self.conf = conf
            self.name = name
            self.description = description
        
        def model_dir(self):
            return os.path.join(self.app_dir, "model")
    
    class InferTask:
        def __init__(self, **kwargs):
            pass
    
    class TrainTask:
        def __init__(self, **kwargs):
            pass
    
    class Strategy:
        def __init__(self):
            pass
    
    class BasicInferTask(InferTask):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
    
    class BasicTrainTask(TrainTask):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
    
    class Random(Strategy):
        def __init__(self):
            super().__init__()
    
    def device_list():
        return ["cuda:0"] if torch.cuda.is_available() else ["cpu"]
    
    def gpu_memory_map():
        return {"cuda:0": {"total": 8192, "used": 2048}} if torch.cuda.is_available() else {}

logger = logging.getLogger(__name__)


@dataclass
class TaskDefinition:
    """Configuration for a MONAI Label annotation task."""
    name: str
    type: str  # 'segmentation' or 'classification'
    description: str
    labels: Dict[str, int]
    model_path: Optional[str] = None
    config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate task definition."""
        if self.type not in ['segmentation', 'classification']:
            raise ValueError(f"Task type must be 'segmentation' or 'classification', got: {self.type}")
        
        if not self.labels:
            raise ValueError("Task must have at least one label defined")
        
        # Validate label values are integers
        for label_name, label_value in self.labels.items():
            if not isinstance(label_value, int):
                raise ValueError(f"Label value for '{label_name}' must be integer, got: {type(label_value)}")


@dataclass
class MONAILabelConfig:
    """Configuration for MONAI Label server."""
    server_host: str = "0.0.0.0"
    server_port: int = 8000
    studies_path: str = "./data/studies"
    models_path: str = "./models/monai_label"
    app_dir: str = "./monai_label_app"
    auto_update_scoring: bool = True
    scoring_enabled: bool = True
    
    def __post_init__(self):
        """Validate configuration and create directories."""
        # Ensure paths exist
        for path_attr in ['studies_path', 'models_path', 'app_dir']:
            path = getattr(self, path_attr)
            Path(path).mkdir(parents=True, exist_ok=True)
        
        # Validate port range
        if not (1024 <= self.server_port <= 65535):
            raise ValueError(f"Server port must be between 1024-65535, got: {self.server_port}")


class NeuroDxMONAILabelApp(MONAILabelApp):
    """
    Custom MONAI Label application for neurodegenerative disease annotation.
    
    Provides specialized tasks for brain imaging segmentation and classification
    with active learning capabilities.
    """
    
    def __init__(self, app_dir: str, studies: str, conf: Dict[str, Any]):
        """Initialize the NeuroDx MONAI Label application."""
        super().__init__(
            app_dir=app_dir,
            studies=studies,
            conf=conf,
            name="NeuroDx MultiModal",
            description="Neurodegenerative disease annotation with active learning"
        )
        
        self.task_definitions = self._create_task_definitions()
        logger.info(f"Initialized NeuroDx MONAI Label app with {len(self.task_definitions)} tasks")
    
    def _create_task_definitions(self) -> List[TaskDefinition]:
        """Create task definitions for neurodegenerative disease annotation."""
        return [
            TaskDefinition(
                name="brain_segmentation",
                type="segmentation",
                description="Brain region segmentation for neurodegenerative analysis",
                labels={
                    "background": 0,
                    "hippocampus": 1,
                    "amygdala": 2,
                    "cortex": 3,
                    "white_matter": 4,
                    "ventricles": 5,
                    "lesion": 6
                },
                config={
                    "spatial_size": [96, 96, 96],
                    "pixdim": [1.0, 1.0, 1.0],
                    "intensity_range": [-1000, 1000]
                }
            ),
            TaskDefinition(
                name="disease_classification",
                type="classification", 
                description="Neurodegenerative disease classification",
                labels={
                    "healthy": 0,
                    "alzheimer": 1,
                    "parkinson": 2,
                    "huntington": 3,
                    "als": 4
                },
                config={
                    "num_classes": 5,
                    "confidence_threshold": 0.8
                }
            ),
            TaskDefinition(
                name="lesion_detection",
                type="segmentation",
                description="Lesion detection and segmentation",
                labels={
                    "background": 0,
                    "lesion": 1
                },
                config={
                    "spatial_size": [128, 128, 128],
                    "roi_size": [64, 64, 64]
                }
            )
        ]
    
    def init_infers(self) -> Dict[str, InferTask]:
        """Initialize inference tasks for each annotation task."""
        infers = {}
        
        for task_def in self.task_definitions:
            if task_def.type == "segmentation":
                infers[task_def.name] = BasicInferTask(
                    path=os.path.join(self.model_dir(), f"{task_def.name}.pt"),
                    network=None,  # Will be loaded from checkpoint
                    labels=task_def.labels,
                    preload=True,
                    config=task_def.config or {}
                )
            elif task_def.type == "classification":
                infers[task_def.name] = BasicInferTask(
                    path=os.path.join(self.model_dir(), f"{task_def.name}.pt"),
                    network=None,
                    labels=task_def.labels,
                    preload=True,
                    config=task_def.config or {}
                )
        
        logger.info(f"Initialized {len(infers)} inference tasks")
        return infers
    
    def init_trainers(self) -> Dict[str, TrainTask]:
        """Initialize training tasks for active learning."""
        trainers = {}
        
        for task_def in self.task_definitions:
            trainers[f"{task_def.name}_train"] = BasicTrainTask(
                model_dir=self.model_dir(),
                network=None,  # Will be configured based on task
                load_path=os.path.join(self.model_dir(), f"{task_def.name}.pt"),
                publish_path=os.path.join(self.model_dir(), f"{task_def.name}.pt"),
                labels=task_def.labels,
                config=task_def.config or {}
            )
        
        logger.info(f"Initialized {len(trainers)} training tasks")
        return trainers
    
    def init_strategies(self) -> Dict[str, Strategy]:
        """Initialize active learning strategies."""
        strategies = {
            "random": Random(),
            # Additional strategies will be added in ActiveLearningEngine
        }
        
        logger.info(f"Initialized {len(strategies)} active learning strategies")
        return strategies


class MONAILabelServer:
    """
    MONAI Label server manager for neurodegenerative disease annotation.
    
    Handles server configuration, startup, and task management for active learning workflows.
    """
    
    def __init__(self, config: Optional[MONAILabelConfig] = None):
        """Initialize MONAI Label server manager."""
        self.config = config or MONAILabelConfig()
        self.app: Optional[NeuroDxMONAILabelApp] = None
        self.is_running = False
        
        logger.info(f"Initialized MONAI Label server manager with config: {self.config}")
    
    def setup_app_directory(self) -> None:
        """Set up the MONAI Label application directory structure."""
        app_path = Path(self.config.app_dir)
        
        # Create main app structure
        (app_path / "lib").mkdir(parents=True, exist_ok=True)
        (app_path / "model").mkdir(parents=True, exist_ok=True)
        (app_path / "logs").mkdir(parents=True, exist_ok=True)
        
        # Create main.py for the app
        main_py_content = '''
import logging
from monailabel.config import settings
from neurodx_app import NeuroDxMONAILabelApp

if __name__ == "__main__":
    app = NeuroDxMONAILabelApp(
        app_dir=settings.MONAI_LABEL_APP_DIR,
        studies=settings.MONAI_LABEL_STUDIES,
        conf=settings.MONAI_LABEL_APP_CONF
    )
'''
        
        with open(app_path / "main.py", "w") as f:
            f.write(main_py_content)
        
        # Create app configuration
        app_config = {
            "name": "NeuroDx MultiModal",
            "description": "Neurodegenerative disease annotation with active learning",
            "version": "1.0.0",
            "labels": {
                "brain_segmentation": {
                    "background": 0,
                    "hippocampus": 1,
                    "amygdala": 2,
                    "cortex": 3,
                    "white_matter": 4,
                    "ventricles": 5,
                    "lesion": 6
                },
                "disease_classification": {
                    "healthy": 0,
                    "alzheimer": 1,
                    "parkinson": 2,
                    "huntington": 3,
                    "als": 4
                }
            }
        }
        
        with open(app_path / "app.json", "w") as f:
            json.dump(app_config, f, indent=2)
        
        logger.info(f"Set up MONAI Label app directory at: {app_path}")
    
    def initialize_app(self) -> NeuroDxMONAILabelApp:
        """Initialize the MONAI Label application."""
        if self.app is not None:
            return self.app
        
        # Set up app directory if it doesn't exist
        self.setup_app_directory()
        
        # Configure MONAI Label settings
        app_conf = {
            "models": self.config.models_path,
            "auto_update_scoring": self.config.auto_update_scoring,
            "scoring_enabled": self.config.scoring_enabled
        }
        
        # Initialize the app
        self.app = NeuroDxMONAILabelApp(
            app_dir=self.config.app_dir,
            studies=self.config.studies_path,
            conf=app_conf
        )
        
        logger.info("Initialized NeuroDx MONAI Label application")
        return self.app
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get server configuration and status information."""
        gpu_info = {}
        try:
            gpu_info = {
                "devices": device_list(),
                "memory": gpu_memory_map()
            }
        except Exception as e:
            logger.warning(f"Could not get GPU info: {e}")
        
        return {
            "config": asdict(self.config),
            "app_initialized": self.app is not None,
            "is_running": self.is_running,
            "gpu_info": gpu_info,
            "task_definitions": [asdict(task) for task in (self.app.task_definitions if self.app else [])]
        }
    
    def validate_configuration(self) -> List[str]:
        """Validate server configuration and return any issues."""
        issues = []
        
        # Check if required directories exist and are writable
        for path_name, path_value in [
            ("studies_path", self.config.studies_path),
            ("models_path", self.config.models_path),
            ("app_dir", self.config.app_dir)
        ]:
            path_obj = Path(path_value)
            if not path_obj.exists():
                try:
                    path_obj.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    issues.append(f"Cannot create {path_name} directory '{path_value}': {e}")
            elif not os.access(path_value, os.W_OK):
                issues.append(f"Directory '{path_value}' is not writable")
        
        # Check port availability
        import socket
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((self.config.server_host, self.config.server_port))
        except OSError:
            issues.append(f"Port {self.config.server_port} is already in use")
        
        return issues
    
    def start_server(self) -> bool:
        """Start the MONAI Label server."""
        if self.is_running:
            logger.warning("Server is already running")
            return True
        
        # Validate configuration
        issues = self.validate_configuration()
        if issues:
            logger.error(f"Configuration validation failed: {issues}")
            return False
        
        try:
            # Initialize app if not already done
            if self.app is None:
                self.initialize_app()
            
            # Note: In a real implementation, this would start the actual MONAI Label server
            # For now, we'll mark it as running for testing purposes
            self.is_running = True
            
            logger.info(f"MONAI Label server started on {self.config.server_host}:{self.config.server_port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start MONAI Label server: {e}")
            return False
    
    def stop_server(self) -> bool:
        """Stop the MONAI Label server."""
        if not self.is_running:
            logger.warning("Server is not running")
            return True
        
        try:
            # Note: In a real implementation, this would stop the actual server
            self.is_running = False
            logger.info("MONAI Label server stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop MONAI Label server: {e}")
            return False
    
    def get_task_definitions(self) -> List[TaskDefinition]:
        """Get all available task definitions."""
        if self.app is None:
            self.initialize_app()
        
        return self.app.task_definitions if self.app else []
    
    def add_task_definition(self, task_def: TaskDefinition) -> bool:
        """Add a new task definition to the server."""
        if self.app is None:
            self.initialize_app()
        
        try:
            if self.app:
                self.app.task_definitions.append(task_def)
                logger.info(f"Added task definition: {task_def.name}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to add task definition: {e}")
            return False