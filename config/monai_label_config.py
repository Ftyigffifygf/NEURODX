"""
Configuration settings for MONAI Label integration.

This module provides configuration classes for MONAI Label server setup,
task definitions, and active learning parameters.
"""

from typing import Dict, List, Optional, Any
try:
    from pydantic_settings import BaseSettings
    from pydantic import Field, validator
except ImportError:
    try:
        from pydantic import BaseSettings, Field, validator
    except ImportError:
        # Mock for testing
        class BaseSettings:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
            
            class Config:
                env_prefix = ""
                case_sensitive = False
        
        def Field(default=None, description=""):
            return default
        
        def validator(field_name):
            def decorator(func):
                return func
            return decorator
from pathlib import Path


class MONAILabelServerConfig(BaseSettings):
    """Configuration for MONAI Label server."""
    
    # Server settings
    host: str = Field(default="0.0.0.0", description="Server host address")
    port: int = Field(default=8000, description="Server port")
    
    # Directory paths
    studies_path: str = Field(default="./data/studies", description="Path to studies directory")
    models_path: str = Field(default="./models/monai_label", description="Path to models directory")
    app_dir: str = Field(default="./monai_label_app", description="MONAI Label app directory")
    annotations_path: str = Field(default="./data/annotations", description="Path to annotations storage")
    
    # Server features
    auto_update_scoring: bool = Field(default=True, description="Enable automatic scoring updates")
    scoring_enabled: bool = Field(default=True, description="Enable scoring functionality")
    preload_models: bool = Field(default=True, description="Preload models on startup")
    
    # Active learning settings
    uncertainty_threshold: float = Field(default=0.8, description="Uncertainty threshold for active learning")
    max_samples_per_round: int = Field(default=10, description="Maximum samples per active learning round")
    
    @validator('port')
    def validate_port(cls, v):
        """Validate port range."""
        if not (1024 <= v <= 65535):
            raise ValueError('Port must be between 1024 and 65535')
        return v
    
    @validator('uncertainty_threshold')
    def validate_uncertainty_threshold(cls, v):
        """Validate uncertainty threshold."""
        if not (0.0 <= v <= 1.0):
            raise ValueError('Uncertainty threshold must be between 0.0 and 1.0')
        return v
    
    class Config:
        env_prefix = "MONAI_LABEL_"
        case_sensitive = False


class TaskConfig(BaseSettings):
    """Configuration for individual annotation tasks."""
    
    # Task identification
    name: str = Field(..., description="Task name")
    type: str = Field(..., description="Task type (segmentation/classification)")
    description: str = Field(default="", description="Task description")
    
    # Labels and classes
    labels: Dict[str, int] = Field(default_factory=dict, description="Label mappings")
    num_classes: Optional[int] = Field(default=None, description="Number of classes for classification")
    
    # Model configuration
    model_name: str = Field(default="swin_unetr", description="Model architecture name")
    spatial_size: List[int] = Field(default=[96, 96, 96], description="Input spatial size")
    pixdim: List[float] = Field(default=[1.0, 1.0, 1.0], description="Pixel dimensions")
    
    # Training parameters
    learning_rate: float = Field(default=1e-4, description="Learning rate")
    batch_size: int = Field(default=2, description="Batch size")
    max_epochs: int = Field(default=100, description="Maximum training epochs")
    
    # Inference parameters
    confidence_threshold: float = Field(default=0.5, description="Confidence threshold for predictions")
    overlap_ratio: float = Field(default=0.25, description="Overlap ratio for sliding window inference")
    
    @validator('type')
    def validate_task_type(cls, v):
        """Validate task type."""
        if v not in ['segmentation', 'classification']:
            raise ValueError('Task type must be segmentation or classification')
        return v
    
    @validator('spatial_size')
    def validate_spatial_size(cls, v):
        """Validate spatial size."""
        if len(v) != 3 or any(dim <= 0 for dim in v):
            raise ValueError('Spatial size must be 3 positive integers')
        return v
    
    @validator('confidence_threshold')
    def validate_confidence_threshold(cls, v):
        """Validate confidence threshold."""
        if not (0.0 <= v <= 1.0):
            raise ValueError('Confidence threshold must be between 0.0 and 1.0')
        return v


class ActiveLearningConfig(BaseSettings):
    """Configuration for active learning strategies."""
    
    # Strategy settings
    strategy_name: str = Field(default="uncertainty", description="Active learning strategy")
    uncertainty_method: str = Field(default="entropy", description="Uncertainty calculation method")
    
    # Sampling parameters
    initial_samples: int = Field(default=5, description="Initial number of labeled samples")
    samples_per_iteration: int = Field(default=3, description="Samples to label per iteration")
    max_iterations: int = Field(default=10, description="Maximum active learning iterations")
    
    # Quality thresholds
    min_quality_score: float = Field(default=0.6, description="Minimum annotation quality score")
    consensus_threshold: float = Field(default=0.8, description="Inter-annotator agreement threshold")
    
    # Diversity settings
    diversity_weight: float = Field(default=0.3, description="Weight for diversity in sample selection")
    cluster_based_sampling: bool = Field(default=True, description="Enable cluster-based diverse sampling")
    
    @validator('strategy_name')
    def validate_strategy_name(cls, v):
        """Validate strategy name."""
        valid_strategies = ['uncertainty', 'diversity', 'hybrid', 'random']
        if v not in valid_strategies:
            raise ValueError(f'Strategy must be one of: {valid_strategies}')
        return v
    
    @validator('uncertainty_method')
    def validate_uncertainty_method(cls, v):
        """Validate uncertainty method."""
        valid_methods = ['entropy', 'variance', 'margin', 'least_confidence']
        if v not in valid_methods:
            raise ValueError(f'Uncertainty method must be one of: {valid_methods}')
        return v
    
    class Config:
        env_prefix = "ACTIVE_LEARNING_"
        case_sensitive = False


class MONAILabelIntegrationConfig(BaseSettings):
    """Main configuration class for MONAI Label integration."""
    
    # Sub-configurations
    server: MONAILabelServerConfig = Field(default_factory=MONAILabelServerConfig)
    active_learning: ActiveLearningConfig = Field(default_factory=ActiveLearningConfig)
    
    # Task configurations
    tasks: Dict[str, TaskConfig] = Field(default_factory=dict)
    
    # Integration settings
    enable_auto_annotation: bool = Field(default=True, description="Enable automatic pre-annotation")
    enable_quality_assessment: bool = Field(default=True, description="Enable annotation quality assessment")
    enable_federated_learning: bool = Field(default=False, description="Enable federated learning")
    
    # Security and compliance
    require_authentication: bool = Field(default=True, description="Require user authentication")
    audit_logging: bool = Field(default=True, description="Enable audit logging")
    data_encryption: bool = Field(default=True, description="Enable data encryption")
    
    def __init__(self, **kwargs):
        """Initialize with default task configurations."""
        super().__init__(**kwargs)
        
        # Set up default tasks if none provided
        if not self.tasks:
            self.tasks = self._create_default_tasks()
    
    def _create_default_tasks(self) -> Dict[str, TaskConfig]:
        """Create default task configurations for neurodegenerative disease annotation."""
        return {
            "brain_segmentation": TaskConfig(
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
                spatial_size=[96, 96, 96],
                pixdim=[1.0, 1.0, 1.0],
                confidence_threshold=0.8
            ),
            "disease_classification": TaskConfig(
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
                num_classes=5,
                spatial_size=[96, 96, 96],
                confidence_threshold=0.9
            ),
            "lesion_detection": TaskConfig(
                name="lesion_detection",
                type="segmentation",
                description="Lesion detection and segmentation",
                labels={
                    "background": 0,
                    "lesion": 1
                },
                spatial_size=[128, 128, 128],
                confidence_threshold=0.7
            )
        }
    
    def get_task_config(self, task_name: str) -> Optional[TaskConfig]:
        """Get configuration for a specific task."""
        return self.tasks.get(task_name)
    
    def add_task_config(self, task_config: TaskConfig) -> None:
        """Add a new task configuration."""
        self.tasks[task_config.name] = task_config
    
    def validate_configuration(self) -> List[str]:
        """Validate the entire configuration and return any issues."""
        issues = []
        
        # Validate directory paths
        for path_name, path_value in [
            ("studies_path", self.server.studies_path),
            ("models_path", self.server.models_path),
            ("app_dir", self.server.app_dir),
            ("annotations_path", self.server.annotations_path)
        ]:
            path_obj = Path(path_value)
            try:
                path_obj.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                issues.append(f"Cannot create {path_name} directory '{path_value}': {e}")
        
        # Validate task configurations
        for task_name, task_config in self.tasks.items():
            if not task_config.labels:
                issues.append(f"Task '{task_name}' has no labels defined")
            
            if task_config.type == "classification" and task_config.num_classes is None:
                issues.append(f"Classification task '{task_name}' must specify num_classes")
        
        return issues
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False