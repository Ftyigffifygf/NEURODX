"""
Federated Learning Configuration

Configuration settings for MONAI federated learning server and client components.
"""

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings
from pydantic import Field, validator
from typing import List, Dict, Optional
import os


from pydantic import BaseModel

class FederatedLearningConfig(BaseModel):
    """Configuration for MONAI federated learning"""
    
    # Server configuration
    server_host: str = Field(default="0.0.0.0", env="FL_SERVER_HOST")
    server_port: int = Field(default=8080, env="FL_SERVER_PORT")
    server_endpoint: str = Field(default="https://fl-server.neurodx.local", env="FL_SERVER_ENDPOINT")
    
    # Security configuration
    enable_encryption: bool = Field(default=True, env="FL_ENABLE_ENCRYPTION")
    key_directory: str = Field(default="keys", env="FL_KEY_DIRECTORY")
    certificate_path: Optional[str] = Field(default=None, env="FL_CERTIFICATE_PATH")
    private_key_path: Optional[str] = Field(default=None, env="FL_PRIVATE_KEY_PATH")
    
    # Federated learning parameters
    min_participants: int = Field(default=2, env="FL_MIN_PARTICIPANTS")
    max_participants: int = Field(default=10, env="FL_MAX_PARTICIPANTS")
    round_timeout: int = Field(default=3600, env="FL_ROUND_TIMEOUT")  # seconds
    aggregation_strategy: str = Field(default="weighted_averaging", env="FL_AGGREGATION_STRATEGY")
    
    # Model training parameters
    local_epochs: int = Field(default=5, env="FL_LOCAL_EPOCHS")
    batch_size: int = Field(default=4, env="FL_BATCH_SIZE")
    learning_rate: float = Field(default=1e-4, env="FL_LEARNING_RATE")
    
    # Node configuration
    node_id: Optional[str] = Field(default=None, env="FL_NODE_ID")
    institution_name: Optional[str] = Field(default=None, env="FL_INSTITUTION_NAME")
    
    # Predefined healthcare institutions
    healthcare_institutions: Dict[str, Dict[str, str]] = {
        "HOSP_A": {
            "name": "Hospital A",
            "endpoint": "https://hosp-a.neurodx.local:8080",
            "region": "North America"
        },
        "HOSP_B": {
            "name": "Hospital B", 
            "endpoint": "https://hosp-b.neurodx.local:8080",
            "region": "Europe"
        },
        "CLINIC_C": {
            "name": "Clinic C",
            "endpoint": "https://clinic-c.neurodx.local:8080", 
            "region": "Asia Pacific"
        }
    }
    
    # Communication settings
    connection_timeout: int = Field(default=30, env="FL_CONNECTION_TIMEOUT")
    retry_attempts: int = Field(default=3, env="FL_RETRY_ATTEMPTS")
    retry_delay: int = Field(default=5, env="FL_RETRY_DELAY")
    
    # Monitoring and logging
    enable_metrics: bool = Field(default=True, env="FL_ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="FL_METRICS_PORT")
    log_level: str = Field(default="INFO", env="FL_LOG_LEVEL")
    
    @validator('aggregation_strategy')
    def validate_aggregation_strategy(cls, v):
        """Validate aggregation strategy"""
        valid_strategies = ['federated_averaging', 'weighted_averaging', 'median_aggregation']
        if v not in valid_strategies:
            raise ValueError(f"Aggregation strategy must be one of: {valid_strategies}")
        return v
    
    @validator('node_id')
    def validate_node_id(cls, v):
        """Validate node ID format"""
        if v and not (v.startswith('HOSP_') or v.startswith('CLINIC_')):
            raise ValueError("Node ID must start with 'HOSP_' or 'CLINIC_'")
        return v
    
    @validator('min_participants')
    def validate_min_participants(cls, v):
        """Validate minimum participants"""
        if v < 2:
            raise ValueError("Minimum participants must be at least 2")
        return v
    
    @validator('local_epochs')
    def validate_local_epochs(cls, v):
        """Validate local epochs"""
        if v < 1:
            raise ValueError("Local epochs must be at least 1")
        return v
    
    @validator('batch_size')
    def validate_batch_size(cls, v):
        """Validate batch size"""
        if v < 1:
            raise ValueError("Batch size must be at least 1")
        return v
    
    class Config:
        case_sensitive = False


class FederatedServerConfig(FederatedLearningConfig):
    """Configuration specific to federated learning server"""
    
    # Server-specific settings
    enable_web_ui: bool = Field(default=True, env="FL_ENABLE_WEB_UI")
    web_ui_port: int = Field(default=8081, env="FL_WEB_UI_PORT")
    
    # Database settings for server
    database_url: str = Field(default="postgresql://localhost/neurodx_federated", env="FL_DATABASE_URL")
    redis_url: str = Field(default="redis://localhost:6379/0", env="FL_REDIS_URL")
    
    # Model storage
    model_storage_path: str = Field(default="models/federated", env="FL_MODEL_STORAGE_PATH")
    checkpoint_interval: int = Field(default=5, env="FL_CHECKPOINT_INTERVAL")  # rounds
    
    # Security settings
    require_authentication: bool = Field(default=True, env="FL_REQUIRE_AUTH")
    jwt_secret_key: str = Field(default="your-secret-key", env="FL_JWT_SECRET")
    token_expiry_hours: int = Field(default=24, env="FL_TOKEN_EXPIRY_HOURS")


class FederatedClientConfig(FederatedLearningConfig):
    """Configuration specific to federated learning client"""
    
    # Client-specific settings
    data_directory: str = Field(default="data/local", env="FL_DATA_DIRECTORY")
    cache_directory: str = Field(default="cache/federated", env="FL_CACHE_DIRECTORY")
    
    # Local model settings
    model_cache_size: int = Field(default=5, env="FL_MODEL_CACHE_SIZE")
    enable_local_validation: bool = Field(default=True, env="FL_ENABLE_LOCAL_VALIDATION")
    validation_split: float = Field(default=0.2, env="FL_VALIDATION_SPLIT")
    
    # Data privacy settings
    enable_differential_privacy: bool = Field(default=False, env="FL_ENABLE_DP")
    privacy_epsilon: float = Field(default=1.0, env="FL_PRIVACY_EPSILON")
    privacy_delta: float = Field(default=1e-5, env="FL_PRIVACY_DELTA")
    
    @validator('validation_split')
    def validate_validation_split(cls, v):
        """Validate validation split"""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Validation split must be between 0.0 and 1.0")
        return v
    
    @validator('privacy_epsilon')
    def validate_privacy_epsilon(cls, v):
        """Validate differential privacy epsilon"""
        if v <= 0:
            raise ValueError("Privacy epsilon must be positive")
        return v


def get_federated_server_config() -> FederatedServerConfig:
    """Get federated learning server configuration"""
    return FederatedServerConfig()


def get_federated_client_config() -> FederatedClientConfig:
    """Get federated learning client configuration"""
    return FederatedClientConfig()


def get_federated_config() -> FederatedLearningConfig:
    """Get general federated learning configuration"""
    return FederatedLearningConfig()