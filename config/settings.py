"""
Configuration management for NeuroDx-MultiModal system.
"""

import os
from typing import Optional, List
from pathlib import Path
from pydantic import Field, validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class NVIDIAConfig(BaseSettings):
    """NVIDIA API and GPU configuration."""
    
    # Palmyra-Med-70B Configuration
    palmyra_api_key: str = Field("test_key", env="NVIDIA_PALMYRA_API_KEY")
    palmyra_base_url: str = Field("https://integrate.api.nvidia.com/v1", env="NVIDIA_PALMYRA_BASE_URL")
    palmyra_model: str = Field("nvidia/palmyra-med-70b", env="NVIDIA_PALMYRA_MODEL")
    palmyra_max_tokens: int = Field(1024, env="NVIDIA_PALMYRA_MAX_TOKENS")
    palmyra_temperature: float = Field(0.1, env="NVIDIA_PALMYRA_TEMPERATURE")
    
    # Genomics Configuration
    genomics_workflow_path: str = Field("./genomics-analysis-blueprint", env="NVIDIA_GENOMICS_WORKFLOW_PATH")
    genomics_reference_genome: str = Field("GRCh38", env="GENOMICS_REFERENCE_GENOME")
    genomics_analysis_type: str = Field("neurodegenerative", env="GENOMICS_ANALYSIS_TYPE")
    genomics_quality_threshold: float = Field(30.0, env="GENOMICS_QUALITY_THRESHOLD")
    
    # GPU Configuration
    cuda_visible_devices: str = Field("0", env="CUDA_VISIBLE_DEVICES")
    nvidia_visible_devices: str = Field("0", env="NVIDIA_VISIBLE_DEVICES")
    cuda_device_order: str = Field("PCI_BUS_ID", env="CUDA_DEVICE_ORDER")


class MONAIConfig(BaseSettings):
    """MONAI framework configuration."""
    
    data_directory: Path = Field(Path("./data"), env="MONAI_DATA_DIRECTORY")
    model_cache: Path = Field(Path("./models"), env="MONAI_MODEL_CACHE")
    log_level: str = Field("INFO", env="MONAI_LOG_LEVEL")
    
    # Model Configuration
    swin_unetr_img_size: tuple = (96, 96, 96)
    swin_unetr_in_channels: int = 4  # Multi-modal fusion channels
    swin_unetr_out_channels: int = 3  # Segmentation classes
    swin_unetr_feature_size: int = 48
    
    # Training Configuration
    batch_size: int = Field(2, env="MONAI_BATCH_SIZE")
    learning_rate: float = Field(1e-4, env="MONAI_LEARNING_RATE")
    max_epochs: int = Field(100, env="MONAI_MAX_EPOCHS")
    
    @validator("data_directory", "model_cache")
    def create_directories(cls, v):
        v.mkdir(parents=True, exist_ok=True)
        return v


class DatabaseConfig(BaseSettings):
    """Database configuration."""
    
    # PostgreSQL
    database_url: str = Field(..., env="DATABASE_URL")
    
    # Redis
    redis_url: str = Field("redis://localhost:6379/0", env="REDIS_URL")
    
    # InfluxDB for time-series data
    influxdb_url: str = Field("http://localhost:8086", env="INFLUXDB_URL")
    influxdb_token: str = Field("test_token", env="INFLUXDB_TOKEN")
    influxdb_org: str = Field("neurodx", env="INFLUXDB_ORG")
    influxdb_bucket: str = Field("sensor_data", env="INFLUXDB_BUCKET")


class StorageConfig(BaseSettings):
    """Object storage configuration."""
    
    # MinIO/S3 Configuration
    minio_endpoint: str = Field("localhost:9000", env="MINIO_ENDPOINT")
    minio_access_key: str = Field("minioadmin", env="MINIO_ACCESS_KEY")
    minio_secret_key: str = Field("minioadmin", env="MINIO_SECRET_KEY")
    minio_bucket_images: str = Field("medical-images", env="MINIO_BUCKET_IMAGES")
    minio_bucket_models: str = Field("ml-models", env="MINIO_BUCKET_MODELS")
    minio_secure: bool = Field(False, env="MINIO_SECURE")


class APIConfig(BaseSettings):
    """API server configuration."""
    
    flask_env: str = Field("development", env="FLASK_ENV")
    flask_debug: bool = Field(True, env="FLASK_DEBUG")
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(5000, env="API_PORT")
    secret_key: str = Field(..., env="SECRET_KEY")
    
    # CORS Configuration
    cors_origins: List[str] = Field(["http://localhost:3000"], env="CORS_ORIGINS")


class HealthcareConfig(BaseSettings):
    """Healthcare system integration configuration."""
    
    # FHIR Configuration
    fhir_server_url: str = Field("http://localhost:8080/fhir", env="FHIR_SERVER_URL")
    
    # HL7 Configuration
    hl7_interface_host: str = Field("localhost", env="HL7_INTERFACE_HOST")
    hl7_interface_port: int = Field(2575, env="HL7_INTERFACE_PORT")


class SecurityConfig(BaseSettings):
    """Security and compliance configuration."""
    
    encryption_key: str = Field(..., env="ENCRYPTION_KEY")
    jwt_secret_key: str = Field(..., env="JWT_SECRET_KEY")
    hipaa_audit_log_path: Path = Field(Path("./logs/audit.log"), env="HIPAA_AUDIT_LOG_PATH")
    
    # Session Configuration
    session_timeout: int = Field(3600, env="SESSION_TIMEOUT")  # 1 hour
    max_login_attempts: int = Field(5, env="MAX_LOGIN_ATTEMPTS")
    
    @validator("hipaa_audit_log_path")
    def create_log_directory(cls, v):
        v.parent.mkdir(parents=True, exist_ok=True)
        return v


class LoggingConfig(BaseSettings):
    """Logging configuration."""
    
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_format: str = Field("json", env="LOG_FORMAT")
    log_file: Optional[Path] = Field(None, env="LOG_FILE")
    
    # Monitoring
    prometheus_port: int = Field(9090, env="PROMETHEUS_PORT")
    grafana_port: int = Field(3000, env="GRAFANA_PORT")


class FederatedLearningConfig(BaseSettings):
    """Federated learning configuration."""
    
    server_host: str = Field("localhost", env="FEDERATED_LEARNING_SERVER_HOST")
    server_port: int = Field(8080, env="FEDERATED_LEARNING_SERVER_PORT")
    node_id: str = Field("node_1", env="FEDERATED_NODE_ID")
    encryption_key: str = Field(..., env="FEDERATED_ENCRYPTION_KEY")
    
    # Training Configuration
    rounds: int = Field(10, env="FEDERATED_ROUNDS")
    min_clients: int = Field(2, env="FEDERATED_MIN_CLIENTS")
    fraction_fit: float = Field(0.8, env="FEDERATED_FRACTION_FIT")


class Settings(BaseSettings):
    """Main application settings."""
    
    # Component configurations
    nvidia: NVIDIAConfig = NVIDIAConfig()
    monai: MONAIConfig = MONAIConfig()
    database: DatabaseConfig = DatabaseConfig()
    storage: StorageConfig = StorageConfig()
    api: APIConfig = APIConfig()
    healthcare: HealthcareConfig = HealthcareConfig()
    security: SecurityConfig = SecurityConfig()
    logging: LoggingConfig = LoggingConfig()
    federated: FederatedLearningConfig = FederatedLearningConfig()
    
    # Application Metadata
    app_name: str = "NeuroDx-MultiModal"
    app_version: str = "0.1.0"
    environment: str = Field("development", env="ENVIRONMENT")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra environment variables


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings


def validate_nvidia_setup() -> bool:
    """Validate NVIDIA API setup."""
    try:
        # Check if API key is provided
        if not settings.nvidia.palmyra_api_key or settings.nvidia.palmyra_api_key == "your_nvidia_palmyra_api_key_here":
            return False
        
        # Check if genomics workflow path exists
        workflow_path = Path(settings.nvidia.genomics_workflow_path)
        if not workflow_path.exists():
            return False
        
        return True
    except Exception:
        return False


def validate_monai_setup() -> bool:
    """Validate MONAI framework setup."""
    try:
        # Check if required directories exist
        if not settings.monai.data_directory.exists():
            return False
        
        if not settings.monai.model_cache.exists():
            return False
        
        return True
    except Exception:
        return False