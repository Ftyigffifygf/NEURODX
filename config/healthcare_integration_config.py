"""
Configuration for healthcare system integrations.

This module provides configuration classes for FHIR, HL7, and wearable device integrations.
"""

from typing import Dict, List, Optional
try:
    from pydantic_settings import BaseSettings
    from pydantic import Field
except ImportError:
    from pydantic import BaseSettings, Field
from src.services.healthcare_integration.fhir_client import FHIRConfig
from src.services.healthcare_integration.hl7_interface import HL7Config
from src.services.healthcare_integration.wearable_sdk_manager import DeviceConfig


class HealthcareIntegrationSettings(BaseSettings):
    """Healthcare integration configuration settings."""
    
    # FHIR Configuration
    fhir_enabled: bool = Field(default=False, env="FHIR_ENABLED")
    fhir_base_url: str = Field(default="", env="FHIR_BASE_URL")
    fhir_auth_type: str = Field(default="bearer", env="FHIR_AUTH_TYPE")
    fhir_username: Optional[str] = Field(default=None, env="FHIR_USERNAME")
    fhir_password: Optional[str] = Field(default=None, env="FHIR_PASSWORD")
    fhir_token: Optional[str] = Field(default=None, env="FHIR_TOKEN")
    fhir_client_id: Optional[str] = Field(default=None, env="FHIR_CLIENT_ID")
    fhir_client_secret: Optional[str] = Field(default=None, env="FHIR_CLIENT_SECRET")
    fhir_timeout: int = Field(default=30, env="FHIR_TIMEOUT")
    fhir_verify_ssl: bool = Field(default=True, env="FHIR_VERIFY_SSL")
    
    # HL7 Configuration
    hl7_enabled: bool = Field(default=False, env="HL7_ENABLED")
    hl7_host: str = Field(default="localhost", env="HL7_HOST")
    hl7_port: int = Field(default=2575, env="HL7_PORT")
    hl7_encoding: str = Field(default="utf-8", env="HL7_ENCODING")
    hl7_timeout: int = Field(default=30, env="HL7_TIMEOUT")
    hl7_max_connections: int = Field(default=10, env="HL7_MAX_CONNECTIONS")
    
    # Wearable Device Configuration
    wearable_devices_enabled: bool = Field(default=False, env="WEARABLE_DEVICES_ENABLED")
    
    # Fitbit Configuration
    fitbit_enabled: bool = Field(default=False, env="FITBIT_ENABLED")
    fitbit_api_endpoint: str = Field(default="https://api.fitbit.com", env="FITBIT_API_ENDPOINT")
    fitbit_client_id: Optional[str] = Field(default=None, env="FITBIT_CLIENT_ID")
    fitbit_client_secret: Optional[str] = Field(default=None, env="FITBIT_CLIENT_SECRET")
    fitbit_access_token: Optional[str] = Field(default=None, env="FITBIT_ACCESS_TOKEN")
    fitbit_refresh_token: Optional[str] = Field(default=None, env="FITBIT_REFRESH_TOKEN")
    
    # Generic WebSocket Device Configuration
    websocket_devices: Dict[str, Dict] = Field(default_factory=dict, env="WEBSOCKET_DEVICES")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields from environment
    
    def get_fhir_config(self) -> Optional[FHIRConfig]:
        """Get FHIR configuration if enabled."""
        if not self.fhir_enabled or not self.fhir_base_url:
            return None
        
        return FHIRConfig(
            base_url=self.fhir_base_url,
            auth_type=self.fhir_auth_type,
            username=self.fhir_username,
            password=self.fhir_password,
            token=self.fhir_token,
            client_id=self.fhir_client_id,
            client_secret=self.fhir_client_secret,
            timeout=self.fhir_timeout,
            verify_ssl=self.fhir_verify_ssl
        )
    
    def get_hl7_config(self) -> Optional[HL7Config]:
        """Get HL7 configuration if enabled."""
        if not self.hl7_enabled:
            return None
        
        return HL7Config(
            host=self.hl7_host,
            port=self.hl7_port,
            encoding=self.hl7_encoding,
            timeout=self.hl7_timeout,
            max_connections=self.hl7_max_connections
        )
    
    def get_fitbit_config(self) -> Optional[DeviceConfig]:
        """Get Fitbit device configuration if enabled."""
        if not self.fitbit_enabled or not self.fitbit_client_id:
            return None
        
        return DeviceConfig(
            device_type="fitbit",
            api_endpoint=self.fitbit_api_endpoint,
            auth_type="oauth2",
            client_id=self.fitbit_client_id,
            client_secret=self.fitbit_client_secret,
            access_token=self.fitbit_access_token,
            refresh_token=self.fitbit_refresh_token,
            data_types=["heart_rate", "steps", "sleep"]
        )
    
    def get_websocket_device_configs(self) -> List[DeviceConfig]:
        """Get WebSocket device configurations."""
        configs = []
        
        for device_id, device_config in self.websocket_devices.items():
            config = DeviceConfig(
                device_type=device_config.get("device_type", "generic"),
                api_endpoint=device_config.get("api_endpoint", ""),
                auth_type=device_config.get("auth_type", "websocket"),
                websocket_url=device_config.get("websocket_url"),
                api_key=device_config.get("api_key"),
                client_id=device_config.get("client_id"),
                data_types=device_config.get("data_types", ["heart_rate", "eeg"])
            )
            configs.append(config)
        
        return configs