"""
Unit tests for healthcare integration services.

Tests FHIR client, HL7 interface, and wearable SDK manager functionality.
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import requests

from src.services.healthcare_integration.fhir_client import FHIRClient, FHIRConfig, FHIRAuthenticationError, FHIRValidationError
from src.services.healthcare_integration.hl7_interface import HL7Interface, HL7Config, HL7Message, HL7ValidationError
from src.services.healthcare_integration.wearable_sdk_manager import (
    WearableSDKManager, DeviceConfig, FitbitSDK, GenericWebSocketSDK, WearableDataPoint, DeviceStatus
)
from src.services.healthcare_integration.healthcare_integration_service import HealthcareIntegrationService
from src.config.healthcare_integration_config import HealthcareIntegrationSettings
from src.models.patient import PatientRecord, WearableSession
from src.models.diagnostics import DiagnosticResult


class TestFHIRClient:
    """Test cases for FHIR client."""
    
    def test_fhir_config_creation(self):
        """Test FHIR configuration creation."""
        config = FHIRConfig(
            base_url="https://fhir.example.com",
            auth_type="bearer",
            token="test_token"
        )
        
        assert config.base_url == "https://fhir.example.com"
        assert config.auth_type == "bearer"
        assert config.token == "test_token"
        assert config.timeout == 30
        assert config.verify_ssl is True
    
    def test_fhir_client_initialization(self):
        """Test FHIR client initialization."""
        config = FHIRConfig(
            base_url="https://fhir.example.com",
            auth_type="bearer",
            token="test_token"
        )
        
        client = FHIRClient(config)
        
        assert client.config == config
        assert client.session is not None
        assert "Authorization" in client.session.headers
        assert client.session.headers["Authorization"] == "Bearer test_token"
    
    def test_fhir_authentication_error(self):
        """Test FHIR authentication error handling."""
        config = FHIRConfig(
            base_url="https://fhir.example.com",
            auth_type="basic"
            # Missing username and password
        )
        
        with pytest.raises(FHIRAuthenticationError):
            FHIRClient(config)
    
    @patch('requests.Session.get')
    def test_get_patient_success(self, mock_get):
        """Test successful patient retrieval."""
        config = FHIRConfig(
            base_url="https://fhir.example.com",
            auth_type="bearer",
            token="test_token"
        )
        
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "resourceType": "Patient",
            "id": "123",
            "name": [{"family": "Doe", "given": ["John"]}]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        client = FHIRClient(config)
        patient = client.get_patient("123")
        
        assert patient is not None
        assert patient["resourceType"] == "Patient"
        assert patient["id"] == "123"
        mock_get.assert_called_once()
    
    @patch('requests.Session.get')
    def test_get_patient_not_found(self, mock_get):
        """Test patient not found scenario."""
        config = FHIRConfig(
            base_url="https://fhir.example.com",
            auth_type="bearer",
            token="test_token"
        )
        
        # Mock 404 response
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        client = FHIRClient(config)
        patient = client.get_patient("nonexistent")
        
        assert patient is None
    
    def test_convert_diagnostic_result_to_fhir(self):
        """Test conversion of diagnostic result to FHIR format."""
        config = FHIRConfig(
            base_url="https://fhir.example.com",
            auth_type="bearer",
            token="test_token"
        )
        
        client = FHIRClient(config)
        
        # Create test diagnostic result
        diagnostic_result = DiagnosticResult(
            patient_id="PAT_20241010_12345",
            timestamp=datetime.now(),
            segmentation_mask=None,
            classification_probabilities={"Alzheimer's": 0.85, "Parkinson's": 0.15},
            confidence_scores={"overall": 0.92},
            explainability_maps={},
            metrics=None
        )
        
        fhir_report = client._convert_to_fhir_diagnostic_report(diagnostic_result, "fhir_patient_123")
        
        assert fhir_report["resourceType"] == "DiagnosticReport"
        assert fhir_report["status"] == "final"
        assert fhir_report["subject"]["reference"] == "Patient/fhir_patient_123"
        assert len(fhir_report["extension"]) == 1  # confidence scores
        assert "High probability (85.00%) of Alzheimer's" in fhir_report["conclusion"]
    
    def test_fhir_resource_validation(self):
        """Test FHIR resource validation."""
        config = FHIRConfig(
            base_url="https://fhir.example.com",
            auth_type="bearer",
            token="test_token"
        )
        
        client = FHIRClient(config)
        
        # Valid resource
        valid_resource = {
            "resourceType": "DiagnosticReport",
            "status": "final",
            "code": {"text": "Test"},
            "subject": {"reference": "Patient/123"}
        }
        
        # Should not raise exception
        client._validate_fhir_resource(valid_resource, "DiagnosticReport")
        
        # Invalid resource - missing required field
        invalid_resource = {
            "resourceType": "DiagnosticReport",
            "status": "final"
            # Missing code and subject
        }
        
        with pytest.raises(FHIRValidationError):
            client._validate_fhir_resource(invalid_resource, "DiagnosticReport")


class TestHL7Interface:
    """Test cases for HL7 interface."""
    
    def test_hl7_config_creation(self):
        """Test HL7 configuration creation."""
        config = HL7Config(
            host="localhost",
            port=2575,
            encoding="utf-8"
        )
        
        assert config.host == "localhost"
        assert config.port == 2575
        assert config.encoding == "utf-8"
        assert config.field_separator == "|"
    
    def test_hl7_message_parsing(self):
        """Test HL7 message parsing."""
        config = HL7Config()
        
        # Sample HL7 MSH segment
        message_text = "MSH|^~\\&|SendingApp|SendingFac|ReceivingApp|ReceivingFac|20241010120000||ADT^A01^ADT_A01|12345|P|2.5"
        
        message = HL7Message(message_text, config)
        
        assert len(message.segments) == 1
        assert message.segments[0]["id"] == "MSH"
        assert message.message_type == "ADT"
        assert message.control_id == "12345"
    
    def test_hl7_message_validation(self):
        """Test HL7 message validation."""
        config = HL7Config()
        
        # Valid message with MSH segment
        valid_message_text = "MSH|^~\\&|SendingApp|SendingFac|ReceivingApp|ReceivingFac|20241010120000||ADT^A01^ADT_A01|12345|P|2.5"
        valid_message = HL7Message(valid_message_text, config)
        
        # Should not raise exception
        assert valid_message.validate() is True
        
        # Invalid message without MSH segment
        invalid_message = HL7Message("", config)
        
        with pytest.raises(HL7ValidationError):
            invalid_message.validate()
    
    def test_hl7_message_creation(self):
        """Test HL7 message creation."""
        config = HL7Config()
        
        message = HL7Message(config=config)
        
        # Add MSH segment
        msh_fields = [
            "^~\\&",
            "NeuroDx",
            "AI-System",
            "Hospital",
            "HIS",
            "20241010120000",
            "",
            "ORU^R01^ORU_R01",
            "MSG001",
            "P",
            "2.5"
        ]
        
        message.add_segment("MSH", msh_fields)
        
        # Add PID segment
        pid_fields = ["1", "", "PAT123", "", "Doe^John", "", "19800101", "M"]
        message.add_segment("PID", pid_fields)
        
        message_text = message.to_string()
        
        assert "MSH|^~\\&|NeuroDx" in message_text
        assert "PID|1||PAT123" in message_text
    
    def test_hl7_interface_initialization(self):
        """Test HL7 interface initialization."""
        config = HL7Config(host="localhost", port=2575)
        
        interface = HL7Interface(config)
        
        assert interface.config == config
        assert interface.is_running is False
        assert len(interface.message_handlers) > 0  # Default handlers registered
    
    def test_create_diagnostic_result_message(self):
        """Test creation of diagnostic result HL7 message."""
        config = HL7Config()
        interface = HL7Interface(config)
        
        # Create test diagnostic result
        diagnostic_result = DiagnosticResult(
            patient_id="PAT_20241010_12345",
            timestamp=datetime.now(),
            segmentation_mask=None,
            classification_probabilities={"Alzheimer's": 0.85},
            confidence_scores={"overall": 0.92},
            explainability_maps={},
            metrics=None
        )
        
        hl7_message = interface.create_diagnostic_result_message(diagnostic_result, "PAT123")
        
        # Check message structure
        msh_segment = hl7_message.get_segment("MSH")
        pid_segment = hl7_message.get_segment("PID")
        obr_segment = hl7_message.get_segment("OBR")
        obx_segments = hl7_message.get_segments("OBX")
        
        assert msh_segment is not None
        assert pid_segment is not None
        assert obr_segment is not None
        assert len(obx_segments) >= 2  # At least classification and confidence


class TestWearableSDKManager:
    """Test cases for wearable SDK manager."""
    
    def test_device_config_creation(self):
        """Test device configuration creation."""
        config = DeviceConfig(
            device_type="fitbit",
            api_endpoint="https://api.fitbit.com",
            auth_type="oauth2",
            client_id="test_client",
            data_types=["heart_rate", "steps"]
        )
        
        assert config.device_type == "fitbit"
        assert config.api_endpoint == "https://api.fitbit.com"
        assert config.auth_type == "oauth2"
        assert config.data_types == ["heart_rate", "steps"]
    
    def test_wearable_data_point_creation(self):
        """Test wearable data point creation."""
        data_point = WearableDataPoint(
            device_id="fitbit_123",
            device_type="fitbit",
            data_type="heart_rate",
            timestamp=datetime.now(),
            value=75.0,
            unit="bpm",
            quality_score=0.95
        )
        
        assert data_point.device_id == "fitbit_123"
        assert data_point.device_type == "fitbit"
        assert data_point.data_type == "heart_rate"
        assert data_point.value == 75.0
        assert data_point.unit == "bpm"
        assert data_point.quality_score == 0.95
    
    def test_device_status_creation(self):
        """Test device status creation."""
        status = DeviceStatus(
            device_id="fitbit_123",
            device_type="fitbit",
            is_connected=True,
            last_sync=datetime.now(),
            battery_level=0.85,
            signal_quality=0.92
        )
        
        assert status.device_id == "fitbit_123"
        assert status.device_type == "fitbit"
        assert status.is_connected is True
        assert status.battery_level == 0.85
        assert status.signal_quality == 0.92
    
    def test_sdk_manager_initialization(self):
        """Test SDK manager initialization."""
        manager = WearableSDKManager()
        
        assert len(manager.sdks) == 0
        assert len(manager.device_configs) == 0
        assert len(manager.data_callbacks) == 0
        assert len(manager.status_callbacks) == 0
    
    def test_device_registration(self):
        """Test device registration."""
        manager = WearableSDKManager()
        
        config = DeviceConfig(
            device_type="fitbit",
            api_endpoint="https://api.fitbit.com",
            auth_type="oauth2",
            client_id="test_client"
        )
        
        manager.register_device("fitbit_1", config)
        
        assert "fitbit_1" in manager.device_configs
        assert "fitbit_1" in manager.sdks
        assert isinstance(manager.sdks["fitbit_1"], FitbitSDK)
    
    @pytest.mark.asyncio
    async def test_fitbit_sdk_initialization(self):
        """Test Fitbit SDK initialization."""
        config = DeviceConfig(
            device_type="fitbit",
            api_endpoint="https://api.fitbit.com",
            auth_type="oauth2",
            access_token="test_token"
        )
        
        sdk = FitbitSDK(config)
        
        assert sdk.config == config
        assert sdk.is_connected is False
        assert sdk.user_id is None
    
    @pytest.mark.asyncio
    @patch('requests.Session.get')
    async def test_fitbit_authentication(self, mock_get):
        """Test Fitbit authentication."""
        config = DeviceConfig(
            device_type="fitbit",
            api_endpoint="https://api.fitbit.com",
            auth_type="oauth2",
            access_token="test_token"
        )
        
        # Mock successful authentication response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "user": {"encodedId": "ABC123"}
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        sdk = FitbitSDK(config)
        success = await sdk.authenticate()
        
        assert success is True
        assert sdk.user_id == "ABC123"
    
    def test_convert_to_wearable_session(self):
        """Test conversion of data points to wearable session."""
        manager = WearableSDKManager()
        
        # Create test data points
        now = datetime.now()
        data_points = [
            WearableDataPoint(
                device_id="fitbit_123",
                device_type="fitbit",
                data_type="heart_rate",
                timestamp=now,
                value=75.0,
                unit="bpm"
            ),
            WearableDataPoint(
                device_id="fitbit_123",
                device_type="fitbit",
                data_type="heart_rate",
                timestamp=now + timedelta(seconds=1),
                value=76.0,
                unit="bpm"
            )
        ]
        
        session = manager.convert_to_wearable_session("fitbit_123", data_points)
        
        assert session.device_type == "Fitbit"
        assert session.start_time == now
        assert session.end_time == now + timedelta(seconds=1)
        assert session.sampling_rate == 2.0  # 2 points in 1 second
        assert "heart_rate" in session.raw_data
        assert "heart_rate_mean" in session.processed_features
        assert session.processed_features["heart_rate_mean"] == 75.5


class TestHealthcareIntegrationService:
    """Test cases for healthcare integration service."""
    
    def test_service_initialization_minimal(self):
        """Test service initialization with minimal configuration."""
        settings = HealthcareIntegrationSettings()
        service = HealthcareIntegrationService(settings)
        
        assert service.settings == settings
        assert service.fhir_client is None  # Not enabled
        assert service.hl7_interface is None  # Not enabled
        assert service.wearable_manager is None  # Not enabled
    
    def test_service_initialization_with_fhir(self):
        """Test service initialization with FHIR enabled."""
        settings = HealthcareIntegrationSettings(
            fhir_enabled=True,
            fhir_base_url="https://fhir.example.com",
            fhir_auth_type="bearer",
            fhir_token="test_token"
        )
        
        service = HealthcareIntegrationService(settings)
        
        assert service.fhir_client is not None
        assert isinstance(service.fhir_client, FHIRClient)
    
    def test_service_initialization_with_hl7(self):
        """Test service initialization with HL7 enabled."""
        settings = HealthcareIntegrationSettings(
            hl7_enabled=True,
            hl7_host="localhost",
            hl7_port=2575
        )
        
        service = HealthcareIntegrationService(settings)
        
        assert service.hl7_interface is not None
        assert isinstance(service.hl7_interface, HL7Interface)
    
    def test_service_initialization_with_wearables(self):
        """Test service initialization with wearables enabled."""
        settings = HealthcareIntegrationSettings(
            wearable_devices_enabled=True,
            fitbit_enabled=True,
            fitbit_client_id="test_client",
            fitbit_api_endpoint="https://api.fitbit.com"
        )
        
        service = HealthcareIntegrationService(settings)
        
        assert service.wearable_manager is not None
        assert isinstance(service.wearable_manager, WearableSDKManager)
        assert "fitbit" in service.wearable_manager.device_configs
    
    @pytest.mark.asyncio
    async def test_get_patient_data_no_fhir(self):
        """Test patient data retrieval when FHIR is not available."""
        settings = HealthcareIntegrationSettings()
        service = HealthcareIntegrationService(settings)
        
        patient_data = await service.get_patient_data("123")
        
        assert patient_data is None
    
    @pytest.mark.asyncio
    @patch.object(FHIRClient, 'get_patient')
    async def test_get_patient_data_with_fhir(self, mock_get_patient):
        """Test patient data retrieval with FHIR enabled."""
        settings = HealthcareIntegrationSettings(
            fhir_enabled=True,
            fhir_base_url="https://fhir.example.com",
            fhir_auth_type="bearer",
            fhir_token="test_token"
        )
        
        # Mock patient data
        mock_patient_data = {
            "resourceType": "Patient",
            "id": "123",
            "name": [{"family": "Doe", "given": ["John"]}]
        }
        mock_get_patient.return_value = mock_patient_data
        
        service = HealthcareIntegrationService(settings)
        service.integration_status.fhir_connected = True
        
        patient_data = await service.get_patient_data("123")
        
        assert patient_data == mock_patient_data
        mock_get_patient.assert_called_once_with("123")
    
    @pytest.mark.asyncio
    async def test_submit_diagnostic_result(self):
        """Test diagnostic result submission."""
        settings = HealthcareIntegrationSettings(
            fhir_enabled=True,
            fhir_base_url="https://fhir.example.com",
            fhir_auth_type="bearer",
            fhir_token="test_token",
            hl7_enabled=True
        )
        
        service = HealthcareIntegrationService(settings)
        service.integration_status.fhir_connected = True
        service.integration_status.hl7_connected = True
        
        # Create test diagnostic result
        diagnostic_result = DiagnosticResult(
            patient_id="PAT_20241010_12345",
            timestamp=datetime.now(),
            segmentation_mask=None,
            classification_probabilities={"Alzheimer's": 0.85},
            confidence_scores={"overall": 0.92},
            explainability_maps={},
            metrics=None
        )
        
        with patch.object(service.fhir_client, 'create_diagnostic_report', return_value="report_123"):
            results = await service.submit_diagnostic_result(diagnostic_result, "patient_123")
        
        assert results["fhir"] is True
        assert results["hl7"] is True
    
    def test_integration_status_tracking(self):
        """Test integration status tracking."""
        settings = HealthcareIntegrationSettings()
        service = HealthcareIntegrationService(settings)
        
        status = service.get_integration_status()
        
        assert status.fhir_connected is False
        assert status.hl7_connected is False
        assert len(status.wearable_devices_connected) == 0
        assert len(status.error_messages) == 0


if __name__ == "__main__":
    pytest.main([__file__])