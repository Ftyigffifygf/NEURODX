"""
Healthcare integration service that coordinates FHIR, HL7, and wearable device integrations.

This service provides a unified interface for all healthcare system integrations,
managing connections and data exchange across different protocols and devices.
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass

from src.config.healthcare_integration_config import HealthcareIntegrationSettings
from src.services.healthcare_integration.fhir_client import FHIRClient, FHIRConfig
from src.services.healthcare_integration.hl7_interface import HL7Interface, HL7Config
from src.services.healthcare_integration.wearable_sdk_manager import (
    WearableSDKManager, DeviceConfig, WearableDataPoint, DeviceStatus
)
from src.models.patient import PatientRecord, WearableSession
from src.models.diagnostics import DiagnosticResult


@dataclass
class IntegrationStatus:
    """Overall healthcare integration status."""
    fhir_connected: bool = False
    hl7_connected: bool = False
    wearable_devices_connected: Dict[str, bool] = None
    last_sync: Optional[datetime] = None
    error_messages: List[str] = None
    
    def __post_init__(self):
        if self.wearable_devices_connected is None:
            self.wearable_devices_connected = {}
        if self.error_messages is None:
            self.error_messages = []


class HealthcareIntegrationService:
    """
    Unified healthcare integration service.
    
    Coordinates FHIR API, HL7 interface, and wearable device integrations
    to provide seamless healthcare system connectivity.
    """
    
    def __init__(self, settings: Optional[HealthcareIntegrationSettings] = None):
        """
        Initialize healthcare integration service.
        
        Args:
            settings: Healthcare integration settings
        """
        self.settings = settings or HealthcareIntegrationSettings()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.fhir_client: Optional[FHIRClient] = None
        self.hl7_interface: Optional[HL7Interface] = None
        self.wearable_manager: Optional[WearableSDKManager] = None
        
        # Status tracking
        self.integration_status = IntegrationStatus()
        self.status_callbacks: List[Callable[[IntegrationStatus], None]] = []
        
        # Initialize components based on configuration
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize healthcare integration components based on configuration."""
        try:
            # Initialize FHIR client
            fhir_config = self.settings.get_fhir_config()
            if fhir_config:
                self.fhir_client = FHIRClient(fhir_config)
                self.logger.info("FHIR client initialized")
            
            # Initialize HL7 interface
            hl7_config = self.settings.get_hl7_config()
            if hl7_config:
                self.hl7_interface = HL7Interface(hl7_config)
                self.logger.info("HL7 interface initialized")
            
            # Initialize wearable device manager
            if self.settings.wearable_devices_enabled:
                self.wearable_manager = WearableSDKManager()
                
                # Register Fitbit if configured
                fitbit_config = self.settings.get_fitbit_config()
                if fitbit_config:
                    self.wearable_manager.register_device("fitbit", fitbit_config)
                
                # Register WebSocket devices
                websocket_configs = self.settings.get_websocket_device_configs()
                for i, config in enumerate(websocket_configs):
                    device_id = f"websocket_{i}"
                    self.wearable_manager.register_device(device_id, config)
                
                # Set up callbacks
                self.wearable_manager.add_data_callback(self._handle_wearable_data)
                self.wearable_manager.add_status_callback(self._handle_wearable_status)
                
                self.logger.info("Wearable device manager initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize healthcare integration components: {e}")
            self.integration_status.error_messages.append(str(e))
    
    async def connect_all_systems(self) -> IntegrationStatus:
        """
        Connect to all configured healthcare systems.
        
        Returns:
            Integration status after connection attempts
        """
        self.logger.info("Connecting to healthcare systems...")
        
        # Test FHIR connection
        if self.fhir_client:
            try:
                self.integration_status.fhir_connected = self.fhir_client.test_connection()
                if self.integration_status.fhir_connected:
                    self.logger.info("FHIR connection successful")
                else:
                    self.logger.warning("FHIR connection failed")
            except Exception as e:
                self.logger.error(f"FHIR connection error: {e}")
                self.integration_status.error_messages.append(f"FHIR: {e}")
        
        # Start HL7 server
        if self.hl7_interface:
            try:
                self.hl7_interface.start_server()
                self.integration_status.hl7_connected = True
                self.logger.info("HL7 server started successfully")
            except Exception as e:
                self.logger.error(f"HL7 server startup error: {e}")
                self.integration_status.error_messages.append(f"HL7: {e}")
        
        # Connect wearable devices
        if self.wearable_manager:
            try:
                connection_results = await self.wearable_manager.connect_all_devices()
                self.integration_status.wearable_devices_connected = connection_results
                
                connected_count = sum(1 for connected in connection_results.values() if connected)
                total_count = len(connection_results)
                self.logger.info(f"Wearable devices connected: {connected_count}/{total_count}")
                
            except Exception as e:
                self.logger.error(f"Wearable device connection error: {e}")
                self.integration_status.error_messages.append(f"Wearables: {e}")
        
        self.integration_status.last_sync = datetime.now()
        
        # Notify status callbacks
        for callback in self.status_callbacks:
            try:
                callback(self.integration_status)
            except Exception as e:
                self.logger.error(f"Error in status callback: {e}")
        
        return self.integration_status
    
    async def disconnect_all_systems(self) -> None:
        """Disconnect from all healthcare systems."""
        self.logger.info("Disconnecting from healthcare systems...")
        
        # Stop HL7 server
        if self.hl7_interface:
            self.hl7_interface.stop_server()
            self.integration_status.hl7_connected = False
        
        # Disconnect wearable devices
        if self.wearable_manager:
            await self.wearable_manager.disconnect_all_devices()
            self.integration_status.wearable_devices_connected = {}
        
        self.integration_status.fhir_connected = False
        self.logger.info("Disconnected from all healthcare systems")
    
    async def get_patient_data(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve patient data from FHIR system.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            Patient data or None if not found
        """
        if not self.fhir_client or not self.integration_status.fhir_connected:
            self.logger.warning("FHIR client not available for patient data retrieval")
            return None
        
        try:
            return self.fhir_client.get_patient(patient_id)
        except Exception as e:
            self.logger.error(f"Failed to retrieve patient data: {e}")
            return None
    
    async def submit_diagnostic_result(self, diagnostic_result: DiagnosticResult, patient_id: str) -> Dict[str, bool]:
        """
        Submit diagnostic result to healthcare systems.
        
        Args:
            diagnostic_result: Diagnostic result to submit
            patient_id: Patient identifier
            
        Returns:
            Dictionary indicating success for each system
        """
        results = {
            "fhir": False,
            "hl7": False
        }
        
        # Submit to FHIR
        if self.fhir_client and self.integration_status.fhir_connected:
            try:
                report_id = self.fhir_client.create_diagnostic_report(diagnostic_result, patient_id)
                results["fhir"] = bool(report_id)
                self.logger.info(f"Submitted diagnostic result to FHIR: {report_id}")
            except Exception as e:
                self.logger.error(f"Failed to submit to FHIR: {e}")
        
        # Submit to HL7
        if self.hl7_interface and self.integration_status.hl7_connected:
            try:
                hl7_message = self.hl7_interface.create_diagnostic_result_message(diagnostic_result, patient_id)
                # In a real implementation, you would send this to configured HL7 endpoints
                # For now, just log that the message was created
                results["hl7"] = True
                self.logger.info("Created HL7 diagnostic result message")
            except Exception as e:
                self.logger.error(f"Failed to create HL7 message: {e}")
        
        return results
    
    async def get_wearable_data(self, device_id: str, start_time: datetime, end_time: datetime, data_types: List[str]) -> List[WearableDataPoint]:
        """
        Retrieve historical wearable data.
        
        Args:
            device_id: Device identifier
            start_time: Start time for data retrieval
            end_time: End time for data retrieval
            data_types: Types of data to retrieve
            
        Returns:
            List of wearable data points
        """
        if not self.wearable_manager:
            self.logger.warning("Wearable manager not available")
            return []
        
        try:
            return await self.wearable_manager.get_historical_data(device_id, start_time, end_time, data_types)
        except Exception as e:
            self.logger.error(f"Failed to retrieve wearable data: {e}")
            return []
    
    async def get_all_wearable_data(self, start_time: datetime, end_time: datetime, data_types: List[str]) -> Dict[str, List[WearableDataPoint]]:
        """
        Retrieve historical data from all wearable devices.
        
        Args:
            start_time: Start time for data retrieval
            end_time: End time for data retrieval
            data_types: Types of data to retrieve
            
        Returns:
            Dictionary mapping device_id to data points
        """
        if not self.wearable_manager:
            return {}
        
        try:
            return await self.wearable_manager.get_aggregated_historical_data(start_time, end_time, data_types)
        except Exception as e:
            self.logger.error(f"Failed to retrieve aggregated wearable data: {e}")
            return {}
    
    async def start_wearable_streaming(self, device_ids: Optional[List[str]] = None) -> bool:
        """
        Start real-time wearable data streaming.
        
        Args:
            device_ids: Specific device IDs to stream from (None for all)
            
        Returns:
            True if streaming started successfully
        """
        if not self.wearable_manager:
            return False
        
        try:
            await self.wearable_manager.start_real_time_streaming(device_ids)
            self.logger.info("Started wearable data streaming")
            return True
        except Exception as e:
            self.logger.error(f"Failed to start wearable streaming: {e}")
            return False
    
    async def stop_wearable_streaming(self, device_ids: Optional[List[str]] = None) -> bool:
        """
        Stop real-time wearable data streaming.
        
        Args:
            device_ids: Specific device IDs to stop streaming (None for all)
            
        Returns:
            True if streaming stopped successfully
        """
        if not self.wearable_manager:
            return False
        
        try:
            await self.wearable_manager.stop_real_time_streaming(device_ids)
            self.logger.info("Stopped wearable data streaming")
            return True
        except Exception as e:
            self.logger.error(f"Failed to stop wearable streaming: {e}")
            return False
    
    async def get_device_statuses(self) -> Dict[str, DeviceStatus]:
        """
        Get status of all wearable devices.
        
        Returns:
            Dictionary mapping device_id to device status
        """
        if not self.wearable_manager:
            return {}
        
        try:
            return await self.wearable_manager.get_all_device_statuses()
        except Exception as e:
            self.logger.error(f"Failed to get device statuses: {e}")
            return {}
    
    def add_status_callback(self, callback: Callable[[IntegrationStatus], None]) -> None:
        """Add callback for integration status updates."""
        self.status_callbacks.append(callback)
    
    def _handle_wearable_data(self, device_id: str, data_points: List[WearableDataPoint]) -> None:
        """Handle real-time wearable data."""
        self.logger.debug(f"Received {len(data_points)} data points from {device_id}")
        
        # Here you could implement real-time processing, alerts, etc.
        # For now, just log the data reception
        
        # Update last sync time
        self.integration_status.last_sync = datetime.now()
    
    def _handle_wearable_status(self, device_id: str, status: DeviceStatus) -> None:
        """Handle wearable device status updates."""
        self.logger.debug(f"Status update from {device_id}: connected={status.is_connected}")
        
        # Update integration status
        self.integration_status.wearable_devices_connected[device_id] = status.is_connected
        
        # Log any errors
        if status.error_message:
            self.logger.warning(f"Device {device_id} error: {status.error_message}")
    
    def get_integration_status(self) -> IntegrationStatus:
        """Get current integration status."""
        return self.integration_status
    
    async def test_all_connections(self) -> Dict[str, bool]:
        """
        Test connections to all healthcare systems.
        
        Returns:
            Dictionary indicating connection status for each system
        """
        results = {
            "fhir": False,
            "hl7": False,
            "wearables": {}
        }
        
        # Test FHIR
        if self.fhir_client:
            results["fhir"] = self.fhir_client.test_connection()
        
        # Test HL7 (check if server is running)
        if self.hl7_interface:
            results["hl7"] = self.hl7_interface.is_running
        
        # Test wearable devices
        if self.wearable_manager:
            device_statuses = await self.get_device_statuses()
            for device_id, status in device_statuses.items():
                results["wearables"][device_id] = status.is_connected
        
        return results