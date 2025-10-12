"""
Wearable device SDK integration manager.

This module provides unified interface for integrating with major wearable device
platforms including Fitbit, Apple HealthKit, Garmin, and other health monitoring devices.
"""

import logging
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import requests
from requests.auth import HTTPBasicAuth
import websockets
import threading
import time
import numpy as np

from src.models.patient import WearableSession


@dataclass
class DeviceConfig:
    """Configuration for wearable device integration."""
    device_type: str  # "fitbit", "apple_health", "garmin", "polar", "generic"
    api_endpoint: str
    auth_type: str  # "oauth2", "api_key", "basic", "websocket"
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    api_key: Optional[str] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    websocket_url: Optional[str] = None
    timeout: int = 30
    rate_limit: int = 100  # requests per minute
    data_types: List[str] = None  # ["heart_rate", "steps", "sleep", "eeg"]


@dataclass
class DeviceStatus:
    """Device connection and health status."""
    device_id: str
    device_type: str
    is_connected: bool
    last_sync: Optional[datetime]
    battery_level: Optional[float]
    signal_quality: Optional[float]
    error_message: Optional[str] = None
    data_points_received: int = 0
    connection_uptime: Optional[timedelta] = None


@dataclass
class WearableDataPoint:
    """Individual wearable sensor data point."""
    device_id: str
    device_type: str
    data_type: str  # "heart_rate", "steps", "sleep_stage", "eeg_alpha", etc.
    timestamp: datetime
    value: Union[float, int, str]
    unit: str
    quality_score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class WearableDeviceSDK(ABC):
    """Abstract base class for wearable device SDK implementations."""
    
    def __init__(self, config: DeviceConfig):
        """Initialize device SDK with configuration."""
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{config.device_type}")
        self.is_connected = False
        self.last_sync = None
        self.data_callback: Optional[Callable] = None
        self.status_callback: Optional[Callable] = None
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to device/service."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from device/service."""
        pass
    
    @abstractmethod
    async def authenticate(self) -> bool:
        """Authenticate with device/service."""
        pass
    
    @abstractmethod
    async def get_device_status(self) -> DeviceStatus:
        """Get current device status."""
        pass
    
    @abstractmethod
    async def get_historical_data(self, start_time: datetime, end_time: datetime, data_types: List[str]) -> List[WearableDataPoint]:
        """Get historical data for specified time range."""
        pass
    
    @abstractmethod
    async def start_real_time_streaming(self) -> None:
        """Start real-time data streaming."""
        pass
    
    @abstractmethod
    async def stop_real_time_streaming(self) -> None:
        """Stop real-time data streaming."""
        pass
    
    def set_data_callback(self, callback: Callable[[List[WearableDataPoint]], None]) -> None:
        """Set callback for receiving data points."""
        self.data_callback = callback
    
    def set_status_callback(self, callback: Callable[[DeviceStatus], None]) -> None:
        """Set callback for device status updates."""
        self.status_callback = callback


class FitbitSDK(WearableDeviceSDK):
    """Fitbit Web API SDK implementation."""
    
    def __init__(self, config: DeviceConfig):
        super().__init__(config)
        self.session = requests.Session()
        self.user_id = None
    
    async def connect(self) -> bool:
        """Connect to Fitbit Web API."""
        try:
            if await self.authenticate():
                self.is_connected = True
                self.last_sync = datetime.now()
                self.logger.info("Connected to Fitbit API")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to connect to Fitbit: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Fitbit API."""
        self.is_connected = False
        self.session.close()
        self.logger.info("Disconnected from Fitbit API")
    
    async def authenticate(self) -> bool:
        """Authenticate with Fitbit OAuth2."""
        if not self.config.access_token:
            # Would typically implement OAuth2 flow here
            self.logger.error("No access token provided for Fitbit authentication")
            return False
        
        # Set authorization header
        self.session.headers.update({
            'Authorization': f'Bearer {self.config.access_token}',
            'Accept': 'application/json'
        })
        
        # Test authentication with profile request
        try:
            response = self.session.get(f"{self.config.api_endpoint}/1/user/-/profile.json")
            response.raise_for_status()
            
            profile_data = response.json()
            self.user_id = profile_data.get('user', {}).get('encodedId')
            
            self.logger.info(f"Authenticated with Fitbit for user: {self.user_id}")
            return True
            
        except requests.RequestException as e:
            self.logger.error(f"Fitbit authentication failed: {e}")
            return False
    
    async def get_device_status(self) -> DeviceStatus:
        """Get Fitbit device status."""
        try:
            response = self.session.get(f"{self.config.api_endpoint}/1/user/-/devices.json")
            response.raise_for_status()
            
            devices_data = response.json()
            
            # Get first device (could be enhanced to handle multiple devices)
            if devices_data:
                device = devices_data[0]
                return DeviceStatus(
                    device_id=device.get('id', 'unknown'),
                    device_type='fitbit',
                    is_connected=self.is_connected,
                    last_sync=datetime.fromisoformat(device.get('lastSyncTime', '').replace('Z', '+00:00')) if device.get('lastSyncTime') else None,
                    battery_level=device.get('batteryLevel'),
                    signal_quality=None,  # Fitbit doesn't provide this
                    data_points_received=0  # Would track this in real implementation
                )
            else:
                return DeviceStatus(
                    device_id='no_device',
                    device_type='fitbit',
                    is_connected=False,
                    last_sync=None,
                    battery_level=None,
                    signal_quality=None,
                    error_message="No devices found"
                )
                
        except Exception as e:
            self.logger.error(f"Failed to get Fitbit device status: {e}")
            return DeviceStatus(
                device_id='error',
                device_type='fitbit',
                is_connected=False,
                last_sync=None,
                battery_level=None,
                signal_quality=None,
                error_message=str(e)
            )
    
    async def get_historical_data(self, start_time: datetime, end_time: datetime, data_types: List[str]) -> List[WearableDataPoint]:
        """Get historical data from Fitbit."""
        data_points = []
        
        for data_type in data_types:
            try:
                if data_type == "heart_rate":
                    data_points.extend(await self._get_heart_rate_data(start_time, end_time))
                elif data_type == "steps":
                    data_points.extend(await self._get_steps_data(start_time, end_time))
                elif data_type == "sleep":
                    data_points.extend(await self._get_sleep_data(start_time, end_time))
                else:
                    self.logger.warning(f"Unsupported data type for Fitbit: {data_type}")
                    
            except Exception as e:
                self.logger.error(f"Failed to get {data_type} data from Fitbit: {e}")
        
        return data_points
    
    async def _get_heart_rate_data(self, start_time: datetime, end_time: datetime) -> List[WearableDataPoint]:
        """Get heart rate data from Fitbit."""
        data_points = []
        
        # Fitbit API requires date-by-date requests
        current_date = start_time.date()
        end_date = end_time.date()
        
        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            
            try:
                response = self.session.get(
                    f"{self.config.api_endpoint}/1/user/-/activities/heart/date/{date_str}/1d/1min.json"
                )
                response.raise_for_status()
                
                heart_data = response.json()
                
                # Process intraday data
                if 'activities-heart-intraday' in heart_data:
                    intraday_data = heart_data['activities-heart-intraday']['dataset']
                    
                    for point in intraday_data:
                        timestamp = datetime.combine(current_date, datetime.strptime(point['time'], '%H:%M:%S').time())
                        
                        # Filter by time range
                        if start_time <= timestamp <= end_time:
                            data_points.append(WearableDataPoint(
                                device_id=self.user_id or 'fitbit_user',
                                device_type='fitbit',
                                data_type='heart_rate',
                                timestamp=timestamp,
                                value=point['value'],
                                unit='bpm'
                            ))
                
            except Exception as e:
                self.logger.error(f"Failed to get heart rate data for {date_str}: {e}")
            
            current_date += timedelta(days=1)
        
        return data_points
    
    async def _get_steps_data(self, start_time: datetime, end_time: datetime) -> List[WearableDataPoint]:
        """Get steps data from Fitbit."""
        # Similar implementation to heart rate but for steps
        # Simplified for brevity
        return []
    
    async def _get_sleep_data(self, start_time: datetime, end_time: datetime) -> List[WearableDataPoint]:
        """Get sleep data from Fitbit."""
        # Similar implementation for sleep data
        # Simplified for brevity
        return []
    
    async def start_real_time_streaming(self) -> None:
        """Start real-time streaming (Fitbit doesn't support real-time streaming)."""
        self.logger.warning("Fitbit does not support real-time streaming")
    
    async def stop_real_time_streaming(self) -> None:
        """Stop real-time streaming."""
        pass


class GenericWebSocketSDK(WearableDeviceSDK):
    """Generic WebSocket SDK for real-time wearable data streaming."""
    
    def __init__(self, config: DeviceConfig):
        super().__init__(config)
        self.websocket = None
        self.streaming_task = None
    
    async def connect(self) -> bool:
        """Connect to WebSocket endpoint."""
        try:
            if not self.config.websocket_url:
                raise ValueError("WebSocket URL not configured")
            
            self.websocket = await websockets.connect(
                self.config.websocket_url,
                timeout=self.config.timeout
            )
            
            self.is_connected = True
            self.last_sync = datetime.now()
            self.logger.info(f"Connected to WebSocket: {self.config.websocket_url}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to WebSocket: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from WebSocket."""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        
        if self.streaming_task:
            self.streaming_task.cancel()
            self.streaming_task = None
        
        self.is_connected = False
        self.logger.info("Disconnected from WebSocket")
    
    async def authenticate(self) -> bool:
        """Authenticate with WebSocket service."""
        if not self.websocket:
            return False
        
        try:
            # Send authentication message
            auth_message = {
                "type": "auth",
                "api_key": self.config.api_key,
                "client_id": self.config.client_id
            }
            
            await self.websocket.send(json.dumps(auth_message))
            
            # Wait for auth response
            response = await asyncio.wait_for(self.websocket.recv(), timeout=10)
            auth_response = json.loads(response)
            
            if auth_response.get("status") == "authenticated":
                self.logger.info("WebSocket authentication successful")
                return True
            else:
                self.logger.error(f"WebSocket authentication failed: {auth_response}")
                return False
                
        except Exception as e:
            self.logger.error(f"WebSocket authentication error: {e}")
            return False
    
    async def get_device_status(self) -> DeviceStatus:
        """Get device status via WebSocket."""
        if not self.websocket:
            return DeviceStatus(
                device_id='websocket',
                device_type=self.config.device_type,
                is_connected=False,
                last_sync=None,
                battery_level=None,
                signal_quality=None,
                error_message="Not connected"
            )
        
        try:
            # Request status
            status_request = {"type": "get_status"}
            await self.websocket.send(json.dumps(status_request))
            
            # Wait for response
            response = await asyncio.wait_for(self.websocket.recv(), timeout=5)
            status_data = json.loads(response)
            
            return DeviceStatus(
                device_id=status_data.get('device_id', 'websocket'),
                device_type=self.config.device_type,
                is_connected=self.is_connected,
                last_sync=self.last_sync,
                battery_level=status_data.get('battery_level'),
                signal_quality=status_data.get('signal_quality'),
                data_points_received=status_data.get('data_points_received', 0)
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get WebSocket device status: {e}")
            return DeviceStatus(
                device_id='websocket',
                device_type=self.config.device_type,
                is_connected=self.is_connected,
                last_sync=self.last_sync,
                battery_level=None,
                signal_quality=None,
                error_message=str(e)
            )
    
    async def get_historical_data(self, start_time: datetime, end_time: datetime, data_types: List[str]) -> List[WearableDataPoint]:
        """Get historical data via WebSocket."""
        if not self.websocket:
            return []
        
        try:
            # Request historical data
            history_request = {
                "type": "get_historical_data",
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "data_types": data_types
            }
            
            await self.websocket.send(json.dumps(history_request))
            
            # Wait for response
            response = await asyncio.wait_for(self.websocket.recv(), timeout=30)
            history_data = json.loads(response)
            
            # Convert to WearableDataPoint objects
            data_points = []
            for point_data in history_data.get('data_points', []):
                data_points.append(WearableDataPoint(
                    device_id=point_data['device_id'],
                    device_type=self.config.device_type,
                    data_type=point_data['data_type'],
                    timestamp=datetime.fromisoformat(point_data['timestamp']),
                    value=point_data['value'],
                    unit=point_data['unit'],
                    quality_score=point_data.get('quality_score'),
                    metadata=point_data.get('metadata')
                ))
            
            return data_points
            
        except Exception as e:
            self.logger.error(f"Failed to get historical data via WebSocket: {e}")
            return []
    
    async def start_real_time_streaming(self) -> None:
        """Start real-time data streaming."""
        if not self.websocket:
            raise RuntimeError("Not connected to WebSocket")
        
        # Send streaming start request
        stream_request = {
            "type": "start_streaming",
            "data_types": self.config.data_types or ["heart_rate", "eeg"]
        }
        
        await self.websocket.send(json.dumps(stream_request))
        
        # Start streaming task
        self.streaming_task = asyncio.create_task(self._streaming_loop())
        self.logger.info("Started real-time streaming")
    
    async def stop_real_time_streaming(self) -> None:
        """Stop real-time data streaming."""
        if self.websocket:
            stop_request = {"type": "stop_streaming"}
            await self.websocket.send(json.dumps(stop_request))
        
        if self.streaming_task:
            self.streaming_task.cancel()
            self.streaming_task = None
        
        self.logger.info("Stopped real-time streaming")
    
    async def _streaming_loop(self) -> None:
        """Main streaming loop to receive real-time data."""
        try:
            while self.websocket and self.is_connected:
                message = await self.websocket.recv()
                data = json.loads(message)
                
                if data.get('type') == 'data_point':
                    # Convert to WearableDataPoint
                    data_point = WearableDataPoint(
                        device_id=data['device_id'],
                        device_type=self.config.device_type,
                        data_type=data['data_type'],
                        timestamp=datetime.fromisoformat(data['timestamp']),
                        value=data['value'],
                        unit=data['unit'],
                        quality_score=data.get('quality_score'),
                        metadata=data.get('metadata')
                    )
                    
                    # Call data callback if set
                    if self.data_callback:
                        self.data_callback([data_point])
                
                elif data.get('type') == 'status_update':
                    # Handle status updates
                    if self.status_callback:
                        status = DeviceStatus(
                            device_id=data['device_id'],
                            device_type=self.config.device_type,
                            is_connected=data.get('is_connected', True),
                            last_sync=datetime.fromisoformat(data['timestamp']),
                            battery_level=data.get('battery_level'),
                            signal_quality=data.get('signal_quality')
                        )
                        self.status_callback(status)
                
        except asyncio.CancelledError:
            self.logger.info("Streaming loop cancelled")
        except Exception as e:
            self.logger.error(f"Streaming loop error: {e}")


class WearableSDKManager:
    """
    Manager for multiple wearable device SDK integrations.
    
    Provides unified interface for managing connections to multiple wearable devices
    and aggregating data from different sources.
    """
    
    def __init__(self):
        """Initialize SDK manager."""
        self.logger = logging.getLogger(__name__)
        self.sdks: Dict[str, WearableDeviceSDK] = {}
        self.device_configs: Dict[str, DeviceConfig] = {}
        self.data_callbacks: List[Callable] = []
        self.status_callbacks: List[Callable] = []
    
    def register_device(self, device_id: str, config: DeviceConfig) -> None:
        """
        Register wearable device configuration.
        
        Args:
            device_id: Unique device identifier
            config: Device configuration
        """
        self.device_configs[device_id] = config
        
        # Create appropriate SDK instance
        if config.device_type == "fitbit":
            sdk = FitbitSDK(config)
        elif config.device_type in ["generic", "eeg", "polar"]:
            sdk = GenericWebSocketSDK(config)
        else:
            # Default to generic WebSocket SDK
            sdk = GenericWebSocketSDK(config)
        
        # Set callbacks
        sdk.set_data_callback(self._handle_device_data)
        sdk.set_status_callback(self._handle_device_status)
        
        self.sdks[device_id] = sdk
        self.logger.info(f"Registered device: {device_id} ({config.device_type})")
    
    async def connect_device(self, device_id: str) -> bool:
        """
        Connect to specific device.
        
        Args:
            device_id: Device identifier
            
        Returns:
            True if connection successful
        """
        if device_id not in self.sdks:
            self.logger.error(f"Device not registered: {device_id}")
            return False
        
        sdk = self.sdks[device_id]
        success = await sdk.connect()
        
        if success:
            self.logger.info(f"Connected to device: {device_id}")
        else:
            self.logger.error(f"Failed to connect to device: {device_id}")
        
        return success
    
    async def connect_all_devices(self) -> Dict[str, bool]:
        """
        Connect to all registered devices.
        
        Returns:
            Dictionary mapping device_id to connection success status
        """
        results = {}
        
        for device_id in self.sdks:
            results[device_id] = await self.connect_device(device_id)
        
        return results
    
    async def disconnect_device(self, device_id: str) -> None:
        """Disconnect from specific device."""
        if device_id in self.sdks:
            await self.sdks[device_id].disconnect()
            self.logger.info(f"Disconnected from device: {device_id}")
    
    async def disconnect_all_devices(self) -> None:
        """Disconnect from all devices."""
        for device_id in self.sdks:
            await self.disconnect_device(device_id)
    
    async def get_device_status(self, device_id: str) -> Optional[DeviceStatus]:
        """Get status for specific device."""
        if device_id not in self.sdks:
            return None
        
        return await self.sdks[device_id].get_device_status()
    
    async def get_all_device_statuses(self) -> Dict[str, DeviceStatus]:
        """Get status for all devices."""
        statuses = {}
        
        for device_id in self.sdks:
            status = await self.get_device_status(device_id)
            if status:
                statuses[device_id] = status
        
        return statuses
    
    async def get_historical_data(self, device_id: str, start_time: datetime, end_time: datetime, data_types: List[str]) -> List[WearableDataPoint]:
        """Get historical data from specific device."""
        if device_id not in self.sdks:
            return []
        
        return await self.sdks[device_id].get_historical_data(start_time, end_time, data_types)
    
    async def get_aggregated_historical_data(self, start_time: datetime, end_time: datetime, data_types: List[str]) -> Dict[str, List[WearableDataPoint]]:
        """Get historical data from all devices."""
        aggregated_data = {}
        
        for device_id in self.sdks:
            data = await self.get_historical_data(device_id, start_time, end_time, data_types)
            if data:
                aggregated_data[device_id] = data
        
        return aggregated_data
    
    async def start_real_time_streaming(self, device_ids: Optional[List[str]] = None) -> None:
        """Start real-time streaming for specified devices (or all if None)."""
        target_devices = device_ids or list(self.sdks.keys())
        
        for device_id in target_devices:
            if device_id in self.sdks:
                try:
                    await self.sdks[device_id].start_real_time_streaming()
                    self.logger.info(f"Started streaming for device: {device_id}")
                except Exception as e:
                    self.logger.error(f"Failed to start streaming for {device_id}: {e}")
    
    async def stop_real_time_streaming(self, device_ids: Optional[List[str]] = None) -> None:
        """Stop real-time streaming for specified devices (or all if None)."""
        target_devices = device_ids or list(self.sdks.keys())
        
        for device_id in target_devices:
            if device_id in self.sdks:
                try:
                    await self.sdks[device_id].stop_real_time_streaming()
                    self.logger.info(f"Stopped streaming for device: {device_id}")
                except Exception as e:
                    self.logger.error(f"Failed to stop streaming for {device_id}: {e}")
    
    def add_data_callback(self, callback: Callable[[str, List[WearableDataPoint]], None]) -> None:
        """Add callback for receiving data from any device."""
        self.data_callbacks.append(callback)
    
    def add_status_callback(self, callback: Callable[[str, DeviceStatus], None]) -> None:
        """Add callback for receiving status updates from any device."""
        self.status_callbacks.append(callback)
    
    def _handle_device_data(self, data_points: List[WearableDataPoint]) -> None:
        """Handle data from device SDKs."""
        if data_points:
            device_id = data_points[0].device_id
            
            # Call all registered data callbacks
            for callback in self.data_callbacks:
                try:
                    callback(device_id, data_points)
                except Exception as e:
                    self.logger.error(f"Error in data callback: {e}")
    
    def _handle_device_status(self, status: DeviceStatus) -> None:
        """Handle status updates from device SDKs."""
        device_id = status.device_id
        
        # Call all registered status callbacks
        for callback in self.status_callbacks:
            try:
                callback(device_id, status)
            except Exception as e:
                self.logger.error(f"Error in status callback: {e}")
    
    def convert_to_wearable_session(self, device_id: str, data_points: List[WearableDataPoint]) -> WearableSession:
        """
        Convert wearable data points to WearableSession model.
        
        Args:
            device_id: Device identifier
            data_points: List of data points
            
        Returns:
            WearableSession object
        """
        if not data_points:
            raise ValueError("No data points provided")
        
        # Group data points by type
        grouped_data = {}
        for point in data_points:
            if point.data_type not in grouped_data:
                grouped_data[point.data_type] = []
            grouped_data[point.data_type].append(point.value)
        
        # Determine device type and session info
        first_point = data_points[0]
        last_point = data_points[-1]
        
        # Calculate sampling rate (simplified)
        if len(data_points) > 1:
            time_diff = (last_point.timestamp - first_point.timestamp).total_seconds()
            sampling_rate = len(data_points) / time_diff if time_diff > 0 else 1.0
            # Ensure sampling rate is within valid range for HeartRate (0.1Hz to 10Hz)
            sampling_rate = max(0.1, min(10.0, sampling_rate))
        else:
            sampling_rate = 1.0
        
        # Map device type to valid format
        device_type_mapping = {
            "fitbit": "HeartRate",
            "eeg": "EEG", 
            "heart_rate": "HeartRate",
            "sleep": "Sleep",
            "gait": "Gait"
        }
        
        mapped_device_type = device_type_mapping.get(first_point.device_type.lower(), "HeartRate")
        
        # Create session
        session = WearableSession(
            session_id=f"WEAR_{mapped_device_type}_{first_point.timestamp.strftime('%Y%m%d_%H%M%S')}",
            device_type=mapped_device_type,
            start_time=first_point.timestamp,
            end_time=last_point.timestamp,
            sampling_rate=sampling_rate,
            raw_data=np.array([point.value for point in data_points]),  # Convert to numpy array
            processed_features=self._extract_session_features(data_points)
        )
        
        return session
    
    def _extract_session_features(self, data_points: List[WearableDataPoint]) -> Dict[str, float]:
        """Extract statistical features from data points."""
        features = {}
        
        # Group by data type
        grouped_data = {}
        for point in data_points:
            if point.data_type not in grouped_data:
                grouped_data[point.data_type] = []
            if isinstance(point.value, (int, float)):
                grouped_data[point.data_type].append(point.value)
        
        # Calculate features for each data type
        for data_type, values in grouped_data.items():
            if values:
                features[f"{data_type}_mean"] = sum(values) / len(values)
                features[f"{data_type}_min"] = min(values)
                features[f"{data_type}_max"] = max(values)
                features[f"{data_type}_count"] = len(values)
                
                # Calculate standard deviation
                mean_val = features[f"{data_type}_mean"]
                variance = sum((x - mean_val) ** 2 for x in values) / len(values)
                features[f"{data_type}_std"] = variance ** 0.5
        
        return features