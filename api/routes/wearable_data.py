"""
Wearable data integration API endpoints.
Implements endpoints for wearable sensor data ingestion, real-time streaming, and data synchronization.
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import json
import numpy as np

from flask import Blueprint, request, jsonify, stream_with_context, Response
from werkzeug.datastructures import FileStorage

from src.services.wearable_sensor.sensor_data_collector import SensorDataCollector
from src.services.wearable_sensor.temporal_synchronizer import TemporalSynchronizer
from src.services.data_fusion.multi_modal_fusion import MultiModalFusion
from src.models.patient import WearableSession
from src.config.settings import get_settings
from src.utils.logging_config import get_logger

logger = get_logger(__name__)
wearable_bp = Blueprint('wearable_data', __name__)

# Global instances
sensor_collector = SensorDataCollector()
temporal_synchronizer = TemporalSynchronizer()
data_fusion = MultiModalFusion()

# In-memory storage for wearable sessions (in production, use database)
wearable_sessions = {}
streaming_sessions = {}


def generate_session_id(device_type: str) -> str:
    """Generate a unique session ID for wearable data."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"WEAR_{device_type}_{timestamp}"


def validate_device_type(device_type: str) -> bool:
    """Validate if device type is supported."""
    valid_types = ['EEG', 'HeartRate', 'Sleep', 'Gait']
    return device_type in valid_types


def parse_sensor_data(data: Dict[str, Any], device_type: str) -> np.ndarray:
    """Parse sensor data from request into numpy array."""
    if 'data' not in data:
        raise ValueError("Missing 'data' field in sensor data")
    
    sensor_data = data['data']
    
    # Handle different data formats
    if isinstance(sensor_data, list):
        return np.array(sensor_data, dtype=np.float32)
    elif isinstance(sensor_data, dict):
        # For multi-channel data (e.g., EEG)
        if device_type == 'EEG' and 'channels' in sensor_data:
            channels = sensor_data['channels']
            return np.array([channels[ch] for ch in sorted(channels.keys())], dtype=np.float32)
        else:
            raise ValueError(f"Unsupported data format for {device_type}")
    else:
        raise ValueError("Sensor data must be a list or dictionary")


@wearable_bp.route('/upload', methods=['POST'])
def upload_wearable_data():
    """
    Upload wearable sensor data for processing.
    
    Expected JSON body:
    - device_type: Type of wearable device (EEG, HeartRate, Sleep, Gait)
    - patient_id: Patient identifier
    - start_time: Session start time (ISO format)
    - end_time: Session end time (ISO format)
    - sampling_rate: Data sampling rate in Hz
    - data: Sensor data array or multi-channel data
    - device_info: Optional device information
    
    Returns:
        JSON response with session information
    """
    try:
        # Validate request
        if not request.is_json:
            return jsonify({
                'error': 'Invalid content type',
                'message': 'Request must contain JSON data'
            }), 400
        
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['device_type', 'patient_id', 'start_time', 'end_time', 'sampling_rate', 'data']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'error': 'Missing required field',
                    'message': f'Field "{field}" is required',
                    'required_fields': required_fields
                }), 400
        
        # Validate device type
        device_type = data['device_type']
        if not validate_device_type(device_type):
            return jsonify({
                'error': 'Invalid device type',
                'message': f'Device type must be one of: EEG, HeartRate, Sleep, Gait',
                'provided': device_type
            }), 400
        
        # Parse timestamps
        try:
            start_time = datetime.fromisoformat(data['start_time'].replace('Z', '+00:00'))
            end_time = datetime.fromisoformat(data['end_time'].replace('Z', '+00:00'))
        except ValueError as e:
            return jsonify({
                'error': 'Invalid timestamp format',
                'message': 'Timestamps must be in ISO format',
                'details': str(e)
            }), 400
        
        # Validate sampling rate
        sampling_rate = float(data['sampling_rate'])
        if sampling_rate <= 0:
            return jsonify({
                'error': 'Invalid sampling rate',
                'message': 'Sampling rate must be positive'
            }), 400
        
        # Parse sensor data
        try:
            sensor_data = parse_sensor_data(data, device_type)
        except ValueError as e:
            return jsonify({
                'error': 'Invalid sensor data',
                'message': str(e)
            }), 400
        
        # Generate session ID
        session_id = generate_session_id(device_type)
        
        # Create wearable session
        device_info = data.get('device_info', {})
        wearable_session = WearableSession(
            session_id=session_id,
            device_type=device_type,
            start_time=start_time,
            end_time=end_time,
            sampling_rate=sampling_rate,
            raw_data=sensor_data,
            device_manufacturer=device_info.get('manufacturer'),
            device_model=device_info.get('model'),
            firmware_version=device_info.get('firmware_version')
        )
        
        # Store session
        wearable_sessions[session_id] = {
            'session': wearable_session,
            'patient_id': data['patient_id'],
            'upload_timestamp': datetime.now().isoformat(),
            'status': 'uploaded',
            'processed_features': {}
        }
        
        logger.info(f"Wearable data uploaded: {session_id} for patient {data['patient_id']}")
        
        return jsonify({
            'message': 'Wearable data uploaded successfully',
            'session_id': session_id,
            'patient_id': data['patient_id'],
            'device_type': device_type,
            'data_shape': sensor_data.shape,
            'duration_seconds': (end_time - start_time).total_seconds(),
            'upload_timestamp': datetime.now().isoformat(),
            'processing_url': f'/api/v1/wearable/process/{session_id}'
        }), 201
        
    except Exception as e:
        logger.error(f"Wearable upload error: {str(e)}")
        return jsonify({
            'error': 'Upload failed',
            'message': 'An unexpected error occurred during upload',
            'details': str(e)
        }), 500


@wearable_bp.route('/stream/start', methods=['POST'])
def start_streaming_session():
    """
    Start a real-time data streaming session.
    
    Expected JSON body:
    - device_type: Type of wearable device
    - patient_id: Patient identifier
    - sampling_rate: Expected sampling rate
    - buffer_size: Optional buffer size for streaming
    
    Returns:
        JSON response with streaming session information
    """
    try:
        if not request.is_json:
            return jsonify({
                'error': 'Invalid content type',
                'message': 'Request must contain JSON data'
            }), 400
        
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['device_type', 'patient_id', 'sampling_rate']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'error': 'Missing required field',
                    'message': f'Field "{field}" is required'
                }), 400
        
        device_type = data['device_type']
        if not validate_device_type(device_type):
            return jsonify({
                'error': 'Invalid device type',
                'message': f'Device type must be one of: EEG, HeartRate, Sleep, Gait'
            }), 400
        
        # Generate streaming session ID
        stream_id = f"STREAM_{device_type}_{uuid.uuid4().hex[:8]}"
        
        # Initialize streaming session
        streaming_sessions[stream_id] = {
            'device_type': device_type,
            'patient_id': data['patient_id'],
            'sampling_rate': float(data['sampling_rate']),
            'buffer_size': data.get('buffer_size', 1000),
            'start_time': datetime.now(),
            'status': 'active',
            'data_buffer': [],
            'last_update': datetime.now()
        }
        
        logger.info(f"Streaming session started: {stream_id}")
        
        return jsonify({
            'message': 'Streaming session started',
            'stream_id': stream_id,
            'device_type': device_type,
            'patient_id': data['patient_id'],
            'sampling_rate': data['sampling_rate'],
            'stream_url': f'/api/v1/wearable/stream/{stream_id}/data',
            'status_url': f'/api/v1/wearable/stream/{stream_id}/status'
        }), 201
        
    except Exception as e:
        logger.error(f"Start streaming error: {str(e)}")
        return jsonify({
            'error': 'Failed to start streaming',
            'message': 'An unexpected error occurred',
            'details': str(e)
        }), 500


@wearable_bp.route('/stream/<stream_id>/data', methods=['POST'])
def stream_data(stream_id: str):
    """
    Stream real-time sensor data to an active session.
    
    Args:
        stream_id: Streaming session identifier
        
    Expected JSON body:
    - timestamp: Data timestamp (ISO format)
    - data: Sensor data points
    
    Returns:
        JSON response with streaming status
    """
    try:
        if stream_id not in streaming_sessions:
            return jsonify({
                'error': 'Stream not found',
                'message': f'No active stream found with ID: {stream_id}'
            }), 404
        
        session = streaming_sessions[stream_id]
        
        if session['status'] != 'active':
            return jsonify({
                'error': 'Stream not active',
                'message': f'Stream status is: {session["status"]}'
            }), 400
        
        if not request.is_json:
            return jsonify({
                'error': 'Invalid content type',
                'message': 'Request must contain JSON data'
            }), 400
        
        data = request.get_json()
        
        # Validate data
        if 'timestamp' not in data or 'data' not in data:
            return jsonify({
                'error': 'Missing required fields',
                'message': 'Both "timestamp" and "data" fields are required'
            }), 400
        
        # Parse timestamp
        try:
            timestamp = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
        except ValueError:
            return jsonify({
                'error': 'Invalid timestamp format',
                'message': 'Timestamp must be in ISO format'
            }), 400
        
        # Add data to buffer
        data_point = {
            'timestamp': timestamp,
            'data': data['data']
        }
        
        session['data_buffer'].append(data_point)
        session['last_update'] = datetime.now()
        
        # Maintain buffer size
        if len(session['data_buffer']) > session['buffer_size']:
            session['data_buffer'] = session['data_buffer'][-session['buffer_size']:]
        
        return jsonify({
            'message': 'Data received',
            'stream_id': stream_id,
            'buffer_size': len(session['data_buffer']),
            'last_update': session['last_update'].isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Stream data error: {str(e)}")
        return jsonify({
            'error': 'Failed to process stream data',
            'message': 'An unexpected error occurred',
            'details': str(e)
        }), 500


@wearable_bp.route('/process/<session_id>', methods=['POST'])
def process_wearable_data(session_id: str):
    """
    Process wearable sensor data for feature extraction and synchronization.
    
    Args:
        session_id: Wearable session identifier
        
    Expected JSON body (optional):
    - processing_options: Optional processing configuration
    
    Returns:
        JSON response with processing results
    """
    try:
        if session_id not in wearable_sessions:
            return jsonify({
                'error': 'Session not found',
                'message': f'No session found with ID: {session_id}'
            }), 404
        
        session_data = wearable_sessions[session_id]
        wearable_session = session_data['session']
        
        # Get processing options
        processing_options = {}
        if request.is_json:
            processing_options = request.get_json() or {}
        
        # Process sensor data using sensor collector
        try:
            processed_features = sensor_collector.extract_features(
                wearable_session.raw_data,
                wearable_session.device_type,
                wearable_session.sampling_rate
            )
            
            # Update session with processed features
            session_data['processed_features'] = processed_features
            session_data['status'] = 'processed'
            
            logger.info(f"Wearable data processed: {session_id}")
            
            return jsonify({
                'message': 'Wearable data processed successfully',
                'session_id': session_id,
                'device_type': wearable_session.device_type,
                'processed_features': processed_features,
                'processing_timestamp': datetime.now().isoformat()
            }), 200
            
        except Exception as e:
            session_data['status'] = 'failed'
            logger.error(f"Processing failed for session {session_id}: {str(e)}")
            return jsonify({
                'error': 'Processing failed',
                'message': f'Failed to process wearable data: {str(e)}',
                'session_id': session_id
            }), 500
        
    except Exception as e:
        logger.error(f"Process wearable data error: {str(e)}")
        return jsonify({
            'error': 'Processing request failed',
            'message': 'An unexpected error occurred',
            'details': str(e)
        }), 500


@wearable_bp.route('/synchronize', methods=['POST'])
def synchronize_multi_modal_data():
    """
    Synchronize multiple wearable data sessions for multi-modal fusion.
    
    Expected JSON body:
    - session_ids: List of wearable session IDs to synchronize
    - patient_id: Patient identifier
    - synchronization_options: Optional synchronization configuration
    
    Returns:
        JSON response with synchronized data information
    """
    try:
        if not request.is_json:
            return jsonify({
                'error': 'Invalid content type',
                'message': 'Request must contain JSON data'
            }), 400
        
        data = request.get_json()
        
        # Validate required fields
        if 'session_ids' not in data or 'patient_id' not in data:
            return jsonify({
                'error': 'Missing required fields',
                'message': 'Both "session_ids" and "patient_id" are required'
            }), 400
        
        session_ids = data['session_ids']
        patient_id = data['patient_id']
        
        if not isinstance(session_ids, list) or len(session_ids) < 2:
            return jsonify({
                'error': 'Invalid session IDs',
                'message': 'At least 2 session IDs are required for synchronization'
            }), 400
        
        # Validate all sessions exist and belong to the same patient
        sessions = []
        for session_id in session_ids:
            if session_id not in wearable_sessions:
                return jsonify({
                    'error': 'Session not found',
                    'message': f'Session {session_id} not found'
                }), 404
            
            session_data = wearable_sessions[session_id]
            if session_data['patient_id'] != patient_id:
                return jsonify({
                    'error': 'Patient ID mismatch',
                    'message': f'Session {session_id} belongs to different patient'
                }), 400
            
            sessions.append(session_data['session'])
        
        # Perform temporal synchronization
        sync_options = data.get('synchronization_options', {})
        
        try:
            synchronized_data = temporal_synchronizer.synchronize_sessions(
                sessions,
                **sync_options
            )
            
            # Generate synchronization ID
            sync_id = f"SYNC_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Store synchronized data (in production, save to database)
            sync_result = {
                'sync_id': sync_id,
                'patient_id': patient_id,
                'session_ids': session_ids,
                'synchronized_data': synchronized_data,
                'sync_timestamp': datetime.now().isoformat(),
                'status': 'completed'
            }
            
            logger.info(f"Multi-modal synchronization completed: {sync_id}")
            
            return jsonify({
                'message': 'Multi-modal data synchronized successfully',
                'sync_id': sync_id,
                'patient_id': patient_id,
                'synchronized_sessions': len(session_ids),
                'sync_timestamp': datetime.now().isoformat(),
                'fusion_url': f'/api/v1/wearable/fusion/{sync_id}'
            }), 200
            
        except Exception as e:
            logger.error(f"Synchronization failed: {str(e)}")
            return jsonify({
                'error': 'Synchronization failed',
                'message': f'Failed to synchronize data: {str(e)}'
            }), 500
        
    except Exception as e:
        logger.error(f"Synchronize data error: {str(e)}")
        return jsonify({
            'error': 'Synchronization request failed',
            'message': 'An unexpected error occurred',
            'details': str(e)
        }), 500


@wearable_bp.route('/sessions', methods=['GET'])
def list_wearable_sessions():
    """
    List wearable data sessions with optional filtering.
    
    Query parameters:
    - patient_id: Filter by patient ID
    - device_type: Filter by device type
    - status: Filter by processing status
    
    Returns:
        JSON response with list of sessions
    """
    try:
        # Get query parameters
        patient_id_filter = request.args.get('patient_id')
        device_type_filter = request.args.get('device_type')
        status_filter = request.args.get('status')
        
        # Filter sessions
        filtered_sessions = []
        for session_id, session_data in wearable_sessions.items():
            wearable_session = session_data['session']
            
            # Apply filters
            if patient_id_filter and session_data['patient_id'] != patient_id_filter:
                continue
            if device_type_filter and wearable_session.device_type != device_type_filter:
                continue
            if status_filter and session_data['status'] != status_filter:
                continue
            
            filtered_sessions.append({
                'session_id': session_id,
                'patient_id': session_data['patient_id'],
                'device_type': wearable_session.device_type,
                'start_time': wearable_session.start_time.isoformat(),
                'end_time': wearable_session.end_time.isoformat(),
                'duration_seconds': (wearable_session.end_time - wearable_session.start_time).total_seconds(),
                'sampling_rate': wearable_session.sampling_rate,
                'status': session_data['status'],
                'upload_timestamp': session_data['upload_timestamp']
            })
        
        # Sort by upload time (newest first)
        filtered_sessions.sort(key=lambda x: x['upload_timestamp'], reverse=True)
        
        return jsonify({
            'sessions': filtered_sessions,
            'total_count': len(filtered_sessions),
            'filters_applied': {
                'patient_id': patient_id_filter,
                'device_type': device_type_filter,
                'status': status_filter
            }
        }), 200
        
    except Exception as e:
        logger.error(f"List sessions error: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve sessions',
            'message': 'An unexpected error occurred',
            'details': str(e)
        }), 500


@wearable_bp.route('/session/<session_id>', methods=['GET'])
def get_session_details(session_id: str):
    """
    Get detailed information about a wearable session.
    
    Args:
        session_id: Session identifier
        
    Returns:
        JSON response with session details
    """
    try:
        if session_id not in wearable_sessions:
            return jsonify({
                'error': 'Session not found',
                'message': f'No session found with ID: {session_id}'
            }), 404
        
        session_data = wearable_sessions[session_id]
        wearable_session = session_data['session']
        
        session_details = {
            'session_id': session_id,
            'patient_id': session_data['patient_id'],
            'device_info': {
                'device_type': wearable_session.device_type,
                'manufacturer': wearable_session.device_manufacturer,
                'model': wearable_session.device_model,
                'firmware_version': wearable_session.firmware_version
            },
            'timing': {
                'start_time': wearable_session.start_time.isoformat(),
                'end_time': wearable_session.end_time.isoformat(),
                'duration_seconds': (wearable_session.end_time - wearable_session.start_time).total_seconds(),
                'sampling_rate': wearable_session.sampling_rate
            },
            'data_info': {
                'shape': wearable_session.raw_data.shape if wearable_session.raw_data is not None else None,
                'dtype': str(wearable_session.raw_data.dtype) if wearable_session.raw_data is not None else None
            },
            'processing': {
                'status': session_data['status'],
                'upload_timestamp': session_data['upload_timestamp'],
                'processed_features': session_data.get('processed_features', {})
            }
        }
        
        return jsonify(session_details), 200
        
    except Exception as e:
        logger.error(f"Get session details error: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve session details',
            'message': 'An unexpected error occurred',
            'details': str(e)
        }), 500