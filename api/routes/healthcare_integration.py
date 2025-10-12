"""
API routes for healthcare system integrations.

This module provides REST API endpoints for FHIR, HL7, and wearable device integrations.
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from flask import Blueprint, request, jsonify, current_app
from dataclasses import asdict

from src.services.healthcare_integration.healthcare_integration_service import HealthcareIntegrationService
from src.config.healthcare_integration_config import HealthcareIntegrationSettings
from src.models.diagnostics import DiagnosticResult
from src.models.patient import PatientRecord


# Create blueprint
healthcare_bp = Blueprint('healthcare', __name__, url_prefix='/api/healthcare')

# Global service instance
healthcare_service: Optional[HealthcareIntegrationService] = None

def get_healthcare_service() -> HealthcareIntegrationService:
    """Get or create healthcare integration service instance."""
    global healthcare_service
    
    if healthcare_service is None:
        settings = HealthcareIntegrationSettings()
        healthcare_service = HealthcareIntegrationService(settings)
    
    return healthcare_service


@healthcare_bp.route('/status', methods=['GET'])
def get_integration_status():
    """
    Get healthcare integration status.
    
    Returns:
        JSON response with integration status for all systems
    """
    try:
        service = get_healthcare_service()
        status = service.get_integration_status()
        
        return jsonify({
            'success': True,
            'data': {
                'fhir_connected': status.fhir_connected,
                'hl7_connected': status.hl7_connected,
                'wearable_devices_connected': status.wearable_devices_connected,
                'last_sync': status.last_sync.isoformat() if status.last_sync else None,
                'error_messages': status.error_messages
            }
        })
        
    except Exception as e:
        logging.error(f"Failed to get integration status: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@healthcare_bp.route('/connect', methods=['POST'])
def connect_systems():
    """
    Connect to all configured healthcare systems.
    
    Returns:
        JSON response with connection results
    """
    try:
        service = get_healthcare_service()
        
        # Run async connection in event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            status = loop.run_until_complete(service.connect_all_systems())
        finally:
            loop.close()
        
        return jsonify({
            'success': True,
            'data': {
                'fhir_connected': status.fhir_connected,
                'hl7_connected': status.hl7_connected,
                'wearable_devices_connected': status.wearable_devices_connected,
                'error_messages': status.error_messages
            }
        })
        
    except Exception as e:
        logging.error(f"Failed to connect to healthcare systems: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@healthcare_bp.route('/disconnect', methods=['POST'])
def disconnect_systems():
    """
    Disconnect from all healthcare systems.
    
    Returns:
        JSON response confirming disconnection
    """
    try:
        service = get_healthcare_service()
        
        # Run async disconnection in event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(service.disconnect_all_systems())
        finally:
            loop.close()
        
        return jsonify({
            'success': True,
            'message': 'Disconnected from all healthcare systems'
        })
        
    except Exception as e:
        logging.error(f"Failed to disconnect from healthcare systems: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@healthcare_bp.route('/test-connections', methods=['POST'])
def test_connections():
    """
    Test connections to all healthcare systems.
    
    Returns:
        JSON response with connection test results
    """
    try:
        service = get_healthcare_service()
        
        # Run async test in event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            results = loop.run_until_complete(service.test_all_connections())
        finally:
            loop.close()
        
        return jsonify({
            'success': True,
            'data': results
        })
        
    except Exception as e:
        logging.error(f"Failed to test healthcare connections: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@healthcare_bp.route('/fhir/patient/<patient_id>', methods=['GET'])
def get_fhir_patient(patient_id: str):
    """
    Retrieve patient data from FHIR system.
    
    Args:
        patient_id: FHIR patient identifier
        
    Returns:
        JSON response with patient data
    """
    try:
        service = get_healthcare_service()
        
        # Run async operation in event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            patient_data = loop.run_until_complete(service.get_patient_data(patient_id))
        finally:
            loop.close()
        
        if patient_data:
            return jsonify({
                'success': True,
                'data': patient_data
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Patient not found'
            }), 404
        
    except Exception as e:
        logging.error(f"Failed to retrieve FHIR patient {patient_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@healthcare_bp.route('/diagnostic-result', methods=['POST'])
def submit_diagnostic_result():
    """
    Submit diagnostic result to healthcare systems.
    
    Expected JSON payload:
    {
        "patient_id": "string",
        "fhir_patient_id": "string",
        "diagnostic_result": {
            "patient_id": "string",
            "timestamp": "ISO datetime",
            "classification_probabilities": {"condition": probability},
            "confidence_scores": {"metric": score},
            "explainability_maps": {},
            "metrics": {}
        }
    }
    
    Returns:
        JSON response with submission results
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        # Extract required fields
        patient_id = data.get('patient_id')
        fhir_patient_id = data.get('fhir_patient_id')
        diagnostic_data = data.get('diagnostic_result')
        
        if not all([patient_id, fhir_patient_id, diagnostic_data]):
            return jsonify({
                'success': False,
                'error': 'Missing required fields: patient_id, fhir_patient_id, diagnostic_result'
            }), 400
        
        # Create DiagnosticResult object
        diagnostic_result = DiagnosticResult(
            patient_id=diagnostic_data['patient_id'],
            timestamp=datetime.fromisoformat(diagnostic_data['timestamp'].replace('Z', '+00:00')),
            segmentation_mask=diagnostic_data.get('segmentation_mask'),
            classification_probabilities=diagnostic_data.get('classification_probabilities', {}),
            confidence_scores=diagnostic_data.get('confidence_scores', {}),
            explainability_maps=diagnostic_data.get('explainability_maps', {}),
            metrics=diagnostic_data.get('metrics')
        )
        
        service = get_healthcare_service()
        
        # Run async submission in event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            results = loop.run_until_complete(
                service.submit_diagnostic_result(diagnostic_result, fhir_patient_id)
            )
        finally:
            loop.close()
        
        return jsonify({
            'success': True,
            'data': results
        })
        
    except Exception as e:
        logging.error(f"Failed to submit diagnostic result: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@healthcare_bp.route('/wearable/devices', methods=['GET'])
def get_device_statuses():
    """
    Get status of all wearable devices.
    
    Returns:
        JSON response with device statuses
    """
    try:
        service = get_healthcare_service()
        
        # Run async operation in event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            statuses = loop.run_until_complete(service.get_device_statuses())
        finally:
            loop.close()
        
        # Convert DeviceStatus objects to dictionaries
        status_data = {}
        for device_id, status in statuses.items():
            status_data[device_id] = {
                'device_id': status.device_id,
                'device_type': status.device_type,
                'is_connected': status.is_connected,
                'last_sync': status.last_sync.isoformat() if status.last_sync else None,
                'battery_level': status.battery_level,
                'signal_quality': status.signal_quality,
                'error_message': status.error_message,
                'data_points_received': status.data_points_received,
                'connection_uptime': str(status.connection_uptime) if status.connection_uptime else None
            }
        
        return jsonify({
            'success': True,
            'data': status_data
        })
        
    except Exception as e:
        logging.error(f"Failed to get device statuses: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@healthcare_bp.route('/wearable/data', methods=['GET'])
def get_wearable_data():
    """
    Get historical wearable data.
    
    Query parameters:
    - device_id: Device identifier (optional, if not provided gets data from all devices)
    - start_time: Start time (ISO format)
    - end_time: End time (ISO format)
    - data_types: Comma-separated list of data types (e.g., "heart_rate,steps")
    
    Returns:
        JSON response with wearable data
    """
    try:
        # Parse query parameters
        device_id = request.args.get('device_id')
        start_time_str = request.args.get('start_time')
        end_time_str = request.args.get('end_time')
        data_types_str = request.args.get('data_types', 'heart_rate,steps')
        
        if not start_time_str or not end_time_str:
            return jsonify({
                'success': False,
                'error': 'start_time and end_time parameters are required'
            }), 400
        
        # Parse timestamps
        start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
        end_time = datetime.fromisoformat(end_time_str.replace('Z', '+00:00'))
        data_types = [dt.strip() for dt in data_types_str.split(',')]
        
        service = get_healthcare_service()
        
        # Run async operation in event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            if device_id:
                # Get data from specific device
                data_points = loop.run_until_complete(
                    service.get_wearable_data(device_id, start_time, end_time, data_types)
                )
                data = {device_id: data_points}
            else:
                # Get data from all devices
                data = loop.run_until_complete(
                    service.get_all_wearable_data(start_time, end_time, data_types)
                )
        finally:
            loop.close()
        
        # Convert WearableDataPoint objects to dictionaries
        response_data = {}
        for dev_id, points in data.items():
            response_data[dev_id] = []
            for point in points:
                response_data[dev_id].append({
                    'device_id': point.device_id,
                    'device_type': point.device_type,
                    'data_type': point.data_type,
                    'timestamp': point.timestamp.isoformat(),
                    'value': point.value,
                    'unit': point.unit,
                    'quality_score': point.quality_score,
                    'metadata': point.metadata
                })
        
        return jsonify({
            'success': True,
            'data': response_data
        })
        
    except Exception as e:
        logging.error(f"Failed to get wearable data: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@healthcare_bp.route('/wearable/streaming/start', methods=['POST'])
def start_wearable_streaming():
    """
    Start real-time wearable data streaming.
    
    Optional JSON payload:
    {
        "device_ids": ["device1", "device2"]  // Optional, if not provided streams from all devices
    }
    
    Returns:
        JSON response confirming streaming start
    """
    try:
        data = request.get_json() or {}
        device_ids = data.get('device_ids')
        
        service = get_healthcare_service()
        
        # Run async operation in event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            success = loop.run_until_complete(service.start_wearable_streaming(device_ids))
        finally:
            loop.close()
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Wearable data streaming started'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to start wearable streaming'
            }), 500
        
    except Exception as e:
        logging.error(f"Failed to start wearable streaming: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@healthcare_bp.route('/wearable/streaming/stop', methods=['POST'])
def stop_wearable_streaming():
    """
    Stop real-time wearable data streaming.
    
    Optional JSON payload:
    {
        "device_ids": ["device1", "device2"]  // Optional, if not provided stops streaming from all devices
    }
    
    Returns:
        JSON response confirming streaming stop
    """
    try:
        data = request.get_json() or {}
        device_ids = data.get('device_ids')
        
        service = get_healthcare_service()
        
        # Run async operation in event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            success = loop.run_until_complete(service.stop_wearable_streaming(device_ids))
        finally:
            loop.close()
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Wearable data streaming stopped'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to stop wearable streaming'
            }), 500
        
    except Exception as e:
        logging.error(f"Failed to stop wearable streaming: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@healthcare_bp.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@healthcare_bp.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors."""
    return jsonify({
        'success': False,
        'error': 'Method not allowed'
    }), 405


@healthcare_bp.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500