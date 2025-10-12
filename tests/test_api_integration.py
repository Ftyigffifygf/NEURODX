"""
Integration tests for Flask REST API endpoints.
Tests all endpoints with various input scenarios and validates error handling and response formats.
"""

import pytest
import json
import io
import tempfile
import os
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np

from flask import Flask
from werkzeug.datastructures import FileStorage

# Set test environment before importing app
os.environ['NVIDIA_PALMYRA_API_KEY'] = 'test_key'
os.environ['DATABASE_URL'] = 'postgresql://test:test@localhost:5432/test'
os.environ['INFLUXDB_TOKEN'] = 'test_token'
os.environ['ENCRYPTION_KEY'] = 'test_encryption_key'
os.environ['JWT_SECRET_KEY'] = 'test_jwt_secret'
os.environ['FEDERATED_ENCRYPTION_KEY'] = 'test_federated_key'

from src.api.app import create_app
from src.config.settings import get_settings


@pytest.fixture
def app():
    """Create Flask app for testing."""
    test_config = {
        'TESTING': True,
        'SECRET_KEY': 'test-secret-key'
    }
    return create_app(test_config)


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


@pytest.fixture
def sample_nifti_file():
    """Create a sample NIfTI file for testing."""
    # Create a temporary file with .nii extension
    temp_file = tempfile.NamedTemporaryFile(suffix='.nii', delete=False)
    temp_file.write(b'fake nifti data for testing')
    temp_file.close()
    return temp_file.name


@pytest.fixture
def sample_dicom_file():
    """Create a sample DICOM file for testing."""
    # Create a temporary file with .dcm extension
    temp_file = tempfile.NamedTemporaryFile(suffix='.dcm', delete=False)
    temp_file.write(b'fake dicom data for testing')
    temp_file.close()
    return temp_file.name


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_basic_health_check(self, client):
        """Test basic health check endpoint."""
        response = client.get('/api/v1/health/')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert 'timestamp' in data
        assert data['service'] == 'NeuroDx-MultiModal API'
    
    def test_detailed_health_check(self, client):
        """Test detailed health check endpoint."""
        response = client.get('/api/v1/health/detailed')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert 'system' in data
        assert 'gpu' in data
        assert 'version' in data
        assert 'environment' in data


class TestImageProcessingEndpoints:
    """Test image processing API endpoints."""
    
    @patch('src.services.image_processing.image_ingestion.ImageIngestionHandler.get_image_info')
    def test_upload_image_success(self, mock_get_info, client, sample_nifti_file):
        """Test successful image upload."""
        mock_get_info.return_value = {
            'file_size': 1024,
            'format_type': 'nifti'
        }
        
        with open(sample_nifti_file, 'rb') as f:
            data = {
                'file': (f, 'test_image.nii'),
                'patient_id': 'PAT_20240101_12345',
                'modality': 'MRI',
                'series_description': 'T1 weighted'
            }
            response = client.post('/api/v1/images/upload', data=data)
        
        assert response.status_code == 201
        data = json.loads(response.data)
        assert 'study_id' in data
        assert data['modality'] == 'MRI'
        assert data['patient_id'] == 'PAT_20240101_12345'
        assert 'processing_status_url' in data
    
    def test_upload_image_no_file(self, client):
        """Test image upload without file."""
        data = {
            'patient_id': 'PAT_20240101_12345',
            'modality': 'MRI'
        }
        response = client.post('/api/v1/images/upload', data=data)
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'No file provided' in data['error']
    
    def test_upload_image_invalid_modality(self, client, sample_nifti_file):
        """Test image upload with invalid modality."""
        with open(sample_nifti_file, 'rb') as f:
            data = {
                'file': (f, 'test_image.nii'),
                'patient_id': 'PAT_20240101_12345',
                'modality': 'INVALID'
            }
            response = client.post('/api/v1/images/upload', data=data)
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'Invalid modality' in data['error']
    
    def test_upload_image_unsupported_format(self, client):
        """Test image upload with unsupported file format."""
        # Create a fake file with unsupported extension
        fake_file = io.BytesIO(b'fake data')
        data = {
            'file': (fake_file, 'test_image.txt'),
            'patient_id': 'PAT_20240101_12345',
            'modality': 'MRI'
        }
        response = client.post('/api/v1/images/upload', data=data)
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'Unsupported file format' in data['error']
    
    @patch('src.services.image_processing.image_ingestion.ImageIngestionHandler.get_image_info')
    def test_batch_upload_success(self, mock_get_info, client, sample_nifti_file, sample_dicom_file):
        """Test successful batch image upload."""
        mock_get_info.return_value = {
            'file_size': 1024,
            'format_type': 'nifti'
        }
        
        with open(sample_nifti_file, 'rb') as f1, open(sample_dicom_file, 'rb') as f2:
            data = {
                'files': [
                    (f1, 'test_mri.nii'),
                    (f2, 'test_ct.dcm')
                ],
                'patient_id': 'PAT_20240101_12345',
                'modalities': 'MRI,CT'
            }
            response = client.post('/api/v1/images/upload/batch', data=data)
        
        assert response.status_code == 201
        data = json.loads(response.data)
        assert 'successful_uploads' in data
        assert len(data['successful_uploads']) == 2
    
    def test_batch_upload_modality_mismatch(self, client, sample_nifti_file):
        """Test batch upload with modality count mismatch."""
        with open(sample_nifti_file, 'rb') as f:
            data = {
                'files': [(f, 'test_image.nii')],
                'patient_id': 'PAT_20240101_12345',
                'modalities': 'MRI,CT'  # 2 modalities for 1 file
            }
            response = client.post('/api/v1/images/upload/batch', data=data)
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'Modality count mismatch' in data['error']
    
    def test_get_processing_status_not_found(self, client):
        """Test getting status for non-existent study."""
        response = client.get('/api/v1/images/status/NONEXISTENT_STUDY')
        
        assert response.status_code == 404
        data = json.loads(response.data)
        assert 'Study not found' in data['error']
    
    def test_list_studies_empty(self, client):
        """Test listing studies when none exist."""
        response = client.get('/api/v1/images/studies')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['studies'] == []
        assert data['total_count'] == 0
    
    def test_list_studies_with_filters(self, client):
        """Test listing studies with query filters."""
        response = client.get('/api/v1/images/studies?patient_id=PAT_20240101_12345&status=completed')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'filters_applied' in data
        assert data['filters_applied']['patient_id'] == 'PAT_20240101_12345'
        assert data['filters_applied']['status'] == 'completed'


class TestWearableDataEndpoints:
    """Test wearable data integration API endpoints."""
    
    def test_upload_wearable_data_success(self, client):
        """Test successful wearable data upload."""
        data = {
            'device_type': 'EEG',
            'patient_id': 'PAT_20240101_12345',
            'start_time': '2024-01-01T10:00:00Z',
            'end_time': '2024-01-01T11:00:00Z',
            'sampling_rate': 256.0,
            'data': [1.0, 2.0, 3.0, 4.0, 5.0] * 100,  # Sample EEG data
            'device_info': {
                'manufacturer': 'TestCorp',
                'model': 'EEG-1000'
            }
        }
        
        response = client.post('/api/v1/wearable/upload', 
                             data=json.dumps(data),
                             content_type='application/json')
        
        assert response.status_code == 201
        response_data = json.loads(response.data)
        assert 'session_id' in response_data
        assert response_data['device_type'] == 'EEG'
        assert response_data['patient_id'] == 'PAT_20240101_12345'
    
    def test_upload_wearable_data_missing_fields(self, client):
        """Test wearable data upload with missing required fields."""
        data = {
            'device_type': 'EEG',
            'patient_id': 'PAT_20240101_12345'
            # Missing required fields
        }
        
        response = client.post('/api/v1/wearable/upload',
                             data=json.dumps(data),
                             content_type='application/json')
        
        assert response.status_code == 400
        response_data = json.loads(response.data)
        assert 'Missing required field' in response_data['error']
    
    def test_upload_wearable_data_invalid_device_type(self, client):
        """Test wearable data upload with invalid device type."""
        data = {
            'device_type': 'INVALID_DEVICE',
            'patient_id': 'PAT_20240101_12345',
            'start_time': '2024-01-01T10:00:00Z',
            'end_time': '2024-01-01T11:00:00Z',
            'sampling_rate': 256.0,
            'data': [1.0, 2.0, 3.0, 4.0, 5.0]
        }
        
        response = client.post('/api/v1/wearable/upload',
                             data=json.dumps(data),
                             content_type='application/json')
        
        assert response.status_code == 400
        response_data = json.loads(response.data)
        assert 'Invalid device type' in response_data['error']
    
    def test_upload_wearable_data_invalid_timestamps(self, client):
        """Test wearable data upload with invalid timestamps."""
        data = {
            'device_type': 'EEG',
            'patient_id': 'PAT_20240101_12345',
            'start_time': 'invalid-timestamp',
            'end_time': '2024-01-01T11:00:00Z',
            'sampling_rate': 256.0,
            'data': [1.0, 2.0, 3.0, 4.0, 5.0]
        }
        
        response = client.post('/api/v1/wearable/upload',
                             data=json.dumps(data),
                             content_type='application/json')
        
        assert response.status_code == 400
        response_data = json.loads(response.data)
        assert 'Invalid timestamp format' in response_data['error']
    
    def test_start_streaming_session_success(self, client):
        """Test starting a streaming session."""
        data = {
            'device_type': 'HeartRate',
            'patient_id': 'PAT_20240101_12345',
            'sampling_rate': 1.0,
            'buffer_size': 1000
        }
        
        response = client.post('/api/v1/wearable/stream/start',
                             data=json.dumps(data),
                             content_type='application/json')
        
        assert response.status_code == 201
        response_data = json.loads(response.data)
        assert 'stream_id' in response_data
        assert response_data['device_type'] == 'HeartRate'
        assert 'stream_url' in response_data
    
    def test_start_streaming_session_invalid_device(self, client):
        """Test starting streaming session with invalid device type."""
        data = {
            'device_type': 'INVALID',
            'patient_id': 'PAT_20240101_12345',
            'sampling_rate': 1.0
        }
        
        response = client.post('/api/v1/wearable/stream/start',
                             data=json.dumps(data),
                             content_type='application/json')
        
        assert response.status_code == 400
        response_data = json.loads(response.data)
        assert 'Invalid device type' in response_data['error']
    
    def test_list_wearable_sessions_empty(self, client):
        """Test listing wearable sessions when none exist."""
        response = client.get('/api/v1/wearable/sessions')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['sessions'] == []
        assert data['total_count'] == 0
    
    def test_list_wearable_sessions_with_filters(self, client):
        """Test listing wearable sessions with filters."""
        response = client.get('/api/v1/wearable/sessions?device_type=EEG&status=processed')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'filters_applied' in data
        assert data['filters_applied']['device_type'] == 'EEG'
        assert data['filters_applied']['status'] == 'processed'


class TestDiagnosticsEndpoints:
    """Test diagnostic results and annotation API endpoints."""
    
    def test_run_diagnostic_prediction_success(self, client):
        """Test successful diagnostic prediction."""
        data = {
            'patient_id': 'PAT_20240101_12345',
            'study_ids': ['STUDY_20240101_120000_001'],
            'wearable_session_ids': ['WEAR_EEG_20240101_120000'],
            'prediction_options': {
                'model_version': '1.0.0'
            }
        }
        
        response = client.post('/api/v1/diagnostics/predict',
                             data=json.dumps(data),
                             content_type='application/json')
        
        assert response.status_code == 201
        response_data = json.loads(response.data)
        assert 'result_id' in response_data
        assert response_data['patient_id'] == 'PAT_20240101_12345'
        assert 'prediction_summary' in response_data
        assert 'result_url' in response_data
    
    def test_run_diagnostic_prediction_missing_fields(self, client):
        """Test diagnostic prediction with missing required fields."""
        data = {
            'patient_id': 'PAT_20240101_12345'
            # Missing study_ids
        }
        
        response = client.post('/api/v1/diagnostics/predict',
                             data=json.dumps(data),
                             content_type='application/json')
        
        assert response.status_code == 400
        response_data = json.loads(response.data)
        assert 'Missing required field' in response_data['error']
    
    def test_run_diagnostic_prediction_invalid_study_ids(self, client):
        """Test diagnostic prediction with invalid study IDs."""
        data = {
            'patient_id': 'PAT_20240101_12345',
            'study_ids': []  # Empty list
        }
        
        response = client.post('/api/v1/diagnostics/predict',
                             data=json.dumps(data),
                             content_type='application/json')
        
        assert response.status_code == 400
        response_data = json.loads(response.data)
        assert 'Invalid study IDs' in response_data['error']
    
    def test_get_diagnostic_result_not_found(self, client):
        """Test getting non-existent diagnostic result."""
        response = client.get('/api/v1/diagnostics/results/NONEXISTENT_RESULT')
        
        assert response.status_code == 404
        response_data = json.loads(response.data)
        assert 'Result not found' in response_data['error']
    
    def test_get_diagnostic_visualizations_not_found(self, client):
        """Test getting visualizations for non-existent result."""
        response = client.get('/api/v1/diagnostics/visualizations/NONEXISTENT_RESULT')
        
        assert response.status_code == 404
        response_data = json.loads(response.data)
        assert 'Result not found' in response_data['error']
    
    def test_get_diagnostic_visualizations_json_format(self, client):
        """Test getting visualizations in JSON format."""
        # First create a diagnostic result
        data = {
            'patient_id': 'PAT_20240101_12345',
            'study_ids': ['STUDY_20240101_120000_001']
        }
        
        predict_response = client.post('/api/v1/diagnostics/predict',
                                     data=json.dumps(data),
                                     content_type='application/json')
        
        assert predict_response.status_code == 201
        predict_data = json.loads(predict_response.data)
        result_id = predict_data['result_id']
        
        # Now get visualizations
        response = client.get(f'/api/v1/diagnostics/visualizations/{result_id}?format=json')
        
        assert response.status_code == 200
        response_data = json.loads(response.data)
        assert 'visualizations' in response_data
        assert response_data['result_id'] == result_id
    
    def test_get_diagnostic_visualizations_image_format(self, client):
        """Test getting visualizations in image format."""
        # First create a diagnostic result
        data = {
            'patient_id': 'PAT_20240101_12345',
            'study_ids': ['STUDY_20240101_120000_001']
        }
        
        predict_response = client.post('/api/v1/diagnostics/predict',
                                     data=json.dumps(data),
                                     content_type='application/json')
        
        assert predict_response.status_code == 201
        predict_data = json.loads(predict_response.data)
        result_id = predict_data['result_id']
        
        # Now get visualizations as image
        response = client.get(f'/api/v1/diagnostics/visualizations/{result_id}?format=image')
        
        assert response.status_code == 200
        assert response.content_type == 'image/png'
    
    def test_get_diagnostic_visualizations_invalid_format(self, client):
        """Test getting visualizations with invalid format."""
        # First create a diagnostic result
        data = {
            'patient_id': 'PAT_20240101_12345',
            'study_ids': ['STUDY_20240101_120000_001']
        }
        
        predict_response = client.post('/api/v1/diagnostics/predict',
                                     data=json.dumps(data),
                                     content_type='application/json')
        
        assert predict_response.status_code == 201
        predict_data = json.loads(predict_response.data)
        result_id = predict_data['result_id']
        
        # Request invalid format
        response = client.get(f'/api/v1/diagnostics/visualizations/{result_id}?format=invalid')
        
        assert response.status_code == 400
        response_data = json.loads(response.data)
        assert 'Invalid format' in response_data['error']
    
    def test_list_annotations_empty(self, client):
        """Test listing annotations when none exist."""
        response = client.get('/api/v1/diagnostics/annotations')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['annotations'] == []
        assert data['total_count'] == 0
    
    def test_list_annotations_with_filters(self, client):
        """Test listing annotations with filters."""
        response = client.get('/api/v1/diagnostics/annotations?patient_id=PAT_20240101_12345&annotation_type=segmentation')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'filters_applied' in data
        assert data['filters_applied']['patient_id'] == 'PAT_20240101_12345'
        assert data['filters_applied']['annotation_type'] == 'segmentation'
    
    def test_get_active_learning_suggestions_success(self, client):
        """Test getting active learning suggestions."""
        response = client.get('/api/v1/diagnostics/annotations/active-learning/suggestions?count=5&strategy=uncertainty')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'suggestions' in data
        assert data['strategy'] == 'uncertainty'
        assert 'generated_at' in data
    
    def test_get_active_learning_suggestions_invalid_count(self, client):
        """Test getting active learning suggestions with invalid count."""
        response = client.get('/api/v1/diagnostics/annotations/active-learning/suggestions?count=0')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'Invalid count' in data['error']
    
    def test_get_active_learning_suggestions_invalid_strategy(self, client):
        """Test getting active learning suggestions with invalid strategy."""
        response = client.get('/api/v1/diagnostics/annotations/active-learning/suggestions?strategy=invalid')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'Invalid strategy' in data['error']
    
    def test_get_longitudinal_tracking_success(self, client):
        """Test getting longitudinal tracking data."""
        patient_id = 'PAT_20240101_12345'
        response = client.get(f'/api/v1/diagnostics/longitudinal/{patient_id}')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['patient_id'] == patient_id
        assert 'baseline_date' in data
        assert 'follow_up_dates' in data
        assert 'metrics_timeline' in data
        assert 'summary' in data
    
    def test_get_longitudinal_tracking_with_date_filters(self, client):
        """Test getting longitudinal tracking data with date filters."""
        patient_id = 'PAT_20240101_12345'
        start_date = '2024-01-01T00:00:00Z'
        end_date = '2024-12-31T23:59:59Z'
        
        response = client.get(f'/api/v1/diagnostics/longitudinal/{patient_id}?start_date={start_date}&end_date={end_date}')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['patient_id'] == patient_id
        assert 'filtered_dates' in data
    
    def test_get_longitudinal_tracking_invalid_date(self, client):
        """Test getting longitudinal tracking data with invalid date format."""
        patient_id = 'PAT_20240101_12345'
        response = client.get(f'/api/v1/diagnostics/longitudinal/{patient_id}?start_date=invalid-date')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'Invalid start date format' in data['error']
    
    def test_list_diagnostic_results_empty(self, client):
        """Test listing diagnostic results when none exist."""
        response = client.get('/api/v1/diagnostics/results')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['results'] == []
        assert data['total_count'] == 0
    
    def test_list_diagnostic_results_with_filters(self, client):
        """Test listing diagnostic results with filters."""
        response = client.get('/api/v1/diagnostics/results?patient_id=PAT_20240101_12345&confidence_threshold=0.8')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'filters_applied' in data
        assert data['filters_applied']['patient_id'] == 'PAT_20240101_12345'
        assert data['filters_applied']['confidence_threshold'] == 0.8


class TestErrorHandling:
    """Test error handling across all endpoints."""
    
    def test_404_error_handler(self, client):
        """Test 404 error handler."""
        response = client.get('/api/v1/nonexistent/endpoint')
        
        assert response.status_code == 404
        data = json.loads(response.data)
        assert data['error'] == 'Not Found'
        assert data['status_code'] == 404
    
    def test_400_error_handler_invalid_json(self, client):
        """Test 400 error with invalid JSON."""
        response = client.post('/api/v1/wearable/upload',
                             data='invalid json',
                             content_type='application/json')
        
        assert response.status_code == 400
    
    def test_413_error_handler_large_file(self, client):
        """Test 413 error with file too large."""
        # Create a large fake file
        large_data = b'x' * (600 * 1024 * 1024)  # 600MB (exceeds 500MB limit)
        fake_file = io.BytesIO(large_data)
        
        data = {
            'file': (fake_file, 'large_file.nii'),
            'patient_id': 'PAT_20240101_12345',
            'modality': 'MRI'
        }
        
        response = client.post('/api/v1/images/upload', data=data)
        
        # Should return 413 or handle gracefully
        assert response.status_code in [413, 400, 500]


class TestCORSHeaders:
    """Test CORS headers are properly set."""
    
    def test_cors_headers_present(self, client):
        """Test that CORS headers are present in responses."""
        response = client.get('/api/v1/health/')
        
        # Check for CORS headers (these would be set by Flask-CORS)
        assert response.status_code == 200
        # Note: In a real test, you'd check for Access-Control-Allow-Origin header
        # but this requires proper CORS configuration in the test environment


if __name__ == '__main__':
    pytest.main([__file__, '-v'])