"""
End-to-end integration tests for NeuroDx-MultiModal system.
Tests complete diagnostic workflows from upload to results.
"""

import pytest
import asyncio
import tempfile
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import io

from src.api.app import create_app
from src.services.image_processing.image_ingestion import ImageIngestionHandler
from src.services.wearable_sensor.sensor_data_collector import SensorDataCollector
from src.services.data_fusion.multi_modal_fusion import MultiModalFusion
from src.services.ml_inference.inference_engine import InferenceEngine
from src.services.security.auth_service import AuthenticationService
from src.services.security.rbac_service import RBACService, Permission
from src.services.monai_label.monai_label_server import MONAILabelServer
from src.models.patient import PatientRecord, Demographics, ImagingStudy, WearableSession
from src.models.diagnostics import DiagnosticResult, ModelMetrics


class TestEndToEndDiagnosticWorkflow:
    """Test complete diagnostic workflows from data upload to results."""
    
    @pytest.fixture
    def app(self):
        """Create Flask test app."""
        app = create_app(testing=True)
        app.config['TESTING'] = True
        return app
    
    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return app.test_client()
    
    @pytest.fixture
    def auth_headers(self):
        """Create authentication headers for API requests."""
        auth_service = AuthenticationService()
        
        # Create test user
        user_id = auth_service.create_user(
            username="testclinician",
            email="clinician@test.com",
            password="TestPassword123!",
            roles=["clinician"]
        )
        
        # Authenticate user
        auth_result = auth_service.authenticate_user(
            username="testclinician",
            password="TestPassword123!",
            ip_address="127.0.0.1",
            user_agent="test-client"
        )
        
        return {"Authorization": f"Bearer {auth_result['token']}"}
    
    @pytest.fixture
    def sample_medical_image(self):
        """Create sample medical image data."""
        # Create synthetic 3D medical image (96x96x96)
        image_data = np.random.rand(96, 96, 96).astype(np.float32)
        
        # Save as temporary NIfTI file
        temp_file = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False)
        
        # Mock NIfTI file structure
        return {
            'file_path': temp_file.name,
            'data': image_data,
            'modality': 'MRI',
            'study_id': 'STUDY_20241012_143000_001'
        }
    
    @pytest.fixture
    def sample_wearable_data(self):
        """Create sample wearable sensor data."""
        # Generate synthetic EEG data
        duration_seconds = 300  # 5 minutes
        sampling_rate = 256  # Hz
        n_samples = duration_seconds * sampling_rate
        
        eeg_data = np.random.randn(8, n_samples) * 50  # 8 channels
        
        # Generate heart rate data
        hr_samples = duration_seconds // 5  # Every 5 seconds
        heart_rate_data = 70 + np.random.randn(hr_samples) * 10
        
        return {
            'eeg': {
                'data': eeg_data,
                'sampling_rate': sampling_rate,
                'channels': ['Fp1', 'Fp2', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2']
            },
            'heart_rate': {
                'data': heart_rate_data,
                'sampling_rate': 0.2,  # Every 5 seconds
                'timestamps': [datetime.now() + timedelta(seconds=i*5) for i in range(hr_samples)]
            }
        }
    
    def test_complete_diagnostic_workflow(self, client, auth_headers, sample_medical_image, sample_wearable_data):
        """Test complete diagnostic workflow from upload to results."""
        
        # Step 1: Create patient record
        patient_data = {
            'patient_id': 'PAT_20241012_00001',
            'demographics': {
                'age': 65,
                'gender': 'M',
                'weight_kg': 75.0,
                'height_cm': 175.0
            }
        }
        
        response = client.post('/api/patients', 
                             json=patient_data, 
                             headers=auth_headers)
        assert response.status_code == 201
        
        # Step 2: Upload medical image
        with open(sample_medical_image['file_path'], 'rb') as f:
            image_upload_data = {
                'patient_id': patient_data['patient_id'],
                'modality': sample_medical_image['modality'],
                'study_id': sample_medical_image['study_id']
            }
            
            response = client.post('/api/images/upload',
                                 data=image_upload_data,
                                 files={'image': f},
                                 headers=auth_headers)
        
        assert response.status_code == 202  # Accepted for processing
        upload_result = response.get_json()
        processing_id = upload_result['processing_id']
        
        # Step 3: Upload wearable data
        wearable_upload_data = {
            'patient_id': patient_data['patient_id'],
            'session_id': 'WEAR_EEG_20241012_143000',
            'device_type': 'EEG',
            'data': sample_wearable_data['eeg']['data'].tolist(),
            'sampling_rate': sample_wearable_data['eeg']['sampling_rate'],
            'channels': sample_wearable_data['eeg']['channels']
        }
        
        response = client.post('/api/wearable/upload',
                             json=wearable_upload_data,
                             headers=auth_headers)
        assert response.status_code == 202
        
        # Step 4: Check processing status
        response = client.get(f'/api/processing/{processing_id}/status',
                            headers=auth_headers)
        assert response.status_code == 200
        
        # Step 5: Trigger diagnostic inference (mock completion)
        with patch('src.services.ml_inference.inference_engine.InferenceEngine.run_inference') as mock_inference:
            # Mock successful inference result
            mock_inference.return_value = {
                'segmentation_mask': np.random.rand(96, 96, 96),
                'classification_probabilities': {
                    'healthy': 0.2,
                    'mild_cognitive_impairment': 0.3,
                    'alzheimers_disease': 0.5
                },
                'confidence_scores': {
                    'segmentation': 0.85,
                    'classification': 0.78
                }
            }
            
            response = client.post(f'/api/diagnostics/run',
                                 json={'patient_id': patient_data['patient_id']},
                                 headers=auth_headers)
            assert response.status_code == 200
            
            diagnostic_result = response.get_json()
            assert 'result_id' in diagnostic_result
            assert 'classification_probabilities' in diagnostic_result
            assert diagnostic_result['classification_probabilities']['alzheimers_disease'] == 0.5
    
    def test_multi_modal_data_fusion_workflow(self, sample_medical_image, sample_wearable_data):
        """Test multi-modal data fusion process."""
        
        # Initialize services
        image_handler = ImageIngestionHandler()
        sensor_collector = SensorDataCollector()
        fusion_service = MultiModalFusion()
        
        # Process medical image
        with patch.object(image_handler, 'load_and_validate_image') as mock_load:
            mock_load.return_value = {
                'data': sample_medical_image['data'],
                'metadata': {
                    'modality': sample_medical_image['modality'],
                    'spacing': (1.0, 1.0, 1.0),
                    'origin': (0.0, 0.0, 0.0)
                }
            }
            
            image_result = image_handler.process_image(sample_medical_image['file_path'])
            assert image_result is not None
        
        # Process wearable data
        wearable_session = WearableSession(
            session_id="WEAR_EEG_20241012_143000",
            device_type="EEG",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(minutes=5),
            sampling_rate=256.0,
            raw_data=sample_wearable_data['eeg']['data'],
            processed_features={}
        )
        
        sensor_features = sensor_collector.extract_features(wearable_session)
        assert sensor_features is not None
        
        # Fuse multi-modal data
        fused_data = fusion_service.fuse_modalities(
            imaging_data=image_result,
            wearable_data=sensor_features
        )
        
        assert fused_data is not None
        assert 'imaging_features' in fused_data
        assert 'wearable_features' in fused_data
        assert 'fused_tensor' in fused_data
    
    def test_active_learning_annotation_workflow(self, client, auth_headers):
        """Test MONAI Label active learning workflow."""
        
        # Step 1: Initialize MONAI Label session
        response = client.post('/api/monai-label/init',
                             json={'app_name': 'neurodx'},
                             headers=auth_headers)
        assert response.status_code == 200
        
        # Step 2: Get next sample for annotation
        response = client.get('/api/monai-label/next-sample',
                            headers=auth_headers)
        assert response.status_code == 200
        
        next_sample = response.get_json()
        assert 'sample_id' in next_sample
        assert 'image_path' in next_sample
        
        # Step 3: Submit annotation
        annotation_data = {
            'sample_id': next_sample['sample_id'],
            'annotations': {
                'segmentation': {
                    'hippocampus': [[10, 10, 10], [20, 20, 20]],  # Mock coordinates
                    'ventricles': [[30, 30, 30], [40, 40, 40]]
                },
                'classification': 'mild_cognitive_impairment'
            },
            'quality_score': 0.9
        }
        
        response = client.post('/api/monai-label/submit-annotation',
                             json=annotation_data,
                             headers=auth_headers)
        assert response.status_code == 201
        
        # Step 4: Trigger model update
        response = client.post('/api/monai-label/update-model',
                             headers=auth_headers)
        assert response.status_code == 202  # Accepted for background processing
    
    def test_federated_learning_coordination(self):
        """Test federated learning coordination across nodes."""
        from src.services.federated_learning.federated_server import FederatedLearningServer
        from src.services.federated_learning.federated_client import FederatedLearningClient
        
        # Initialize federated learning server
        server = FederatedLearningServer(
            server_id="central_server",
            min_clients=2,
            rounds=3
        )
        
        # Initialize federated clients (simulating different hospitals)
        client_a = FederatedLearningClient(
            node_id="hospital_a",
            server_address="localhost:8080"
        )
        
        client_b = FederatedLearningClient(
            node_id="hospital_b", 
            server_address="localhost:8080"
        )
        
        # Mock model parameters
        mock_params = {
            'layer1.weight': np.random.randn(64, 3, 3, 3),
            'layer1.bias': np.random.randn(64),
            'layer2.weight': np.random.randn(128, 64, 3, 3, 3),
            'layer2.bias': np.random.randn(128)
        }
        
        # Test federated training round
        with patch.object(client_a, 'train_local_model') as mock_train_a, \
             patch.object(client_b, 'train_local_model') as mock_train_b:
            
            mock_train_a.return_value = mock_params
            mock_train_b.return_value = mock_params
            
            # Start federated training
            server.start_training_round()
            
            # Clients participate in training
            client_a.participate_in_round()
            client_b.participate_in_round()
            
            # Server aggregates results
            aggregated_params = server.aggregate_model_updates()
            
            assert aggregated_params is not None
            assert 'layer1.weight' in aggregated_params
            assert 'layer2.weight' in aggregated_params
    
    def test_longitudinal_patient_tracking(self, client, auth_headers):
        """Test longitudinal patient progression tracking."""
        
        patient_id = 'PAT_20241012_00001'
        
        # Create baseline diagnostic result
        baseline_data = {
            'patient_id': patient_id,
            'timestamp': '2024-01-01T10:00:00Z',
            'classification_probabilities': {
                'healthy': 0.7,
                'mild_cognitive_impairment': 0.2,
                'alzheimers_disease': 0.1
            },
            'cognitive_scores': {
                'mmse': 28,
                'moca': 26
            }
        }
        
        response = client.post('/api/diagnostics/results',
                             json=baseline_data,
                             headers=auth_headers)
        assert response.status_code == 201
        
        # Create follow-up diagnostic result (6 months later)
        followup_data = {
            'patient_id': patient_id,
            'timestamp': '2024-07-01T10:00:00Z',
            'classification_probabilities': {
                'healthy': 0.4,
                'mild_cognitive_impairment': 0.4,
                'alzheimers_disease': 0.2
            },
            'cognitive_scores': {
                'mmse': 25,
                'moca': 23
            }
        }
        
        response = client.post('/api/diagnostics/results',
                             json=followup_data,
                             headers=auth_headers)
        assert response.status_code == 201
        
        # Get longitudinal analysis
        response = client.get(f'/api/patients/{patient_id}/longitudinal',
                            headers=auth_headers)
        assert response.status_code == 200
        
        longitudinal_data = response.get_json()
        assert 'progression_analysis' in longitudinal_data
        assert 'trend_indicators' in longitudinal_data
        assert len(longitudinal_data['diagnostic_history']) == 2
        
        # Verify progression detection
        progression = longitudinal_data['progression_analysis']
        assert progression['cognitive_decline_detected'] == True
        assert progression['risk_increase'] > 0
    
    def test_healthcare_system_integration(self, client, auth_headers):
        """Test integration with healthcare systems (FHIR/HL7)."""
        
        # Test FHIR patient data exchange
        fhir_patient_data = {
            'resourceType': 'Patient',
            'id': 'PAT_20241012_00001',
            'identifier': [
                {
                    'system': 'http://hospital.example.com/patient-ids',
                    'value': 'MRN123456'
                }
            ],
            'name': [
                {
                    'family': 'Doe',
                    'given': ['John']
                }
            ],
            'gender': 'male',
            'birthDate': '1959-01-01'
        }
        
        response = client.post('/api/fhir/patients',
                             json=fhir_patient_data,
                             headers=auth_headers)
        assert response.status_code == 201
        
        # Test diagnostic report generation in FHIR format
        diagnostic_request = {
            'patient_id': 'PAT_20241012_00001',
            'format': 'fhir'
        }
        
        response = client.post('/api/fhir/diagnostic-reports',
                             json=diagnostic_request,
                             headers=auth_headers)
        assert response.status_code == 200
        
        fhir_report = response.get_json()
        assert fhir_report['resourceType'] == 'DiagnosticReport'
        assert 'subject' in fhir_report
        assert 'result' in fhir_report
    
    def test_system_performance_under_load(self, client, auth_headers):
        """Test system performance under concurrent load."""
        import concurrent.futures
        import time
        
        def make_diagnostic_request():
            """Make a diagnostic API request."""
            request_data = {
                'patient_id': f'PAT_20241012_{np.random.randint(10000, 99999):05d}',
                'modality': 'MRI'
            }
            
            response = client.post('/api/diagnostics/run',
                                 json=request_data,
                                 headers=auth_headers)
            return response.status_code
        
        # Test concurrent requests
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_diagnostic_request) for _ in range(50)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        end_time = time.time()
        
        # Verify performance
        successful_requests = sum(1 for status in results if status in [200, 202])
        assert successful_requests >= 45  # At least 90% success rate
        
        total_time = end_time - start_time
        assert total_time < 30  # Should complete within 30 seconds
        
        throughput = len(results) / total_time
        assert throughput > 1.0  # At least 1 request per second
    
    def test_error_handling_and_recovery(self, client, auth_headers):
        """Test system error handling and recovery mechanisms."""
        
        # Test invalid patient data
        invalid_patient_data = {
            'patient_id': 'INVALID_ID',  # Wrong format
            'demographics': {
                'age': -5,  # Invalid age
                'gender': 'X'  # Invalid gender
            }
        }
        
        response = client.post('/api/patients',
                             json=invalid_patient_data,
                             headers=auth_headers)
        assert response.status_code == 400
        
        error_response = response.get_json()
        assert 'error' in error_response
        assert 'validation_errors' in error_response
        
        # Test missing authentication
        response = client.get('/api/patients/PAT_20241012_00001')
        assert response.status_code == 401
        
        # Test insufficient permissions
        # Create user with limited permissions
        auth_service = AuthenticationService()
        limited_user_id = auth_service.create_user(
            username="limiteduser",
            email="limited@test.com",
            password="TestPassword123!",
            roles=["viewer"]  # Read-only role
        )
        
        limited_auth = auth_service.authenticate_user(
            username="limiteduser",
            password="TestPassword123!",
            ip_address="127.0.0.1",
            user_agent="test-client"
        )
        
        limited_headers = {"Authorization": f"Bearer {limited_auth['token']}"}
        
        # Try to create patient with limited permissions
        response = client.post('/api/patients',
                             json={'patient_id': 'PAT_20241012_00002'},
                             headers=limited_headers)
        assert response.status_code == 403  # Forbidden
    
    def test_data_privacy_and_anonymization(self, client, auth_headers):
        """Test data privacy and anonymization features."""
        
        # Create patient with sensitive data
        patient_data = {
            'patient_id': 'PAT_20241012_00001',
            'demographics': {
                'age': 65,
                'gender': 'M'
            },
            'sensitive_info': {
                'ssn': '123-45-6789',
                'phone': '555-123-4567',
                'address': '123 Main St, Anytown, USA'
            }
        }
        
        response = client.post('/api/patients',
                             json=patient_data,
                             headers=auth_headers)
        assert response.status_code == 201
        
        # Request anonymized data for research
        response = client.get('/api/patients/PAT_20241012_00001/anonymized',
                            headers=auth_headers)
        assert response.status_code == 200
        
        anonymized_data = response.get_json()
        
        # Verify PII is removed/anonymized
        assert 'ssn' not in str(anonymized_data)
        assert '123-45-6789' not in str(anonymized_data)
        assert '555-123-4567' not in str(anonymized_data)
        assert 'anonymized_id' in anonymized_data
        assert anonymized_data['patient_id'] != 'PAT_20241012_00001'
    
    def test_audit_trail_completeness(self, client, auth_headers):
        """Test that all operations are properly audited."""
        from src.services.security.audit_logger import AuditLogger
        
        audit_logger = AuditLogger()
        
        # Perform various operations
        patient_data = {'patient_id': 'PAT_20241012_00001'}
        client.post('/api/patients', json=patient_data, headers=auth_headers)
        client.get('/api/patients/PAT_20241012_00001', headers=auth_headers)
        
        # Check audit logs
        logs = audit_logger.get_audit_logs(patient_id='PAT_20241012_00001')
        
        assert len(logs) >= 2  # At least create and read operations
        
        # Verify log structure
        for log in logs:
            assert 'timestamp' in log
            assert 'user_id' in log
            assert 'action' in log
            assert 'resource' in log
            assert 'patient_id' in log


@pytest.mark.asyncio
async def test_real_time_processing_pipeline():
    """Test real-time data processing pipeline."""
    
    # Mock real-time data stream
    async def mock_data_stream():
        """Simulate real-time wearable data stream."""
        for i in range(10):
            yield {
                'timestamp': datetime.now(),
                'heart_rate': 70 + np.random.randn() * 5,
                'eeg_sample': np.random.randn(8) * 50
            }
            await asyncio.sleep(0.1)  # 10 Hz data
    
    # Process real-time stream
    processed_samples = []
    
    async for data_sample in mock_data_stream():
        # Simulate real-time processing
        processed_sample = {
            'timestamp': data_sample['timestamp'],
            'heart_rate_normalized': (data_sample['heart_rate'] - 70) / 10,
            'eeg_power': np.mean(np.abs(data_sample['eeg_sample']))
        }
        processed_samples.append(processed_sample)
    
    assert len(processed_samples) == 10
    assert all('heart_rate_normalized' in sample for sample in processed_samples)
    assert all('eeg_power' in sample for sample in processed_samples)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])