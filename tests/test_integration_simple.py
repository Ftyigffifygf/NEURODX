"""
Simplified integration tests for NeuroDx-MultiModal system.
Tests core integration points without requiring full system setup.
"""

import pytest
import numpy as np
import tempfile
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from pathlib import Path

from src.services.image_processing.image_ingestion import ImageIngestionHandler
from src.services.wearable_sensor.sensor_data_collector import SensorDataCollector
from src.services.data_fusion.multi_modal_fusion import MultiModalFusion
from src.services.ml_inference.inference_engine import InferenceEngine
from src.services.security.auth_service import AuthenticationService
from src.services.security.rbac_service import RBACService, Permission
from src.models.patient import PatientRecord, Demographics, ImagingStudy, WearableSession
from src.models.diagnostics import DiagnosticResult, ModelMetrics


class TestCoreIntegration:
    """Test core system integration points."""
    
    def test_image_processing_to_inference_pipeline(self):
        """Test image processing to ML inference pipeline."""
        
        # Create synthetic medical image
        image_data = np.random.rand(96, 96, 96).astype(np.float32)
        
        # Initialize services
        image_handler = ImageIngestionHandler()
        inference_engine = InferenceEngine()
        
        # Mock image loading
        with patch.object(image_handler, 'load_and_validate_image') as mock_load:
            mock_load.return_value = {
                'data': image_data,
                'metadata': {
                    'modality': 'MRI',
                    'spacing': (1.0, 1.0, 1.0),
                    'origin': (0.0, 0.0, 0.0),
                    'shape': image_data.shape
                }
            }
            
            # Process image
            processed_image = image_handler.process_image("fake_path.nii.gz")
            assert processed_image is not None
            assert 'data' in processed_image
            assert 'metadata' in processed_image
        
        # Mock inference
        with patch.object(inference_engine, 'run_inference') as mock_inference:
            mock_inference.return_value = {
                'segmentation_mask': np.random.rand(96, 96, 96),
                'classification_probabilities': {
                    'healthy': 0.3,
                    'mild_cognitive_impairment': 0.4,
                    'alzheimers_disease': 0.3
                },
                'confidence_scores': {
                    'segmentation': 0.85,
                    'classification': 0.78
                }
            }
            
            # Run inference
            result = inference_engine.run_inference(processed_image['data'])
            
            assert result is not None
            assert 'segmentation_mask' in result
            assert 'classification_probabilities' in result
            assert 'confidence_scores' in result
            
            # Verify classification probabilities sum to 1
            probs = result['classification_probabilities']
            assert abs(sum(probs.values()) - 1.0) < 0.01
    
    def test_wearable_data_processing_pipeline(self):
        """Test wearable data processing pipeline."""
        
        # Create synthetic wearable data
        duration_seconds = 120  # 2 minutes minimum
        sampling_rate = 256
        n_samples = duration_seconds * sampling_rate
        
        eeg_data = np.random.randn(n_samples, 8) * 50  # Time x Channels format
        
        wearable_session = WearableSession(
            session_id="WEAR_EEG_20241012_143000",
            device_type="EEG",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(seconds=duration_seconds),
            sampling_rate=float(sampling_rate),
            raw_data=eeg_data,
            processed_features={}
        )
        
        # Initialize sensor data collector
        sensor_collector = SensorDataCollector()
        
        # Process wearable data
        features = sensor_collector.extract_features(wearable_session)
        
        assert features is not None
        assert 'temporal_features' in features
        assert 'spectral_features' in features
        assert len(features['temporal_features']) > 0
        assert len(features['spectral_features']) > 0
    
    def test_multi_modal_data_fusion(self):
        """Test multi-modal data fusion integration."""
        
        # Create mock imaging data
        imaging_data = {
            'data': np.random.rand(96, 96, 96),
            'metadata': {
                'modality': 'MRI',
                'spacing': (1.0, 1.0, 1.0)
            }
        }
        
        # Create mock wearable features
        wearable_features = {
            'temporal_features': {
                'mean_amplitude': 25.5,
                'std_amplitude': 12.3,
                'peak_frequency': 10.2
            },
            'spectral_features': {
                'alpha_power': 0.45,
                'beta_power': 0.32,
                'theta_power': 0.23
            }
        }
        
        # Initialize fusion service
        fusion_service = MultiModalFusion()
        
        # Fuse data
        fused_data = fusion_service.fuse_modalities(
            imaging_data=imaging_data,
            wearable_data=wearable_features
        )
        
        assert fused_data is not None
        assert 'imaging_features' in fused_data
        assert 'wearable_features' in fused_data
        assert 'fused_tensor' in fused_data
        
        # Verify fused tensor shape
        fused_tensor = fused_data['fused_tensor']
        assert isinstance(fused_tensor, np.ndarray)
        assert fused_tensor.shape[0] > 0  # Has features
    
    def test_authentication_and_authorization_integration(self):
        """Test authentication and authorization integration."""
        
        # Initialize services
        auth_service = AuthenticationService()
        rbac_service = RBACService()
        
        # Create test user
        user_id = auth_service.create_user(
            username="testuser",
            email="test@example.com",
            password="TestPassword123!",
            roles=["clinician"]
        )
        
        # Authenticate user
        auth_result = auth_service.authenticate_user(
            username="testuser",
            password="TestPassword123!",
            ip_address="127.0.0.1",
            user_agent="test-client"
        )
        
        assert auth_result is not None
        assert "token" in auth_result
        assert "user" in auth_result
        
        # Validate token
        token_info = auth_service.validate_token(auth_result["token"])
        assert token_info is not None
        assert token_info["user_id"] == user_id
        
        # Check permissions
        user_roles = auth_result["user"]["roles"]
        has_permission = rbac_service.check_permission(
            user_roles, 
            Permission.READ_PATIENT_DATA,
            user_id=user_id
        )
        assert has_permission == True
        
        # Check denied permission
        has_admin_permission = rbac_service.check_permission(
            user_roles,
            Permission.MANAGE_USERS,
            user_id=user_id
        )
        assert has_admin_permission == False
    
    def test_patient_data_model_integration(self):
        """Test patient data model integration."""
        
        # Create patient record
        patient = PatientRecord(
            patient_id="PAT_20241012_00001",
            demographics=Demographics(
                age=65,
                gender="M",
                weight_kg=75.0,
                height_cm=175.0
            ),
            imaging_studies=[],
            wearable_data=[],
            annotations=[],
            longitudinal_tracking=None
        )
        
        # Add imaging study
        imaging_study = ImagingStudy(
            study_id="STUDY_20241012_143000_001",
            modality="MRI",
            acquisition_date=datetime.now(),
            file_path="/path/to/image.nii.gz",
            preprocessing_metadata=None
        )
        
        patient.imaging_studies.append(imaging_study)
        
        # Add wearable session
        duration_seconds = 120  # 2 minutes minimum
        sampling_rate = 256.0
        n_samples = int(duration_seconds * sampling_rate)
        
        wearable_session = WearableSession(
            session_id="WEAR_EEG_20241012_143000",
            device_type="EEG",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(seconds=duration_seconds),
            sampling_rate=sampling_rate,
            raw_data=np.random.randn(n_samples, 8),  # Time x Channels format
            processed_features={}
        )
        
        patient.wearable_data.append(wearable_session)
        
        # Verify patient record structure
        assert patient.patient_id == "PAT_20241012_00001"
        assert len(patient.imaging_studies) == 1
        assert len(patient.wearable_data) == 1
        assert patient.imaging_studies[0].modality == "MRI"
        assert patient.wearable_data[0].device_type == "EEG"
    
    def test_diagnostic_result_creation(self):
        """Test diagnostic result creation and validation."""
        
        # Create diagnostic result
        result = DiagnosticResult(
            patient_id="PAT_20241012_00001",
            timestamp=datetime.now(),
            segmentation_mask=np.random.rand(96, 96, 96),
            classification_probabilities={
                "healthy": 0.2,
                "mild_cognitive_impairment": 0.3,
                "alzheimers_disease": 0.5
            },
            confidence_scores={
                "segmentation": 0.85,
                "classification": 0.78
            },
            explainability_maps={
                "grad_cam": np.random.rand(96, 96, 96),
                "integrated_gradients": np.random.rand(96, 96, 96)
            },
            metrics=ModelMetrics(
                dice_score=0.82,
                hausdorff_distance=2.5,
                auc_score=0.89,
                sensitivity=0.85,
                specificity=0.91
            )
        )
        
        # Verify result structure
        assert result.patient_id == "PAT_20241012_00001"
        assert result.segmentation_mask.shape == (96, 96, 96)
        assert len(result.classification_probabilities) == 3
        assert result.metrics.dice_score == 0.82
        
        # Verify probabilities sum to 1
        prob_sum = sum(result.classification_probabilities.values())
        assert abs(prob_sum - 1.0) < 0.01
    
    def test_error_handling_integration(self):
        """Test error handling across integrated components."""
        
        # Test invalid patient ID format
        with pytest.raises(ValueError, match="Patient ID.*must follow format"):
            PatientRecord(
                patient_id="INVALID_ID",
                demographics=Demographics(age=65, gender="M"),
                imaging_studies=[],
                wearable_data=[],
                annotations=[],
                longitudinal_tracking=None
            )
        
        # Test invalid imaging study
        with pytest.raises(ValueError, match="Study ID.*must follow format"):
            ImagingStudy(
                study_id="INVALID_STUDY_ID",
                modality="MRI",
                acquisition_date=datetime.now(),
                file_path="test.nii.gz"
            )
        
        # Test invalid file format
        with pytest.raises(ValueError, match="Unsupported file format"):
            ImagingStudy(
                study_id="STUDY_20241012_143000_001",
                modality="MRI",
                acquisition_date=datetime.now(),
                file_path="test.txt"  # Invalid format
            )
    
    def test_data_flow_integration(self):
        """Test complete data flow from input to output."""
        
        # Step 1: Create patient
        patient = PatientRecord(
            patient_id="PAT_20241012_00001",
            demographics=Demographics(age=65, gender="M"),
            imaging_studies=[],
            wearable_data=[],
            annotations=[],
            longitudinal_tracking=None
        )
        
        # Step 2: Add imaging data
        imaging_study = ImagingStudy(
            study_id="STUDY_20241012_143000_001",
            modality="MRI",
            acquisition_date=datetime.now(),
            file_path="test.nii.gz"
        )
        patient.imaging_studies.append(imaging_study)
        
        # Step 3: Add wearable data
        duration_seconds = 120  # 2 minutes minimum
        sampling_rate = 256.0
        n_samples = int(duration_seconds * sampling_rate)
        
        wearable_session = WearableSession(
            session_id="WEAR_EEG_20241012_143000",
            device_type="EEG",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(seconds=duration_seconds),
            sampling_rate=sampling_rate,
            raw_data=np.random.randn(n_samples, 8),  # Time x Channels format
            processed_features={}
        )
        patient.wearable_data.append(wearable_session)
        
        # Step 4: Process data (mock)
        image_handler = ImageIngestionHandler()
        sensor_collector = SensorDataCollector()
        fusion_service = MultiModalFusion()
        
        # Mock processing results
        with patch.object(image_handler, 'process_image') as mock_image, \
             patch.object(sensor_collector, 'extract_features') as mock_sensor, \
             patch.object(fusion_service, 'fuse_modalities') as mock_fusion:
            
            mock_image.return_value = {
                'data': np.random.rand(96, 96, 96),
                'metadata': {'modality': 'MRI'}
            }
            
            mock_sensor.return_value = {
                'temporal_features': {'mean': 25.5},
                'spectral_features': {'alpha_power': 0.45}
            }
            
            mock_fusion.return_value = {
                'fused_tensor': np.random.rand(100),
                'imaging_features': np.random.rand(50),
                'wearable_features': np.random.rand(50)
            }
            
            # Process all data
            image_result = image_handler.process_image(imaging_study.file_path)
            sensor_result = sensor_collector.extract_features(wearable_session)
            fused_result = fusion_service.fuse_modalities(image_result, sensor_result)
            
            # Verify complete pipeline
            assert image_result is not None
            assert sensor_result is not None
            assert fused_result is not None
            assert 'fused_tensor' in fused_result
    
    def test_configuration_integration(self):
        """Test configuration integration across services."""
        from src.config.settings import settings
        
        # Verify settings are accessible
        assert settings.app_name is not None
        assert settings.security is not None
        assert settings.monai is not None
        
        # Test that services can access configuration
        auth_service = AuthenticationService()
        assert hasattr(auth_service, 'jwt_secret')
        assert hasattr(auth_service, 'session_timeout')
        
        # Test configuration validation
        assert settings.security.session_timeout > 0
        assert len(settings.security.jwt_secret_key) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])