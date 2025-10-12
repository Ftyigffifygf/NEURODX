"""
Unit tests for multi-modal data fusion service.
"""

import pytest
import numpy as np
import torch
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.services.data_fusion.multi_modal_fusion import (
    MultiModalFusion, FusionConfig, FusionStrategy, ModalityWeight, FusionResult
)
from src.services.data_fusion.feature_alignment import (
    FeatureAlignment, AlignmentConfig, FeatureRepresentation, AlignmentResult,
    MissingDataStrategy, NormalizationMethod
)
from src.models.patient import (
    PatientRecord, Demographics, ImagingStudy, WearableSession
)


class TestMultiModalFusion:
    """Test cases for MultiModalFusion class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = FusionConfig(
            fusion_strategy=FusionStrategy.EARLY_FUSION,
            temporal_window_seconds=300.0
        )
        self.fusion_service = MultiModalFusion(self.config)
        
        # Create test patient record
        self.patient_record = self._create_test_patient_record()
    
    def _create_test_patient_record(self) -> PatientRecord:
        """Create a test patient record with imaging and wearable data."""
        demographics = Demographics(age=65, gender="M")
        
        # Create imaging studies
        imaging_studies = [
            ImagingStudy(
                study_id="STUDY_20240101_120000_001",
                modality="MRI",
                acquisition_date=datetime(2024, 1, 1, 12, 0, 0),
                file_path="/test/mri.nii"
            ),
            ImagingStudy(
                study_id="STUDY_20240101_120500_002",
                modality="CT",
                acquisition_date=datetime(2024, 1, 1, 12, 5, 0),
                file_path="/test/ct.dcm"
            )
        ]
        
        # Create wearable sessions
        wearable_sessions = [
            WearableSession(
                session_id="WEAR_EEG_20240101_115500",
                device_type="EEG",
                start_time=datetime(2024, 1, 1, 11, 55, 0),
                end_time=datetime(2024, 1, 1, 12, 10, 0),
                sampling_rate=256.0,
                raw_data=np.random.randn(2304, 8)  # 9 minutes * 256 Hz, 8 channels
            ),
            WearableSession(
                session_id="WEAR_HeartRate_20240101_115500",
                device_type="HeartRate",
                start_time=datetime(2024, 1, 1, 11, 55, 0),
                end_time=datetime(2024, 1, 1, 12, 10, 0),
                sampling_rate=1.0,
                raw_data=np.random.uniform(60, 100, 900)  # 15 minutes
            )
        ]
        
        return PatientRecord(
            patient_id="PAT_20240101_00001",
            demographics=demographics,
            imaging_studies=imaging_studies,
            wearable_data=wearable_sessions
        )
    
    def test_fusion_config_initialization(self):
        """Test fusion configuration initialization."""
        config = FusionConfig()
        assert config.fusion_strategy == FusionStrategy.EARLY_FUSION
        assert config.modality_weights == ModalityWeight.EQUAL
        assert config.temporal_window_seconds == 300.0
    
    def test_multi_modal_fusion_initialization(self):
        """Test MultiModalFusion initialization."""
        fusion = MultiModalFusion()
        assert fusion.config is not None
        assert fusion.temporal_synchronizer is not None
        assert fusion.imaging_tensor_size == (96, 96, 96)
        assert fusion.wearable_feature_size == 512
    
    @patch('src.services.data_fusion.multi_modal_fusion.TemporalSynchronizer')
    def test_fuse_patient_data_success(self, mock_synchronizer):
        """Test successful patient data fusion."""
        # Mock temporal synchronizer
        mock_aligned_data = Mock()
        mock_aligned_data.data_streams = {
            "EEG": np.random.randn(100, 8),
            "HeartRate": np.random.randn(100)
        }
        
        mock_temporal_features = Mock()
        mock_temporal_features.feature_vector = np.random.randn(256)
        mock_temporal_features.feature_names = [f"feature_{i}" for i in range(256)]
        
        mock_synchronizer.return_value.align_timestamps.return_value = mock_aligned_data
        mock_synchronizer.return_value.extract_temporal_features.return_value = mock_temporal_features
        
        # Test fusion
        result = self.fusion_service.fuse_patient_data(self.patient_record)
        
        assert isinstance(result, FusionResult)
        assert isinstance(result.fused_tensor, torch.Tensor)
        assert result.fusion_confidence >= 0.0
        assert result.fusion_confidence <= 1.0
        assert len(result.modality_contributions) > 0    

    def test_fuse_patient_data_missing_imaging(self):
        """Test patient data fusion with missing imaging data."""
        # Create patient record with only wearable data
        patient_record = PatientRecord(
            patient_id="PAT_20240101_00002",
            demographics=Demographics(age=45, gender="F"),
            imaging_studies=[],
            wearable_data=self.patient_record.wearable_data
        )
        
        with patch.object(self.fusion_service.temporal_synchronizer, 'align_timestamps') as mock_align:
            with patch.object(self.fusion_service.temporal_synchronizer, 'extract_temporal_features') as mock_extract:
                mock_aligned_data = Mock()
                mock_temporal_features = Mock()
                mock_temporal_features.feature_vector = np.random.randn(256)
                mock_temporal_features.feature_names = [f"feature_{i}" for i in range(256)]
                
                mock_align.return_value = mock_aligned_data
                mock_extract.return_value = mock_temporal_features
                
                result = self.fusion_service.fuse_patient_data(patient_record)
                
                assert isinstance(result, FusionResult)
                assert "imaging" in result.missing_modalities or len(result.missing_modalities) > 0
    
    def test_fuse_study_data_success(self):
        """Test successful study data fusion."""
        imaging_data = {
            "MRI": torch.randn(1, 96, 96, 96),
            "CT": torch.randn(1, 96, 96, 96)
        }
        
        wearable_features = {
            "EEG": np.random.randn(256),
            "HeartRate": np.random.randn(64)
        }
        
        result = self.fusion_service.fuse_study_data(imaging_data, wearable_features)
        
        assert isinstance(result, FusionResult)
        assert isinstance(result.fused_tensor, torch.Tensor)
        assert result.fusion_confidence > 0.0
        assert "imaging" in result.modality_contributions
        assert "wearable" in result.modality_contributions
    
    def test_early_fusion_strategy(self):
        """Test early fusion strategy."""
        imaging_tensor = torch.randn(1, 1, 96, 96, 96)
        wearable_tensor = torch.randn(512)
        
        from src.services.data_fusion.multi_modal_fusion import SpatialTemporalAlignment
        alignment_info = SpatialTemporalAlignment(
            temporal_offset_seconds=0.0,
            spatial_transform_matrix=None,
            alignment_confidence=1.0,
            reference_timestamp=datetime.now(),
            reference_coordinate_system="RAS"
        )
        
        fused = self.fusion_service._early_fusion(imaging_tensor, wearable_tensor, alignment_info)
        
        assert isinstance(fused, torch.Tensor)
        assert fused.shape[0] == 1  # Batch dimension
        assert fused.shape[1] == 2  # Two channels (imaging + wearable)
    
    def test_expand_wearable_features(self):
        """Test wearable feature expansion to spatial dimensions."""
        wearable_tensor = torch.randn(256)
        target_size = (32, 32, 32)
        
        expanded = self.fusion_service._expand_wearable_features(wearable_tensor, target_size)
        
        assert expanded.shape == target_size
        assert isinstance(expanded, torch.Tensor)
    
    def test_missing_modalities_identification(self):
        """Test identification of missing modalities."""
        imaging_studies = [
            ImagingStudy(
                study_id="test", modality="MRI", 
                acquisition_date=datetime.now(), file_path="/test"
            )
        ]
        wearable_sessions = [
            WearableSession(
                session_id="test", device_type="EEG",
                start_time=datetime.now(), end_time=datetime.now(),
                sampling_rate=256.0, raw_data=np.array([1, 2, 3])
            )
        ]
        
        missing = self.fusion_service._identify_missing_modalities(
            imaging_studies, wearable_sessions
        )
        
        assert "CT" in missing
        assert "Ultrasound" in missing
        assert "HeartRate" in missing
        assert "Sleep" in missing
        assert "Gait" in missing


class TestFeatureAlignment:
    """Test cases for FeatureAlignment class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = AlignmentConfig(
            temporal_resolution_seconds=1.0,
            missing_data_strategy=MissingDataStrategy.INTERPOLATION,
            normalization_method=NormalizationMethod.STANDARD
        )
        self.alignment_service = FeatureAlignment(self.config)
    
    def test_alignment_config_initialization(self):
        """Test alignment configuration initialization."""
        config = AlignmentConfig()
        assert config.temporal_resolution_seconds == 1.0
        assert config.missing_data_strategy == MissingDataStrategy.INTERPOLATION
        assert config.normalization_method == NormalizationMethod.STANDARD
    
    def test_feature_alignment_initialization(self):
        """Test FeatureAlignment initialization."""
        alignment = FeatureAlignment()
        assert alignment.config is not None
        assert alignment.temporal_aligner is not None
        assert alignment.spatial_aligner is not None
    
    def test_align_multi_modal_features_success(self):
        """Test successful multi-modal feature alignment."""
        # Create test feature representations
        modality_features = {
            "imaging": FeatureRepresentation(
                features=torch.randn(64, 64, 64),
                timestamps=np.linspace(0, 100, 100),
                metadata={"modality": "MRI"}
            ),
            "wearable": FeatureRepresentation(
                features=torch.randn(100, 8),
                timestamps=np.linspace(0, 100, 100),
                metadata={"modality": "EEG"}
            )
        }
        
        result = self.alignment_service.align_multi_modal_features(modality_features)
        
        assert isinstance(result, AlignmentResult)
        assert len(result.aligned_features) == 2
        assert "imaging" in result.aligned_features
        assert "wearable" in result.aligned_features
        assert all(0.0 <= quality <= 1.0 for quality in result.alignment_quality.values())
    
    def test_create_unified_tensor(self):
        """Test unified tensor creation from aligned features."""
        aligned_features = {
            "modality1": torch.randn(32, 32, 32),
            "modality2": torch.randn(32, 32, 32)
        }
        
        unified = self.alignment_service.create_unified_tensor(aligned_features)
        
        assert isinstance(unified, torch.Tensor)
        assert unified.shape[0] == 2  # Two modalities concatenated
        assert unified.shape[1:] == (32, 32, 32)
    
    def test_handle_partial_data(self):
        """Test handling of partial data with missing modalities."""
        available_modalities = {
            "MRI": FeatureRepresentation(
                features=torch.randn(64, 64, 64),
                metadata={"modality": "MRI"}
            )
        }
        expected_modalities = ["MRI", "CT", "EEG"]
        
        complete_modalities = self.alignment_service.handle_partial_data(
            available_modalities, expected_modalities
        )
        
        assert len(complete_modalities) == 3
        assert "MRI" in complete_modalities
        assert "CT" in complete_modalities
        assert "EEG" in complete_modalities
        assert complete_modalities["CT"].metadata["imputed"] is True
        assert complete_modalities["EEG"].metadata["imputed"] is True
    
    def test_temporal_alignment(self):
        """Test temporal feature alignment."""
        from src.services.data_fusion.feature_alignment import TemporalAligner
        
        aligner = TemporalAligner()
        
        source_features = FeatureRepresentation(
            features=torch.randn(50, 8),
            timestamps=np.linspace(0, 49, 50)
        )
        
        target_features = FeatureRepresentation(
            features=torch.randn(100, 8),
            timestamps=np.linspace(0, 99, 100)
        )
        
        aligned, quality = aligner.align_features(source_features, target_features, self.config)
        
        assert isinstance(aligned, FeatureRepresentation)
        assert aligned.features.shape[0] == 100  # Aligned to target length
        assert 0.0 <= quality <= 1.0
    
    def test_missing_data_detection(self):
        """Test detection of missing data regions."""
        # Create features with NaN regions
        features = torch.randn(100)
        features[20:30] = float('nan')
        features[50:55] = float('nan')
        
        missing_regions = self.alignment_service._detect_missing_data_regions(features)
        
        assert len(missing_regions) == 2
        assert (20, 30) in missing_regions
        assert (50, 55) in missing_regions
    
    def test_normalization_methods(self):
        """Test different normalization methods."""
        features = torch.randn(100, 10) * 10 + 5  # Non-normalized features
        
        # Test standard normalization
        self.config.normalization_method = NormalizationMethod.STANDARD
        normalized, params = self.alignment_service._normalize_features(features, "test")
        
        assert torch.abs(torch.mean(normalized)) < 0.1  # Should be close to 0
        assert torch.abs(torch.std(normalized) - 1.0) < 0.1  # Should be close to 1
        assert "method" in params
    
    def test_cross_modal_normalization(self):
        """Test cross-modal normalization."""
        aligned_features = {
            "modality1": torch.randn(100) * 10,  # Large scale
            "modality2": torch.randn(100) * 0.1  # Small scale
        }
        
        normalized = self.alignment_service._cross_modal_normalization(aligned_features)
        
        # Check that scales are more similar after normalization
        scale1 = torch.std(normalized["modality1"])
        scale2 = torch.std(normalized["modality2"])
        ratio = max(scale1, scale2) / min(scale1, scale2)
        
        assert ratio < 5.0  # Should be more balanced than before


class TestInputTensorBuilder:
    """Test cases for InputTensorBuilder class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.builder = InputTensorBuilder(target_shape=(32, 32, 32))
    
    def test_input_tensor_builder_initialization(self):
        """Test InputTensorBuilder initialization."""
        builder = InputTensorBuilder()
        assert builder.target_shape == (96, 96, 96)
        
        custom_builder = InputTensorBuilder((64, 64, 64))
        assert custom_builder.target_shape == (64, 64, 64)
    
    def test_build_monai_tensor_success(self):
        """Test successful MONAI tensor building."""
        aligned_features = {
            "imaging": torch.randn(32, 32, 32),
            "wearable": torch.randn(32, 32, 32)
        }
        
        tensor = self.builder.build_monai_tensor(aligned_features)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (2, 32, 32, 32)  # 2 channels
    
    def test_build_batch_tensor(self):
        """Test batch tensor building."""
        batch_features = [
            {"modality1": torch.randn(32, 32, 32)},
            {"modality1": torch.randn(32, 32, 32)},
            {"modality1": torch.randn(32, 32, 32)}
        ]
        
        batch_tensor = self.builder.build_batch_tensor(batch_features)
        
        assert isinstance(batch_tensor, torch.Tensor)
        assert batch_tensor.shape[0] == 3  # Batch size
        assert batch_tensor.shape[1] == 1  # One modality
        assert batch_tensor.shape[2:] == (32, 32, 32)
    
    def test_expand_1d_to_3d(self):
        """Test expansion of 1D features to 3D volume."""
        features_1d = torch.randn(256)
        expanded = self.builder._expand_1d_to_3d(features_1d)
        
        assert expanded.shape == (32, 32, 32)
        assert isinstance(expanded, torch.Tensor)
    
    def test_expand_2d_to_3d(self):
        """Test expansion of 2D features to 3D volume."""
        features_2d = torch.randn(32, 32)
        expanded = self.builder._expand_2d_to_3d(features_2d)
        
        assert expanded.shape == (32, 32, 32)
        assert isinstance(expanded, torch.Tensor)
    
    def test_resize_3d_volume(self):
        """Test resizing of 3D volume."""
        features_3d = torch.randn(64, 64, 64)
        resized = self.builder._resize_3d_volume(features_3d)
        
        assert resized.shape == (32, 32, 32)
        assert isinstance(resized, torch.Tensor)
    
    def test_standardize_different_dimensions(self):
        """Test standardization of features with different dimensions."""
        # Test 1D features
        features_1d = torch.randn(100)
        standardized_1d = self.builder._standardize_to_target_shape(features_1d, "test")
        assert standardized_1d.shape == (32, 32, 32)
        
        # Test 2D features
        features_2d = torch.randn(16, 16)
        standardized_2d = self.builder._standardize_to_target_shape(features_2d, "test")
        assert standardized_2d.shape == (32, 32, 32)
        
        # Test 3D features
        features_3d = torch.randn(64, 64, 64)
        standardized_3d = self.builder._standardize_to_target_shape(features_3d, "test")
        assert standardized_3d.shape == (32, 32, 32)


if __name__ == "__main__":
    pytest.main([__file__])