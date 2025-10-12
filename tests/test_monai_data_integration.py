#!/usr/bin/env python3
"""
Comprehensive test suite for MONAI data integration.

Tests all components of the MONAI data handling system including:
- Data loading and management
- Transform pipelines
- Quality assurance
- Performance optimization
"""

import pytest
import numpy as np
import torch
import tempfile
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Import modules to test
from src.services.data_management.monai_data_integration import (
    MONAIDataManager, DatasetConfig, MultiModalDataItem,
    NeuroDxDataset, create_optimized_data_pipeline
)
from src.services.data_management.data_quality_assurance import (
    MedicalImageQualityAssessment, DatasetQualityAssessment,
    QualityMetrics, DatasetQualityReport
)
from src.services.data_management.advanced_transforms import (
    NeuroDxTransformPipeline, TransformConfig,
    create_neurodx_training_pipeline, create_neurodx_inference_pipeline,
    create_multi_modal_training_pipeline
)
from src.models.patient import (
    PatientRecord, Demographics, ImagingStudy, WearableSession
)

# MONAI imports for testing
from monai.data import Dataset as MONAIDataset, CacheDataset
from monai.transforms import Compose, LoadImaged, ToTensord
import nibabel as nib


class TestMONAIDataIntegration:
    """Test suite for MONAI data integration components."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def sample_nifti_image(self, temp_dir):
        """Create sample NIfTI image for testing."""
        # Create synthetic 3D image
        image_data = np.random.rand(64, 64, 32).astype(np.float32)
        
        # Create NIfTI image
        nifti_img = nib.Nifti1Image(image_data, affine=np.eye(4))
        
        # Save to temporary file
        image_path = temp_dir / "test_image.nii.gz"
        nib.save(nifti_img, str(image_path))
        
        return image_path
    
    @pytest.fixture
    def sample_patient_records(self, sample_nifti_image):
        """Create sample patient records for testing."""
        patients = []
        
        for i in range(3):
            # Create demographics
            demographics = Demographics(
                age=65 + i * 5,
                gender="M" if i % 2 == 0 else "F",
                weight_kg=70.0 + i * 5,
                height_cm=170.0 + i * 2
            )
            
            # Create imaging study
            study = ImagingStudy(
                study_id=f"STUDY_20241012_120000_{i:03d}",
                modality="MRI",
                acquisition_date=datetime.now(),
                file_path=str(sample_nifti_image)
            )
            
            # Create wearable session
            wearable = WearableSession(
                session_id=f"WEAR_EEG_20241012_120000",
                device_type="EEG",
                start_time=datetime.now(),
                end_time=datetime.now(),
                sampling_rate=256.0,
                processed_features={"alpha_power": 0.5, "beta_power": 0.3}
            )
            
            # Create patient record
            patient = PatientRecord(
                patient_id=f"PAT_20241012_{i:05d}",
                demographics=demographics,
                imaging_studies=[study],
                wearable_data=[wearable]
            )
            
            patients.append(patient)
        
        return patients
    
    @pytest.fixture
    def dataset_config(self, temp_dir):
        """Create dataset configuration for testing."""
        return DatasetConfig(
            name="test_dataset",
            data_root=temp_dir,
            cache_dir=temp_dir / "cache",
            cache_rate=1.0,
            num_workers=1,  # Use single worker for testing
            batch_size=2,
            seed=42
        )
    
    def test_dataset_config_creation(self, dataset_config):
        """Test dataset configuration creation."""
        assert dataset_config.name == "test_dataset"
        assert dataset_config.cache_rate == 1.0
        assert dataset_config.num_workers == 1
        assert dataset_config.batch_size == 2
        assert dataset_config.seed == 42
    
    def test_monai_data_manager_initialization(self, dataset_config):
        """Test MONAI data manager initialization."""
        data_manager = MONAIDataManager(dataset_config)
        
        assert data_manager.config == dataset_config
        assert isinstance(data_manager.readers, dict)
        assert 'nifti' in data_manager.readers
        assert 'dicom' in data_manager.readers
    
    def test_patient_records_conversion(self, dataset_config, sample_patient_records):
        """Test conversion of patient records to MONAI format."""
        data_manager = MONAIDataManager(dataset_config)
        
        # Convert patient records
        data_items = data_manager._convert_patient_records(sample_patient_records)
        
        assert len(data_items) == 3  # 3 patients, 1 study each
        
        # Check data item structure
        item = data_items[0]
        assert 'patient_id' in item
        assert 'study_id' in item
        assert 'image' in item
        assert 'modality' in item
        assert 'demographics' in item
        assert 'metadata' in item
        assert 'wearable' in item
    
    def test_dataset_creation(self, dataset_config, sample_patient_records):
        """Test MONAI dataset creation from patient records."""
        data_manager = MONAIDataManager(dataset_config)
        
        # Create cache dataset
        dataset = data_manager.create_dataset_from_patient_records(
            sample_patient_records,
            dataset_type="cache"
        )
        
        assert isinstance(dataset, CacheDataset)
        assert len(dataset) == 3
    
    def test_data_loader_creation(self, dataset_config, sample_patient_records):
        """Test data loader creation."""
        data_manager = MONAIDataManager(dataset_config)
        
        # Create dataset
        dataset = data_manager.create_dataset_from_patient_records(
            sample_patient_records,
            dataset_type="standard"
        )
        
        # Create data loader
        data_loader = data_manager.create_data_loader(dataset)
        
        assert data_loader.batch_size == dataset_config.batch_size
        assert len(data_loader) > 0
    
    def test_data_splits_creation(self, dataset_config, sample_patient_records):
        """Test train/validation/test splits creation."""
        data_manager = MONAIDataManager(dataset_config)
        
        # Convert patient records
        data_list = data_manager._convert_patient_records(sample_patient_records)
        
        # Create splits
        splits = data_manager.create_train_val_test_splits(data_list)
        
        assert 'train' in splits
        assert 'validation' in splits
        assert 'test' in splits
        
        # Check that all data is accounted for
        total_split_size = len(splits['train']) + len(splits['validation']) + len(splits['test'])
        assert total_split_size == len(data_list)
    
    def test_cross_validation_splits(self, dataset_config, sample_patient_records):
        """Test cross-validation splits creation."""
        data_manager = MONAIDataManager(dataset_config)
        
        # Convert patient records
        data_list = data_manager._convert_patient_records(sample_patient_records)
        
        # Create CV splits
        cv_splits = data_manager.create_cross_validation_splits(data_list)
        
        assert 'train' in cv_splits
        assert 'validation' in cv_splits
        assert len(cv_splits['train']) > 0
        assert len(cv_splits['validation']) > 0
    
    def test_synthetic_dataset_creation(self, dataset_config):
        """Test synthetic dataset creation."""
        data_manager = MONAIDataManager(dataset_config)
        
        # Create synthetic dataset
        dataset = data_manager.create_synthetic_dataset(
            num_samples=10,
            image_shape=(32, 32, 16),
            num_classes=4
        )
        
        assert len(dataset) == 10
        
        # Test data loading
        sample = dataset[0]
        assert 'image' in sample
        assert sample['image'].shape == (1, 32, 32, 16)  # With channel dimension
    
    def test_dataset_summary(self, dataset_config, sample_patient_records):
        """Test dataset summary generation."""
        data_manager = MONAIDataManager(dataset_config)
        
        # Create dataset
        dataset = data_manager.create_dataset_from_patient_records(
            sample_patient_records,
            dataset_type="standard"
        )
        
        # Get summary
        summary = data_manager.get_dataset_summary(dataset)
        
        assert 'num_samples' in summary
        assert summary['num_samples'] == 3


class TestDataQualityAssurance:
    """Test suite for data quality assurance components."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def sample_nifti_image(self, temp_dir):
        """Create sample NIfTI image for testing."""
        # Create synthetic 3D image with known properties
        image_data = np.random.rand(64, 64, 32).astype(np.float32) * 1000
        
        # Add some structure to make quality metrics meaningful
        image_data[20:40, 20:40, 10:20] += 500  # High intensity region
        image_data[:10, :10, :] = 0  # Background region
        
        # Create NIfTI image with proper header
        affine = np.eye(4)
        affine[0, 0] = 1.0  # 1mm spacing
        affine[1, 1] = 1.0
        affine[2, 2] = 1.0
        
        nifti_img = nib.Nifti1Image(image_data, affine=affine)
        
        # Save to temporary file
        image_path = temp_dir / "test_image.nii.gz"
        nib.save(nifti_img, str(image_path))
        
        return image_path
    
    def test_quality_assessment_initialization(self):
        """Test quality assessment initialization."""
        qa = MedicalImageQualityAssessment()
        
        assert qa.quality_thresholds is not None
        assert 'MRI' in qa.quality_thresholds
        assert 'CT' in qa.quality_thresholds
        assert 'Ultrasound' in qa.quality_thresholds
    
    def test_image_quality_assessment(self, sample_nifti_image):
        """Test individual image quality assessment."""
        qa = MedicalImageQualityAssessment()
        
        # Assess image quality
        metrics = qa.assess_image_quality(sample_nifti_image, modality="MRI")
        
        assert isinstance(metrics, QualityMetrics)
        assert metrics.shape == (64, 64, 32)
        assert metrics.spacing == (1.0, 1.0, 1.0)
        assert metrics.data_type == "float32"
        assert 0 <= metrics.quality_score <= 100
        assert metrics.quality_grade in ['A', 'B', 'C', 'D', 'F']
        assert isinstance(metrics.issues, list)
        assert isinstance(metrics.warnings, list)
    
    def test_quality_metrics_calculation(self, sample_nifti_image):
        """Test specific quality metrics calculation."""
        qa = MedicalImageQualityAssessment()
        metrics = qa.assess_image_quality(sample_nifti_image, modality="MRI")
        
        # Check that metrics are reasonable
        assert metrics.snr > 0
        assert metrics.cnr >= 0
        assert 0 <= metrics.sharpness <= 1
        assert 0 <= metrics.uniformity <= 1
        assert 0 <= metrics.motion_artifacts <= 1
        assert 0 <= metrics.noise_level <= 1
        assert 0 <= metrics.bias_field <= 1
        assert 0 <= metrics.metadata_completeness <= 1
    
    def test_dataset_quality_assessment(self, sample_nifti_image):
        """Test dataset-level quality assessment."""
        # Create sample patient records
        patients = []
        for i in range(2):
            demographics = Demographics(age=65, gender="M")
            study = ImagingStudy(
                study_id=f"STUDY_20241012_120000_{i:03d}",
                modality="MRI",
                acquisition_date=datetime.now(),
                file_path=str(sample_nifti_image)
            )
            patient = PatientRecord(
                patient_id=f"PAT_20241012_{i:05d}",
                demographics=demographics,
                imaging_studies=[study]
            )
            patients.append(patient)
        
        # Assess dataset quality
        dataset_qa = DatasetQualityAssessment()
        report = dataset_qa.assess_dataset_quality(patients)
        
        assert isinstance(report, DatasetQualityReport)
        assert report.total_samples == 2
        assert report.processed_samples <= report.total_samples
        assert isinstance(report.quality_distribution, dict)
        assert isinstance(report.common_issues, dict)
        assert isinstance(report.recommendations, list)
        assert report.processing_time > 0


class TestAdvancedTransforms:
    """Test suite for advanced transform pipelines."""
    
    @pytest.fixture
    def transform_config(self):
        """Create transform configuration for testing."""
        return TransformConfig(
            target_spacing=(1.0, 1.0, 1.0),
            target_size=(32, 32, 16),  # Smaller size for faster testing
            augmentation_probability=0.5,
            spatial_aug_prob=0.3,
            intensity_aug_prob=0.2
        )
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for transform testing."""
        return {
            'image': torch.rand(1, 64, 64, 32),  # BCHW format
            'label': torch.randint(0, 4, (1, 64, 64, 32))
        }
    
    def test_transform_pipeline_initialization(self, transform_config):
        """Test transform pipeline initialization."""
        pipeline = NeuroDxTransformPipeline(transform_config)
        
        assert pipeline.config == transform_config
        assert isinstance(pipeline.modality_configs, dict)
        assert 'MRI' in pipeline.modality_configs
        assert 'CT' in pipeline.modality_configs
    
    def test_preprocessing_pipeline_creation(self, transform_config):
        """Test preprocessing pipeline creation."""
        pipeline = NeuroDxTransformPipeline(transform_config)
        
        # Create preprocessing pipeline
        preprocessing = pipeline.create_preprocessing_pipeline(['image'], modality="MRI")
        
        assert isinstance(preprocessing, Compose)
        assert len(preprocessing.transforms) > 0
    
    def test_augmentation_pipeline_creation(self, transform_config):
        """Test augmentation pipeline creation."""
        pipeline = NeuroDxTransformPipeline(transform_config)
        
        # Create augmentation pipeline for training
        augmentation = pipeline.create_augmentation_pipeline(['image'], modality="MRI", training=True)
        
        assert isinstance(augmentation, Compose)
        
        # Create augmentation pipeline for inference (should be empty)
        inference_aug = pipeline.create_augmentation_pipeline(['image'], modality="MRI", training=False)
        assert len(inference_aug.transforms) == 0
    
    def test_multi_modal_pipeline_creation(self, transform_config):
        """Test multi-modal pipeline creation."""
        pipeline = NeuroDxTransformPipeline(transform_config)
        
        modality_keys = {
            'MRI': ['mri_image'],
            'CT': ['ct_image']
        }
        
        # Create multi-modal pipeline
        multimodal = pipeline.create_multi_modal_pipeline(modality_keys, training=True)
        
        assert isinstance(multimodal, Compose)
        assert len(multimodal.transforms) > 0
    
    def test_post_processing_pipeline_creation(self, transform_config):
        """Test post-processing pipeline creation."""
        pipeline = NeuroDxTransformPipeline(transform_config)
        
        # Create segmentation post-processing
        seg_postproc = pipeline.create_post_processing_pipeline(['pred'], task_type="segmentation")
        assert isinstance(seg_postproc, Compose)
        
        # Create classification post-processing
        cls_postproc = pipeline.create_post_processing_pipeline(['pred'], task_type="classification")
        assert isinstance(cls_postproc, Compose)
    
    def test_inference_pipeline_creation(self, transform_config):
        """Test inference pipeline creation."""
        pipeline = NeuroDxTransformPipeline(transform_config)
        
        # Create inference pipeline
        inference = pipeline.create_inference_pipeline(['image'], modality="MRI")
        
        assert isinstance(inference, Compose)
        assert len(inference.transforms) > 0
    
    def test_validation_pipeline_creation(self, transform_config):
        """Test validation pipeline creation."""
        pipeline = NeuroDxTransformPipeline(transform_config)
        
        # Create validation pipeline
        validation = pipeline.create_validation_pipeline(['image'], modality="MRI")
        
        assert isinstance(validation, Compose)
        assert len(validation.transforms) > 0
    
    def test_custom_transform_creation(self, transform_config):
        """Test custom transform creation."""
        pipeline = NeuroDxTransformPipeline(transform_config)
        
        # Define custom function
        def custom_func(data):
            data['image'] = data['image'] * 2
            return data
        
        # Create custom transform
        custom_transform = pipeline.create_custom_transform(custom_func, ['image'])
        
        assert custom_transform is not None
    
    def test_pipeline_summary(self, transform_config):
        """Test pipeline summary generation."""
        pipeline = NeuroDxTransformPipeline(transform_config)
        
        # Create a pipeline
        preprocessing = pipeline.create_preprocessing_pipeline(['image'], modality="MRI")
        
        # Get summary
        summary = pipeline.get_pipeline_summary(preprocessing)
        
        assert 'num_transforms' in summary
        assert 'transform_types' in summary
        assert 'randomizable_transforms' in summary
        assert summary['num_transforms'] > 0
    
    def test_factory_functions(self):
        """Test factory functions for common pipelines."""
        # Test training pipeline
        training_pipeline = create_neurodx_training_pipeline("MRI")
        assert isinstance(training_pipeline, Compose)
        assert len(training_pipeline.transforms) > 0
        
        # Test inference pipeline
        inference_pipeline = create_neurodx_inference_pipeline("MRI")
        assert isinstance(inference_pipeline, Compose)
        assert len(inference_pipeline.transforms) > 0
        
        # Test multi-modal pipeline
        multimodal_pipeline = create_multi_modal_training_pipeline()
        assert isinstance(multimodal_pipeline, Compose)
        assert len(multimodal_pipeline.transforms) > 0


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple components."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def complete_test_setup(self, temp_dir):
        """Create complete test setup with all components."""
        # Create sample NIfTI image
        image_data = np.random.rand(32, 32, 16).astype(np.float32) * 1000
        nifti_img = nib.Nifti1Image(image_data, affine=np.eye(4))
        image_path = temp_dir / "test_image.nii.gz"
        nib.save(nifti_img, str(image_path))
        
        # Create patient records
        patients = []
        for i in range(2):
            demographics = Demographics(age=65, gender="M")
            study = ImagingStudy(
                study_id=f"STUDY_20241012_120000_{i:03d}",
                modality="MRI",
                acquisition_date=datetime.now(),
                file_path=str(image_path)
            )
            patient = PatientRecord(
                patient_id=f"PAT_20241012_{i:05d}",
                demographics=demographics,
                imaging_studies=[study]
            )
            patients.append(patient)
        
        # Create configuration
        config = DatasetConfig(
            name="integration_test",
            data_root=temp_dir,
            cache_dir=temp_dir / "cache",
            num_workers=1,
            batch_size=1,
            seed=42
        )
        
        return {
            'patients': patients,
            'config': config,
            'image_path': image_path
        }
    
    def test_end_to_end_data_pipeline(self, complete_test_setup):
        """Test complete end-to-end data pipeline."""
        patients = complete_test_setup['patients']
        config = complete_test_setup['config']
        
        # Create optimized data pipeline
        data_manager, data_loaders = create_optimized_data_pipeline(config, patients)
        
        assert isinstance(data_manager, MONAIDataManager)
        assert 'train' in data_loaders
        assert 'validation' in data_loaders
        assert 'test' in data_loaders
        
        # Test data loading
        train_loader = data_loaders['train']
        for batch in train_loader:
            assert 'image' in batch
            break  # Just test first batch
    
    def test_quality_assurance_integration(self, complete_test_setup):
        """Test integration of quality assurance with data pipeline."""
        patients = complete_test_setup['patients']
        
        # Assess dataset quality
        dataset_qa = DatasetQualityAssessment()
        report = dataset_qa.assess_dataset_quality(patients)
        
        assert isinstance(report, DatasetQualityReport)
        assert report.total_samples == 2
        assert report.processed_samples > 0
    
    def test_transform_pipeline_integration(self, complete_test_setup):
        """Test integration of transform pipelines with data loading."""
        patients = complete_test_setup['patients']
        config = complete_test_setup['config']
        
        # Create data manager
        data_manager = MONAIDataManager(config)
        
        # Create dataset with custom transforms
        transform_config = TransformConfig(target_size=(16, 16, 8))
        pipeline = NeuroDxTransformPipeline(transform_config)
        
        # Create preprocessing pipeline
        preprocessing = pipeline.create_preprocessing_pipeline(['image'], modality="MRI")
        
        # This would normally be integrated into the dataset creation
        # For testing, we just verify the pipeline works
        assert isinstance(preprocessing, Compose)
        assert len(preprocessing.transforms) > 0


# Performance and stress tests
class TestPerformanceAndStress:
    """Test performance and stress scenarios."""
    
    def test_large_dataset_handling(self):
        """Test handling of large datasets."""
        # Create synthetic large dataset
        config = DatasetConfig(
            name="large_test",
            data_root=Path("."),
            num_workers=1,
            batch_size=4,
            cache_rate=0.1  # Limited caching for large dataset
        )
        
        data_manager = MONAIDataManager(config)
        
        # Create large synthetic dataset
        dataset = data_manager.create_synthetic_dataset(
            num_samples=100,
            image_shape=(32, 32, 16),
            num_classes=4
        )
        
        assert len(dataset) == 100
        
        # Test data loader performance
        data_loader = data_manager.create_data_loader(dataset)
        
        # Load a few batches to test performance
        batch_count = 0
        for batch in data_loader:
            batch_count += 1
            if batch_count >= 5:  # Test first 5 batches
                break
        
        assert batch_count == 5
    
    def test_memory_efficiency(self):
        """Test memory efficiency of data loading."""
        config = DatasetConfig(
            name="memory_test",
            data_root=Path("."),
            num_workers=1,
            batch_size=2,
            cache_rate=0.0  # No caching to test memory efficiency
        )
        
        data_manager = MONAIDataManager(config)
        
        # Create dataset
        dataset = data_manager.create_synthetic_dataset(
            num_samples=20,
            image_shape=(64, 64, 32),
            num_classes=4
        )
        
        # Test that we can iterate through dataset multiple times
        for iteration in range(3):
            data_loader = data_manager.create_data_loader(dataset)
            batch_count = 0
            for batch in data_loader:
                batch_count += 1
                if batch_count >= 5:
                    break
            
            assert batch_count == 5


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])