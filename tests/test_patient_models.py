"""
Unit tests for patient record and imaging study data models.

This module tests data model creation and validation with various input scenarios
and validates error handling for invalid data formats.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from src.models.patient import (
    Demographics, PreprocessingMetadata, ImagingStudy, WearableSession,
    LongitudinalData, Annotation, PatientRecord,
    validate_medical_imaging_metadata, create_preprocessing_metadata
)


class TestDemographics:
    """Test cases for Demographics data model."""
    
    def test_valid_demographics(self):
        """Test creating valid demographics."""
        demographics = Demographics(
            age=45,
            gender="F",
            weight_kg=65.5,
            height_cm=165.0,
            medical_history=["hypertension", "diabetes"]
        )
        
        assert demographics.age == 45
        assert demographics.gender == "F"
        assert demographics.weight_kg == 65.5
        assert demographics.height_cm == 165.0
        assert len(demographics.medical_history) == 2
    
    def test_demographics_with_minimal_data(self):
        """Test demographics with only required fields."""
        demographics = Demographics(age=30, gender="M")
        
        assert demographics.age == 30
        assert demographics.gender == "M"
        assert demographics.weight_kg is None
        assert demographics.height_cm is None
        assert demographics.medical_history == []


class TestPreprocessingMetadata:
    """Test cases for PreprocessingMetadata data model."""
    
    def test_valid_preprocessing_metadata(self):
        """Test creating valid preprocessing metadata."""
        metadata = PreprocessingMetadata(
            original_spacing=(2.0, 2.0, 3.0),
            resampled_spacing=(1.0, 1.0, 1.0),
            normalization_method="z_score",
            registration_applied=True,
            augmentation_applied=False
        )
        
        assert metadata.original_spacing == (2.0, 2.0, 3.0)
        assert metadata.resampled_spacing == (1.0, 1.0, 1.0)
        assert metadata.normalization_method == "z_score"
        assert metadata.registration_applied is True
        assert metadata.augmentation_applied is False
        assert isinstance(metadata.preprocessing_timestamp, datetime)
    
    def test_default_preprocessing_metadata(self):
        """Test preprocessing metadata with default values."""
        metadata = PreprocessingMetadata(original_spacing=(1.5, 1.5, 2.0))
        
        assert metadata.original_spacing == (1.5, 1.5, 2.0)
        assert metadata.resampled_spacing == (1.0, 1.0, 1.0)
        assert metadata.normalization_method == "z_score"
        assert metadata.registration_applied is False
        assert metadata.augmentation_applied is False


class TestImagingStudy:
    """Test cases for ImagingStudy data model."""
    
    def test_valid_mri_study(self):
        """Test creating valid MRI study."""
        study = ImagingStudy(
            study_id="STUDY_20241010_143000_001",
            modality="MRI",
            acquisition_date=datetime.now(),
            file_path="/data/patient001/mri_t1.nii.gz",
            slice_thickness=1.0,
            pixel_spacing=(0.5, 0.5)
        )
        
        assert study.study_id == "STUDY_20241010_143000_001"
        assert study.modality == "MRI"
        assert study.file_path == "/data/patient001/mri_t1.nii.gz"
        assert study.slice_thickness == 1.0
        assert study.pixel_spacing == (0.5, 0.5)
    
    def test_valid_ct_study(self):
        """Test creating valid CT study."""
        study = ImagingStudy(
            study_id="STUDY_20241010_143000_002",
            modality="CT",
            acquisition_date=datetime.now(),
            file_path="/data/patient001/ct_head.dcm",
            slice_thickness=2.5
        )
        
        assert study.modality == "CT"
        assert study.slice_thickness == 2.5
    
    def test_valid_ultrasound_study(self):
        """Test creating valid ultrasound study."""
        study = ImagingStudy(
            study_id="STUDY_20241010_143000_003",
            modality="Ultrasound",
            acquisition_date=datetime.now(),
            file_path="/data/patient001/ultrasound.nii"
        )
        
        assert study.modality == "Ultrasound"
        assert study.slice_thickness is None  # Ultrasound shouldn't have slice thickness
    
    def test_invalid_study_id_format(self):
        """Test invalid study ID format raises error."""
        with pytest.raises(ValueError, match="Study ID.*must follow format"):
            ImagingStudy(
                study_id="INVALID_ID",
                modality="MRI",
                acquisition_date=datetime.now(),
                file_path="/data/test.nii.gz"
            )
    
    def test_empty_study_id(self):
        """Test empty study ID raises error."""
        with pytest.raises(ValueError, match="Study ID must be a non-empty string"):
            ImagingStudy(
                study_id="",
                modality="MRI",
                acquisition_date=datetime.now(),
                file_path="/data/test.nii.gz"
            )
    
    def test_invalid_file_format(self):
        """Test invalid file format raises error."""
        with pytest.raises(ValueError, match="Unsupported file format"):
            ImagingStudy(
                study_id="STUDY_20241010_143000_001",
                modality="MRI",
                acquisition_date=datetime.now(),
                file_path="/data/test.jpg"
            )
    
    def test_empty_file_path(self):
        """Test empty file path raises error."""
        with pytest.raises(ValueError, match="File path cannot be empty"):
            ImagingStudy(
                study_id="STUDY_20241010_143000_001",
                modality="MRI",
                acquisition_date=datetime.now(),
                file_path=""
            )
    
    def test_invalid_mri_slice_thickness(self):
        """Test invalid MRI slice thickness raises error."""
        with pytest.raises(ValueError, match="MRI slice thickness must be positive"):
            ImagingStudy(
                study_id="STUDY_20241010_143000_001",
                modality="MRI",
                acquisition_date=datetime.now(),
                file_path="/data/test.nii.gz",
                slice_thickness=-1.0
            )
    
    def test_invalid_ct_slice_thickness(self):
        """Test invalid CT slice thickness raises error."""
        with pytest.raises(ValueError, match="CT slice thickness should be between"):
            ImagingStudy(
                study_id="STUDY_20241010_143000_001",
                modality="CT",
                acquisition_date=datetime.now(),
                file_path="/data/test.dcm",
                slice_thickness=15.0  # Too thick
            )
    
    def test_ultrasound_with_slice_thickness(self):
        """Test ultrasound with slice thickness raises error."""
        with pytest.raises(ValueError, match="Ultrasound studies should not have slice thickness"):
            ImagingStudy(
                study_id="STUDY_20241010_143000_001",
                modality="Ultrasound",
                acquisition_date=datetime.now(),
                file_path="/data/test.nii",
                slice_thickness=2.0
            )


class TestWearableSession:
    """Test cases for WearableSession data model."""
    
    def test_valid_eeg_session(self):
        """Test creating valid EEG session."""
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=1)
        raw_data = np.random.randn(3600 * 256)  # 1 hour at 256Hz
        
        session = WearableSession(
            session_id="WEAR_EEG_20241010_143000",
            device_type="EEG",
            start_time=start_time,
            end_time=end_time,
            sampling_rate=256.0,
            raw_data=raw_data,
            device_manufacturer="NeuroSky",
            device_model="MindWave"
        )
        
        assert session.device_type == "EEG"
        assert session.sampling_rate == 256.0
        assert session.raw_data.shape[0] == 3600 * 256
    
    def test_valid_heart_rate_session(self):
        """Test creating valid heart rate session."""
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=24)
        raw_data = np.random.randn(int(24 * 3600 * 1))  # 24 hours at 1Hz
        
        session = WearableSession(
            session_id="WEAR_HeartRate_20241010_143000",
            device_type="HeartRate",
            start_time=start_time,
            end_time=end_time,
            sampling_rate=1.0,
            raw_data=raw_data
        )
        
        assert session.device_type == "HeartRate"
        assert session.sampling_rate == 1.0
    
    def test_invalid_session_id_format(self):
        """Test invalid session ID format raises error."""
        with pytest.raises(ValueError, match="Session ID.*must follow format"):
            WearableSession(
                session_id="INVALID_SESSION",
                device_type="EEG",
                start_time=datetime.now(),
                end_time=datetime.now() + timedelta(hours=1),
                sampling_rate=256.0
            )
    
    def test_invalid_time_range(self):
        """Test invalid time range raises error."""
        start_time = datetime.now()
        end_time = start_time - timedelta(hours=1)  # End before start
        
        with pytest.raises(ValueError, match="End time must be after start time"):
            WearableSession(
                session_id="WEAR_EEG_20241010_143000",
                device_type="EEG",
                start_time=start_time,
                end_time=end_time,
                sampling_rate=256.0
            )
    
    def test_too_short_session(self):
        """Test session duration too short raises error."""
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=30)  # Less than 1 minute
        
        with pytest.raises(ValueError, match="Session duration must be at least 1 minute"):
            WearableSession(
                session_id="WEAR_EEG_20241010_143000",
                device_type="EEG",
                start_time=start_time,
                end_time=end_time,
                sampling_rate=256.0
            )
    
    def test_too_long_session(self):
        """Test session duration too long raises error."""
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=25)  # Longer than 24 hours for EEG
        
        with pytest.raises(ValueError, match="EEG session duration exceeds maximum"):
            WearableSession(
                session_id="WEAR_EEG_20241010_143000",
                device_type="EEG",
                start_time=start_time,
                end_time=end_time,
                sampling_rate=256.0
            )
    
    def test_invalid_sampling_rate(self):
        """Test invalid sampling rate raises error."""
        with pytest.raises(ValueError, match="Sampling rate must be positive"):
            WearableSession(
                session_id="WEAR_EEG_20241010_143000",
                device_type="EEG",
                start_time=datetime.now(),
                end_time=datetime.now() + timedelta(hours=1),
                sampling_rate=-1.0
            )
    
    def test_sampling_rate_out_of_range(self):
        """Test sampling rate out of valid range raises error."""
        with pytest.raises(ValueError, match="EEG sampling rate must be between"):
            WearableSession(
                session_id="WEAR_EEG_20241010_143000",
                device_type="EEG",
                start_time=datetime.now(),
                end_time=datetime.now() + timedelta(hours=1),
                sampling_rate=50.0  # Too low for EEG
            )
    
    def test_raw_data_length_mismatch(self):
        """Test raw data length mismatch raises error."""
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=1)
        raw_data = np.random.randn(1000)  # Wrong length for 1 hour at 256Hz
        
        with pytest.raises(ValueError, match="Raw data length.*doesn't match expected samples"):
            WearableSession(
                session_id="WEAR_EEG_20241010_143000",
                device_type="EEG",
                start_time=start_time,
                end_time=end_time,
                sampling_rate=256.0,
                raw_data=raw_data
            )


class TestPatientRecord:
    """Test cases for PatientRecord data model."""
    
    def test_valid_patient_record(self):
        """Test creating valid patient record."""
        demographics = Demographics(age=45, gender="F")
        patient = PatientRecord(
            patient_id="PAT_20241010_00001",
            demographics=demographics
        )
        
        assert patient.patient_id == "PAT_20241010_00001"
        assert patient.demographics.age == 45
        assert len(patient.imaging_studies) == 0
        assert len(patient.wearable_data) == 0
        assert isinstance(patient.created_at, datetime)
    
    def test_invalid_patient_id_format(self):
        """Test invalid patient ID format raises error."""
        demographics = Demographics(age=45, gender="F")
        
        with pytest.raises(ValueError, match="Patient ID.*must follow format"):
            PatientRecord(
                patient_id="INVALID_ID",
                demographics=demographics
            )
    
    def test_invalid_age(self):
        """Test invalid age raises error."""
        demographics = Demographics(age=200, gender="F")  # Age too high
        
        with pytest.raises(ValueError, match="Age must be between 0 and 150"):
            PatientRecord(
                patient_id="PAT_20241010_00001",
                demographics=demographics
            )
    
    def test_invalid_weight(self):
        """Test invalid weight raises error."""
        demographics = Demographics(age=45, gender="F", weight_kg=-10)
        
        with pytest.raises(ValueError, match="Weight must be between 0 and 1000 kg"):
            PatientRecord(
                patient_id="PAT_20241010_00001",
                demographics=demographics
            )
    
    def test_invalid_height(self):
        """Test invalid height raises error."""
        demographics = Demographics(age=45, gender="F", height_cm=400)
        
        with pytest.raises(ValueError, match="Height must be between 0 and 300 cm"):
            PatientRecord(
                patient_id="PAT_20241010_00001",
                demographics=demographics
            )
    
    def test_add_imaging_study(self):
        """Test adding imaging study to patient record."""
        demographics = Demographics(age=45, gender="F")
        patient = PatientRecord(
            patient_id="PAT_20241010_00001",
            demographics=demographics
        )
        
        study = ImagingStudy(
            study_id="STUDY_20241010_143000_001",
            modality="MRI",
            acquisition_date=datetime.now(),
            file_path="/data/test.nii.gz"
        )
        
        patient.add_imaging_study(study)
        
        assert len(patient.imaging_studies) == 1
        assert patient.imaging_studies[0].study_id == "STUDY_20241010_143000_001"
    
    def test_add_duplicate_study_id(self):
        """Test adding duplicate study ID raises error."""
        demographics = Demographics(age=45, gender="F")
        patient = PatientRecord(
            patient_id="PAT_20241010_00001",
            demographics=demographics
        )
        
        study1 = ImagingStudy(
            study_id="STUDY_20241010_143000_001",
            modality="MRI",
            acquisition_date=datetime.now(),
            file_path="/data/test1.nii.gz"
        )
        
        study2 = ImagingStudy(
            study_id="STUDY_20241010_143000_001",  # Same ID
            modality="CT",
            acquisition_date=datetime.now(),
            file_path="/data/test2.dcm"
        )
        
        patient.add_imaging_study(study1)
        
        with pytest.raises(ValueError, match="Study ID.*already exists"):
            patient.add_imaging_study(study2)
    
    def test_add_wearable_session(self):
        """Test adding wearable session to patient record."""
        demographics = Demographics(age=45, gender="F")
        patient = PatientRecord(
            patient_id="PAT_20241010_00001",
            demographics=demographics
        )
        
        session = WearableSession(
            session_id="WEAR_EEG_20241010_143000",
            device_type="EEG",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=1),
            sampling_rate=256.0
        )
        
        patient.add_wearable_session(session)
        
        assert len(patient.wearable_data) == 1
        assert patient.wearable_data[0].device_type == "EEG"
    
    def test_get_studies_by_modality(self):
        """Test getting studies by modality."""
        demographics = Demographics(age=45, gender="F")
        patient = PatientRecord(
            patient_id="PAT_20241010_00001",
            demographics=demographics
        )
        
        mri_study = ImagingStudy(
            study_id="STUDY_20241010_143000_001",
            modality="MRI",
            acquisition_date=datetime.now(),
            file_path="/data/mri.nii.gz"
        )
        
        ct_study = ImagingStudy(
            study_id="STUDY_20241010_143000_002",
            modality="CT",
            acquisition_date=datetime.now(),
            file_path="/data/ct.dcm"
        )
        
        patient.add_imaging_study(mri_study)
        patient.add_imaging_study(ct_study)
        
        mri_studies = patient.get_studies_by_modality("MRI")
        ct_studies = patient.get_studies_by_modality("CT")
        
        assert len(mri_studies) == 1
        assert len(ct_studies) == 1
        assert mri_studies[0].modality == "MRI"
        assert ct_studies[0].modality == "CT"
    
    def test_get_wearable_sessions_by_type(self):
        """Test getting wearable sessions by device type."""
        demographics = Demographics(age=45, gender="F")
        patient = PatientRecord(
            patient_id="PAT_20241010_00001",
            demographics=demographics
        )
        
        eeg_session = WearableSession(
            session_id="WEAR_EEG_20241010_143000",
            device_type="EEG",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=1),
            sampling_rate=256.0
        )
        
        hr_session = WearableSession(
            session_id="WEAR_HeartRate_20241010_143000",
            device_type="HeartRate",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=24),
            sampling_rate=1.0
        )
        
        patient.add_wearable_session(eeg_session)
        patient.add_wearable_session(hr_session)
        
        eeg_sessions = patient.get_wearable_sessions_by_type("EEG")
        hr_sessions = patient.get_wearable_sessions_by_type("HeartRate")
        
        assert len(eeg_sessions) == 1
        assert len(hr_sessions) == 1
        assert eeg_sessions[0].device_type == "EEG"
        assert hr_sessions[0].device_type == "HeartRate"


class TestValidationFunctions:
    """Test cases for validation utility functions."""
    
    def test_validate_medical_imaging_metadata_valid(self):
        """Test valid medical imaging metadata validation."""
        metadata = {
            'spacing': [1.0, 1.0, 1.0],
            'origin': [0.0, 0.0, 0.0],
            'direction': [1, 0, 0, 0, 1, 0, 0, 0, 1]
        }
        
        result = validate_medical_imaging_metadata(metadata)
        assert result is True
    
    def test_validate_medical_imaging_metadata_missing_field(self):
        """Test missing required field raises error."""
        metadata = {
            'spacing': [1.0, 1.0, 1.0],
            'origin': [0.0, 0.0, 0.0]
            # Missing 'direction'
        }
        
        with pytest.raises(ValueError, match="Missing required metadata field: direction"):
            validate_medical_imaging_metadata(metadata)
    
    def test_validate_medical_imaging_metadata_invalid_spacing(self):
        """Test invalid spacing raises error."""
        metadata = {
            'spacing': [1.0, -1.0, 1.0],  # Negative spacing
            'origin': [0.0, 0.0, 0.0],
            'direction': [1, 0, 0, 0, 1, 0, 0, 0, 1]
        }
        
        with pytest.raises(ValueError, match="All spacing values must be positive"):
            validate_medical_imaging_metadata(metadata)
    
    def test_create_preprocessing_metadata_valid(self):
        """Test creating valid preprocessing metadata."""
        metadata = create_preprocessing_metadata(
            original_spacing=(2.0, 2.0, 3.0),
            resampled_spacing=(1.0, 1.0, 1.0),
            normalization_method="z_score"
        )
        
        assert metadata.original_spacing == (2.0, 2.0, 3.0)
        assert metadata.resampled_spacing == (1.0, 1.0, 1.0)
        assert metadata.normalization_method == "z_score"
    
    def test_create_preprocessing_metadata_invalid_spacing(self):
        """Test invalid spacing raises error."""
        with pytest.raises(ValueError, match="Spacing must be 3 positive values"):
            create_preprocessing_metadata(
                original_spacing=(2.0, -1.0, 3.0)  # Negative value
            )
    
    def test_create_preprocessing_metadata_invalid_method(self):
        """Test invalid normalization method raises error."""
        with pytest.raises(ValueError, match="Normalization method must be one of"):
            create_preprocessing_metadata(
                original_spacing=(2.0, 2.0, 3.0),
                normalization_method="invalid_method"
            )


class TestLongitudinalData:
    """Test cases for LongitudinalData data model."""
    
    def test_valid_longitudinal_data(self):
        """Test creating valid longitudinal data."""
        baseline = datetime.now() - timedelta(days=365)
        follow_ups = [
            baseline + timedelta(days=90),
            baseline + timedelta(days=180),
            baseline + timedelta(days=270)
        ]
        
        longitudinal = LongitudinalData(
            baseline_date=baseline,
            follow_up_dates=follow_ups,
            progression_metrics={"cognitive_score": [100, 95, 90, 85]},
            clinical_assessments={
                baseline: {"mmse": 28, "cdr": 0},
                follow_ups[0]: {"mmse": 26, "cdr": 0.5}
            }
        )
        
        assert longitudinal.baseline_date == baseline
        assert len(longitudinal.follow_up_dates) == 3
        assert "cognitive_score" in longitudinal.progression_metrics
        assert len(longitudinal.clinical_assessments) == 2
    
    def test_empty_longitudinal_data(self):
        """Test longitudinal data with minimal information."""
        baseline = datetime.now() - timedelta(days=30)
        
        longitudinal = LongitudinalData(baseline_date=baseline)
        
        assert longitudinal.baseline_date == baseline
        assert len(longitudinal.follow_up_dates) == 0
        assert len(longitudinal.progression_metrics) == 0
        assert len(longitudinal.clinical_assessments) == 0


class TestAnnotation:
    """Test cases for Annotation data model."""
    
    def test_valid_segmentation_annotation(self):
        """Test creating valid segmentation annotation."""
        annotation = Annotation(
            annotation_id="ANN_20241010_001",
            annotator_id="RADIOLOGIST_001",
            annotation_type="segmentation",
            annotation_data={
                "mask": np.random.randint(0, 3, size=(64, 64, 32)).tolist(),
                "classes": ["background", "lesion", "edema"]
            },
            confidence_score=0.95
        )
        
        assert annotation.annotation_id == "ANN_20241010_001"
        assert annotation.annotation_type == "segmentation"
        assert annotation.confidence_score == 0.95
        assert annotation.validation_status == "pending"
        assert isinstance(annotation.creation_timestamp, datetime)
    
    def test_valid_bounding_box_annotation(self):
        """Test creating valid bounding box annotation."""
        annotation = Annotation(
            annotation_id="ANN_20241010_002",
            annotator_id="RADIOLOGIST_002",
            annotation_type="bounding_box",
            annotation_data={
                "boxes": [
                    {"x": 10, "y": 15, "width": 20, "height": 25, "class": "lesion"},
                    {"x": 50, "y": 60, "width": 15, "height": 18, "class": "tumor"}
                ]
            },
            confidence_score=0.88,
            validation_status="validated"
        )
        
        assert annotation.annotation_type == "bounding_box"
        assert len(annotation.annotation_data["boxes"]) == 2
        assert annotation.validation_status == "validated"
    
    def test_valid_classification_annotation(self):
        """Test creating valid classification annotation."""
        annotation = Annotation(
            annotation_id="ANN_20241010_003",
            annotator_id="NEUROLOGIST_001",
            annotation_type="classification",
            annotation_data={
                "class": "alzheimer",
                "severity": "moderate",
                "confidence": 0.92
            },
            confidence_score=0.92
        )
        
        assert annotation.annotation_type == "classification"
        assert annotation.annotation_data["class"] == "alzheimer"


class TestEdgeCasesAndErrorHandling:
    """Test cases for edge cases and comprehensive error handling."""
    
    def test_imaging_study_with_preprocessing_metadata(self):
        """Test imaging study with preprocessing metadata."""
        preprocessing = PreprocessingMetadata(
            original_spacing=(2.0, 2.0, 3.0),
            resampled_spacing=(1.0, 1.0, 1.0),
            normalization_method="min_max",
            registration_applied=True,
            augmentation_applied=True
        )
        
        study = ImagingStudy(
            study_id="STUDY_20241010_143000_001",
            modality="MRI",
            acquisition_date=datetime.now(),
            file_path="/data/test.nii.gz",
            preprocessing_metadata=preprocessing,
            series_description="T1-weighted MPRAGE",
            scanner_manufacturer="Siemens",
            scanner_model="Prisma 3T",
            slice_thickness=1.0,
            pixel_spacing=(0.5, 0.5)
        )
        
        assert study.preprocessing_metadata is not None
        assert study.preprocessing_metadata.normalization_method == "min_max"
        assert study.series_description == "T1-weighted MPRAGE"
        assert study.scanner_manufacturer == "Siemens"
    
    def test_wearable_session_with_all_optional_fields(self):
        """Test wearable session with all optional fields."""
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=2)
        raw_data = np.random.randn(int(2 * 3600 * 100))  # 2 hours at 100Hz
        
        session = WearableSession(
            session_id="WEAR_Gait_20241010_143000",
            device_type="Gait",
            start_time=start_time,
            end_time=end_time,
            sampling_rate=100.0,
            raw_data=raw_data,
            processed_features={
                "step_count": 2500,
                "cadence": 110.5,
                "stride_length": 1.2,
                "gait_variability": 0.15
            },
            device_manufacturer="Garmin",
            device_model="Vivosmart 5",
            firmware_version="4.20"
        )
        
        assert session.device_type == "Gait"
        assert len(session.processed_features) == 4
        assert session.device_manufacturer == "Garmin"
        assert session.firmware_version == "4.20"
    
    def test_patient_record_with_all_data_types(self):
        """Test patient record with all types of associated data."""
        demographics = Demographics(
            age=65,
            gender="M",
            weight_kg=75.5,
            height_cm=175.0,
            medical_history=["hypertension", "diabetes", "family_history_alzheimer"]
        )
        
        patient = PatientRecord(
            patient_id="PAT_20241010_00001",
            demographics=demographics
        )
        
        # Add imaging studies
        mri_study = ImagingStudy(
            study_id="STUDY_20241010_143000_001",
            modality="MRI",
            acquisition_date=datetime.now(),
            file_path="/data/mri_t1.nii.gz"
        )
        
        ct_study = ImagingStudy(
            study_id="STUDY_20241010_143000_002",
            modality="CT",
            acquisition_date=datetime.now(),
            file_path="/data/ct_head.dcm"
        )
        
        patient.add_imaging_study(mri_study)
        patient.add_imaging_study(ct_study)
        
        # Add wearable sessions
        eeg_session = WearableSession(
            session_id="WEAR_EEG_20241010_143000",
            device_type="EEG",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=1),
            sampling_rate=256.0
        )
        
        patient.add_wearable_session(eeg_session)
        
        # Add longitudinal data
        longitudinal = LongitudinalData(
            baseline_date=datetime.now() - timedelta(days=365),
            follow_up_dates=[datetime.now() - timedelta(days=180)],
            progression_metrics={"cognitive_decline": [0.0, 0.15]}
        )
        patient.longitudinal_tracking = longitudinal
        
        # Add annotations
        annotation = Annotation(
            annotation_id="ANN_20241010_001",
            annotator_id="EXPERT_001",
            annotation_type="segmentation",
            annotation_data={"mask_path": "/annotations/mask_001.nii"},
            confidence_score=0.95
        )
        patient.annotations.append(annotation)
        
        # Verify all data is present
        assert len(patient.imaging_studies) == 2
        assert len(patient.wearable_data) == 1
        assert patient.longitudinal_tracking is not None
        assert len(patient.annotations) == 1
        assert len(patient.demographics.medical_history) == 3
    
    def test_boundary_values_demographics(self):
        """Test boundary values for demographics validation."""
        # Test minimum valid age
        demographics_min = Demographics(age=0, gender="F")
        patient_min = PatientRecord(
            patient_id="PAT_20241010_00001",
            demographics=demographics_min
        )
        assert patient_min.demographics.age == 0
        
        # Test maximum valid age
        demographics_max = Demographics(age=150, gender="M")
        patient_max = PatientRecord(
            patient_id="PAT_20241010_00002",
            demographics=demographics_max
        )
        assert patient_max.demographics.age == 150
        
        # Test minimum valid weight
        demographics_weight = Demographics(
            age=30, gender="F", weight_kg=0.1  # Very small but positive
        )
        patient_weight = PatientRecord(
            patient_id="PAT_20241010_00003",
            demographics=demographics_weight
        )
        assert patient_weight.demographics.weight_kg == 0.1
    
    def test_boundary_values_wearable_sessions(self):
        """Test boundary values for wearable session validation."""
        # Test minimum session duration (1 minute)
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=1)
        
        session_min = WearableSession(
            session_id="WEAR_HeartRate_20241010_143000",
            device_type="HeartRate",
            start_time=start_time,
            end_time=end_time,
            sampling_rate=0.1  # Minimum for HeartRate
        )
        
        assert (session_min.end_time - session_min.start_time).total_seconds() == 60
        
        # Test maximum sampling rates
        session_eeg_max = WearableSession(
            session_id="WEAR_EEG_20241010_143000",
            device_type="EEG",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=1),
            sampling_rate=2048.0  # Maximum for EEG
        )
        
        assert session_eeg_max.sampling_rate == 2048.0
    
    def test_file_format_variations(self):
        """Test various valid file format variations."""
        # Test .nii format
        study_nii = ImagingStudy(
            study_id="STUDY_20241010_143000_001",
            modality="MRI",
            acquisition_date=datetime.now(),
            file_path="/data/brain.nii"
        )
        assert study_nii.file_path.endswith(".nii")
        
        # Test .nii.gz format
        study_nii_gz = ImagingStudy(
            study_id="STUDY_20241010_143000_002",
            modality="MRI",
            acquisition_date=datetime.now(),
            file_path="/data/brain.nii.gz"
        )
        assert study_nii_gz.file_path.endswith(".nii.gz")
        
        # Test .dcm format
        study_dcm = ImagingStudy(
            study_id="STUDY_20241010_143000_003",
            modality="CT",
            acquisition_date=datetime.now(),
            file_path="/data/ct_slice.dcm"
        )
        assert study_dcm.file_path.endswith(".dcm")
        
        # Test .dicom format
        study_dicom = ImagingStudy(
            study_id="STUDY_20241010_143000_004",
            modality="CT",
            acquisition_date=datetime.now(),
            file_path="/data/ct_slice.dicom"
        )
        assert study_dicom.file_path.endswith(".dicom")
    
    def test_raw_data_tolerance_validation(self):
        """Test raw data length validation with tolerance."""
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=1)
        
        # Test data within tolerance (5%)
        expected_samples = int(3600 * 256)  # 1 hour at 256Hz
        tolerance_samples = int(expected_samples * 0.04)  # 4% tolerance (within 5%)
        raw_data = np.random.randn(expected_samples + tolerance_samples)
        
        session = WearableSession(
            session_id="WEAR_EEG_20241010_143000",
            device_type="EEG",
            start_time=start_time,
            end_time=end_time,
            sampling_rate=256.0,
            raw_data=raw_data
        )
        
        assert session.raw_data.shape[0] == expected_samples + tolerance_samples
    
    def test_multiple_error_conditions(self):
        """Test multiple error conditions in sequence."""
        demographics = Demographics(age=45, gender="F")
        patient = PatientRecord(
            patient_id="PAT_20241010_00001",
            demographics=demographics
        )
        
        # Test adding invalid study type
        with pytest.raises(ValueError, match="Study must be an ImagingStudy instance"):
            patient.add_imaging_study("not_a_study")
        
        # Test adding invalid session type
        with pytest.raises(ValueError, match="Session must be a WearableSession instance"):
            patient.add_wearable_session("not_a_session")
        
        # Add valid study first
        study = ImagingStudy(
            study_id="STUDY_20241010_143000_001",
            modality="MRI",
            acquisition_date=datetime.now(),
            file_path="/data/test.nii.gz"
        )
        patient.add_imaging_study(study)
        
        # Test adding duplicate study
        duplicate_study = ImagingStudy(
            study_id="STUDY_20241010_143000_001",  # Same ID
            modality="CT",
            acquisition_date=datetime.now(),
            file_path="/data/test2.dcm"
        )
        
        with pytest.raises(ValueError, match="Study ID.*already exists"):
            patient.add_imaging_study(duplicate_study)
    
    def test_comprehensive_validation_functions(self):
        """Test comprehensive validation of utility functions."""
        # Test metadata validation with numpy arrays
        metadata_with_numpy = {
            'spacing': np.array([1.0, 1.0, 1.0]),
            'origin': np.array([0.0, 0.0, 0.0]),
            'direction': np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])
        }
        
        result = validate_medical_imaging_metadata(metadata_with_numpy)
        assert result is True
        
        # Test preprocessing metadata with all normalization methods
        for method in ["z_score", "min_max", "unit_variance", "none"]:
            metadata = create_preprocessing_metadata(
                original_spacing=(2.0, 2.0, 3.0),
                normalization_method=method
            )
            assert metadata.normalization_method == method
        
        # Test invalid direction matrix length
        invalid_metadata = {
            'spacing': [1.0, 1.0, 1.0],
            'origin': [0.0, 0.0, 0.0],
            'direction': [1, 0, 0, 0, 1]  # Only 5 elements instead of 9
        }
        
        with pytest.raises(ValueError, match="Direction matrix must have 9 elements"):
            validate_medical_imaging_metadata(invalid_metadata)


if __name__ == "__main__":
    pytest.main([__file__])