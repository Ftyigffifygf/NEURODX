"""
Patient record and imaging study data models for NeuroDx-MultiModal system.

This module defines the core data structures for patient records, imaging studies,
and wearable sensor sessions with validation functions for medical imaging metadata.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Literal, Any
import numpy as np
from pathlib import Path
import re


@dataclass
class Demographics:
    """Patient demographic information."""
    age: int
    gender: Literal["M", "F", "O"]  # Male, Female, Other
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None
    medical_history: List[str] = field(default_factory=list)


@dataclass
class PreprocessingMetadata:
    """Metadata for image preprocessing operations."""
    original_spacing: tuple[float, float, float]
    resampled_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    normalization_method: str = "z_score"
    registration_applied: bool = False
    augmentation_applied: bool = False
    preprocessing_timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ImagingStudy:
    """Medical imaging study data model."""
    study_id: str
    modality: Literal["MRI", "CT", "Ultrasound"]
    acquisition_date: datetime
    file_path: str
    preprocessing_metadata: Optional[PreprocessingMetadata] = None
    series_description: Optional[str] = None
    scanner_manufacturer: Optional[str] = None
    scanner_model: Optional[str] = None
    slice_thickness: Optional[float] = None
    pixel_spacing: Optional[tuple[float, float]] = None
    
    def __post_init__(self):
        """Validate imaging study data after initialization."""
        self._validate_study_id()
        self._validate_file_path()
        self._validate_modality_specific_fields()
    
    def _validate_study_id(self):
        """Validate study ID format."""
        if not self.study_id or not isinstance(self.study_id, str):
            raise ValueError("Study ID must be a non-empty string")
        
        # Study ID should follow pattern: STUDY_YYYYMMDD_HHMMSS_XXX
        pattern = r'^STUDY_\d{8}_\d{6}_\d{3}$'
        if not re.match(pattern, self.study_id):
            raise ValueError(
                f"Study ID '{self.study_id}' must follow format: "
                "STUDY_YYYYMMDD_HHMMSS_XXX"
            )
    
    def _validate_file_path(self):
        """Validate file path and format."""
        if not self.file_path:
            raise ValueError("File path cannot be empty")
        
        path = Path(self.file_path)
        valid_extensions = {'.nii', '.nii.gz', '.dcm', '.dicom'}
        
        if path.suffix.lower() not in valid_extensions and not path.name.endswith('.nii.gz'):
            raise ValueError(
                f"Unsupported file format. Supported formats: {valid_extensions}"
            )
    
    def _validate_modality_specific_fields(self):
        """Validate modality-specific requirements."""
        if self.modality == "MRI":
            if self.slice_thickness is not None and self.slice_thickness <= 0:
                raise ValueError("MRI slice thickness must be positive")
        
        elif self.modality == "CT":
            if self.slice_thickness is not None and (self.slice_thickness < 0.5 or self.slice_thickness > 10.0):
                raise ValueError("CT slice thickness should be between 0.5mm and 10.0mm")
        
        elif self.modality == "Ultrasound":
            # Ultrasound typically doesn't have slice thickness
            if self.slice_thickness is not None:
                raise ValueError("Ultrasound studies should not have slice thickness")


@dataclass
class WearableSession:
    """Wearable sensor data session model."""
    session_id: str
    device_type: Literal["EEG", "HeartRate", "Sleep", "Gait"]
    start_time: datetime
    end_time: datetime
    sampling_rate: float
    raw_data: Optional[np.ndarray] = None
    processed_features: Dict[str, float] = field(default_factory=dict)
    device_manufacturer: Optional[str] = None
    device_model: Optional[str] = None
    firmware_version: Optional[str] = None
    
    def __post_init__(self):
        """Validate wearable session data after initialization."""
        self._validate_session_id()
        self._validate_time_range()
        self._validate_sampling_rate()
        self._validate_device_specific_requirements()
    
    def _validate_session_id(self):
        """Validate session ID format."""
        if not self.session_id or not isinstance(self.session_id, str):
            raise ValueError("Session ID must be a non-empty string")
        
        # Session ID pattern: WEAR_DEVICETYPE_YYYYMMDD_HHMMSS
        pattern = r'^WEAR_(EEG|HeartRate|Sleep|Gait)_\d{8}_\d{6}$'
        if not re.match(pattern, self.session_id):
            raise ValueError(
                f"Session ID '{self.session_id}' must follow format: "
                "WEAR_DEVICETYPE_YYYYMMDD_HHMMSS"
            )
    
    def _validate_time_range(self):
        """Validate time range consistency."""
        if self.end_time <= self.start_time:
            raise ValueError("End time must be after start time")
        
        # Check for reasonable session duration
        duration = (self.end_time - self.start_time).total_seconds()
        if duration < 60:  # Less than 1 minute
            raise ValueError("Session duration must be at least 1 minute")
        
        max_durations = {
            "EEG": 24 * 3600,      # 24 hours max
            "HeartRate": 7 * 24 * 3600,  # 7 days max
            "Sleep": 12 * 3600,    # 12 hours max
            "Gait": 2 * 3600       # 2 hours max
        }
        
        if duration > max_durations[self.device_type]:
            raise ValueError(
                f"{self.device_type} session duration exceeds maximum allowed "
                f"({max_durations[self.device_type]} seconds)"
            )
    
    def _validate_sampling_rate(self):
        """Validate sampling rate for device type."""
        if self.sampling_rate <= 0:
            raise ValueError("Sampling rate must be positive")
        
        # Device-specific sampling rate ranges
        valid_ranges = {
            "EEG": (128, 2048),      # 128Hz to 2048Hz
            "HeartRate": (0.1, 10),  # 0.1Hz to 10Hz
            "Sleep": (0.01, 1),      # 0.01Hz to 1Hz
            "Gait": (50, 1000)       # 50Hz to 1000Hz
        }
        
        min_rate, max_rate = valid_ranges[self.device_type]
        if not (min_rate <= self.sampling_rate <= max_rate):
            raise ValueError(
                f"{self.device_type} sampling rate must be between "
                f"{min_rate}Hz and {max_rate}Hz"
            )
    
    def _validate_device_specific_requirements(self):
        """Validate device-specific data requirements."""
        if self.raw_data is not None:
            if not isinstance(self.raw_data, np.ndarray):
                raise ValueError("Raw data must be a numpy array")
            
            # Check data shape consistency with sampling rate and duration
            expected_samples = int(
                (self.end_time - self.start_time).total_seconds() * self.sampling_rate
            )
            
            # Allow 5% tolerance for sampling rate variations
            tolerance = 0.05
            min_samples = int(expected_samples * (1 - tolerance))
            max_samples = int(expected_samples * (1 + tolerance))
            
            if not (min_samples <= self.raw_data.shape[0] <= max_samples):
                raise ValueError(
                    f"Raw data length ({self.raw_data.shape[0]}) doesn't match "
                    f"expected samples ({expected_samples}±{tolerance*100}%)"
                )


@dataclass
class LongitudinalData:
    """Longitudinal tracking data for disease progression monitoring."""
    baseline_date: datetime
    follow_up_dates: List[datetime] = field(default_factory=list)
    progression_metrics: Dict[str, List[float]] = field(default_factory=dict)
    clinical_assessments: Dict[datetime, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class Annotation:
    """Medical annotation data model."""
    annotation_id: str
    annotator_id: str
    annotation_type: Literal["segmentation", "bounding_box", "classification"]
    annotation_data: Dict[str, Any]
    confidence_score: float
    creation_timestamp: datetime = field(default_factory=datetime.now)
    validation_status: Literal["pending", "validated", "rejected"] = "pending"


@dataclass
class PatientRecord:
    """Complete patient record with all associated data."""
    patient_id: str
    demographics: Demographics
    imaging_studies: List[ImagingStudy] = field(default_factory=list)
    wearable_data: List[WearableSession] = field(default_factory=list)
    annotations: List[Annotation] = field(default_factory=list)
    longitudinal_tracking: Optional[LongitudinalData] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate patient record after initialization."""
        self._validate_patient_id()
        self._validate_demographics()
    
    def _validate_patient_id(self):
        """Validate patient ID format."""
        if not self.patient_id or not isinstance(self.patient_id, str):
            raise ValueError("Patient ID must be a non-empty string")
        
        # Patient ID pattern: PAT_YYYYMMDD_XXXXX
        pattern = r'^PAT_\d{8}_\d{5}$'
        if not re.match(pattern, self.patient_id):
            raise ValueError(
                f"Patient ID '{self.patient_id}' must follow format: "
                "PAT_YYYYMMDD_XXXXX"
            )
    
    def _validate_demographics(self):
        """Validate demographic data."""
        if not isinstance(self.demographics, Demographics):
            raise ValueError("Demographics must be a Demographics instance")
        
        if self.demographics.age < 0 or self.demographics.age > 150:
            raise ValueError("Age must be between 0 and 150")
        
        if self.demographics.weight_kg is not None:
            if self.demographics.weight_kg <= 0 or self.demographics.weight_kg > 1000:
                raise ValueError("Weight must be between 0 and 1000 kg")
        
        if self.demographics.height_cm is not None:
            if self.demographics.height_cm <= 0 or self.demographics.height_cm > 300:
                raise ValueError("Height must be between 0 and 300 cm")
    
    def add_imaging_study(self, study: ImagingStudy):
        """Add an imaging study to the patient record."""
        if not isinstance(study, ImagingStudy):
            raise ValueError("Study must be an ImagingStudy instance")
        
        # Check for duplicate study IDs
        existing_ids = {s.study_id for s in self.imaging_studies}
        if study.study_id in existing_ids:
            raise ValueError(f"Study ID '{study.study_id}' already exists")
        
        self.imaging_studies.append(study)
        self.updated_at = datetime.now()
    
    def add_wearable_session(self, session: WearableSession):
        """Add a wearable session to the patient record."""
        if not isinstance(session, WearableSession):
            raise ValueError("Session must be a WearableSession instance")
        
        # Check for duplicate session IDs
        existing_ids = {s.session_id for s in self.wearable_data}
        if session.session_id in existing_ids:
            raise ValueError(f"Session ID '{session.session_id}' already exists")
        
        self.wearable_data.append(session)
        self.updated_at = datetime.now()
    
    def get_studies_by_modality(self, modality: str) -> List[ImagingStudy]:
        """Get all imaging studies of a specific modality."""
        return [study for study in self.imaging_studies if study.modality == modality]
    
    def get_wearable_sessions_by_type(self, device_type: str) -> List[WearableSession]:
        """Get all wearable sessions of a specific device type."""
        return [session for session in self.wearable_data if session.device_type == device_type]


# Validation utility functions
def validate_medical_imaging_metadata(metadata: Dict[str, Any]) -> bool:
    """
    Validate medical imaging metadata for MONAI compatibility.
    
    Args:
        metadata: Dictionary containing imaging metadata
        
    Returns:
        bool: True if metadata is valid
        
    Raises:
        ValueError: If metadata is invalid
    """
    required_fields = ['spacing', 'origin', 'direction']
    
    for field in required_fields:
        if field not in metadata:
            raise ValueError(f"Missing required metadata field: {field}")
    
    # Validate spacing
    spacing = metadata['spacing']
    if not isinstance(spacing, (list, tuple, np.ndarray)) or len(spacing) != 3:
        raise ValueError("Spacing must be a 3-element list, tuple, or numpy array")
    
    if any(s <= 0 for s in spacing):
        raise ValueError("All spacing values must be positive")
    
    # Validate origin
    origin = metadata['origin']
    if not isinstance(origin, (list, tuple, np.ndarray)) or len(origin) != 3:
        raise ValueError("Origin must be a 3-element list, tuple, or numpy array")
    
    # Validate direction matrix
    direction = metadata['direction']
    if not isinstance(direction, (list, tuple, np.ndarray)):
        raise ValueError("Direction must be a list, tuple, or numpy array")
    
    if len(direction) != 9:
        raise ValueError("Direction matrix must have 9 elements (3x3 flattened)")
    
    return True


def create_preprocessing_metadata(
    original_spacing: tuple[float, float, float],
    resampled_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    normalization_method: str = "z_score",
    registration_applied: bool = False,
    augmentation_applied: bool = False
) -> PreprocessingMetadata:
    """
    Create preprocessing metadata with validation.
    
    Args:
        original_spacing: Original image spacing
        resampled_spacing: Target resampling spacing
        normalization_method: Normalization method used
        registration_applied: Whether registration was applied
        augmentation_applied: Whether augmentation was applied
        
    Returns:
        PreprocessingMetadata: Validated preprocessing metadata
    """
    # Validate spacing values
    for spacing in [original_spacing, resampled_spacing]:
        if len(spacing) != 3 or any(s <= 0 for s in spacing):
            raise ValueError("Spacing must be 3 positive values")
    
    # Validate normalization method
    valid_methods = ["z_score", "min_max", "unit_variance", "none"]
    if normalization_method not in valid_methods:
        raise ValueError(f"Normalization method must be one of: {valid_methods}")
    
    return PreprocessingMetadata(
        original_spacing=original_spacing,
        resampled_spacing=resampled_spacing,
        normalization_method=normalization_method,
        registration_applied=registration_applied,
        augmentation_applied=augmentation_applied
    )