"""
Annotation storage and retrieval manager for MONAI Label integration.

This module handles annotation persistence, validation, and retrieval
for neurodegenerative disease annotation workflows.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
import numpy as np

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    # Mock nibabel for testing
    class MockNifti1Image:
        def __init__(self, data, affine):
            self.data = data
            self.affine = affine
        
        def get_fdata(self):
            return self.data
    
    class MockNib:
        Nifti1Image = MockNifti1Image
        
        @staticmethod
        def save(img, filename):
            # Mock save - just create empty file
            Path(filename).touch()
        
        @staticmethod
        def load(filename):
            # Mock load - return dummy image
            return MockNifti1Image(np.zeros((64, 64, 64)), np.eye(4))
    
    nib = MockNib()

from src.models.patient import PatientRecord, ImagingStudy
from src.models.diagnostics import DiagnosticResult

logger = logging.getLogger(__name__)


@dataclass
class Annotation:
    """Represents a single annotation for medical imaging."""
    annotation_id: str
    patient_id: str
    study_id: str
    task_name: str
    annotator_id: str
    annotation_type: str  # 'segmentation', 'classification', 'bounding_box'
    created_at: datetime
    updated_at: datetime
    data: Dict[str, Any]  # Annotation-specific data
    metadata: Optional[Dict[str, Any]] = None
    quality_score: Optional[float] = None
    validation_status: str = "pending"  # 'pending', 'approved', 'rejected'
    
    def __post_init__(self):
        """Validate annotation data."""
        if self.annotation_type not in ['segmentation', 'classification', 'bounding_box']:
            raise ValueError(f"Invalid annotation type: {self.annotation_type}")
        
        if not self.annotation_id:
            raise ValueError("Annotation ID cannot be empty")
        
        if not self.patient_id or not self.study_id:
            raise ValueError("Patient ID and Study ID are required")
        
        if self.quality_score is not None and not (0.0 <= self.quality_score <= 1.0):
            raise ValueError("Quality score must be between 0.0 and 1.0")


@dataclass
class AnnotationSession:
    """Represents an annotation session with multiple annotations."""
    session_id: str
    annotator_id: str
    task_name: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    annotations: List[str] = None  # List of annotation IDs
    session_metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize annotations list if None."""
        if self.annotations is None:
            self.annotations = []


class AnnotationStorage:
    """Handles persistent storage of annotations."""
    
    def __init__(self, storage_path: str):
        """Initialize annotation storage."""
        self.storage_path = Path(storage_path)
        self.annotations_dir = self.storage_path / "annotations"
        self.sessions_dir = self.storage_path / "sessions"
        self.masks_dir = self.storage_path / "masks"
        
        # Create directories
        for dir_path in [self.annotations_dir, self.sessions_dir, self.masks_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized annotation storage at: {self.storage_path}")
    
    def save_annotation(self, annotation: Annotation) -> bool:
        """Save annotation to storage."""
        try:
            annotation_file = self.annotations_dir / f"{annotation.annotation_id}.json"
            
            # Convert annotation to dict for JSON serialization
            annotation_dict = asdict(annotation)
            annotation_dict['created_at'] = annotation.created_at.isoformat()
            annotation_dict['updated_at'] = annotation.updated_at.isoformat()
            
            # Handle numpy arrays in data
            if 'mask' in annotation.data and isinstance(annotation.data['mask'], np.ndarray):
                # Save mask as NIfTI file
                mask_file = self.masks_dir / f"{annotation.annotation_id}_mask.nii.gz"
                mask_img = nib.Nifti1Image(annotation.data['mask'], np.eye(4))
                nib.save(mask_img, str(mask_file))
                
                # Replace mask data with file reference
                annotation_dict['data']['mask'] = str(mask_file)
            
            with open(annotation_file, 'w') as f:
                json.dump(annotation_dict, f, indent=2)
            
            logger.info(f"Saved annotation: {annotation.annotation_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save annotation {annotation.annotation_id}: {e}")
            return False
    
    def load_annotation(self, annotation_id: str) -> Optional[Annotation]:
        """Load annotation from storage."""
        try:
            annotation_file = self.annotations_dir / f"{annotation_id}.json"
            
            if not annotation_file.exists():
                logger.warning(f"Annotation file not found: {annotation_id}")
                return None
            
            with open(annotation_file, 'r') as f:
                annotation_dict = json.load(f)
            
            # Convert datetime strings back to datetime objects
            annotation_dict['created_at'] = datetime.fromisoformat(annotation_dict['created_at'])
            annotation_dict['updated_at'] = datetime.fromisoformat(annotation_dict['updated_at'])
            
            # Load mask if it's a file reference
            if 'mask' in annotation_dict['data'] and isinstance(annotation_dict['data']['mask'], str):
                mask_file = Path(annotation_dict['data']['mask'])
                if mask_file.exists():
                    mask_img = nib.load(str(mask_file))
                    annotation_dict['data']['mask'] = mask_img.get_fdata()
            
            return Annotation(**annotation_dict)
            
        except Exception as e:
            logger.error(f"Failed to load annotation {annotation_id}: {e}")
            return None
    
    def delete_annotation(self, annotation_id: str) -> bool:
        """Delete annotation from storage."""
        try:
            annotation_file = self.annotations_dir / f"{annotation_id}.json"
            mask_file = self.masks_dir / f"{annotation_id}_mask.nii.gz"
            
            if annotation_file.exists():
                annotation_file.unlink()
            
            if mask_file.exists():
                mask_file.unlink()
            
            logger.info(f"Deleted annotation: {annotation_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete annotation {annotation_id}: {e}")
            return False
    
    def list_annotations(self, patient_id: Optional[str] = None, 
                        task_name: Optional[str] = None) -> List[str]:
        """List annotation IDs with optional filtering."""
        try:
            annotation_ids = []
            
            for annotation_file in self.annotations_dir.glob("*.json"):
                annotation_id = annotation_file.stem
                
                # Apply filters if specified
                if patient_id or task_name:
                    annotation = self.load_annotation(annotation_id)
                    if annotation is None:
                        continue
                    
                    if patient_id and annotation.patient_id != patient_id:
                        continue
                    
                    if task_name and annotation.task_name != task_name:
                        continue
                
                annotation_ids.append(annotation_id)
            
            return annotation_ids
            
        except Exception as e:
            logger.error(f"Failed to list annotations: {e}")
            return []


class AnnotationManager:
    """
    Manages annotation storage, retrieval, and validation for MONAI Label integration.
    
    Provides high-level interface for annotation operations including quality assessment,
    validation workflows, and integration with active learning systems.
    """
    
    def __init__(self, storage_path: str):
        """Initialize annotation manager."""
        self.storage = AnnotationStorage(storage_path)
        self.active_sessions: Dict[str, AnnotationSession] = {}
        
        logger.info(f"Initialized annotation manager with storage: {storage_path}")
    
    def create_annotation(self, patient_id: str, study_id: str, task_name: str,
                         annotator_id: str, annotation_type: str, 
                         annotation_data: Dict[str, Any],
                         metadata: Optional[Dict[str, Any]] = None) -> Optional[Annotation]:
        """Create a new annotation."""
        try:
            # Generate unique annotation ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            annotation_id = f"ANN_{task_name}_{patient_id}_{timestamp}"
            
            annotation = Annotation(
                annotation_id=annotation_id,
                patient_id=patient_id,
                study_id=study_id,
                task_name=task_name,
                annotator_id=annotator_id,
                annotation_type=annotation_type,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                data=annotation_data,
                metadata=metadata or {}
            )
            
            # Validate annotation data based on type
            if not self._validate_annotation_data(annotation):
                logger.error(f"Annotation data validation failed for: {annotation_id}")
                return None
            
            # Calculate quality score
            annotation.quality_score = self._calculate_quality_score(annotation)
            
            # Save annotation
            if self.storage.save_annotation(annotation):
                logger.info(f"Created annotation: {annotation_id}")
                return annotation
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to create annotation: {e}")
            return None
    
    def get_annotation(self, annotation_id: str) -> Optional[Annotation]:
        """Retrieve annotation by ID."""
        return self.storage.load_annotation(annotation_id)
    
    def update_annotation(self, annotation_id: str, 
                         updates: Dict[str, Any]) -> Optional[Annotation]:
        """Update existing annotation."""
        try:
            annotation = self.storage.load_annotation(annotation_id)
            if annotation is None:
                logger.warning(f"Annotation not found for update: {annotation_id}")
                return None
            
            # Apply updates
            for key, value in updates.items():
                if hasattr(annotation, key):
                    setattr(annotation, key, value)
            
            annotation.updated_at = datetime.now()
            
            # Re-validate and save
            if self._validate_annotation_data(annotation):
                annotation.quality_score = self._calculate_quality_score(annotation)
                
                if self.storage.save_annotation(annotation):
                    logger.info(f"Updated annotation: {annotation_id}")
                    return annotation
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to update annotation {annotation_id}: {e}")
            return None
    
    def delete_annotation(self, annotation_id: str) -> bool:
        """Delete annotation."""
        return self.storage.delete_annotation(annotation_id)
    
    def get_annotations_for_patient(self, patient_id: str, 
                                   task_name: Optional[str] = None) -> List[Annotation]:
        """Get all annotations for a specific patient."""
        annotation_ids = self.storage.list_annotations(patient_id=patient_id, task_name=task_name)
        annotations = []
        
        for annotation_id in annotation_ids:
            annotation = self.storage.load_annotation(annotation_id)
            if annotation:
                annotations.append(annotation)
        
        return annotations
    
    def get_annotations_for_task(self, task_name: str) -> List[Annotation]:
        """Get all annotations for a specific task."""
        annotation_ids = self.storage.list_annotations(task_name=task_name)
        annotations = []
        
        for annotation_id in annotation_ids:
            annotation = self.storage.load_annotation(annotation_id)
            if annotation:
                annotations.append(annotation)
        
        return annotations
    
    def start_annotation_session(self, annotator_id: str, task_name: str,
                                session_metadata: Optional[Dict[str, Any]] = None) -> str:
        """Start a new annotation session."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"SESSION_{task_name}_{annotator_id}_{timestamp}"
        
        session = AnnotationSession(
            session_id=session_id,
            annotator_id=annotator_id,
            task_name=task_name,
            started_at=datetime.now(),
            session_metadata=session_metadata or {}
        )
        
        self.active_sessions[session_id] = session
        logger.info(f"Started annotation session: {session_id}")
        
        return session_id
    
    def end_annotation_session(self, session_id: str) -> Optional[AnnotationSession]:
        """End an annotation session."""
        if session_id not in self.active_sessions:
            logger.warning(f"Session not found: {session_id}")
            return None
        
        session = self.active_sessions[session_id]
        session.completed_at = datetime.now()
        
        # Save session to storage
        try:
            session_file = self.storage.sessions_dir / f"{session_id}.json"
            session_dict = asdict(session)
            session_dict['started_at'] = session.started_at.isoformat()
            if session.completed_at:
                session_dict['completed_at'] = session.completed_at.isoformat()
            
            with open(session_file, 'w') as f:
                json.dump(session_dict, f, indent=2)
            
            # Remove from active sessions
            del self.active_sessions[session_id]
            
            logger.info(f"Ended annotation session: {session_id}")
            return session
            
        except Exception as e:
            logger.error(f"Failed to end session {session_id}: {e}")
            return None
    
    def add_annotation_to_session(self, session_id: str, annotation_id: str) -> bool:
        """Add annotation to an active session."""
        if session_id not in self.active_sessions:
            logger.warning(f"Session not found: {session_id}")
            return False
        
        session = self.active_sessions[session_id]
        if annotation_id not in session.annotations:
            session.annotations.append(annotation_id)
            logger.info(f"Added annotation {annotation_id} to session {session_id}")
            return True
        
        return False
    
    def validate_annotation(self, annotation_id: str, validator_id: str,
                           validation_result: str, comments: Optional[str] = None) -> bool:
        """Validate an annotation with approval/rejection."""
        try:
            annotation = self.storage.load_annotation(annotation_id)
            if annotation is None:
                return False
            
            annotation.validation_status = validation_result
            annotation.updated_at = datetime.now()
            
            # Add validation metadata
            if annotation.metadata is None:
                annotation.metadata = {}
            
            annotation.metadata.update({
                'validator_id': validator_id,
                'validation_date': datetime.now().isoformat(),
                'validation_comments': comments
            })
            
            return self.storage.save_annotation(annotation)
            
        except Exception as e:
            logger.error(f"Failed to validate annotation {annotation_id}: {e}")
            return False
    
    def get_annotation_statistics(self, task_name: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics about annotations."""
        try:
            annotations = self.get_annotations_for_task(task_name) if task_name else []
            
            if not task_name:
                # Get all annotations
                all_annotation_ids = self.storage.list_annotations()
                annotations = []
                for annotation_id in all_annotation_ids:
                    annotation = self.storage.load_annotation(annotation_id)
                    if annotation:
                        annotations.append(annotation)
            
            total_count = len(annotations)
            
            # Count by validation status
            status_counts = {}
            quality_scores = []
            annotator_counts = {}
            
            for annotation in annotations:
                # Validation status
                status = annotation.validation_status
                status_counts[status] = status_counts.get(status, 0) + 1
                
                # Quality scores
                if annotation.quality_score is not None:
                    quality_scores.append(annotation.quality_score)
                
                # Annotator counts
                annotator = annotation.annotator_id
                annotator_counts[annotator] = annotator_counts.get(annotator, 0) + 1
            
            # Calculate quality statistics
            quality_stats = {}
            if quality_scores:
                quality_stats = {
                    'mean': np.mean(quality_scores),
                    'median': np.median(quality_scores),
                    'std': np.std(quality_scores),
                    'min': np.min(quality_scores),
                    'max': np.max(quality_scores)
                }
            
            return {
                'total_annotations': total_count,
                'validation_status_counts': status_counts,
                'quality_statistics': quality_stats,
                'annotator_counts': annotator_counts,
                'task_name': task_name
            }
            
        except Exception as e:
            logger.error(f"Failed to get annotation statistics: {e}")
            return {}
    
    def _validate_annotation_data(self, annotation: Annotation) -> bool:
        """Validate annotation data based on type."""
        try:
            if annotation.annotation_type == "segmentation":
                return self._validate_segmentation_data(annotation.data)
            elif annotation.annotation_type == "classification":
                return self._validate_classification_data(annotation.data)
            elif annotation.annotation_type == "bounding_box":
                return self._validate_bounding_box_data(annotation.data)
            
            return False
            
        except Exception as e:
            logger.error(f"Annotation validation error: {e}")
            return False
    
    def _validate_segmentation_data(self, data: Dict[str, Any]) -> bool:
        """Validate segmentation annotation data."""
        required_fields = ['mask']
        
        for field in required_fields:
            if field not in data:
                logger.error(f"Missing required field for segmentation: {field}")
                return False
        
        # Validate mask
        mask = data['mask']
        if isinstance(mask, np.ndarray):
            if mask.dtype not in [np.uint8, np.uint16, np.int32]:
                logger.error("Segmentation mask must be integer type")
                return False
        
        return True
    
    def _validate_classification_data(self, data: Dict[str, Any]) -> bool:
        """Validate classification annotation data."""
        required_fields = ['class_label']
        
        for field in required_fields:
            if field not in data:
                logger.error(f"Missing required field for classification: {field}")
                return False
        
        # Validate confidence if present
        if 'confidence' in data:
            confidence = data['confidence']
            if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
                logger.error("Confidence must be a number between 0.0 and 1.0")
                return False
        
        return True
    
    def _validate_bounding_box_data(self, data: Dict[str, Any]) -> bool:
        """Validate bounding box annotation data."""
        required_fields = ['bbox']
        
        for field in required_fields:
            if field not in data:
                logger.error(f"Missing required field for bounding box: {field}")
                return False
        
        # Validate bounding box format
        bbox = data['bbox']
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 6:
            logger.error("Bounding box must be a list/tuple of 6 coordinates [x1,y1,z1,x2,y2,z2]")
            return False
        
        return True
    
    def _calculate_quality_score(self, annotation: Annotation) -> float:
        """Calculate quality score for annotation."""
        try:
            # Basic quality scoring based on annotation completeness and consistency
            score = 0.5  # Base score
            
            # Check if all required fields are present
            if self._validate_annotation_data(annotation):
                score += 0.3
            
            # Check metadata completeness
            if annotation.metadata and len(annotation.metadata) > 0:
                score += 0.1
            
            # Additional scoring based on annotation type
            if annotation.annotation_type == "segmentation":
                mask = annotation.data.get('mask')
                if isinstance(mask, np.ndarray):
                    # Check mask properties
                    unique_values = np.unique(mask)
                    if len(unique_values) > 1:  # Has actual segmentation
                        score += 0.1
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Failed to calculate quality score: {e}")
            return 0.5