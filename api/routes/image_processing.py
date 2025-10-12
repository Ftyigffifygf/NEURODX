"""
Image processing API endpoints for medical image upload and processing.
Implements endpoints for medical image upload, validation, processing status tracking, and result retrieval.
"""

import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage

from flask import Blueprint, request, jsonify, current_app
import torch

from src.services.image_processing.image_ingestion import ImageIngestionHandler, ImageIngestionError, UnsupportedFormatError, CorruptedFileError
from src.services.image_processing.preprocessing_pipeline import PreprocessingPipeline
from src.models.patient import ImagingStudy, PreprocessingMetadata, PatientRecord
from src.config.settings import get_settings
from src.utils.logging_config import get_logger

logger = get_logger(__name__)
image_bp = Blueprint('image_processing', __name__)

# Global instances
image_handler = ImageIngestionHandler()
preprocessing_pipeline = PreprocessingPipeline()

# In-memory storage for processing status (in production, use Redis/database)
processing_status = {}


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    allowed_extensions = {'.nii', '.nii.gz', '.dcm', '.dicom'}
    
    if filename.lower().endswith('.nii.gz'):
        return True
    
    file_ext = Path(filename).suffix.lower()
    return file_ext in allowed_extensions


def generate_study_id() -> str:
    """Generate a unique study ID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:3].upper()
    return f"STUDY_{timestamp}_{unique_id}"


def save_uploaded_file(file: FileStorage, upload_dir: Path) -> Path:
    """
    Save uploaded file to disk with secure filename.
    
    Args:
        file: Uploaded file object
        upload_dir: Directory to save file
        
    Returns:
        Path to saved file
    """
    # Create upload directory if it doesn't exist
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate secure filename
    filename = secure_filename(file.filename)
    if not filename:
        filename = f"upload_{uuid.uuid4().hex}"
    
    # Add timestamp to avoid conflicts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name, ext = os.path.splitext(filename)
    secure_name = f"{name}_{timestamp}{ext}"
    
    file_path = upload_dir / secure_name
    file.save(str(file_path))
    
    return file_path


@image_bp.route('/upload', methods=['POST'])
def upload_image():
    """
    Upload medical image for processing.
    
    Expected form data:
    - file: Medical image file (NIfTI or DICOM)
    - patient_id: Patient identifier (optional)
    - modality: Image modality (MRI, CT, Ultrasound)
    - series_description: Optional series description
    
    Returns:
        JSON response with upload status and study information
    """
    try:
        # Validate request
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file provided',
                'message': 'Please select a file to upload'
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'message': 'Please select a valid file'
            }), 400
        
        # Validate file type
        if not allowed_file(file.filename):
            return jsonify({
                'error': 'Unsupported file format',
                'message': f'Supported formats: .nii, .nii.gz, .dcm, .dicom',
                'filename': file.filename
            }), 400
        
        # Get form data
        patient_id = request.form.get('patient_id')
        modality = request.form.get('modality', '').upper()
        series_description = request.form.get('series_description')
        
        # Validate modality
        valid_modalities = ['MRI', 'CT', 'ULTRASOUND']
        if modality not in valid_modalities:
            return jsonify({
                'error': 'Invalid modality',
                'message': f'Modality must be one of: {", ".join(valid_modalities)}',
                'provided': modality
            }), 400
        
        # Generate study ID
        study_id = generate_study_id()
        
        # Save file
        settings = get_settings()
        upload_dir = settings.monai.data_directory / "uploads" / study_id
        file_path = save_uploaded_file(file, upload_dir)
        
        logger.info(f"File uploaded: {file_path} for study {study_id}")
        
        # Validate file using image handler
        try:
            image_info = image_handler.get_image_info(file_path)
        except (UnsupportedFormatError, CorruptedFileError) as e:
            # Clean up uploaded file
            file_path.unlink(missing_ok=True)
            return jsonify({
                'error': 'File validation failed',
                'message': str(e),
                'filename': file.filename
            }), 400
        except Exception as e:
            # Clean up uploaded file
            file_path.unlink(missing_ok=True)
            return jsonify({
                'error': 'File processing error',
                'message': f'Unable to process file: {str(e)}',
                'filename': file.filename
            }), 500
        
        # Create imaging study record
        imaging_study = ImagingStudy(
            study_id=study_id,
            modality=modality,
            acquisition_date=datetime.now(),
            file_path=str(file_path),
            series_description=series_description
        )
        
        # Initialize processing status
        processing_status[study_id] = {
            'status': 'uploaded',
            'progress': 0,
            'message': 'File uploaded successfully',
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'study_info': {
                'study_id': study_id,
                'patient_id': patient_id,
                'modality': modality,
                'filename': file.filename,
                'file_size': image_info.get('file_size', 0),
                'format_type': image_info.get('format_type'),
                'series_description': series_description
            }
        }
        
        return jsonify({
            'message': 'File uploaded successfully',
            'study_id': study_id,
            'patient_id': patient_id,
            'modality': modality,
            'filename': file.filename,
            'file_size': image_info.get('file_size', 0),
            'format_type': image_info.get('format_type'),
            'upload_timestamp': datetime.now().isoformat(),
            'processing_status_url': f'/api/v1/images/status/{study_id}'
        }), 201
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({
            'error': 'Upload failed',
            'message': 'An unexpected error occurred during upload',
            'details': str(e)
        }), 500


@image_bp.route('/upload/batch', methods=['POST'])
def upload_batch_images():
    """
    Upload multiple medical images for batch processing.
    
    Expected form data:
    - files: Multiple medical image files
    - patient_id: Patient identifier (optional)
    - modalities: Comma-separated list of modalities corresponding to files
    
    Returns:
        JSON response with batch upload status
    """
    try:
        # Validate request
        if 'files' not in request.files:
            return jsonify({
                'error': 'No files provided',
                'message': 'Please select files to upload'
            }), 400
        
        files = request.files.getlist('files')
        if not files or all(f.filename == '' for f in files):
            return jsonify({
                'error': 'No files selected',
                'message': 'Please select valid files'
            }), 400
        
        # Get form data
        patient_id = request.form.get('patient_id')
        modalities_str = request.form.get('modalities', '')
        modalities = [m.strip().upper() for m in modalities_str.split(',') if m.strip()]
        
        # Validate modalities count matches files count
        if len(modalities) != len(files):
            return jsonify({
                'error': 'Modality count mismatch',
                'message': f'Number of modalities ({len(modalities)}) must match number of files ({len(files)})'
            }), 400
        
        # Validate each modality
        valid_modalities = ['MRI', 'CT', 'ULTRASOUND']
        for modality in modalities:
            if modality not in valid_modalities:
                return jsonify({
                    'error': 'Invalid modality',
                    'message': f'All modalities must be one of: {", ".join(valid_modalities)}',
                    'invalid_modality': modality
                }), 400
        
        # Process each file
        results = []
        errors = []
        
        for i, (file, modality) in enumerate(zip(files, modalities)):
            try:
                # Validate file type
                if not allowed_file(file.filename):
                    errors.append({
                        'file_index': i,
                        'filename': file.filename,
                        'error': 'Unsupported file format'
                    })
                    continue
                
                # Generate study ID
                study_id = generate_study_id()
                
                # Save file
                settings = get_settings()
                upload_dir = settings.monai.data_directory / "uploads" / study_id
                file_path = save_uploaded_file(file, upload_dir)
                
                # Validate file
                image_info = image_handler.get_image_info(file_path)
                
                # Initialize processing status
                processing_status[study_id] = {
                    'status': 'uploaded',
                    'progress': 0,
                    'message': 'File uploaded successfully',
                    'created_at': datetime.now().isoformat(),
                    'updated_at': datetime.now().isoformat(),
                    'study_info': {
                        'study_id': study_id,
                        'patient_id': patient_id,
                        'modality': modality,
                        'filename': file.filename,
                        'file_size': image_info.get('file_size', 0),
                        'format_type': image_info.get('format_type')
                    }
                }
                
                results.append({
                    'study_id': study_id,
                    'filename': file.filename,
                    'modality': modality,
                    'file_size': image_info.get('file_size', 0),
                    'format_type': image_info.get('format_type')
                })
                
            except Exception as e:
                errors.append({
                    'file_index': i,
                    'filename': file.filename,
                    'error': str(e)
                })
        
        response_data = {
            'message': f'Batch upload completed: {len(results)} successful, {len(errors)} failed',
            'successful_uploads': results,
            'failed_uploads': errors,
            'patient_id': patient_id,
            'upload_timestamp': datetime.now().isoformat()
        }
        
        status_code = 201 if results else 400
        return jsonify(response_data), status_code
        
    except Exception as e:
        logger.error(f"Batch upload error: {str(e)}")
        return jsonify({
            'error': 'Batch upload failed',
            'message': 'An unexpected error occurred during batch upload',
            'details': str(e)
        }), 500


@image_bp.route('/process/<study_id>', methods=['POST'])
def process_image(study_id: str):
    """
    Start image preprocessing for a specific study.
    
    Args:
        study_id: Study identifier
        
    Expected JSON body:
    - preprocessing_options: Optional preprocessing configuration
    
    Returns:
        JSON response with processing status
    """
    try:
        # Check if study exists
        if study_id not in processing_status:
            return jsonify({
                'error': 'Study not found',
                'message': f'No study found with ID: {study_id}'
            }), 404
        
        study_status = processing_status[study_id]
        
        # Check if already processing or completed
        if study_status['status'] in ['processing', 'completed']:
            return jsonify({
                'message': f'Study is already {study_status["status"]}',
                'study_id': study_id,
                'current_status': study_status['status'],
                'progress': study_status['progress']
            }), 200
        
        # Get preprocessing options from request
        preprocessing_options = {}
        if request.is_json:
            preprocessing_options = request.get_json() or {}
        
        # Update status to processing
        processing_status[study_id].update({
            'status': 'processing',
            'progress': 10,
            'message': 'Starting image preprocessing',
            'updated_at': datetime.now().isoformat()
        })
        
        # Get file path from study info
        file_path = Path(study_status['study_info']['study_id']).parent / "uploads" / study_id
        
        # Start preprocessing (this would typically be done asynchronously)
        try:
            # Load image
            processing_status[study_id].update({
                'progress': 30,
                'message': 'Loading image data',
                'updated_at': datetime.now().isoformat()
            })
            
            # Find the actual image file in the upload directory
            settings = get_settings()
            upload_dir = settings.monai.data_directory / "uploads" / study_id
            image_files = list(upload_dir.glob("*"))
            
            if not image_files:
                raise FileNotFoundError(f"No image files found for study {study_id}")
            
            image_file = image_files[0]  # Take the first (and should be only) file
            
            # Load image using image handler
            image_data = image_handler.load_single_image(image_file)
            
            processing_status[study_id].update({
                'progress': 50,
                'message': 'Applying preprocessing transforms',
                'updated_at': datetime.now().isoformat()
            })
            
            # Apply preprocessing pipeline
            processed_data = preprocessing_pipeline.preprocess_single_image(
                image_data['image'],
                image_data['metadata'],
                **preprocessing_options
            )
            
            processing_status[study_id].update({
                'progress': 80,
                'message': 'Saving processed data',
                'updated_at': datetime.now().isoformat()
            })
            
            # Save processed data (in production, save to storage)
            processed_dir = settings.monai.data_directory / "processed" / study_id
            processed_dir.mkdir(parents=True, exist_ok=True)
            
            # Save tensor data
            processed_file = processed_dir / "processed_image.pt"
            torch.save(processed_data['processed_image'], processed_file)
            
            # Update final status
            processing_status[study_id].update({
                'status': 'completed',
                'progress': 100,
                'message': 'Image preprocessing completed successfully',
                'updated_at': datetime.now().isoformat(),
                'processed_file': str(processed_file),
                'preprocessing_metadata': {
                    'original_shape': processed_data['original_shape'],
                    'processed_shape': processed_data['processed_shape'],
                    'transforms_applied': processed_data['transforms_applied']
                }
            })
            
            return jsonify({
                'message': 'Image processing completed successfully',
                'study_id': study_id,
                'status': 'completed',
                'progress': 100,
                'processed_file': str(processed_file),
                'preprocessing_metadata': processing_status[study_id]['preprocessing_metadata']
            }), 200
            
        except Exception as e:
            # Update status to failed
            processing_status[study_id].update({
                'status': 'failed',
                'progress': 0,
                'message': f'Processing failed: {str(e)}',
                'updated_at': datetime.now().isoformat(),
                'error': str(e)
            })
            
            logger.error(f"Image processing failed for study {study_id}: {str(e)}")
            return jsonify({
                'error': 'Processing failed',
                'message': f'Image processing failed: {str(e)}',
                'study_id': study_id
            }), 500
        
    except Exception as e:
        logger.error(f"Process image error: {str(e)}")
        return jsonify({
            'error': 'Processing request failed',
            'message': 'An unexpected error occurred',
            'details': str(e)
        }), 500


@image_bp.route('/status/<study_id>', methods=['GET'])
def get_processing_status(study_id: str):
    """
    Get processing status for a specific study.
    
    Args:
        study_id: Study identifier
        
    Returns:
        JSON response with current processing status
    """
    try:
        if study_id not in processing_status:
            return jsonify({
                'error': 'Study not found',
                'message': f'No study found with ID: {study_id}'
            }), 404
        
        status_data = processing_status[study_id]
        
        return jsonify({
            'study_id': study_id,
            'status': status_data['status'],
            'progress': status_data['progress'],
            'message': status_data['message'],
            'created_at': status_data['created_at'],
            'updated_at': status_data['updated_at'],
            'study_info': status_data['study_info']
        }), 200
        
    except Exception as e:
        logger.error(f"Get status error: {str(e)}")
        return jsonify({
            'error': 'Status retrieval failed',
            'message': 'An unexpected error occurred',
            'details': str(e)
        }), 500


@image_bp.route('/studies', methods=['GET'])
def list_studies():
    """
    List all studies with their current status.
    
    Query parameters:
    - patient_id: Filter by patient ID
    - status: Filter by processing status
    - modality: Filter by image modality
    
    Returns:
        JSON response with list of studies
    """
    try:
        # Get query parameters
        patient_id_filter = request.args.get('patient_id')
        status_filter = request.args.get('status')
        modality_filter = request.args.get('modality')
        
        # Filter studies
        filtered_studies = []
        for study_id, status_data in processing_status.items():
            study_info = status_data['study_info']
            
            # Apply filters
            if patient_id_filter and study_info.get('patient_id') != patient_id_filter:
                continue
            if status_filter and status_data['status'] != status_filter:
                continue
            if modality_filter and study_info.get('modality') != modality_filter.upper():
                continue
            
            filtered_studies.append({
                'study_id': study_id,
                'patient_id': study_info.get('patient_id'),
                'modality': study_info.get('modality'),
                'filename': study_info.get('filename'),
                'status': status_data['status'],
                'progress': status_data['progress'],
                'created_at': status_data['created_at'],
                'updated_at': status_data['updated_at']
            })
        
        # Sort by creation time (newest first)
        filtered_studies.sort(key=lambda x: x['created_at'], reverse=True)
        
        return jsonify({
            'studies': filtered_studies,
            'total_count': len(filtered_studies),
            'filters_applied': {
                'patient_id': patient_id_filter,
                'status': status_filter,
                'modality': modality_filter
            }
        }), 200
        
    except Exception as e:
        logger.error(f"List studies error: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve studies',
            'message': 'An unexpected error occurred',
            'details': str(e)
        }), 500


@image_bp.route('/info/<study_id>', methods=['GET'])
def get_image_info(study_id: str):
    """
    Get detailed information about an image study.
    
    Args:
        study_id: Study identifier
        
    Returns:
        JSON response with detailed image information
    """
    try:
        if study_id not in processing_status:
            return jsonify({
                'error': 'Study not found',
                'message': f'No study found with ID: {study_id}'
            }), 404
        
        status_data = processing_status[study_id]
        study_info = status_data['study_info']
        
        # Get additional image information if file exists
        settings = get_settings()
        upload_dir = settings.monai.data_directory / "uploads" / study_id
        image_files = list(upload_dir.glob("*"))
        
        detailed_info = {
            'study_id': study_id,
            'basic_info': study_info,
            'processing_status': {
                'status': status_data['status'],
                'progress': status_data['progress'],
                'message': status_data['message'],
                'created_at': status_data['created_at'],
                'updated_at': status_data['updated_at']
            }
        }
        
        if image_files:
            try:
                image_file = image_files[0]
                file_info = image_handler.get_image_info(image_file)
                detailed_info['file_details'] = file_info
            except Exception as e:
                logger.warning(f"Could not get detailed file info for {study_id}: {str(e)}")
        
        # Add preprocessing metadata if available
        if 'preprocessing_metadata' in status_data:
            detailed_info['preprocessing_metadata'] = status_data['preprocessing_metadata']
        
        return jsonify(detailed_info), 200
        
    except Exception as e:
        logger.error(f"Get image info error: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve image information',
            'message': 'An unexpected error occurred',
            'details': str(e)
        }), 500