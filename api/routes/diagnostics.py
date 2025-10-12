"""
Diagnostic results and annotation API endpoints.
Implements endpoints for retrieving diagnostic results, visualizations, MONAI Label annotation interface, and longitudinal tracking.
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import json
import numpy as np
import base64
import io

from flask import Blueprint, request, jsonify, send_file
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from src.services.ml_inference.inference_engine import InferenceEngine
from src.services.explainability.explainability_service import ExplainabilityService
from src.services.monai_label.annotation_manager import AnnotationManager
from src.services.monai_label.active_learning_engine import ActiveLearningEngine
from src.models.diagnostics import DiagnosticResult, ModelMetrics, create_diagnostic_result
from src.models.patient import PatientRecord, Annotation
from src.config.settings import get_settings
from src.utils.logging_config import get_logger

logger = get_logger(__name__)
diagnostics_bp = Blueprint('diagnostics', __name__)

# Global instances (lazy initialization)
inference_engine = None
explainability_service = None
annotation_manager = None
active_learning_engine = None

# In-memory storage for diagnostic results (in production, use database)
diagnostic_results = {}
longitudinal_data = {}


def get_inference_engine():
    """Get or create inference engine instance."""
    global inference_engine
    if inference_engine is None:
        from src.services.ml_inference.swin_unetr_model import ModelManager
        model_manager = ModelManager()
        inference_engine = InferenceEngine(model_manager)
    return inference_engine


def get_explainability_service():
    """Get or create explainability service instance."""
    global explainability_service
    if explainability_service is None:
        explainability_service = ExplainabilityService()
    return explainability_service


def get_annotation_manager():
    """Get or create annotation manager instance."""
    global annotation_manager
    if annotation_manager is None:
        annotation_manager = AnnotationManager()
    return annotation_manager


def get_active_learning_engine():
    """Get or create active learning engine instance."""
    global active_learning_engine
    if active_learning_engine is None:
        active_learning_engine = ActiveLearningEngine()
    return active_learning_engine


def encode_array_to_base64(array: np.ndarray) -> str:
    """Encode numpy array to base64 string for JSON serialization."""
    buffer = io.BytesIO()
    np.save(buffer, array)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')


def decode_base64_to_array(encoded_str: str) -> np.ndarray:
    """Decode base64 string back to numpy array."""
    buffer = io.BytesIO(base64.b64decode(encoded_str))
    return np.load(buffer)


@diagnostics_bp.route('/predict', methods=['POST'])
def run_diagnostic_prediction():
    """
    Run diagnostic prediction on processed imaging and wearable data.
    
    Expected JSON body:
    - patient_id: Patient identifier
    - study_ids: List of imaging study IDs
    - wearable_session_ids: Optional list of wearable session IDs
    - prediction_options: Optional prediction configuration
    
    Returns:
        JSON response with diagnostic results
    """
    try:
        if not request.is_json:
            return jsonify({
                'error': 'Invalid content type',
                'message': 'Request must contain JSON data'
            }), 400
        
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['patient_id', 'study_ids']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'error': 'Missing required field',
                    'message': f'Field "{field}" is required'
                }), 400
        
        patient_id = data['patient_id']
        study_ids = data['study_ids']
        wearable_session_ids = data.get('wearable_session_ids', [])
        prediction_options = data.get('prediction_options', {})
        
        if not isinstance(study_ids, list) or not study_ids:
            return jsonify({
                'error': 'Invalid study IDs',
                'message': 'At least one study ID is required'
            }), 400
        
        # TODO: In a real implementation, load actual processed data
        # For now, create synthetic data for demonstration
        logger.info(f"Running diagnostic prediction for patient {patient_id}")
        
        # Generate synthetic prediction results
        try:
            # Create synthetic segmentation mask and probabilities
            segmentation_mask = np.random.randint(0, 3, size=(96, 96, 96), dtype=np.uint8)
            class_probabilities = {
                'healthy': 0.2,
                'mild_cognitive_impairment': 0.3,
                'alzheimers': 0.5
            }
            
            # Create synthetic metrics
            metrics = ModelMetrics(
                dice_score=0.85,
                hausdorff_distance=2.3,
                auc_score=0.92,
                accuracy=0.88,
                precision=0.87,
                recall=0.89,
                f1_score=0.88
            )
            
            # Determine modalities used
            modalities_used = ['MRI']  # Default, would be determined from actual study data
            if wearable_session_ids:
                modalities_used.extend(['EEG', 'HeartRate'])  # Example wearable modalities
            
            # Create diagnostic result
            diagnostic_result = create_diagnostic_result(
                patient_id=patient_id,
                study_ids=study_ids,
                segmentation_mask=segmentation_mask,
                class_probabilities=class_probabilities,
                metrics=metrics,
                modalities_used=modalities_used,
                wearable_data_included=bool(wearable_session_ids),
                processing_time_seconds=2.5
            )
            
            # Generate result ID
            result_id = f"DIAG_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Store result
            diagnostic_results[result_id] = {
                'result': diagnostic_result,
                'created_at': datetime.now().isoformat(),
                'study_ids': study_ids,
                'wearable_session_ids': wearable_session_ids
            }
            
            logger.info(f"Diagnostic prediction completed: {result_id}")
            
            # Prepare response (encode arrays for JSON)
            response_data = {
                'message': 'Diagnostic prediction completed successfully',
                'result_id': result_id,
                'patient_id': patient_id,
                'prediction_summary': diagnostic_result.get_summary(),
                'requires_manual_review': diagnostic_result.requires_manual_review(),
                'result_url': f'/api/v1/diagnostics/results/{result_id}',
                'visualization_url': f'/api/v1/diagnostics/visualizations/{result_id}'
            }
            
            return jsonify(response_data), 201
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return jsonify({
                'error': 'Prediction failed',
                'message': f'Failed to run diagnostic prediction: {str(e)}'
            }), 500
        
    except Exception as e:
        logger.error(f"Diagnostic prediction error: {str(e)}")
        return jsonify({
            'error': 'Prediction request failed',
            'message': 'An unexpected error occurred',
            'details': str(e)
        }), 500


@diagnostics_bp.route('/results/<result_id>', methods=['GET'])
def get_diagnostic_result(result_id: str):
    """
    Get detailed diagnostic result by ID.
    
    Args:
        result_id: Diagnostic result identifier
        
    Returns:
        JSON response with diagnostic result details
    """
    try:
        if result_id not in diagnostic_results:
            return jsonify({
                'error': 'Result not found',
                'message': f'No diagnostic result found with ID: {result_id}'
            }), 404
        
        result_data = diagnostic_results[result_id]
        diagnostic_result = result_data['result']
        
        # Prepare detailed response
        response_data = {
            'result_id': result_id,
            'patient_id': diagnostic_result.patient_id,
            'timestamp': diagnostic_result.timestamp.isoformat(),
            'created_at': result_data['created_at'],
            'study_ids': diagnostic_result.study_ids,
            'wearable_session_ids': result_data.get('wearable_session_ids', []),
            
            # Classification results
            'classification': {
                'predicted_class': diagnostic_result.classification_result.predicted_class,
                'class_probabilities': diagnostic_result.classification_result.class_probabilities,
                'confidence_score': diagnostic_result.classification_result.confidence_score,
                'disease_stage': diagnostic_result.classification_result.disease_stage.value if diagnostic_result.classification_result.disease_stage else None
            },
            
            # Segmentation info (shape only, not full data)
            'segmentation': {
                'mask_shape': diagnostic_result.segmentation_result.segmentation_mask.shape,
                'class_count': len(diagnostic_result.segmentation_result.class_probabilities)
            },
            
            # Model metrics
            'metrics': {
                'dice_score': diagnostic_result.metrics.dice_score,
                'hausdorff_distance': diagnostic_result.metrics.hausdorff_distance,
                'auc_score': diagnostic_result.metrics.auc_score,
                'accuracy': diagnostic_result.metrics.accuracy,
                'precision': diagnostic_result.metrics.precision,
                'recall': diagnostic_result.metrics.recall,
                'f1_score': diagnostic_result.metrics.f1_score,
                'overall_performance': diagnostic_result.metrics.get_overall_performance_score()
            },
            
            # Clinical information
            'clinical': {
                'diagnostic_confidence': diagnostic_result.diagnostic_confidence.value,
                'clinical_recommendations': diagnostic_result.clinical_recommendations,
                'follow_up_required': diagnostic_result.follow_up_required,
                'requires_manual_review': diagnostic_result.requires_manual_review()
            },
            
            # Processing metadata
            'processing': {
                'modalities_used': diagnostic_result.modalities_used,
                'wearable_data_included': diagnostic_result.wearable_data_included,
                'model_version': diagnostic_result.model_version,
                'processing_time_seconds': diagnostic_result.processing_time_seconds,
                'gpu_memory_used_mb': diagnostic_result.gpu_memory_used_mb
            }
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Get diagnostic result error: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve diagnostic result',
            'message': 'An unexpected error occurred',
            'details': str(e)
        }), 500


@diagnostics_bp.route('/visualizations/<result_id>', methods=['GET'])
def get_diagnostic_visualizations(result_id: str):
    """
    Get diagnostic visualizations including explainability maps.
    
    Args:
        result_id: Diagnostic result identifier
        
    Query parameters:
    - format: Response format ('json' or 'image')
    - visualization_type: Type of visualization ('gradcam', 'integrated_gradients', 'segmentation')
    
    Returns:
        JSON response with visualization data or image file
    """
    try:
        if result_id not in diagnostic_results:
            return jsonify({
                'error': 'Result not found',
                'message': f'No diagnostic result found with ID: {result_id}'
            }), 404
        
        result_data = diagnostic_results[result_id]
        diagnostic_result = result_data['result']
        
        format_type = request.args.get('format', 'json')
        viz_type = request.args.get('visualization_type', 'all')
        
        if format_type == 'json':
            # Return visualization data as JSON
            visualizations = {}
            
            # Generate explainability visualizations if not already present
            if diagnostic_result.explainability_maps is None:
                try:
                    # Generate synthetic explainability data for demonstration
                    grad_cam_maps = {
                        'layer1': np.random.rand(96, 96, 96).astype(np.float32),
                        'layer2': np.random.rand(96, 96, 96).astype(np.float32)
                    }
                    
                    integrated_gradients = np.random.rand(96, 96, 96).astype(np.float32)
                    
                    # Encode arrays for JSON
                    if viz_type in ['gradcam', 'all']:
                        visualizations['grad_cam'] = {
                            layer: encode_array_to_base64(cam_map) 
                            for layer, cam_map in grad_cam_maps.items()
                        }
                    
                    if viz_type in ['integrated_gradients', 'all']:
                        visualizations['integrated_gradients'] = encode_array_to_base64(integrated_gradients)
                    
                except Exception as e:
                    logger.error(f"Failed to generate explainability visualizations: {str(e)}")
            
            # Add segmentation visualization
            if viz_type in ['segmentation', 'all']:
                visualizations['segmentation'] = {
                    'mask_shape': diagnostic_result.segmentation_result.segmentation_mask.shape,
                    'mask_data': encode_array_to_base64(diagnostic_result.segmentation_result.segmentation_mask)
                }
            
            return jsonify({
                'result_id': result_id,
                'visualizations': visualizations,
                'generated_at': datetime.now().isoformat()
            }), 200
            
        elif format_type == 'image':
            # Generate and return image visualization
            try:
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                fig.suptitle(f'Diagnostic Visualizations - {result_id}')
                
                # Plot segmentation mask (middle slice)
                mask = diagnostic_result.segmentation_result.segmentation_mask
                middle_slice = mask.shape[2] // 2
                axes[0, 0].imshow(mask[:, :, middle_slice], cmap='viridis')
                axes[0, 0].set_title('Segmentation Mask')
                axes[0, 0].axis('off')
                
                # Plot class probabilities
                classes = list(diagnostic_result.classification_result.class_probabilities.keys())
                probs = list(diagnostic_result.classification_result.class_probabilities.values())
                axes[0, 1].bar(classes, probs)
                axes[0, 1].set_title('Classification Probabilities')
                axes[0, 1].set_ylabel('Probability')
                
                # Plot synthetic Grad-CAM
                grad_cam = np.random.rand(96, 96)
                axes[1, 0].imshow(grad_cam, cmap='hot')
                axes[1, 0].set_title('Grad-CAM Visualization')
                axes[1, 0].axis('off')
                
                # Plot metrics
                metrics_names = ['Dice', 'AUC', 'Accuracy', 'F1']
                metrics_values = [
                    diagnostic_result.metrics.dice_score,
                    diagnostic_result.metrics.auc_score,
                    diagnostic_result.metrics.accuracy or 0,
                    diagnostic_result.metrics.f1_score or 0
                ]
                axes[1, 1].bar(metrics_names, metrics_values)
                axes[1, 1].set_title('Model Metrics')
                axes[1, 1].set_ylabel('Score')
                axes[1, 1].set_ylim(0, 1)
                
                plt.tight_layout()
                
                # Save to buffer
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
                img_buffer.seek(0)
                plt.close()
                
                return send_file(
                    img_buffer,
                    mimetype='image/png',
                    as_attachment=True,
                    download_name=f'diagnostic_visualization_{result_id}.png'
                )
                
            except Exception as e:
                logger.error(f"Failed to generate image visualization: {str(e)}")
                return jsonify({
                    'error': 'Visualization generation failed',
                    'message': f'Failed to generate image: {str(e)}'
                }), 500
        
        else:
            return jsonify({
                'error': 'Invalid format',
                'message': 'Format must be "json" or "image"'
            }), 400
        
    except Exception as e:
        logger.error(f"Get visualizations error: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve visualizations',
            'message': 'An unexpected error occurred',
            'details': str(e)
        }), 500


@diagnostics_bp.route('/annotations', methods=['GET'])
def list_annotations():
    """
    List available annotations with filtering options.
    
    Query parameters:
    - patient_id: Filter by patient ID
    - annotator_id: Filter by annotator ID
    - annotation_type: Filter by annotation type
    - validation_status: Filter by validation status
    
    Returns:
        JSON response with list of annotations
    """
    try:
        # Get query parameters
        patient_id_filter = request.args.get('patient_id')
        annotator_id_filter = request.args.get('annotator_id')
        annotation_type_filter = request.args.get('annotation_type')
        validation_status_filter = request.args.get('validation_status')
        
        # Get annotations from annotation manager
        annotations = get_annotation_manager().list_annotations(
            patient_id=patient_id_filter,
            annotator_id=annotator_id_filter,
            annotation_type=annotation_type_filter,
            validation_status=validation_status_filter
        )
        
        # Format response
        annotation_list = []
        for annotation in annotations:
            annotation_list.append({
                'annotation_id': annotation.annotation_id,
                'annotator_id': annotation.annotator_id,
                'annotation_type': annotation.annotation_type,
                'confidence_score': annotation.confidence_score,
                'creation_timestamp': annotation.creation_timestamp.isoformat(),
                'validation_status': annotation.validation_status
            })
        
        return jsonify({
            'annotations': annotation_list,
            'total_count': len(annotation_list),
            'filters_applied': {
                'patient_id': patient_id_filter,
                'annotator_id': annotator_id_filter,
                'annotation_type': annotation_type_filter,
                'validation_status': validation_status_filter
            }
        }), 200
        
    except Exception as e:
        logger.error(f"List annotations error: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve annotations',
            'message': 'An unexpected error occurred',
            'details': str(e)
        }), 500


@diagnostics_bp.route('/annotations/active-learning/suggestions', methods=['GET'])
def get_active_learning_suggestions():
    """
    Get active learning suggestions for annotation.
    
    Query parameters:
    - count: Number of suggestions to return (default: 10)
    - strategy: Active learning strategy ('uncertainty', 'diversity', 'hybrid')
    
    Returns:
        JSON response with suggested cases for annotation
    """
    try:
        count = int(request.args.get('count', 10))
        strategy = request.args.get('strategy', 'uncertainty')
        
        if count <= 0 or count > 100:
            return jsonify({
                'error': 'Invalid count',
                'message': 'Count must be between 1 and 100'
            }), 400
        
        valid_strategies = ['uncertainty', 'diversity', 'hybrid']
        if strategy not in valid_strategies:
            return jsonify({
                'error': 'Invalid strategy',
                'message': f'Strategy must be one of: {", ".join(valid_strategies)}'
            }), 400
        
        # Get suggestions from active learning engine
        suggestions = get_active_learning_engine().get_annotation_suggestions(
            count=count,
            strategy=strategy
        )
        
        # Format response
        suggestion_list = []
        for suggestion in suggestions:
            suggestion_list.append({
                'case_id': suggestion['case_id'],
                'patient_id': suggestion['patient_id'],
                'study_ids': suggestion['study_ids'],
                'uncertainty_score': suggestion['uncertainty_score'],
                'priority_score': suggestion['priority_score'],
                'suggested_annotation_type': suggestion['annotation_type'],
                'reason': suggestion['reason']
            })
        
        return jsonify({
            'suggestions': suggestion_list,
            'count': len(suggestion_list),
            'strategy': strategy,
            'generated_at': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Get active learning suggestions error: {str(e)}")
        return jsonify({
            'error': 'Failed to get suggestions',
            'message': 'An unexpected error occurred',
            'details': str(e)
        }), 500


@diagnostics_bp.route('/longitudinal/<patient_id>', methods=['GET'])
def get_longitudinal_tracking(patient_id: str):
    """
    Get longitudinal tracking data for a patient.
    
    Args:
        patient_id: Patient identifier
        
    Query parameters:
    - start_date: Start date for tracking data (ISO format)
    - end_date: End date for tracking data (ISO format)
    - metrics: Comma-separated list of metrics to include
    
    Returns:
        JSON response with longitudinal tracking data
    """
    try:
        # Get query parameters
        start_date_str = request.args.get('start_date')
        end_date_str = request.args.get('end_date')
        metrics_str = request.args.get('metrics')
        
        # Parse dates if provided
        start_date = None
        end_date = None
        
        if start_date_str:
            try:
                start_date = datetime.fromisoformat(start_date_str.replace('Z', '+00:00'))
            except ValueError:
                return jsonify({
                    'error': 'Invalid start date format',
                    'message': 'Start date must be in ISO format'
                }), 400
        
        if end_date_str:
            try:
                end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
            except ValueError:
                return jsonify({
                    'error': 'Invalid end date format',
                    'message': 'End date must be in ISO format'
                }), 400
        
        # Parse metrics filter
        requested_metrics = None
        if metrics_str:
            requested_metrics = [m.strip() for m in metrics_str.split(',')]
        
        # Get longitudinal data for patient
        # In a real implementation, this would query the database
        # For now, generate synthetic longitudinal data
        
        if patient_id not in longitudinal_data:
            # Generate synthetic longitudinal data
            baseline_date = datetime.now() - timedelta(days=365)
            follow_up_dates = [
                baseline_date + timedelta(days=90),
                baseline_date + timedelta(days=180),
                baseline_date + timedelta(days=270),
                baseline_date + timedelta(days=360)
            ]
            
            longitudinal_data[patient_id] = {
                'patient_id': patient_id,
                'baseline_date': baseline_date,
                'follow_up_dates': follow_up_dates,
                'metrics_timeline': {
                    'cognitive_score': [85, 82, 78, 75, 72],
                    'brain_volume': [1200, 1195, 1188, 1180, 1175],
                    'disease_progression': [0.1, 0.2, 0.35, 0.5, 0.65],
                    'medication_response': [0.8, 0.75, 0.7, 0.65, 0.6]
                },
                'clinical_assessments': {
                    date.isoformat(): {
                        'mmse_score': max(10, 30 - i * 3),
                        'cdr_score': min(3, i * 0.5),
                        'clinical_notes': f'Assessment {i + 1}'
                    }
                    for i, date in enumerate([baseline_date] + follow_up_dates)
                }
            }
        
        patient_longitudinal = longitudinal_data[patient_id]
        
        # Apply date filters
        filtered_dates = [patient_longitudinal['baseline_date']] + patient_longitudinal['follow_up_dates']
        
        if start_date:
            filtered_dates = [d for d in filtered_dates if d >= start_date]
        if end_date:
            filtered_dates = [d for d in filtered_dates if d <= end_date]
        
        # Apply metrics filter
        metrics_timeline = patient_longitudinal['metrics_timeline']
        if requested_metrics:
            metrics_timeline = {
                metric: values for metric, values in metrics_timeline.items()
                if metric in requested_metrics
            }
        
        # Format response
        response_data = {
            'patient_id': patient_id,
            'baseline_date': patient_longitudinal['baseline_date'].isoformat(),
            'follow_up_dates': [d.isoformat() for d in patient_longitudinal['follow_up_dates']],
            'filtered_dates': [d.isoformat() for d in filtered_dates],
            'metrics_timeline': metrics_timeline,
            'clinical_assessments': patient_longitudinal['clinical_assessments'],
            'summary': {
                'total_assessments': len(filtered_dates),
                'tracking_duration_days': (filtered_dates[-1] - filtered_dates[0]).days if len(filtered_dates) > 1 else 0,
                'available_metrics': list(metrics_timeline.keys())
            }
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Get longitudinal tracking error: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve longitudinal data',
            'message': 'An unexpected error occurred',
            'details': str(e)
        }), 500


@diagnostics_bp.route('/results', methods=['GET'])
def list_diagnostic_results():
    """
    List diagnostic results with filtering options.
    
    Query parameters:
    - patient_id: Filter by patient ID
    - start_date: Filter results after this date
    - end_date: Filter results before this date
    - confidence_threshold: Minimum confidence threshold
    
    Returns:
        JSON response with list of diagnostic results
    """
    try:
        # Get query parameters
        patient_id_filter = request.args.get('patient_id')
        start_date_str = request.args.get('start_date')
        end_date_str = request.args.get('end_date')
        confidence_threshold = request.args.get('confidence_threshold', type=float)
        
        # Parse dates
        start_date = None
        end_date = None
        
        if start_date_str:
            try:
                start_date = datetime.fromisoformat(start_date_str.replace('Z', '+00:00'))
            except ValueError:
                return jsonify({
                    'error': 'Invalid start date format',
                    'message': 'Start date must be in ISO format'
                }), 400
        
        if end_date_str:
            try:
                end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
            except ValueError:
                return jsonify({
                    'error': 'Invalid end date format',
                    'message': 'End date must be in ISO format'
                }), 400
        
        # Filter results
        filtered_results = []
        for result_id, result_data in diagnostic_results.items():
            diagnostic_result = result_data['result']
            
            # Apply filters
            if patient_id_filter and diagnostic_result.patient_id != patient_id_filter:
                continue
            
            if start_date and diagnostic_result.timestamp < start_date:
                continue
            
            if end_date and diagnostic_result.timestamp > end_date:
                continue
            
            if confidence_threshold and diagnostic_result.classification_result.confidence_score < confidence_threshold:
                continue
            
            filtered_results.append({
                'result_id': result_id,
                'patient_id': diagnostic_result.patient_id,
                'timestamp': diagnostic_result.timestamp.isoformat(),
                'predicted_class': diagnostic_result.classification_result.predicted_class,
                'confidence_score': diagnostic_result.classification_result.confidence_score,
                'diagnostic_confidence': diagnostic_result.diagnostic_confidence.value,
                'requires_manual_review': diagnostic_result.requires_manual_review(),
                'study_ids': diagnostic_result.study_ids,
                'modalities_used': diagnostic_result.modalities_used
            })
        
        # Sort by timestamp (newest first)
        filtered_results.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return jsonify({
            'results': filtered_results,
            'total_count': len(filtered_results),
            'filters_applied': {
                'patient_id': patient_id_filter,
                'start_date': start_date_str,
                'end_date': end_date_str,
                'confidence_threshold': confidence_threshold
            }
        }), 200
        
    except Exception as e:
        logger.error(f"List diagnostic results error: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve diagnostic results',
            'message': 'An unexpected error occurred',
            'details': str(e)
        }), 500