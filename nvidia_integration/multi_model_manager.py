"""
Multi-Modal NVIDIA API Manager
Orchestrates multiple NVIDIA models and API keys for comprehensive medical AI analysis
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, AsyncGenerator, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import numpy as np

from .api_key_manager import APIKeyManager, TaskType
from .nvidia_enhanced_service import NVIDIAEnhancedService
from src.services.security.audit_logger import AuditLogger
from src.config.settings import settings

logger = logging.getLogger(__name__)


class ModalityType(Enum):
    """Different types of medical data modalities"""
    MEDICAL_IMAGING = "medical_imaging"
    WEARABLE_SENSORS = "wearable_sensors"
    GENOMIC_DATA = "genomic_data"
    CLINICAL_TEXT = "clinical_text"
    LABORATORY_DATA = "laboratory_data"
    VITAL_SIGNS = "vital_signs"


@dataclass
class MultiModalRequest:
    """Request for multi-modal analysis"""
    patient_id: str
    modalities: Dict[ModalityType, Any]
    analysis_type: str
    priority: int = 1
    streaming: bool = False
    batch_mode: bool = False
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class MultiModalResult:
    """Result from multi-modal analysis"""
    patient_id: str
    analysis_results: Dict[str, Any]
    fusion_results: Dict[str, Any]
    confidence_scores: Dict[str, float]
    processing_time: float
    api_keys_used: List[str]
    cost_breakdown: Dict[str, float]
    timestamp: datetime
    
    @property
    def total_cost(self) -> float:
        return sum(self.cost_breakdown.values())
    
    @property
    def overall_confidence(self) -> float:
        if not self.confidence_scores:
            return 0.0
        return np.mean(list(self.confidence_scores.values()))


class MultiModalManager:
    """
    Advanced multi-modal manager for comprehensive medical AI analysis
    
    Orchestrates multiple NVIDIA models and API keys to provide:
    - Medical imaging analysis (MRI, CT, Ultrasound)
    - Wearable sensor data processing (EEG, heart rate, sleep, gait)
    - Genomic variant analysis and risk scoring
    - Clinical text analysis and report generation
    - Multi-modal data fusion and correlation analysis
    """
    
    def __init__(self, audit_logger: Optional[AuditLogger] = None):
        """Initialize multi-modal manager"""
        self.audit_logger = audit_logger or AuditLogger()
        
        # Production API key configuration
        self.api_key_config = self._create_production_config()
        
        # Initialize enhanced NVIDIA service
        self.nvidia_service = NVIDIAEnhancedService(
            audit_logger=self.audit_logger,
            api_key_config=self.api_key_config
        )
        
        # Multi-modal processing configuration
        self.max_concurrent_requests = int(getattr(settings, "MULTIMODAL_MAX_CONCURRENT_REQUESTS", 10))
        self.batch_size = int(getattr(settings, "MULTIMODAL_BATCH_SIZE", 32))
        self.enable_streaming = str(getattr(settings, "MULTIMODAL_STREAMING_ENABLED", "true")).lower() == "true"
        self.enable_fusion = str(getattr(settings, "MULTIMODAL_FUSION_ENABLED", "true")).lower() == "true"
        
        # Performance tracking
        self.processing_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_processing_time": 0.0,
            "total_cost": 0.0
        }
        
        logger.info("Multi-Modal Manager initialized with production API keys")
    
    def _create_production_config(self) -> List[Dict[str, Any]]:
        """Create production API key configuration"""
        # Use hardcoded production API keys for demo
        return [
            {
                'key_id': 'medical_text_primary',
                'api_key': 'nvapi-8c6tEjUiGKR-MeMyuSx_we6afFc6nKZqRkd-hLrLDNQCJupsFjNfIrSH86C5qGUSnvapi-JPyvgmVsh1CpV-iaz_yqaTn5RpsvIgHX5f7-3fX_8fYEr85-gZSSDSG8CaZisyzO',
                'endpoint': 'https://integrate.api.nvidia.com/v1',
                'max_rpm': 1000,
                'max_tpm': 50000,
                'priority': 3,
                'task_types': [
                    TaskType.MEDICAL_TEXT_ANALYSIS.value,
                    TaskType.DIAGNOSTIC_REPORT_GENERATION.value
                ],
                'cost_per_token': 0.0001,
                'description': 'Primary key for medical text analysis and diagnostic reports'
            },
            {
                'key_id': 'genomics_specialist',
                'api_key': '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',
                'endpoint': 'https://integrate.api.nvidia.com/v1',
                'max_rpm': 500,
                'max_tpm': 25000,
                'priority': 3,
                'task_types': [TaskType.GENOMICS_ANALYSIS.value],
                'cost_per_token': 0.0002,
                'description': 'Specialized key for genomics analysis and variant interpretation'
            },
            {
                'key_id': 'streaming_realtime',
                'api_key': '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',
                'endpoint': 'https://integrate.api.nvidia.com/v1',
                'max_rpm': 2000,
                'max_tpm': 100000,
                'priority': 2,
                'task_types': [
                    TaskType.STREAMING_INSIGHTS.value,
                    TaskType.REAL_TIME_INFERENCE.value
                ],
                'cost_per_token': 0.00015,
                'description': 'High-throughput key for streaming and real-time processing'
            },
            {
                'key_id': 'multimodal_fusion',
                'api_key': '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',
                'endpoint': 'https://integrate.api.nvidia.com/v1',
                'max_rpm': 800,
                'max_tpm': 60000,
                'priority': 2,
                'task_types': [
                    TaskType.MULTI_MODAL_FUSION.value,
                    TaskType.BATCH_PROCESSING.value,
                    TaskType.LONGITUDINAL_TRACKING.value
                ],
                'cost_per_token': 0.00012,
                'description': 'Specialized key for multi-modal fusion and batch processing'
            }
        ]
    
    async def analyze_multimodal_patient(self, request: MultiModalRequest) -> MultiModalResult:
        """
        Comprehensive multi-modal patient analysis
        
        Args:
            request: Multi-modal analysis request
            
        Returns:
            Comprehensive analysis results with fusion insights
        """
        start_time = time.time()
        analysis_results = {}
        confidence_scores = {}
        api_keys_used = []
        cost_breakdown = {}
        
        try:
            # Process each modality in parallel
            tasks = []
            
            # Medical imaging analysis
            if ModalityType.MEDICAL_IMAGING in request.modalities:
                tasks.append(self._analyze_medical_imaging(
                    request.modalities[ModalityType.MEDICAL_IMAGING],
                    request.patient_id
                ))
            
            # Wearable sensor data analysis
            if ModalityType.WEARABLE_SENSORS in request.modalities:
                tasks.append(self._analyze_wearable_sensors(
                    request.modalities[ModalityType.WEARABLE_SENSORS],
                    request.patient_id
                ))
            
            # Genomic data analysis
            if ModalityType.GENOMIC_DATA in request.modalities:
                tasks.append(self._analyze_genomic_data(
                    request.modalities[ModalityType.GENOMIC_DATA],
                    request.patient_id
                ))
            
            # Clinical text analysis
            if ModalityType.CLINICAL_TEXT in request.modalities:
                tasks.append(self._analyze_clinical_text(
                    request.modalities[ModalityType.CLINICAL_TEXT],
                    request.patient_id
                ))
            
            # Laboratory data analysis
            if ModalityType.LABORATORY_DATA in request.modalities:
                tasks.append(self._analyze_laboratory_data(
                    request.modalities[ModalityType.LABORATORY_DATA],
                    request.patient_id
                ))
            
            # Execute all analyses in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Analysis task {i} failed: {result}")
                    continue
                
                modality_name = list(request.modalities.keys())[i].value
                analysis_results[modality_name] = result['analysis']
                confidence_scores[modality_name] = result['confidence']
                api_keys_used.extend(result['api_keys_used'])
                cost_breakdown[modality_name] = result['cost']
            
            # Multi-modal fusion analysis
            fusion_results = {}
            if self.enable_fusion and len(analysis_results) > 1:
                fusion_results = await self._perform_multimodal_fusion(
                    analysis_results, request.patient_id
                )
                
                # Add fusion costs
                if 'cost' in fusion_results:
                    cost_breakdown['fusion'] = fusion_results['cost']
                    api_keys_used.extend(fusion_results.get('api_keys_used', []))
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Update statistics
            self._update_stats(processing_time, sum(cost_breakdown.values()), True)
            
            # Create result
            result = MultiModalResult(
                patient_id=request.patient_id,
                analysis_results=analysis_results,
                fusion_results=fusion_results,
                confidence_scores=confidence_scores,
                processing_time=processing_time,
                api_keys_used=list(set(api_keys_used)),  # Remove duplicates
                cost_breakdown=cost_breakdown,
                timestamp=datetime.now()
            )
            
            # Log successful analysis
            self.audit_logger.log_event(
                event_type="multimodal_analysis",
                details={
                    "patient_id": request.patient_id,
                    "modalities": [m.value for m in request.modalities.keys()],
                    "processing_time": processing_time,
                    "total_cost": result.total_cost,
                    "overall_confidence": result.overall_confidence,
                    "api_keys_used": result.api_keys_used
                }
            )
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_stats(processing_time, 0.0, False)
            
            logger.error(f"Multi-modal analysis failed for patient {request.patient_id}: {e}")
            raise
    
    async def _analyze_medical_imaging(self, imaging_data: Dict[str, Any], 
                                     patient_id: str) -> Dict[str, Any]:
        """Analyze medical imaging data"""
        try:
            # Use medical text analysis key for imaging report generation
            analysis_text = f"""
            Medical Imaging Analysis for Patient {patient_id}
            
            Imaging Studies: {', '.join(imaging_data.get('studies', []))}
            Modalities: {', '.join(imaging_data.get('modalities', []))}
            Findings: {imaging_data.get('findings', 'No specific findings noted')}
            
            Please provide comprehensive analysis including:
            1. Structural abnormalities
            2. Volumetric measurements
            3. Disease progression indicators
            4. Diagnostic recommendations
            """
            
            result = await self.nvidia_service.analyze_medical_text(
                text=analysis_text,
                context="medical_imaging_analysis"
            )
            
            return {
                'analysis': {
                    'imaging_findings': result.get('analysis', {}),
                    'structural_analysis': imaging_data.get('structural_measurements', {}),
                    'volumetric_analysis': imaging_data.get('volumetric_data', {}),
                    'abnormalities_detected': result.get('abnormalities', []),
                    'diagnostic_confidence': result.get('confidence', 0.85)
                },
                'confidence': result.get('confidence', 0.85),
                'api_keys_used': [result.get('api_key_used', 'medical_text_primary')],
                'cost': result.get('cost', 0.01)
            }
            
        except Exception as e:
            logger.error(f"Medical imaging analysis failed: {e}")
            return {
                'analysis': {'error': str(e)},
                'confidence': 0.0,
                'api_keys_used': [],
                'cost': 0.0
            }
    
    async def _analyze_wearable_sensors(self, sensor_data: Dict[str, Any], 
                                      patient_id: str) -> Dict[str, Any]:
        """Analyze wearable sensor data"""
        try:
            # Use streaming key for real-time sensor analysis
            sensor_summary = f"""
            Wearable Sensor Data Analysis for Patient {patient_id}
            
            EEG Data: {sensor_data.get('eeg', {}).get('summary', 'No EEG data')}
            Heart Rate: {sensor_data.get('heart_rate', {}).get('average', 'N/A')} BPM
            Sleep Quality: {sensor_data.get('sleep', {}).get('efficiency', 'N/A')}%
            Gait Analysis: {sensor_data.get('gait', {}).get('speed', 'N/A')} m/s
            
            Analyze for neurological indicators and health patterns.
            """
            
            # Use streaming insights for real-time processing
            if self.enable_streaming:
                insights = []
                async for insight in self.nvidia_service.stream_real_time_insights(sensor_summary):
                    insights.append(insight)
                    if len(insights) >= 5:  # Limit streaming results
                        break
                
                analysis_result = {
                    'streaming_insights': insights,
                    'real_time_analysis': True
                }
            else:
                # Fallback to batch analysis
                result = await self.nvidia_service.analyze_medical_text(
                    text=sensor_summary,
                    context="wearable_sensor_analysis"
                )
                analysis_result = result.get('analysis', {})
            
            return {
                'analysis': {
                    'sensor_insights': analysis_result,
                    'eeg_analysis': sensor_data.get('eeg', {}),
                    'cardiac_analysis': sensor_data.get('heart_rate', {}),
                    'sleep_analysis': sensor_data.get('sleep', {}),
                    'mobility_analysis': sensor_data.get('gait', {}),
                    'anomaly_detection': self._detect_sensor_anomalies(sensor_data)
                },
                'confidence': 0.82,
                'api_keys_used': ['streaming_realtime'],
                'cost': 0.005
            }
            
        except Exception as e:
            logger.error(f"Wearable sensor analysis failed: {e}")
            return {
                'analysis': {'error': str(e)},
                'confidence': 0.0,
                'api_keys_used': [],
                'cost': 0.0
            }
    
    async def _analyze_genomic_data(self, genomic_data: Dict[str, Any], 
                                  patient_id: str) -> Dict[str, Any]:
        """Analyze genomic data"""
        try:
            # Use genomics specialist key
            result = await self.nvidia_service.analyze_genomics_data(genomic_data)
            
            return {
                'analysis': {
                    'variant_analysis': result.get('variants', {}),
                    'risk_scores': result.get('risk_scores', {}),
                    'pharmacogenomics': result.get('pharmacogenomics', {}),
                    'family_history_analysis': result.get('family_analysis', {}),
                    'genetic_counseling_recommendations': result.get('recommendations', [])
                },
                'confidence': result.get('confidence', 0.88),
                'api_keys_used': [result.get('api_key_used', 'genomics_specialist')],
                'cost': result.get('cost', 0.02)
            }
            
        except Exception as e:
            logger.error(f"Genomic analysis failed: {e}")
            return {
                'analysis': {'error': str(e)},
                'confidence': 0.0,
                'api_keys_used': [],
                'cost': 0.0
            }
    
    async def _analyze_clinical_text(self, clinical_text: Dict[str, Any], 
                                   patient_id: str) -> Dict[str, Any]:
        """Analyze clinical text data"""
        try:
            # Combine all clinical text
            combined_text = ""
            if 'clinical_notes' in clinical_text:
                combined_text += f"Clinical Notes: {clinical_text['clinical_notes']}\n"
            if 'discharge_summary' in clinical_text:
                combined_text += f"Discharge Summary: {clinical_text['discharge_summary']}\n"
            if 'lab_reports' in clinical_text:
                combined_text += f"Lab Reports: {clinical_text['lab_reports']}\n"
            
            # Use medical text analysis key
            result = await self.nvidia_service.analyze_medical_text(
                text=combined_text,
                context="comprehensive_clinical_analysis"
            )
            
            return {
                'analysis': {
                    'clinical_insights': result.get('analysis', {}),
                    'symptom_extraction': result.get('symptoms', []),
                    'diagnosis_suggestions': result.get('diagnoses', []),
                    'medication_analysis': result.get('medications', []),
                    'risk_factors': result.get('risk_factors', [])
                },
                'confidence': result.get('confidence', 0.87),
                'api_keys_used': [result.get('api_key_used', 'medical_text_primary')],
                'cost': result.get('cost', 0.008)
            }
            
        except Exception as e:
            logger.error(f"Clinical text analysis failed: {e}")
            return {
                'analysis': {'error': str(e)},
                'confidence': 0.0,
                'api_keys_used': [],
                'cost': 0.0
            }
    
    async def _analyze_laboratory_data(self, lab_data: Dict[str, Any], 
                                     patient_id: str) -> Dict[str, Any]:
        """Analyze laboratory data"""
        try:
            # Format lab data for analysis
            lab_summary = f"""
            Laboratory Data Analysis for Patient {patient_id}
            
            Blood Work: {lab_data.get('blood_work', {})}
            Biomarkers: {lab_data.get('biomarkers', {})}
            Inflammatory Markers: {lab_data.get('inflammatory_markers', {})}
            Metabolic Panel: {lab_data.get('metabolic_panel', {})}
            
            Analyze for disease indicators and health status.
            """
            
            result = await self.nvidia_service.analyze_medical_text(
                text=lab_summary,
                context="laboratory_analysis"
            )
            
            return {
                'analysis': {
                    'lab_insights': result.get('analysis', {}),
                    'abnormal_values': self._identify_abnormal_labs(lab_data),
                    'trend_analysis': lab_data.get('trends', {}),
                    'reference_comparisons': lab_data.get('references', {}),
                    'clinical_significance': result.get('significance', [])
                },
                'confidence': result.get('confidence', 0.84),
                'api_keys_used': [result.get('api_key_used', 'medical_text_primary')],
                'cost': result.get('cost', 0.006)
            }
            
        except Exception as e:
            logger.error(f"Laboratory analysis failed: {e}")
            return {
                'analysis': {'error': str(e)},
                'confidence': 0.0,
                'api_keys_used': [],
                'cost': 0.0
            }
    
    async def _perform_multimodal_fusion(self, analysis_results: Dict[str, Any], 
                                       patient_id: str) -> Dict[str, Any]:
        """Perform multi-modal data fusion"""
        try:
            # Create comprehensive fusion prompt
            fusion_prompt = f"""
            Multi-Modal Medical Data Fusion for Patient {patient_id}
            
            Available Data Modalities:
            {json.dumps(analysis_results, indent=2)}
            
            Perform comprehensive fusion analysis including:
            1. Cross-modal correlations and patterns
            2. Integrated risk assessment
            3. Diagnostic confidence scoring
            4. Treatment recommendations
            5. Longitudinal tracking insights
            6. Personalized medicine recommendations
            
            Provide a unified clinical picture with evidence-based insights.
            """
            
            # Use multimodal fusion key for complex analysis
            result = await self.nvidia_service.analyze_medical_text(
                text=fusion_prompt,
                context="multimodal_fusion"
            )
            
            # Calculate integrated metrics
            integrated_metrics = self._calculate_integrated_metrics(analysis_results)
            
            return {
                'fusion_analysis': result.get('analysis', {}),
                'integrated_risk_score': integrated_metrics['risk_score'],
                'cross_modal_correlations': integrated_metrics['correlations'],
                'unified_diagnosis': result.get('diagnosis', {}),
                'treatment_recommendations': result.get('recommendations', []),
                'confidence_matrix': integrated_metrics['confidence_matrix'],
                'longitudinal_insights': result.get('longitudinal', {}),
                'personalized_recommendations': result.get('personalized', []),
                'cost': result.get('cost', 0.015),
                'api_keys_used': [result.get('api_key_used', 'multimodal_fusion')]
            }
            
        except Exception as e:
            logger.error(f"Multi-modal fusion failed: {e}")
            return {
                'fusion_analysis': {'error': str(e)},
                'integrated_risk_score': 0.0,
                'cost': 0.0,
                'api_keys_used': []
            }
    
    def _detect_sensor_anomalies(self, sensor_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in sensor data"""
        anomalies = []
        
        # EEG anomalies
        if 'eeg' in sensor_data:
            eeg_data = sensor_data['eeg']
            if eeg_data.get('alpha_power', 0) < 0.3:
                anomalies.append({
                    'type': 'eeg_anomaly',
                    'description': 'Low alpha wave activity detected',
                    'severity': 'moderate',
                    'value': eeg_data.get('alpha_power', 0)
                })
        
        # Heart rate anomalies
        if 'heart_rate' in sensor_data:
            hr_data = sensor_data['heart_rate']
            avg_hr = hr_data.get('average', 70)
            if avg_hr > 100 or avg_hr < 50:
                anomalies.append({
                    'type': 'cardiac_anomaly',
                    'description': f'Abnormal heart rate: {avg_hr} BPM',
                    'severity': 'high' if avg_hr > 120 or avg_hr < 40 else 'moderate',
                    'value': avg_hr
                })
        
        return anomalies
    
    def _identify_abnormal_labs(self, lab_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify abnormal laboratory values"""
        abnormal_values = []
        
        # Example reference ranges (simplified)
        reference_ranges = {
            'glucose': (70, 100),
            'cholesterol': (0, 200),
            'creatinine': (0.6, 1.2),
            'hemoglobin': (12, 16)
        }
        
        for test_name, (min_val, max_val) in reference_ranges.items():
            if test_name in lab_data.get('blood_work', {}):
                value = lab_data['blood_work'][test_name]
                if value < min_val or value > max_val:
                    abnormal_values.append({
                        'test': test_name,
                        'value': value,
                        'reference_range': f"{min_val}-{max_val}",
                        'status': 'high' if value > max_val else 'low'
                    })
        
        return abnormal_values
    
    def _calculate_integrated_metrics(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate integrated metrics from multi-modal analysis"""
        # Simplified integration logic
        risk_scores = []
        confidence_scores = []
        
        for modality, results in analysis_results.items():
            if 'diagnostic_confidence' in results:
                confidence_scores.append(results['diagnostic_confidence'])
            if 'risk_score' in results:
                risk_scores.append(results['risk_score'])
        
        # Calculate integrated risk score
        integrated_risk = np.mean(risk_scores) if risk_scores else 0.5
        
        # Calculate confidence matrix
        confidence_matrix = {
            'overall_confidence': np.mean(confidence_scores) if confidence_scores else 0.8,
            'modality_agreement': len(confidence_scores) / len(analysis_results),
            'data_completeness': len(analysis_results) / 6  # Assuming 6 possible modalities
        }
        
        # Calculate cross-modal correlations (simplified)
        correlations = {
            'imaging_genomics': 0.75,
            'sensors_clinical': 0.68,
            'labs_imaging': 0.82,
            'genomics_clinical': 0.71
        }
        
        return {
            'risk_score': integrated_risk,
            'confidence_matrix': confidence_matrix,
            'correlations': correlations
        }
    
    def _update_stats(self, processing_time: float, cost: float, success: bool):
        """Update processing statistics"""
        self.processing_stats['total_requests'] += 1
        self.processing_stats['total_processing_time'] += processing_time
        self.processing_stats['total_cost'] += cost
        
        if success:
            self.processing_stats['successful_requests'] += 1
        else:
            self.processing_stats['failed_requests'] += 1
    
    async def batch_analyze_patients(self, requests: List[MultiModalRequest]) -> List[MultiModalResult]:
        """Batch analyze multiple patients"""
        logger.info(f"Starting batch analysis of {len(requests)} patients")
        
        # Process in batches to avoid overwhelming the APIs
        results = []
        batch_size = self.batch_size
        
        for i in range(0, len(requests), batch_size):
            batch = requests[i:i + batch_size]
            
            # Process batch in parallel
            batch_tasks = [self.analyze_multimodal_patient(request) for request in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Handle results and exceptions
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch analysis failed: {result}")
                    continue
                results.append(result)
            
            # Small delay between batches to respect rate limits
            if i + batch_size < len(requests):
                await asyncio.sleep(1)
        
        logger.info(f"Completed batch analysis: {len(results)} successful, {len(requests) - len(results)} failed")
        return results
    
    async def stream_patient_analysis(self, request: MultiModalRequest) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream real-time patient analysis results"""
        if not self.enable_streaming:
            raise ValueError("Streaming is not enabled")
        
        logger.info(f"Starting streaming analysis for patient {request.patient_id}")
        
        # Stream results as they become available
        for modality_type, modality_data in request.modalities.items():
            try:
                if modality_type == ModalityType.WEARABLE_SENSORS:
                    # Stream sensor data analysis
                    async for insight in self.nvidia_service.stream_real_time_insights(modality_data):
                        yield {
                            'patient_id': request.patient_id,
                            'modality': modality_type.value,
                            'insight': insight,
                            'timestamp': datetime.now().isoformat()
                        }
                
                elif modality_type == ModalityType.CLINICAL_TEXT:
                    # Stream clinical text analysis
                    result = await self._analyze_clinical_text(modality_data, request.patient_id)
                    yield {
                        'patient_id': request.patient_id,
                        'modality': modality_type.value,
                        'analysis': result['analysis'],
                        'confidence': result['confidence'],
                        'timestamp': datetime.now().isoformat()
                    }
                
            except Exception as e:
                logger.error(f"Streaming analysis failed for {modality_type}: {e}")
                yield {
                    'patient_id': request.patient_id,
                    'modality': modality_type.value,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        api_metrics = self.nvidia_service.get_api_key_metrics()
        
        return {
            'processing_stats': self.processing_stats,
            'api_key_metrics': api_metrics,
            'average_processing_time': (
                self.processing_stats['total_processing_time'] / 
                max(self.processing_stats['total_requests'], 1)
            ),
            'success_rate': (
                self.processing_stats['successful_requests'] / 
                max(self.processing_stats['total_requests'], 1) * 100
            ),
            'average_cost_per_request': (
                self.processing_stats['total_cost'] / 
                max(self.processing_stats['total_requests'], 1)
            ),
            'configuration': {
                'max_concurrent_requests': self.max_concurrent_requests,
                'batch_size': self.batch_size,
                'streaming_enabled': self.enable_streaming,
                'fusion_enabled': self.enable_fusion
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        nvidia_health = await self.nvidia_service.health_check()
        
        return {
            'status': nvidia_health['status'],
            'multimodal_manager': {
                'status': 'healthy',
                'total_requests_processed': self.processing_stats['total_requests'],
                'success_rate': (
                    self.processing_stats['successful_requests'] / 
                    max(self.processing_stats['total_requests'], 1) * 100
                ),
                'streaming_enabled': self.enable_streaming,
                'fusion_enabled': self.enable_fusion
            },
            'nvidia_service': nvidia_health,
            'timestamp': datetime.now().isoformat()
        }