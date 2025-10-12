#!/usr/bin/env python3
"""
Multi-Modal NeuroDx System Demo
Demonstrates comprehensive medical AI analysis using multiple NVIDIA API keys
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our multi-modal system
from src.services.nvidia_integration.multi_model_manager import (
    MultiModalManager, MultiModalRequest, ModalityType
)
from src.services.security.audit_logger import AuditLogger


def print_header(title: str):
    """Print a formatted header"""
    print(f"\n{'='*80}")
    print(f"🧠 {title}")
    print(f"{'='*80}")


def print_results(results: Dict[str, Any], title: str):
    """Print analysis results in a formatted way"""
    print(f"\n📊 {title}")
    print("-" * 60)
    
    if isinstance(results, dict):
        for key, value in results.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            elif isinstance(value, list):
                print(f"  {key}: {len(value)} items")
                for item in value[:3]:  # Show first 3 items
                    print(f"    - {item}")
                if len(value) > 3:
                    print(f"    ... and {len(value) - 3} more")
            else:
                print(f"  {key}: {value}")
    else:
        print(f"  {results}")


def create_sample_patient_data() -> Dict[ModalityType, Any]:
    """Create comprehensive sample patient data"""
    return {
        ModalityType.MEDICAL_IMAGING: {
            'studies': ['MRI_Brain', 'CT_Head'],
            'modalities': ['T1_weighted', 'T2_weighted', 'FLAIR'],
            'findings': 'Mild hippocampal atrophy, white matter hyperintensities',
            'structural_measurements': {
                'hippocampal_volume': 2850,  # mm³
                'ventricular_volume': 45000,  # mm³
                'cortical_thickness': 2.8  # mm
            },
            'volumetric_data': {
                'total_brain_volume': 1200000,  # mm³
                'gray_matter_volume': 600000,  # mm³
                'white_matter_volume': 500000  # mm³
            }
        },
        
        ModalityType.WEARABLE_SENSORS: {
            'eeg': {
                'summary': 'Reduced alpha wave activity, increased theta waves',
                'alpha_power': 0.35,
                'beta_power': 0.28,
                'theta_power': 0.42,
                'delta_power': 0.15
            },
            'heart_rate': {
                'average': 72,
                'resting': 65,
                'max': 145,
                'variability': 45
            },
            'sleep': {
                'efficiency': 68.5,
                'deep_sleep_percentage': 15.2,
                'rem_percentage': 18.7,
                'wake_episodes': 12
            },
            'gait': {
                'speed': 1.05,  # m/s
                'stride_length': 1.2,  # m
                'cadence': 105,  # steps/min
                'variability': 0.08
            }
        },
        
        ModalityType.GENOMIC_DATA: {
            'variants': [
                {'gene': 'APOE', 'variant': 'ε4/ε3', 'significance': 'pathogenic'},
                {'gene': 'PSEN1', 'variant': 'c.146C>T', 'significance': 'likely_benign'},
                {'gene': 'APP', 'variant': 'c.2149G>A', 'significance': 'uncertain'}
            ],
            'polygenic_risk_scores': {
                'alzheimers': 0.75,
                'parkinsons': 0.23,
                'frontotemporal_dementia': 0.18
            },
            'family_history': {
                'alzheimers_family_history': True,
                'age_of_onset_family': 68,
                'affected_relatives': 2
            }
        },
        
        ModalityType.CLINICAL_TEXT: {
            'clinical_notes': """
            Patient presents with progressive memory loss over 18 months. 
            Difficulty with word finding, occasional disorientation to time.
            Family reports personality changes and increased anxiety.
            MMSE score: 24/30. Clock drawing test shows mild impairment.
            """,
            'discharge_summary': """
            68-year-old female with mild cognitive impairment.
            Comprehensive neuropsychological testing completed.
            Recommended follow-up in 6 months with repeat imaging.
            Started on cholinesterase inhibitor therapy.
            """,
            'lab_reports': """
            Vitamin B12: 350 pg/mL (normal)
            TSH: 2.1 mIU/L (normal)
            Inflammatory markers: CRP 1.2 mg/L (normal)
            """
        },
        
        ModalityType.LABORATORY_DATA: {
            'blood_work': {
                'glucose': 95,  # mg/dL
                'cholesterol': 185,  # mg/dL
                'creatinine': 0.9,  # mg/dL
                'hemoglobin': 13.5  # g/dL
            },
            'biomarkers': {
                'amyloid_beta_42': 450,  # pg/mL (low)
                'tau': 380,  # pg/mL (elevated)
                'phospho_tau': 65,  # pg/mL (elevated)
                'nfl': 25  # pg/mL (elevated)
            },
            'inflammatory_markers': {
                'crp': 1.2,  # mg/L
                'esr': 18,  # mm/hr
                'il6': 2.1  # pg/mL
            },
            'metabolic_panel': {
                'sodium': 140,  # mEq/L
                'potassium': 4.2,  # mEq/L
                'chloride': 102,  # mEq/L
                'bun': 18  # mg/dL
            }
        }
    }


async def demonstrate_single_patient_analysis():
    """Demonstrate comprehensive single patient analysis"""
    
    print_header("Single Patient Multi-Modal Analysis")
    
    # Initialize multi-modal manager
    audit_logger = AuditLogger()
    manager = MultiModalManager(audit_logger)
    
    print("✅ Multi-Modal Manager initialized with production API keys")
    
    # Create sample patient data
    patient_data = create_sample_patient_data()
    
    print(f"📋 Created sample patient data with {len(patient_data)} modalities:")
    for modality in patient_data.keys():
        print(f"   • {modality.value}")
    
    # Create analysis request
    request = MultiModalRequest(
        patient_id="PAT_20241012_DEMO_001",
        modalities=patient_data,
        analysis_type="comprehensive_neurological_assessment",
        priority=2,
        streaming=False,
        batch_mode=False
    )
    
    print(f"\n🔄 Starting comprehensive analysis for patient {request.patient_id}...")
    
    try:
        # Perform multi-modal analysis
        result = await manager.analyze_multimodal_patient(request)
        
        print(f"✅ Analysis completed successfully!")
        print(f"⏱️  Processing time: {result.processing_time:.2f} seconds")
        print(f"💰 Total cost: ${result.total_cost:.6f}")
        print(f"🎯 Overall confidence: {result.overall_confidence:.1%}")
        print(f"🔑 API keys used: {', '.join(result.api_keys_used)}")
        
        # Show detailed results
        print_header("Analysis Results by Modality")
        
        for modality, analysis in result.analysis_results.items():
            print_results(analysis, f"{modality.upper()} Analysis")
        
        # Show fusion results
        if result.fusion_results:
            print_header("Multi-Modal Fusion Results")
            print_results(result.fusion_results, "Integrated Analysis")
        
        # Show cost breakdown
        print_header("Cost Breakdown")
        for modality, cost in result.cost_breakdown.items():
            print(f"  {modality}: ${cost:.6f}")
        print(f"  TOTAL: ${result.total_cost:.6f}")
        
        # Show confidence scores
        print_header("Confidence Scores")
        for modality, confidence in result.confidence_scores.items():
            print(f"  {modality}: {confidence:.1%}")
        print(f"  OVERALL: {result.overall_confidence:.1%}")
        
        return result
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"❌ Analysis failed: {e}")
        return None


async def demonstrate_streaming_analysis():
    """Demonstrate streaming real-time analysis"""
    
    print_header("Streaming Real-Time Analysis")
    
    # Initialize manager with streaming enabled
    manager = MultiModalManager()
    
    # Create streaming request with sensor data
    streaming_data = {
        ModalityType.WEARABLE_SENSORS: {
            'real_time_eeg': 'Continuous EEG monitoring showing irregular patterns',
            'heart_rate_stream': 'Real-time heart rate: 85 BPM, increasing trend',
            'activity_data': 'Patient movement detected, gait analysis in progress'
        },
        ModalityType.CLINICAL_TEXT: {
            'live_notes': 'Patient reporting increased confusion, nurse assessment ongoing'
        }
    }
    
    request = MultiModalRequest(
        patient_id="PAT_20241012_STREAM_001",
        modalities=streaming_data,
        analysis_type="real_time_monitoring",
        streaming=True
    )
    
    print(f"🔄 Starting streaming analysis for patient {request.patient_id}...")
    
    try:
        insight_count = 0
        async for insight in manager.stream_patient_analysis(request):
            insight_count += 1
            print(f"📡 Streaming Insight #{insight_count}:")
            print(f"   Patient: {insight['patient_id']}")
            print(f"   Modality: {insight['modality']}")
            print(f"   Timestamp: {insight['timestamp']}")
            
            if 'insight' in insight:
                print(f"   Insight: {insight['insight']}")
            elif 'analysis' in insight:
                print(f"   Analysis: {list(insight['analysis'].keys())}")
            elif 'error' in insight:
                print(f"   Error: {insight['error']}")
            
            # Limit streaming demo
            if insight_count >= 5:
                break
        
        print(f"✅ Streaming analysis completed with {insight_count} insights")
        
    except Exception as e:
        logger.error(f"Streaming analysis failed: {e}")
        print(f"❌ Streaming analysis failed: {e}")


async def demonstrate_batch_processing():
    """Demonstrate batch processing of multiple patients"""
    
    print_header("Batch Processing Demo")
    
    manager = MultiModalManager()
    
    # Create multiple patient requests
    requests = []
    for i in range(3):
        patient_data = create_sample_patient_data()
        # Vary the data slightly for each patient
        patient_data[ModalityType.WEARABLE_SENSORS]['heart_rate']['average'] += i * 5
        
        request = MultiModalRequest(
            patient_id=f"PAT_20241012_BATCH_{i+1:03d}",
            modalities=patient_data,
            analysis_type="batch_neurological_screening",
            batch_mode=True
        )
        requests.append(request)
    
    print(f"📊 Created batch of {len(requests)} patient analysis requests")
    
    try:
        # Process batch
        results = await manager.batch_analyze_patients(requests)
        
        print(f"✅ Batch processing completed!")
        print(f"📈 Processed: {len(results)} patients")
        
        # Summary statistics
        total_cost = sum(r.total_cost for r in results)
        avg_processing_time = sum(r.processing_time for r in results) / len(results)
        avg_confidence = sum(r.overall_confidence for r in results) / len(results)
        
        print(f"💰 Total batch cost: ${total_cost:.6f}")
        print(f"⏱️  Average processing time: {avg_processing_time:.2f} seconds")
        print(f"🎯 Average confidence: {avg_confidence:.1%}")
        
        # Show per-patient summary
        print("\n📋 Per-Patient Summary:")
        for result in results:
            print(f"  {result.patient_id}: "
                  f"${result.total_cost:.6f}, "
                  f"{result.processing_time:.1f}s, "
                  f"{result.overall_confidence:.1%} confidence")
        
        return results
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        print(f"❌ Batch processing failed: {e}")
        return []


async def demonstrate_performance_monitoring():
    """Demonstrate performance monitoring and metrics"""
    
    print_header("Performance Monitoring & Metrics")
    
    manager = MultiModalManager()
    
    # Get performance metrics
    metrics = manager.get_performance_metrics()
    
    print("📊 System Performance Metrics:")
    print(f"  Total Requests: {metrics['processing_stats']['total_requests']}")
    print(f"  Successful Requests: {metrics['processing_stats']['successful_requests']}")
    print(f"  Failed Requests: {metrics['processing_stats']['failed_requests']}")
    print(f"  Success Rate: {metrics['success_rate']:.1f}%")
    print(f"  Average Processing Time: {metrics['average_processing_time']:.2f}s")
    print(f"  Average Cost per Request: ${metrics['average_cost_per_request']:.6f}")
    print(f"  Total Cost: ${metrics['processing_stats']['total_cost']:.6f}")
    
    print("\n🔧 Configuration:")
    config = metrics['configuration']
    print(f"  Max Concurrent Requests: {config['max_concurrent_requests']}")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Streaming Enabled: {config['streaming_enabled']}")
    print(f"  Fusion Enabled: {config['fusion_enabled']}")
    
    # API Key Metrics
    if 'api_key_metrics' in metrics:
        print_header("API Key Usage Metrics")
        
        api_metrics = metrics['api_key_metrics']
        print(f"💰 Total API Cost: ${api_metrics['total_cost']:.6f}")
        print(f"📊 Total API Requests: {api_metrics['total_requests']}")
        print(f"🎯 Overall Success Rate: {api_metrics['overall_success_rate']:.1f}%")
        
        print("\n🔑 Per-Key Metrics:")
        for key_id, key_metrics in api_metrics['key_metrics'].items():
            print(f"  {key_id}:")
            print(f"    Requests: {key_metrics['total_requests']}")
            print(f"    Success Rate: {key_metrics['success_rate']:.1f}%")
            print(f"    Avg Response Time: {key_metrics['average_response_time']:.1f}ms")
            print(f"    Current RPM: {key_metrics['current_rpm']}")
    
    # Health check
    print_header("System Health Check")
    
    health = await manager.health_check()
    print(f"🏥 Overall Status: {health['status'].upper()}")
    print(f"📊 Multi-Modal Manager Status: {health['multimodal_manager']['status']}")
    print(f"🔄 Total Requests Processed: {health['multimodal_manager']['total_requests_processed']}")
    print(f"✅ Success Rate: {health['multimodal_manager']['success_rate']:.1f}%")
    print(f"📡 Streaming Enabled: {health['multimodal_manager']['streaming_enabled']}")
    print(f"🔗 Fusion Enabled: {health['multimodal_manager']['fusion_enabled']}")


async def main():
    """Main demonstration function"""
    
    print("🧠 NeuroDx Multi-Modal System Demonstration")
    print("=" * 80)
    print("This demo showcases comprehensive medical AI analysis using")
    print("multiple NVIDIA API keys for different specialized tasks.")
    print("=" * 80)
    
    try:
        # Single patient analysis
        single_result = await demonstrate_single_patient_analysis()
        
        # Wait between demos
        await asyncio.sleep(2)
        
        # Streaming analysis
        await demonstrate_streaming_analysis()
        
        # Wait between demos
        await asyncio.sleep(2)
        
        # Batch processing
        batch_results = await demonstrate_batch_processing()
        
        # Wait between demos
        await asyncio.sleep(2)
        
        # Performance monitoring
        await demonstrate_performance_monitoring()
        
        # Final summary
        print_header("Demo Summary")
        
        total_patients = 1 + len(batch_results) if batch_results else 1
        total_cost = (single_result.total_cost if single_result else 0) + \
                    sum(r.total_cost for r in batch_results) if batch_results else 0
        
        print(f"✅ Successfully demonstrated multi-modal medical AI system")
        print(f"📊 Total patients analyzed: {total_patients}")
        print(f"💰 Total demonstration cost: ${total_cost:.6f}")
        print(f"🔑 API keys utilized: Medical Text, Genomics, Streaming, Multi-Modal")
        print(f"🎯 Key capabilities demonstrated:")
        print(f"   • Medical imaging analysis")
        print(f"   • Wearable sensor data processing")
        print(f"   • Genomic variant analysis")
        print(f"   • Clinical text analysis")
        print(f"   • Laboratory data interpretation")
        print(f"   • Multi-modal data fusion")
        print(f"   • Real-time streaming analysis")
        print(f"   • Batch processing")
        print(f"   • Performance monitoring")
        
        print("\n🎉 Multi-Modal Demo Completed Successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"❌ Demo failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())