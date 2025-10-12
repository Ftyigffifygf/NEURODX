#!/usr/bin/env python3
"""
Demonstration of Task-Specific API Key Management
Shows how different API keys are allocated for different tasks
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our services
from src.services.nvidia_integration.api_key_manager import (
    APIKeyManager, TaskType, create_default_api_key_config
)
from src.services.nvidia_integration.nvidia_enhanced_service import NVIDIAEnhancedService
from src.services.security.audit_logger import AuditLogger


def print_header(title: str):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"🔑 {title}")
    print(f"{'='*60}")


def print_metrics_table(metrics_data: Dict[str, Any]):
    """Print API key metrics in a formatted table"""
    print(f"\n{'Key ID':<25} {'Requests':<10} {'Success%':<10} {'Tokens':<10} {'RPM':<8} {'Status':<10}")
    print("-" * 80)
    
    for key_id, metrics in metrics_data["key_metrics"].items():
        status = "Active" if metrics["current_rpm"] >= 0 else "Inactive"
        print(f"{key_id:<25} {metrics['total_requests']:<10} "
              f"{metrics['success_rate']:<10.1f} {metrics['total_tokens']:<10} "
              f"{metrics['current_rpm']:<8} {status:<10}")


async def demonstrate_task_allocation():
    """Demonstrate task-specific API key allocation"""
    
    print_header("Task-Specific API Key Management Demo")
    
    # Initialize audit logger
    audit_logger = AuditLogger()
    
    # Create custom API key configuration for demo
    api_key_config = [
        {
            'key_id': 'medical_text_specialist',
            'api_key': 'nvapi-medical-text-key-demo',
            'endpoint': 'https://integrate.api.nvidia.com/v1',
            'max_rpm': 1000,
            'max_tpm': 50000,
            'priority': 3,
            'task_types': [TaskType.MEDICAL_TEXT_ANALYSIS.value],
            'cost_per_token': 0.0001
        },
        {
            'key_id': 'diagnostic_report_specialist',
            'api_key': 'nvapi-diagnostic-report-key-demo',
            'endpoint': 'https://integrate.api.nvidia.com/v1',
            'max_rpm': 800,
            'max_tpm': 40000,
            'priority': 3,
            'task_types': [TaskType.DIAGNOSTIC_REPORT_GENERATION.value],
            'cost_per_token': 0.00012
        },
        {
            'key_id': 'genomics_specialist',
            'api_key': 'nvapi-genomics-key-demo',
            'endpoint': 'https://integrate.api.nvidia.com/v1',
            'max_rpm': 500,
            'max_tpm': 25000,
            'priority': 3,
            'task_types': [TaskType.GENOMICS_ANALYSIS.value],
            'cost_per_token': 0.0002
        },
        {
            'key_id': 'streaming_specialist',
            'api_key': 'nvapi-streaming-key-demo',
            'endpoint': 'https://integrate.api.nvidia.com/v1',
            'max_rpm': 2000,
            'max_tpm': 100000,
            'priority': 2,
            'task_types': [TaskType.STREAMING_INSIGHTS.value, TaskType.REAL_TIME_INFERENCE.value],
            'cost_per_token': 0.00015
        },
        {
            'key_id': 'batch_processing_specialist',
            'api_key': 'nvapi-batch-key-demo',
            'endpoint': 'https://integrate.api.nvidia.com/v1',
            'max_rpm': 200,
            'max_tpm': 200000,
            'priority': 1,
            'task_types': [TaskType.BATCH_PROCESSING.value],
            'cost_per_token': 0.00008
        },
        {
            'key_id': 'general_purpose_backup',
            'api_key': 'nvapi-general-key-demo',
            'endpoint': 'https://integrate.api.nvidia.com/v1',
            'max_rpm': 600,
            'max_tpm': 30000,
            'priority': 1,
            'task_types': [],  # Can handle any task as backup
            'cost_per_token': 0.0001
        }
    ]
    
    # Initialize enhanced service with custom config
    enhanced_service = NVIDIAEnhancedService(
        audit_logger=audit_logger,
        api_key_config=api_key_config
    )
    
    print("✅ Enhanced NVIDIA service initialized with 6 specialized API keys")
    
    # Show initial API key configuration
    print_header("Initial API Key Configuration")
    
    for key_id, key_config in enhanced_service.api_key_manager.api_keys.items():
        task_types = [t.value for t in key_config.task_types]
        print(f"🔑 {key_id}")
        print(f"   Tasks: {task_types if task_types else ['Any (backup)']}")
        print(f"   Limits: {key_config.max_rpm} RPM, {key_config.max_tpm} TPM")
        print(f"   Priority: {key_config.priority}")
        print(f"   Cost: ${key_config.cost_per_token:.6f} per token")
    
    # Demonstrate task-specific key allocation
    print_header("Task-Specific Key Allocation Demo")
    
    tasks_to_test = [
        (TaskType.MEDICAL_TEXT_ANALYSIS, 150, "Analyzing medical report text"),
        (TaskType.DIAGNOSTIC_REPORT_GENERATION, 800, "Generating diagnostic report"),
        (TaskType.GENOMICS_ANALYSIS, 300, "Analyzing genetic variants"),
        (TaskType.STREAMING_INSIGHTS, 50, "Streaming real-time insights"),
        (TaskType.BATCH_PROCESSING, 2000, "Processing batch of 100 studies"),
        (TaskType.MULTI_MODAL_FUSION, 400, "Fusing imaging and sensor data")
    ]
    
    for task_type, estimated_tokens, description in tasks_to_test:
        print(f"\n🎯 Task: {description}")
        print(f"   Type: {task_type.value}")
        print(f"   Estimated tokens: {estimated_tokens}")
        
        # Get API key for this task
        key_info = enhanced_service.api_key_manager.get_api_key_for_task(
            task_type, estimated_tokens
        )
        
        if key_info:
            key_id, api_key, endpoint = key_info
            print(f"   ✅ Allocated key: {key_id}")
            print(f"   🔗 Endpoint: {endpoint}")
            
            # Simulate request completion
            enhanced_service.api_key_manager.record_request(
                key_id=key_id,
                tokens_used=estimated_tokens,
                response_time_ms=150 + (estimated_tokens * 0.1),
                success=True
            )
        else:
            print(f"   ❌ No available key for this task")
    
    # Show updated metrics
    print_header("API Key Usage Metrics After Simulation")
    
    metrics = enhanced_service.get_api_key_metrics()
    print_metrics_table(metrics)
    
    # Show cost analysis
    print_header("Cost Analysis")
    
    total_cost = metrics["total_cost"]
    print(f"💰 Total cost: ${total_cost:.6f}")
    print(f"📊 Total requests: {metrics['total_requests']}")
    print(f"🎯 Total tokens: {metrics['total_tokens']}")
    print(f"✅ Overall success rate: {metrics['overall_success_rate']:.1f}%")
    
    print("\n💡 Cost breakdown by key:")
    for key_id, cost in metrics["cost_analysis"].items():
        print(f"   {key_id}: ${cost:.6f}")
    
    # Show task distribution
    print_header("Task Distribution Analysis")
    
    for task_type, distribution in metrics["task_distribution"].items():
        if distribution:  # Only show tasks that have been used
            print(f"\n📋 {task_type}:")
            for key_id, request_count in distribution.items():
                if request_count > 0:
                    print(f"   {key_id}: {request_count} requests")
    
    # Demonstrate load balancing
    print_header("Load Balancing Demonstration")
    
    print("🔄 Simulating multiple requests for the same task type...")
    
    # Make multiple requests for medical text analysis
    for i in range(5):
        key_info = enhanced_service.api_key_manager.get_api_key_for_task(
            TaskType.MEDICAL_TEXT_ANALYSIS, 100
        )
        
        if key_info:
            key_id, _, _ = key_info
            print(f"   Request {i+1}: Allocated to {key_id}")
            
            # Simulate request
            enhanced_service.api_key_manager.record_request(
                key_id=key_id,
                tokens_used=100,
                response_time_ms=120,
                success=True
            )
    
    # Demonstrate rate limiting
    print_header("Rate Limiting Demonstration")
    
    print("⚡ Testing rate limit detection...")
    
    # Try to make many requests quickly
    medical_key_id = None
    for key_id, config in enhanced_service.api_key_manager.api_keys.items():
        if TaskType.MEDICAL_TEXT_ANALYSIS in config.task_types:
            medical_key_id = key_id
            break
    
    if medical_key_id:
        # Simulate many requests to trigger rate limiting
        for i in range(10):
            enhanced_service.api_key_manager.record_request(
                key_id=medical_key_id,
                tokens_used=100,
                response_time_ms=100,
                success=True
            )
        
        # Check if key is still available
        key_info = enhanced_service.api_key_manager.get_api_key_for_task(
            TaskType.MEDICAL_TEXT_ANALYSIS, 100
        )
        
        if key_info:
            allocated_key_id, _, _ = key_info
            if allocated_key_id == medical_key_id:
                print(f"   ✅ Key {medical_key_id} still available (within limits)")
            else:
                print(f"   🔄 Switched to backup key: {allocated_key_id}")
        else:
            print(f"   ⚠️ No keys available (rate limited)")
    
    # Show optimization recommendations
    print_header("Optimization Recommendations")
    
    optimization = enhanced_service.optimize_api_key_allocation()
    
    print(f"🎯 Optimization Score: {optimization['optimization_score']}/100")
    
    if optimization["recommendations"]:
        print("\n💡 Recommendations:")
        for rec in optimization["recommendations"]:
            print(f"   {rec['type'].upper()}: {rec['message']}")
    else:
        print("✅ No optimization recommendations - system is well balanced!")
    
    # Final health check
    print_header("System Health Check")
    
    health = await enhanced_service.health_check()
    
    print(f"🏥 Overall Status: {health['status'].upper()}")
    print(f"📊 Health Score: {health['health_score']}/100")
    print(f"🔑 API Keys: {health['api_key_status']['active_keys']}/{health['api_key_status']['total_keys']} active")
    
    for service, status in health["service_status"].items():
        status_emoji = "✅" if status == "healthy" else "⚠️" if status == "degraded" else "❌"
        print(f"   {status_emoji} {service}: {status}")


async def demonstrate_failover_scenario():
    """Demonstrate API key failover when keys become unavailable"""
    
    print_header("API Key Failover Demonstration")
    
    # Create a simple API key manager for failover demo
    failover_config = [
        {
            'key_id': 'primary_key',
            'api_key': 'nvapi-primary-key',
            'endpoint': 'https://integrate.api.nvidia.com/v1',
            'max_rpm': 100,  # Low limit to trigger failover
            'max_tpm': 5000,
            'priority': 3,
            'task_types': [TaskType.MEDICAL_TEXT_ANALYSIS.value],
            'cost_per_token': 0.0001
        },
        {
            'key_id': 'backup_key',
            'api_key': 'nvapi-backup-key',
            'endpoint': 'https://integrate.api.nvidia.com/v1',
            'max_rpm': 1000,
            'max_tpm': 50000,
            'priority': 2,
            'task_types': [TaskType.MEDICAL_TEXT_ANALYSIS.value],
            'cost_per_token': 0.00012
        },
        {
            'key_id': 'emergency_key',
            'api_key': 'nvapi-emergency-key',
            'endpoint': 'https://integrate.api.nvidia.com/v1',
            'max_rpm': 500,
            'max_tpm': 25000,
            'priority': 1,
            'task_types': [],  # General purpose backup
            'cost_per_token': 0.00015
        }
    ]
    
    api_key_manager = APIKeyManager(failover_config)
    
    print("🔧 Created failover scenario with 3 keys:")
    print("   1. Primary key (high priority, low limits)")
    print("   2. Backup key (medium priority, high limits)")
    print("   3. Emergency key (low priority, general purpose)")
    
    # Simulate requests until primary key hits rate limit
    print("\n🔄 Simulating requests to trigger failover...")
    
    for request_num in range(15):
        key_info = api_key_manager.get_api_key_for_task(
            TaskType.MEDICAL_TEXT_ANALYSIS, 100
        )
        
        if key_info:
            key_id, _, _ = key_info
            
            # Record the request
            api_key_manager.record_request(
                key_id=key_id,
                tokens_used=100,
                response_time_ms=150,
                success=True
            )
            
            print(f"   Request {request_num + 1:2d}: Using {key_id}")
            
            # Show current usage
            metrics = api_key_manager.get_key_metrics(key_id)
            if metrics:
                print(f"                Current RPM: {metrics.current_rpm}")
        else:
            print(f"   Request {request_num + 1:2d}: No keys available!")
        
        # Small delay to simulate real requests
        await asyncio.sleep(0.1)
    
    print("\n📊 Final metrics after failover test:")
    all_metrics = api_key_manager.get_all_metrics()
    
    for key_id, metrics in all_metrics.items():
        print(f"   {key_id}:")
        print(f"     Requests: {metrics.total_requests}")
        print(f"     Current RPM: {metrics.current_rpm}")
        print(f"     Success Rate: {metrics.success_rate:.1f}%")


async def main():
    """Main demonstration function"""
    
    print("🧠 NeuroDx-MultiModal API Key Management Demo")
    print("=" * 60)
    print("This demo shows how different API keys are allocated")
    print("for different tasks to optimize performance and cost.")
    
    try:
        # Run main demonstration
        await demonstrate_task_allocation()
        
        # Wait a moment
        await asyncio.sleep(1)
        
        # Run failover demonstration
        await demonstrate_failover_scenario()
        
        print_header("Demo Complete")
        print("✅ Task-specific API key management demonstration completed!")
        print("💡 Key benefits demonstrated:")
        print("   • Automatic task-specific key allocation")
        print("   • Load balancing across multiple keys")
        print("   • Rate limit detection and failover")
        print("   • Cost optimization and usage tracking")
        print("   • Real-time metrics and health monitoring")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"❌ Demo failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())