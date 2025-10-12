#!/usr/bin/env python3
"""
Simple test of the API Key Manager functionality
"""

import asyncio
import logging
from src.services.nvidia_integration.api_key_manager import (
    APIKeyManager, TaskType, create_default_api_key_config
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def print_header(title: str):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"🔑 {title}")
    print(f"{'='*60}")


def main():
    """Test API key manager functionality"""
    
    print_header("API Key Manager Test")
    
    # Create API key configuration
    config = create_default_api_key_config()
    
    # Initialize API key manager
    api_key_manager = APIKeyManager(config)
    
    print(f"✅ Initialized API Key Manager with {len(api_key_manager.api_keys)} keys")
    
    # Show configuration
    print_header("API Key Configuration")
    
    for key_id, key_config in api_key_manager.api_keys.items():
        task_types = [t.value for t in key_config.task_types]
        print(f"🔑 {key_id}")
        print(f"   Tasks: {task_types if task_types else ['Any (backup)']}")
        print(f"   Limits: {key_config.max_rpm} RPM, {key_config.max_tpm} TPM")
        print(f"   Priority: {key_config.priority}")
        print(f"   Cost: ${key_config.cost_per_token:.6f} per token")
    
    # Test task allocation
    print_header("Task Allocation Test")
    
    tasks_to_test = [
        (TaskType.MEDICAL_TEXT_ANALYSIS, 150, "Medical text analysis"),
        (TaskType.DIAGNOSTIC_REPORT_GENERATION, 800, "Diagnostic report generation"),
        (TaskType.GENOMICS_ANALYSIS, 300, "Genomics analysis"),
        (TaskType.STREAMING_INSIGHTS, 50, "Streaming insights"),
        (TaskType.BATCH_PROCESSING, 2000, "Batch processing"),
        (TaskType.MULTI_MODAL_FUSION, 400, "Multi-modal fusion")
    ]
    
    for task_type, estimated_tokens, description in tasks_to_test:
        print(f"\n🎯 {description}")
        print(f"   Task type: {task_type.value}")
        print(f"   Estimated tokens: {estimated_tokens}")
        
        # Get API key for this task
        key_info = api_key_manager.get_api_key_for_task(task_type, estimated_tokens)
        
        if key_info:
            key_id, api_key, endpoint = key_info
            print(f"   ✅ Allocated key: {key_id}")
            print(f"   🔗 Endpoint: {endpoint}")
            
            # Simulate successful request
            api_key_manager.record_request(
                key_id=key_id,
                tokens_used=estimated_tokens,
                response_time_ms=150 + (estimated_tokens * 0.1),
                success=True
            )
            print(f"   📊 Request recorded successfully")
        else:
            print(f"   ❌ No available key for this task")
    
    # Show metrics
    print_header("Usage Metrics")
    
    all_metrics = api_key_manager.get_all_metrics()
    
    print(f"{'Key ID':<25} {'Requests':<10} {'Success%':<10} {'Tokens':<10} {'Avg RT':<10}")
    print("-" * 75)
    
    for key_id, metrics in all_metrics.items():
        print(f"{key_id:<25} {metrics.total_requests:<10} "
              f"{metrics.success_rate:<10.1f} {metrics.total_tokens:<10} "
              f"{metrics.average_response_time:<10.1f}")
    
    # Show cost analysis
    print_header("Cost Analysis")
    
    cost_analysis = api_key_manager.get_cost_analysis()
    total_cost = sum(cost_analysis.values())
    total_requests = sum(m.total_requests for m in all_metrics.values())
    total_tokens = sum(m.total_tokens for m in all_metrics.values())
    
    print(f"💰 Total cost: ${total_cost:.6f}")
    print(f"📊 Total requests: {total_requests}")
    print(f"🎯 Total tokens: {total_tokens}")
    
    print("\n💡 Cost breakdown by key:")
    for key_id, cost in cost_analysis.items():
        if cost > 0:
            print(f"   {key_id}: ${cost:.6f}")
    
    # Test load balancing
    print_header("Load Balancing Test")
    
    print("🔄 Making multiple requests for the same task type...")
    
    for i in range(5):
        key_info = api_key_manager.get_api_key_for_task(
            TaskType.MEDICAL_TEXT_ANALYSIS, 100
        )
        
        if key_info:
            key_id, _, _ = key_info
            print(f"   Request {i+1}: Allocated to {key_id}")
            
            # Record request
            api_key_manager.record_request(
                key_id=key_id,
                tokens_used=100,
                response_time_ms=120,
                success=True
            )
    
    # Final metrics
    print_header("Final Metrics")
    
    final_metrics = api_key_manager.get_all_metrics()
    active_keys = sum(1 for k in api_key_manager.api_keys.values() if k.is_active)
    total_keys = len(api_key_manager.api_keys)
    
    print(f"🔑 Active keys: {active_keys}/{total_keys}")
    print(f"📈 Total requests processed: {sum(m.total_requests for m in final_metrics.values())}")
    print(f"✅ Overall success rate: {sum(m.successful_requests for m in final_metrics.values()) / max(sum(m.total_requests for m in final_metrics.values()), 1) * 100:.1f}%")
    
    print_header("Test Complete")
    print("✅ API Key Manager test completed successfully!")


if __name__ == "__main__":
    main()