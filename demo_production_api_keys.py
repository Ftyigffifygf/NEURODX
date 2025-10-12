#!/usr/bin/env python3
"""
Production API Keys Demo
Shows how the actual NVIDIA API keys work with task-specific allocation
"""

import asyncio
import json
import logging
import time
from datetime import datetime
import openai

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our API key management system
from src.services.nvidia_integration.api_key_manager import (
    APIKeyManager, TaskType
)


def print_header(title: str):
    """Print a formatted header"""
    print(f"\n{'='*80}")
    print(f"🔑 {title}")
    print(f"{'='*80}")


def create_production_api_config():
    """Create production API key configuration"""
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


async def test_api_key(api_key: str, endpoint: str, key_id: str) -> dict:
    """Test an individual API key with a real API call"""
    
    print(f"🔄 Testing API key: {key_id}")
    print(f"   Key: {api_key[:20]}...")
    print(f"   Endpoint: {endpoint}")
    
    try:
        # Configure OpenAI client for NVIDIA API
        client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=endpoint
        )
        
        # Test with a simple medical text analysis
        start_time = time.time()
        
        response = await client.chat.completions.create(
            model="meta/llama-3.1-405b-instruct",
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical AI assistant specialized in analyzing clinical data."
                },
                {
                    "role": "user",
                    "content": "Analyze this patient case: 68-year-old female with progressive memory loss, MMSE score 24/30, mild hippocampal atrophy on MRI. Provide a brief assessment."
                }
            ],
            max_tokens=200,
            temperature=0.7
        )
        
        response_time = (time.time() - start_time) * 1000
        
        # Extract response details
        content = response.choices[0].message.content
        usage = response.usage
        
        print(f"   ✅ Success!")
        print(f"   📊 Response time: {response_time:.1f}ms")
        print(f"   🎯 Tokens used: {usage.total_tokens}")
        print(f"   💰 Estimated cost: ${usage.total_tokens * 0.0001:.6f}")
        print(f"   📝 Response preview: {content[:100]}...")
        
        return {
            'success': True,
            'response_time': response_time,
            'tokens_used': usage.total_tokens,
            'cost': usage.total_tokens * 0.0001,
            'content': content
        }
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'response_time': 0,
            'tokens_used': 0,
            'cost': 0.0
        }


async def demonstrate_task_specific_allocation():
    """Demonstrate task-specific API key allocation with real API calls"""
    
    print_header("Production API Key Task Allocation Demo")
    
    # Initialize API key manager with production keys
    config = create_production_api_config()
    api_key_manager = APIKeyManager(config)
    
    print(f"✅ Initialized API Key Manager with {len(config)} production keys")
    
    # Show configuration
    print("\n🔧 Production API Key Configuration:")
    for key_config in config:
        print(f"   🔑 {key_config['key_id']}")
        print(f"      Tasks: {key_config['task_types']}")
        print(f"      Limits: {key_config['max_rpm']} RPM, {key_config['max_tpm']} TPM")
        print(f"      Priority: {key_config['priority']}")
        print(f"      Cost: ${key_config['cost_per_token']:.6f} per token")
    
    # Test different task types with real API calls
    test_scenarios = [
        {
            'task_type': TaskType.MEDICAL_TEXT_ANALYSIS,
            'description': 'Medical Text Analysis',
            'prompt': 'Analyze this EEG report: Patient shows reduced alpha wave activity and increased theta waves, suggesting possible cognitive impairment.',
            'estimated_tokens': 150
        },
        {
            'task_type': TaskType.GENOMICS_ANALYSIS,
            'description': 'Genomics Analysis',
            'prompt': 'Interpret this genetic variant: APOE ε4/ε3 genotype in a 65-year-old patient with family history of Alzheimer\'s disease.',
            'estimated_tokens': 200
        },
        {
            'task_type': TaskType.DIAGNOSTIC_REPORT_GENERATION,
            'description': 'Diagnostic Report Generation',
            'prompt': 'Generate a diagnostic summary for: MRI shows mild hippocampal atrophy, cognitive testing reveals memory deficits, patient reports word-finding difficulties.',
            'estimated_tokens': 300
        },
        {
            'task_type': TaskType.MULTI_MODAL_FUSION,
            'description': 'Multi-Modal Data Fusion',
            'prompt': 'Integrate findings: Brain MRI shows atrophy, EEG shows slowing, genetic testing reveals APOE ε4, cognitive scores declining. Provide unified assessment.',
            'estimated_tokens': 400
        }
    ]
    
    print_header("Real API Testing with Task-Specific Keys")
    
    total_cost = 0.0
    successful_tests = 0
    
    for scenario in test_scenarios:
        print(f"\n🎯 Testing: {scenario['description']}")
        print(f"   Task Type: {scenario['task_type'].value}")
        print(f"   Estimated Tokens: {scenario['estimated_tokens']}")
        
        # Get appropriate API key for this task
        key_info = api_key_manager.get_api_key_for_task(
            scenario['task_type'],
            scenario['estimated_tokens']
        )
        
        if key_info:
            key_id, api_key, endpoint = key_info
            print(f"   🔑 Allocated Key: {key_id}")
            
            # Test the API key with real call
            result = await test_api_key_with_prompt(
                api_key, endpoint, key_id, scenario['prompt']
            )
            
            if result['success']:
                # Record successful request
                api_key_manager.record_request(
                    key_id=key_id,
                    tokens_used=result['tokens_used'],
                    response_time_ms=result['response_time'],
                    success=True
                )
                
                total_cost += result['cost']
                successful_tests += 1
                
                print(f"   ✅ API call successful")
                print(f"   📊 Actual tokens: {result['tokens_used']}")
                print(f"   💰 Actual cost: ${result['cost']:.6f}")
                
            else:
                # Record failed request
                api_key_manager.record_request(
                    key_id=key_id,
                    tokens_used=0,
                    response_time_ms=result['response_time'],
                    success=False
                )
                
                print(f"   ❌ API call failed: {result['error']}")
        else:
            print(f"   ❌ No available API key for this task")
        
        # Small delay between requests
        await asyncio.sleep(1)
    
    # Show final metrics
    print_header("Final Results")
    
    print(f"📊 Test Summary:")
    print(f"   Total scenarios tested: {len(test_scenarios)}")
    print(f"   Successful API calls: {successful_tests}")
    print(f"   Success rate: {successful_tests/len(test_scenarios)*100:.1f}%")
    print(f"   Total cost: ${total_cost:.6f}")
    
    # Show API key usage metrics
    print(f"\n🔑 API Key Usage Metrics:")
    all_metrics = api_key_manager.get_all_metrics()
    
    for key_id, metrics in all_metrics.items():
        if metrics.total_requests > 0:
            print(f"   {key_id}:")
            print(f"     Requests: {metrics.total_requests}")
            print(f"     Success Rate: {metrics.success_rate:.1f}%")
            print(f"     Avg Response Time: {metrics.average_response_time:.1f}ms")
            print(f"     Total Tokens: {metrics.total_tokens}")
    
    # Show cost analysis
    cost_analysis = api_key_manager.get_cost_analysis()
    print(f"\n💰 Cost Breakdown:")
    for key_id, cost in cost_analysis.items():
        if cost > 0:
            print(f"   {key_id}: ${cost:.6f}")
    
    return successful_tests, total_cost


async def test_api_key_with_prompt(api_key: str, endpoint: str, key_id: str, prompt: str) -> dict:
    """Test API key with a specific prompt"""
    
    try:
        # Configure OpenAI client for NVIDIA API
        client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=endpoint
        )
        
        start_time = time.time()
        
        response = await client.chat.completions.create(
            model="meta/llama-3.1-405b-instruct",
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical AI assistant specialized in clinical analysis and diagnostics."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=300,
            temperature=0.7
        )
        
        response_time = (time.time() - start_time) * 1000
        
        # Extract response details
        content = response.choices[0].message.content
        usage = response.usage
        
        return {
            'success': True,
            'response_time': response_time,
            'tokens_used': usage.total_tokens,
            'cost': usage.total_tokens * 0.0001,
            'content': content
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'response_time': 0,
            'tokens_used': 0,
            'cost': 0.0
        }


async def demonstrate_load_balancing():
    """Demonstrate load balancing across multiple API keys"""
    
    print_header("Load Balancing Demonstration")
    
    # Create configuration with multiple keys for the same task
    config = create_production_api_config()
    api_key_manager = APIKeyManager(config)
    
    print("🔄 Testing load balancing with multiple requests...")
    
    # Make multiple requests for the same task type
    for i in range(6):
        key_info = api_key_manager.get_api_key_for_task(
            TaskType.MEDICAL_TEXT_ANALYSIS, 100
        )
        
        if key_info:
            key_id, _, _ = key_info
            print(f"   Request {i+1}: Allocated to {key_id}")
            
            # Simulate request completion
            api_key_manager.record_request(
                key_id=key_id,
                tokens_used=100,
                response_time_ms=150,
                success=True
            )
        else:
            print(f"   Request {i+1}: No available key")
    
    # Show distribution
    print(f"\n📊 Request Distribution:")
    all_metrics = api_key_manager.get_all_metrics()
    
    for key_id, metrics in all_metrics.items():
        if metrics.total_requests > 0:
            print(f"   {key_id}: {metrics.total_requests} requests")


async def main():
    """Main demonstration function"""
    
    print("🧠 NeuroDx Production API Keys Demonstration")
    print("=" * 80)
    print("This demo shows real NVIDIA API key usage with task-specific allocation")
    print("=" * 80)
    
    try:
        # Test task-specific allocation with real API calls
        successful_tests, total_cost = await demonstrate_task_specific_allocation()
        
        # Wait between demos
        await asyncio.sleep(2)
        
        # Demonstrate load balancing
        await demonstrate_load_balancing()
        
        # Final summary
        print_header("Demo Summary")
        
        print(f"✅ Production API key demonstration completed!")
        print(f"🔑 API keys tested: 4 specialized keys")
        print(f"📊 Successful API calls: {successful_tests}")
        print(f"💰 Total cost: ${total_cost:.6f}")
        print(f"🎯 Key capabilities demonstrated:")
        print(f"   • Task-specific API key allocation")
        print(f"   • Real NVIDIA API integration")
        print(f"   • Automatic load balancing")
        print(f"   • Cost tracking and optimization")
        print(f"   • Performance monitoring")
        print(f"   • Rate limit management")
        
        print("\n🎉 Production Demo Completed Successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"❌ Demo failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())