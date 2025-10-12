"""
Advanced API Key Management for NVIDIA Services
Handles task-specific API key allocation, load balancing, and usage tracking
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Different types of tasks that require API keys"""
    MEDICAL_TEXT_ANALYSIS = "medical_text_analysis"
    DIAGNOSTIC_REPORT_GENERATION = "diagnostic_report_generation"
    GENOMICS_ANALYSIS = "genomics_analysis"
    LONGITUDINAL_TRACKING = "longitudinal_tracking"
    MULTI_MODAL_FUSION = "multi_modal_fusion"
    STREAMING_INSIGHTS = "streaming_insights"
    BATCH_PROCESSING = "batch_processing"
    REAL_TIME_INFERENCE = "real_time_inference"


@dataclass
class APIKeyMetrics:
    """Metrics for tracking API key usage"""
    key_id: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    last_used: Optional[datetime] = None
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    rate_limit_hits: int = 0
    current_rpm: int = 0
    current_tpm: int = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage"""
        if self.total_requests == 0:
            return 100.0
        return (self.successful_requests / self.total_requests) * 100
    
    @property
    def average_response_time(self) -> float:
        """Calculate average response time in milliseconds"""
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)


@dataclass
class APIKeyConfig:
    """Configuration for an API key"""
    key_id: str
    api_key: str
    endpoint: str
    max_rpm: int = 1000  # Requests per minute
    max_tpm: int = 50000  # Tokens per minute
    priority: int = 1  # Higher number = higher priority
    task_types: List[TaskType] = field(default_factory=list)
    is_active: bool = True
    cost_per_token: float = 0.0001  # Cost in USD per token


class APIKeyManager:
    """
    Advanced API key management system for task-specific allocation
    """
    
    def __init__(self, api_keys_config: List[Dict[str, Any]]):
        """
        Initialize API key manager with configuration
        
        Args:
            api_keys_config: List of API key configurations
        """
        self.api_keys: Dict[str, APIKeyConfig] = {}
        self.metrics: Dict[str, APIKeyMetrics] = {}
        self.task_assignments: Dict[TaskType, List[str]] = defaultdict(list)
        self.lock = threading.RLock()
        
        # Load balancing state
        self.round_robin_counters: Dict[TaskType, int] = defaultdict(int)
        self.usage_windows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=60))  # 60-second windows
        
        # Initialize from configuration
        self._load_configuration(api_keys_config)
        
        # Start background monitoring
        self._start_monitoring()
        
        logger.info(f"API Key Manager initialized with {len(self.api_keys)} keys")
    
    def _load_configuration(self, config: List[Dict[str, Any]]):
        """Load API key configuration"""
        for key_config in config:
            key_id = key_config.get('key_id', f"key_{len(self.api_keys)}")
            
            # Parse task types
            task_types = []
            for task_name in key_config.get('task_types', []):
                try:
                    task_types.append(TaskType(task_name))
                except ValueError:
                    logger.warning(f"Unknown task type: {task_name}")
            
            # Create API key configuration
            api_key_config = APIKeyConfig(
                key_id=key_id,
                api_key=key_config['api_key'],
                endpoint=key_config.get('endpoint', 'https://integrate.api.nvidia.com/v1'),
                max_rpm=key_config.get('max_rpm', 1000),
                max_tpm=key_config.get('max_tpm', 50000),
                priority=key_config.get('priority', 1),
                task_types=task_types,
                is_active=key_config.get('is_active', True),
                cost_per_token=key_config.get('cost_per_token', 0.0001)
            )
            
            self.api_keys[key_id] = api_key_config
            self.metrics[key_id] = APIKeyMetrics(key_id=key_id)
            
            # Assign to task types
            for task_type in task_types:
                self.task_assignments[task_type].append(key_id)
        
        # Sort task assignments by priority
        for task_type in self.task_assignments:
            self.task_assignments[task_type].sort(
                key=lambda k: self.api_keys[k].priority, 
                reverse=True
            )
    
    def get_api_key_for_task(self, task_type: TaskType, 
                           estimated_tokens: int = 100) -> Optional[Tuple[str, str, str]]:
        """
        Get the best API key for a specific task
        
        Args:
            task_type: Type of task requiring API key
            estimated_tokens: Estimated token usage for the request
            
        Returns:
            Tuple of (key_id, api_key, endpoint) or None if no key available
        """
        with self.lock:
            available_keys = self._get_available_keys_for_task(task_type, estimated_tokens)
            
            if not available_keys:
                logger.warning(f"No available API keys for task: {task_type}")
                return None
            
            # Select best key using load balancing strategy
            selected_key_id = self._select_best_key(available_keys, task_type)
            
            if selected_key_id:
                key_config = self.api_keys[selected_key_id]
                return selected_key_id, key_config.api_key, key_config.endpoint
            
            return None
    
    def _get_available_keys_for_task(self, task_type: TaskType, 
                                   estimated_tokens: int) -> List[str]:
        """Get list of available API keys for a task"""
        available_keys = []
        
        # Get keys assigned to this task type
        assigned_keys = self.task_assignments.get(task_type, [])
        
        # If no specific assignment, use all keys
        if not assigned_keys:
            assigned_keys = list(self.api_keys.keys())
        
        for key_id in assigned_keys:
            key_config = self.api_keys[key_id]
            metrics = self.metrics[key_id]
            
            # Check if key is active
            if not key_config.is_active:
                continue
            
            # Check rate limits
            if not self._check_rate_limits(key_id, estimated_tokens):
                continue
            
            available_keys.append(key_id)
        
        return available_keys
    
    def _check_rate_limits(self, key_id: str, estimated_tokens: int) -> bool:
        """Check if API key is within rate limits"""
        key_config = self.api_keys[key_id]
        metrics = self.metrics[key_id]
        
        current_time = time.time()
        
        # Clean old usage records (older than 1 minute)
        usage_window = self.usage_windows[key_id]
        while usage_window and usage_window[0]['timestamp'] < current_time - 60:
            usage_window.popleft()
        
        # Calculate current usage
        current_requests = len(usage_window)
        current_tokens = sum(record['tokens'] for record in usage_window)
        
        # Check limits
        if current_requests >= key_config.max_rpm:
            return False
        
        if current_tokens + estimated_tokens > key_config.max_tpm:
            return False
        
        return True
    
    def _select_best_key(self, available_keys: List[str], task_type: TaskType) -> Optional[str]:
        """Select the best API key using load balancing strategy"""
        if not available_keys:
            return None
        
        # Strategy 1: Round-robin for equal distribution
        if len(available_keys) > 1:
            counter = self.round_robin_counters[task_type]
            selected_key = available_keys[counter % len(available_keys)]
            self.round_robin_counters[task_type] = counter + 1
            return selected_key
        
        # Strategy 2: Least loaded key
        best_key = min(available_keys, key=lambda k: self.metrics[k].current_rpm)
        return best_key
    
    def record_request(self, key_id: str, tokens_used: int, 
                      response_time_ms: float, success: bool):
        """Record API request metrics"""
        with self.lock:
            if key_id not in self.metrics:
                return
            
            metrics = self.metrics[key_id]
            current_time = time.time()
            
            # Update metrics
            metrics.total_requests += 1
            metrics.total_tokens += tokens_used
            metrics.last_used = datetime.now()
            metrics.response_times.append(response_time_ms)
            
            if success:
                metrics.successful_requests += 1
            else:
                metrics.failed_requests += 1
            
            # Record usage for rate limiting
            self.usage_windows[key_id].append({
                'timestamp': current_time,
                'tokens': tokens_used,
                'success': success
            })
    
    def record_rate_limit_hit(self, key_id: str):
        """Record a rate limit hit for an API key"""
        with self.lock:
            if key_id in self.metrics:
                self.metrics[key_id].rate_limit_hits += 1
                logger.warning(f"Rate limit hit for API key: {key_id}")
    
    def get_key_metrics(self, key_id: str) -> Optional[APIKeyMetrics]:
        """Get metrics for a specific API key"""
        return self.metrics.get(key_id)
    
    def get_all_metrics(self) -> Dict[str, APIKeyMetrics]:
        """Get metrics for all API keys"""
        return self.metrics.copy()
    
    def get_task_distribution(self) -> Dict[TaskType, Dict[str, int]]:
        """Get request distribution by task type and API key"""
        distribution = {}
        
        for task_type in TaskType:
            distribution[task_type] = {}
            assigned_keys = self.task_assignments.get(task_type, [])
            
            for key_id in assigned_keys:
                metrics = self.metrics.get(key_id)
                if metrics:
                    distribution[task_type][key_id] = metrics.total_requests
        
        return distribution
    
    def get_cost_analysis(self) -> Dict[str, float]:
        """Get cost analysis for all API keys"""
        costs = {}
        
        for key_id, metrics in self.metrics.items():
            key_config = self.api_keys[key_id]
            total_cost = metrics.total_tokens * key_config.cost_per_token
            costs[key_id] = total_cost
        
        return costs
    
    def deactivate_key(self, key_id: str, reason: str = "Manual deactivation"):
        """Deactivate an API key"""
        with self.lock:
            if key_id in self.api_keys:
                self.api_keys[key_id].is_active = False
                logger.info(f"Deactivated API key {key_id}: {reason}")
    
    def activate_key(self, key_id: str):
        """Activate an API key"""
        with self.lock:
            if key_id in self.api_keys:
                self.api_keys[key_id].is_active = True
                logger.info(f"Activated API key: {key_id}")
    
    def _start_monitoring(self):
        """Start background monitoring thread"""
        def monitor():
            while True:
                try:
                    self._update_current_usage()
                    self._check_key_health()
                    time.sleep(10)  # Update every 10 seconds
                except Exception as e:
                    logger.error(f"Error in API key monitoring: {e}")
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
    
    def _update_current_usage(self):
        """Update current usage metrics"""
        current_time = time.time()
        
        with self.lock:
            for key_id in self.api_keys:
                usage_window = self.usage_windows[key_id]
                
                # Clean old records
                while usage_window and usage_window[0]['timestamp'] < current_time - 60:
                    usage_window.popleft()
                
                # Update current metrics
                metrics = self.metrics[key_id]
                metrics.current_rpm = len(usage_window)
                metrics.current_tpm = sum(record['tokens'] for record in usage_window)
    
    def _check_key_health(self):
        """Check health of API keys and auto-deactivate problematic ones"""
        with self.lock:
            for key_id, metrics in self.metrics.items():
                key_config = self.api_keys[key_id]
                
                # Auto-deactivate keys with very low success rate
                if (metrics.total_requests > 10 and 
                    metrics.success_rate < 50 and 
                    key_config.is_active):
                    
                    self.deactivate_key(key_id, f"Low success rate: {metrics.success_rate:.1f}%")
                
                # Auto-deactivate keys with too many rate limit hits
                if (metrics.rate_limit_hits > 10 and 
                    key_config.is_active):
                    
                    self.deactivate_key(key_id, f"Too many rate limit hits: {metrics.rate_limit_hits}")


def create_default_api_key_config() -> List[Dict[str, Any]]:
    """Create default API key configuration for testing"""
    return [
        {
            'key_id': 'medical_analysis_primary',
            'api_key': 'nvapi-medical-analysis-key-1',
            'endpoint': 'https://integrate.api.nvidia.com/v1',
            'max_rpm': 1000,
            'max_tpm': 50000,
            'priority': 3,
            'task_types': [
                TaskType.MEDICAL_TEXT_ANALYSIS.value,
                TaskType.DIAGNOSTIC_REPORT_GENERATION.value
            ],
            'cost_per_token': 0.0001
        },
        {
            'key_id': 'genomics_primary',
            'api_key': 'nvapi-genomics-key-1',
            'endpoint': 'https://integrate.api.nvidia.com/v1',
            'max_rpm': 500,
            'max_tpm': 25000,
            'priority': 3,
            'task_types': [
                TaskType.GENOMICS_ANALYSIS.value
            ],
            'cost_per_token': 0.0002
        },
        {
            'key_id': 'streaming_primary',
            'api_key': 'nvapi-streaming-key-1',
            'endpoint': 'https://integrate.api.nvidia.com/v1',
            'max_rpm': 2000,
            'max_tpm': 100000,
            'priority': 2,
            'task_types': [
                TaskType.STREAMING_INSIGHTS.value,
                TaskType.REAL_TIME_INFERENCE.value
            ],
            'cost_per_token': 0.00015
        },
        {
            'key_id': 'batch_processing',
            'api_key': 'nvapi-batch-key-1',
            'endpoint': 'https://integrate.api.nvidia.com/v1',
            'max_rpm': 200,
            'max_tpm': 200000,
            'priority': 1,
            'task_types': [
                TaskType.BATCH_PROCESSING.value,
                TaskType.LONGITUDINAL_TRACKING.value
            ],
            'cost_per_token': 0.00008
        },
        {
            'key_id': 'general_purpose',
            'api_key': 'nvapi-general-key-1',
            'endpoint': 'https://integrate.api.nvidia.com/v1',
            'max_rpm': 800,
            'max_tpm': 40000,
            'priority': 1,
            'task_types': [
                TaskType.MULTI_MODAL_FUSION.value
            ],
            'cost_per_token': 0.0001
        }
    ]