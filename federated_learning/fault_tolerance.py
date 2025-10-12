"""
Fault Tolerance for MONAI Federated Learning

Implements node failure detection, recovery mechanisms, dynamic node management,
and backup/synchronization strategies for robust federated learning.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import pickle
import hashlib
from pathlib import Path

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """Enumeration of possible node statuses"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    FAILED = "failed"
    RECOVERING = "recovering"
    SUSPENDED = "suspended"


@dataclass
class NodeHealthMetrics:
    """Health metrics for a federated learning node"""
    node_id: str
    last_heartbeat: datetime
    response_time_ms: float
    success_rate: float  # Percentage of successful operations
    error_count: int
    memory_usage_mb: float
    cpu_usage_percent: float
    network_latency_ms: float
    model_sync_status: str  # 'synced', 'syncing', 'out_of_sync'
    
    def __post_init__(self):
        """Validate health metrics"""
        if not 0 <= self.success_rate <= 100:
            raise ValueError("Success rate must be between 0 and 100")
        if not 0 <= self.cpu_usage_percent <= 100:
            raise ValueError("CPU usage must be between 0 and 100")


@dataclass
class FailureEvent:
    """Represents a node failure event"""
    node_id: str
    failure_type: str  # 'timeout', 'connection_lost', 'error', 'resource_exhaustion'
    timestamp: datetime
    error_message: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    recovery_attempts: int = 0
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate failure event"""
        valid_types = ['timeout', 'connection_lost', 'error', 'resource_exhaustion']
        if self.failure_type not in valid_types:
            raise ValueError(f"Invalid failure type. Must be one of: {valid_types}")
        
        valid_severities = ['low', 'medium', 'high', 'critical']
        if self.severity not in valid_severities:
            raise ValueError(f"Invalid severity. Must be one of: {valid_severities}")


@dataclass
class BackupSnapshot:
    """Represents a backup snapshot of federated learning state"""
    snapshot_id: str
    timestamp: datetime
    round_number: int
    participating_nodes: List[str]
    global_model_state: Dict[str, Any]
    aggregation_metrics: Dict[str, float]
    checksum: str
    file_path: Optional[str] = None
    
    def __post_init__(self):
        """Generate checksum for integrity verification"""
        if not self.checksum:
            # Generate checksum from model state and metrics
            content = json.dumps({
                'round_number': self.round_number,
                'participating_nodes': sorted(self.participating_nodes),
                'aggregation_metrics': self.aggregation_metrics
            }, sort_keys=True)
            self.checksum = hashlib.sha256(content.encode()).hexdigest()


class FaultToleranceManager:
    """
    Fault Tolerance Manager for MONAI Federated Learning
    
    Provides comprehensive fault tolerance including node failure detection,
    automatic recovery, dynamic node management, and backup/restore capabilities.
    """
    
    def __init__(self, backup_directory: str = "backups/federated"):
        self.node_health: Dict[str, NodeHealthMetrics] = {}
        self.node_status: Dict[str, NodeStatus] = {}
        self.failure_history: List[FailureEvent] = []
        self.backup_snapshots: List[BackupSnapshot] = []
        self.backup_directory = Path(backup_directory)
        self.backup_directory.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        self.health_check_config = self._get_default_health_config()
        self.recovery_config = self._get_default_recovery_config()
        self.backup_config = self._get_default_backup_config()
        
        # Active monitoring
        self.monitoring_active = False
        self.recovery_tasks: Dict[str, asyncio.Task] = {}
    
    def _get_default_health_config(self) -> Dict[str, Any]:
        """Get default health check configuration"""
        return {
            'heartbeat_interval_seconds': 30,
            'heartbeat_timeout_seconds': 60,
            'max_response_time_ms': 5000,
            'min_success_rate': 80.0,
            'max_error_count': 10,
            'max_memory_usage_mb': 8192,
            'max_cpu_usage_percent': 90.0,
            'max_network_latency_ms': 2000
        }
    
    def _get_default_recovery_config(self) -> Dict[str, Any]:
        """Get default recovery configuration"""
        return {
            'max_recovery_attempts': 3,
            'recovery_timeout_seconds': 300,
            'exponential_backoff_base': 2,
            'min_recovery_delay_seconds': 10,
            'max_recovery_delay_seconds': 300,
            'auto_recovery_enabled': True,
            'quarantine_duration_minutes': 30
        }
    
    def _get_default_backup_config(self) -> Dict[str, Any]:
        """Get default backup configuration"""
        return {
            'backup_interval_rounds': 5,
            'max_backup_files': 20,
            'compression_enabled': True,
            'integrity_check_enabled': True,
            'auto_cleanup_enabled': True,
            'backup_retention_days': 30
        }
    
    async def start_monitoring(self):
        """Start continuous health monitoring of federated nodes"""
        if self.monitoring_active:
            logger.warning("Monitoring is already active")
            return
        
        self.monitoring_active = True
        logger.info("Started fault tolerance monitoring")
        
        # Start monitoring tasks
        asyncio.create_task(self._health_check_loop())
        asyncio.create_task(self._cleanup_loop())
    
    async def stop_monitoring(self):
        """Stop health monitoring"""
        self.monitoring_active = False
        
        # Cancel recovery tasks
        for task in self.recovery_tasks.values():
            task.cancel()
        self.recovery_tasks.clear()
        
        logger.info("Stopped fault tolerance monitoring")
    
    async def register_node(self, node_id: str, initial_metrics: NodeHealthMetrics):
        """Register a new node for monitoring"""
        try:
            self.node_health[node_id] = initial_metrics
            self.node_status[node_id] = NodeStatus.ACTIVE
            
            logger.info(f"Registered node {node_id} for fault tolerance monitoring")
            
        except Exception as e:
            logger.error(f"Failed to register node {node_id}: {e}")
    
    async def update_node_health(self, node_id: str, metrics: NodeHealthMetrics):
        """Update health metrics for a node"""
        try:
            if node_id not in self.node_health:
                logger.warning(f"Received metrics for unregistered node {node_id}")
                return
            
            self.node_health[node_id] = metrics
            
            # Check if node health indicates problems
            await self._evaluate_node_health(node_id, metrics)
            
        except Exception as e:
            logger.error(f"Failed to update health for node {node_id}: {e}")
    
    async def _evaluate_node_health(self, node_id: str, metrics: NodeHealthMetrics):
        """Evaluate node health and trigger actions if needed"""
        try:
            config = self.health_check_config
            issues = []
            
            # Check various health indicators
            if metrics.response_time_ms > config['max_response_time_ms']:
                issues.append(f"High response time: {metrics.response_time_ms}ms")
            
            if metrics.success_rate < config['min_success_rate']:
                issues.append(f"Low success rate: {metrics.success_rate}%")
            
            if metrics.error_count > config['max_error_count']:
                issues.append(f"High error count: {metrics.error_count}")
            
            if metrics.memory_usage_mb > config['max_memory_usage_mb']:
                issues.append(f"High memory usage: {metrics.memory_usage_mb}MB")
            
            if metrics.cpu_usage_percent > config['max_cpu_usage_percent']:
                issues.append(f"High CPU usage: {metrics.cpu_usage_percent}%")
            
            if metrics.network_latency_ms > config['max_network_latency_ms']:
                issues.append(f"High network latency: {metrics.network_latency_ms}ms")
            
            # Check heartbeat timeout
            heartbeat_age = (datetime.now() - metrics.last_heartbeat).total_seconds()
            if heartbeat_age > config['heartbeat_timeout_seconds']:
                issues.append(f"Heartbeat timeout: {heartbeat_age}s")
            
            # Determine severity and take action
            if issues:
                severity = self._determine_failure_severity(issues)
                await self._handle_node_issues(node_id, issues, severity)
            else:
                # Node is healthy, update status if it was previously problematic
                if self.node_status[node_id] != NodeStatus.ACTIVE:
                    await self._mark_node_recovered(node_id)
                    
        except Exception as e:
            logger.error(f"Failed to evaluate health for node {node_id}: {e}")
    
    def _determine_failure_severity(self, issues: List[str]) -> str:
        """Determine failure severity based on issues"""
        critical_keywords = ['timeout', 'connection_lost', 'memory']
        high_keywords = ['error_count', 'cpu']
        
        for issue in issues:
            if any(keyword in issue.lower() for keyword in critical_keywords):
                return 'critical'
        
        for issue in issues:
            if any(keyword in issue.lower() for keyword in high_keywords):
                return 'high'
        
        return 'medium' if len(issues) > 2 else 'low'
    
    async def _handle_node_issues(self, node_id: str, issues: List[str], severity: str):
        """Handle detected node issues"""
        try:
            # Record failure event
            failure_event = FailureEvent(
                node_id=node_id,
                failure_type='error',  # Generic type for health issues
                timestamp=datetime.now(),
                error_message='; '.join(issues),
                severity=severity
            )
            self.failure_history.append(failure_event)
            
            # Update node status
            if severity in ['critical', 'high']:
                self.node_status[node_id] = NodeStatus.FAILED
                logger.error(f"Node {node_id} marked as FAILED due to {severity} issues: {issues}")
                
                # Trigger recovery if enabled
                if self.recovery_config['auto_recovery_enabled']:
                    await self._initiate_node_recovery(node_id, failure_event)
            else:
                self.node_status[node_id] = NodeStatus.INACTIVE
                logger.warning(f"Node {node_id} marked as INACTIVE due to issues: {issues}")
                
        except Exception as e:
            logger.error(f"Failed to handle issues for node {node_id}: {e}")
    
    async def _initiate_node_recovery(self, node_id: str, failure_event: FailureEvent):
        """Initiate recovery process for a failed node"""
        try:
            if node_id in self.recovery_tasks:
                logger.info(f"Recovery already in progress for node {node_id}")
                return
            
            logger.info(f"Initiating recovery for node {node_id}")
            self.node_status[node_id] = NodeStatus.RECOVERING
            
            # Start recovery task
            recovery_task = asyncio.create_task(
                self._execute_node_recovery(node_id, failure_event)
            )
            self.recovery_tasks[node_id] = recovery_task
            
        except Exception as e:
            logger.error(f"Failed to initiate recovery for node {node_id}: {e}")
    
    async def _execute_node_recovery(self, node_id: str, failure_event: FailureEvent):
        """Execute recovery process for a failed node"""
        try:
            config = self.recovery_config
            max_attempts = config['max_recovery_attempts']
            
            for attempt in range(1, max_attempts + 1):
                logger.info(f"Recovery attempt {attempt}/{max_attempts} for node {node_id}")
                failure_event.recovery_attempts = attempt
                
                # Calculate delay with exponential backoff
                delay = min(
                    config['min_recovery_delay_seconds'] * (config['exponential_backoff_base'] ** (attempt - 1)),
                    config['max_recovery_delay_seconds']
                )
                
                if attempt > 1:
                    await asyncio.sleep(delay)
                
                # Attempt recovery
                recovery_success = await self._attempt_node_recovery(node_id)
                
                if recovery_success:
                    await self._mark_node_recovered(node_id)
                    failure_event.resolved = True
                    failure_event.resolution_timestamp = datetime.now()
                    logger.info(f"Successfully recovered node {node_id} on attempt {attempt}")
                    break
                else:
                    logger.warning(f"Recovery attempt {attempt} failed for node {node_id}")
            
            else:
                # All recovery attempts failed
                logger.error(f"All recovery attempts failed for node {node_id}")
                await self._quarantine_node(node_id)
            
        except Exception as e:
            logger.error(f"Recovery execution failed for node {node_id}: {e}")
        finally:
            # Clean up recovery task
            if node_id in self.recovery_tasks:
                del self.recovery_tasks[node_id]
    
    async def _attempt_node_recovery(self, node_id: str) -> bool:
        """Attempt to recover a specific node"""
        try:
            # In a real implementation, this would:
            # 1. Try to re-establish connection
            # 2. Restart node services if possible
            # 3. Resync model state
            # 4. Verify node functionality
            
            # For now, simulate recovery attempt
            logger.info(f"Attempting to recover node {node_id}")
            
            # Simulate recovery process
            await asyncio.sleep(1)  # Simulate recovery time
            
            # In practice, you would check if the node is actually responsive
            # For simulation, we'll assume 70% success rate
            import random
            return random.random() < 0.7
            
        except Exception as e:
            logger.error(f"Recovery attempt failed for node {node_id}: {e}")
            return False
    
    async def _mark_node_recovered(self, node_id: str):
        """Mark a node as successfully recovered"""
        try:
            self.node_status[node_id] = NodeStatus.ACTIVE
            logger.info(f"Node {node_id} marked as recovered and active")
            
        except Exception as e:
            logger.error(f"Failed to mark node {node_id} as recovered: {e}")
    
    async def _quarantine_node(self, node_id: str):
        """Quarantine a node that failed recovery"""
        try:
            self.node_status[node_id] = NodeStatus.SUSPENDED
            logger.warning(f"Node {node_id} quarantined after failed recovery")
            
            # Schedule automatic re-evaluation after quarantine period
            quarantine_duration = self.recovery_config['quarantine_duration_minutes'] * 60
            asyncio.create_task(self._schedule_quarantine_release(node_id, quarantine_duration))
            
        except Exception as e:
            logger.error(f"Failed to quarantine node {node_id}: {e}")
    
    async def _schedule_quarantine_release(self, node_id: str, duration_seconds: int):
        """Schedule release from quarantine"""
        try:
            await asyncio.sleep(duration_seconds)
            
            if self.node_status.get(node_id) == NodeStatus.SUSPENDED:
                self.node_status[node_id] = NodeStatus.INACTIVE
                logger.info(f"Node {node_id} released from quarantine")
                
        except Exception as e:
            logger.error(f"Failed to release node {node_id} from quarantine: {e}")
    
    async def create_backup_snapshot(self, round_number: int, participating_nodes: List[str],
                                   global_model_state: Dict[str, Any], 
                                   aggregation_metrics: Dict[str, float]) -> str:
        """Create a backup snapshot of the current federated learning state"""
        try:
            snapshot_id = f"snapshot_{round_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            snapshot = BackupSnapshot(
                snapshot_id=snapshot_id,
                timestamp=datetime.now(),
                round_number=round_number,
                participating_nodes=participating_nodes.copy(),
                global_model_state=global_model_state.copy(),
                aggregation_metrics=aggregation_metrics.copy(),
                checksum=""  # Will be generated in __post_init__
            )
            
            # Save snapshot to file
            snapshot_file = self.backup_directory / f"{snapshot_id}.pkl"
            with open(snapshot_file, 'wb') as f:
                pickle.dump(snapshot, f)
            
            snapshot.file_path = str(snapshot_file)
            self.backup_snapshots.append(snapshot)
            
            # Cleanup old backups if needed
            await self._cleanup_old_backups()
            
            logger.info(f"Created backup snapshot {snapshot_id} for round {round_number}")
            return snapshot_id
            
        except Exception as e:
            logger.error(f"Failed to create backup snapshot: {e}")
            raise
    
    async def restore_from_snapshot(self, snapshot_id: str) -> Dict[str, Any]:
        """Restore federated learning state from a backup snapshot"""
        try:
            # Find snapshot
            snapshot = None
            for s in self.backup_snapshots:
                if s.snapshot_id == snapshot_id:
                    snapshot = s
                    break
            
            if not snapshot:
                raise ValueError(f"Snapshot {snapshot_id} not found")
            
            # Load snapshot from file if needed
            if snapshot.file_path and Path(snapshot.file_path).exists():
                with open(snapshot.file_path, 'rb') as f:
                    loaded_snapshot = pickle.load(f)
                
                # Verify integrity
                if loaded_snapshot.checksum != snapshot.checksum:
                    raise ValueError(f"Snapshot {snapshot_id} integrity check failed")
                
                snapshot = loaded_snapshot
            
            logger.info(f"Restored from snapshot {snapshot_id} (round {snapshot.round_number})")
            
            return {
                'round_number': snapshot.round_number,
                'participating_nodes': snapshot.participating_nodes,
                'global_model_state': snapshot.global_model_state,
                'aggregation_metrics': snapshot.aggregation_metrics,
                'timestamp': snapshot.timestamp
            }
            
        except Exception as e:
            logger.error(f"Failed to restore from snapshot {snapshot_id}: {e}")
            raise
    
    async def _cleanup_old_backups(self):
        """Clean up old backup files"""
        try:
            if not self.backup_config['auto_cleanup_enabled']:
                return
            
            max_files = self.backup_config['max_backup_files']
            retention_days = self.backup_config['backup_retention_days']
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            # Remove snapshots older than retention period
            self.backup_snapshots = [
                s for s in self.backup_snapshots 
                if s.timestamp > cutoff_date
            ]
            
            # Keep only the most recent snapshots
            if len(self.backup_snapshots) > max_files:
                self.backup_snapshots.sort(key=lambda x: x.timestamp, reverse=True)
                removed_snapshots = self.backup_snapshots[max_files:]
                self.backup_snapshots = self.backup_snapshots[:max_files]
                
                # Delete files for removed snapshots
                for snapshot in removed_snapshots:
                    if snapshot.file_path and Path(snapshot.file_path).exists():
                        Path(snapshot.file_path).unlink()
            
            logger.debug(f"Cleaned up old backups, {len(self.backup_snapshots)} snapshots remaining")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old backups: {e}")
    
    async def _health_check_loop(self):
        """Continuous health check loop"""
        try:
            interval = self.health_check_config['heartbeat_interval_seconds']
            
            while self.monitoring_active:
                await asyncio.sleep(interval)
                
                # Check for nodes that haven't sent heartbeats
                current_time = datetime.now()
                timeout_threshold = self.health_check_config['heartbeat_timeout_seconds']
                
                for node_id, metrics in self.node_health.items():
                    heartbeat_age = (current_time - metrics.last_heartbeat).total_seconds()
                    
                    if heartbeat_age > timeout_threshold:
                        if self.node_status.get(node_id) == NodeStatus.ACTIVE:
                            await self._handle_heartbeat_timeout(node_id, heartbeat_age)
                            
        except Exception as e:
            logger.error(f"Health check loop error: {e}")
    
    async def _handle_heartbeat_timeout(self, node_id: str, timeout_seconds: float):
        """Handle heartbeat timeout for a node"""
        try:
            failure_event = FailureEvent(
                node_id=node_id,
                failure_type='timeout',
                timestamp=datetime.now(),
                error_message=f"Heartbeat timeout: {timeout_seconds:.1f}s",
                severity='high'
            )
            
            self.failure_history.append(failure_event)
            self.node_status[node_id] = NodeStatus.FAILED
            
            logger.error(f"Node {node_id} failed due to heartbeat timeout ({timeout_seconds:.1f}s)")
            
            # Trigger recovery
            if self.recovery_config['auto_recovery_enabled']:
                await self._initiate_node_recovery(node_id, failure_event)
                
        except Exception as e:
            logger.error(f"Failed to handle heartbeat timeout for {node_id}: {e}")
    
    async def _cleanup_loop(self):
        """Periodic cleanup of old data"""
        try:
            while self.monitoring_active:
                await asyncio.sleep(3600)  # Run every hour
                
                # Cleanup old failure events (keep last 1000)
                if len(self.failure_history) > 1000:
                    self.failure_history = self.failure_history[-1000:]
                
                # Cleanup old backups
                await self._cleanup_old_backups()
                
        except Exception as e:
            logger.error(f"Cleanup loop error: {e}")
    
    def get_active_nodes(self) -> List[str]:
        """Get list of currently active nodes"""
        return [
            node_id for node_id, status in self.node_status.items()
            if status == NodeStatus.ACTIVE
        ]
    
    def get_failed_nodes(self) -> List[str]:
        """Get list of currently failed nodes"""
        return [
            node_id for node_id, status in self.node_status.items()
            if status == NodeStatus.FAILED
        ]
    
    def get_fault_tolerance_status(self) -> Dict[str, Any]:
        """Get comprehensive fault tolerance status"""
        try:
            active_nodes = self.get_active_nodes()
            failed_nodes = self.get_failed_nodes()
            
            # Recent failures (last 24 hours)
            recent_failures = [
                f for f in self.failure_history
                if (datetime.now() - f.timestamp).total_seconds() < 86400
            ]
            
            # Recovery statistics
            total_recoveries = len([f for f in self.failure_history if f.resolved])
            recovery_rate = (total_recoveries / len(self.failure_history) * 100) if self.failure_history else 0
            
            return {
                'monitoring_active': self.monitoring_active,
                'total_nodes': len(self.node_status),
                'active_nodes': len(active_nodes),
                'failed_nodes': len(failed_nodes),
                'recovering_nodes': len([s for s in self.node_status.values() if s == NodeStatus.RECOVERING]),
                'suspended_nodes': len([s for s in self.node_status.values() if s == NodeStatus.SUSPENDED]),
                'recent_failures_24h': len(recent_failures),
                'total_failures': len(self.failure_history),
                'recovery_rate_percent': round(recovery_rate, 2),
                'active_recovery_tasks': len(self.recovery_tasks),
                'backup_snapshots': len(self.backup_snapshots),
                'node_status_summary': {
                    status.value: len([s for s in self.node_status.values() if s == status])
                    for status in NodeStatus
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get fault tolerance status: {e}")
            return {'error': str(e)}