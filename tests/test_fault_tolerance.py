"""
Tests for Fault Tolerance in MONAI Federated Learning

Tests node failure detection, recovery mechanisms, and backup/restore functionality.
"""

import pytest
import asyncio
import tempfile
import shutil
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path

from src.services.federated_learning.fault_tolerance import (
    FaultToleranceManager, NodeHealthMetrics, FailureEvent, BackupSnapshot, NodeStatus
)


class TestNodeHealthMetrics:
    """Test NodeHealthMetrics data class"""
    
    def test_valid_metrics_creation(self):
        """Test creating valid node health metrics"""
        metrics = NodeHealthMetrics(
            node_id="HOSP_A",
            last_heartbeat=datetime.now(),
            response_time_ms=150.0,
            success_rate=95.5,
            error_count=2,
            memory_usage_mb=2048.0,
            cpu_usage_percent=45.0,
            network_latency_ms=50.0,
            model_sync_status="synced"
        )
        
        assert metrics.node_id == "HOSP_A"
        assert metrics.success_rate == 95.5
        assert metrics.model_sync_status == "synced"
    
    def test_invalid_success_rate(self):
        """Test validation of success rate range"""
        with pytest.raises(ValueError, match="Success rate must be between 0 and 100"):
            NodeHealthMetrics(
                node_id="HOSP_A",
                last_heartbeat=datetime.now(),
                response_time_ms=150.0,
                success_rate=150.0,  # Invalid > 100
                error_count=2,
                memory_usage_mb=2048.0,
                cpu_usage_percent=45.0,
                network_latency_ms=50.0,
                model_sync_status="synced"
            )
    
    def test_invalid_cpu_usage(self):
        """Test validation of CPU usage range"""
        with pytest.raises(ValueError, match="CPU usage must be between 0 and 100"):
            NodeHealthMetrics(
                node_id="HOSP_A",
                last_heartbeat=datetime.now(),
                response_time_ms=150.0,
                success_rate=95.0,
                error_count=2,
                memory_usage_mb=2048.0,
                cpu_usage_percent=150.0,  # Invalid > 100
                network_latency_ms=50.0,
                model_sync_status="synced"
            )


class TestFailureEvent:
    """Test FailureEvent data class"""
    
    def test_valid_failure_event_creation(self):
        """Test creating a valid failure event"""
        event = FailureEvent(
            node_id="HOSP_A",
            failure_type="timeout",
            timestamp=datetime.now(),
            error_message="Connection timeout after 60 seconds",
            severity="high"
        )
        
        assert event.node_id == "HOSP_A"
        assert event.failure_type == "timeout"
        assert event.severity == "high"
        assert event.recovery_attempts == 0
        assert event.resolved is False
    
    def test_invalid_failure_type(self):
        """Test validation of failure type"""
        with pytest.raises(ValueError, match="Invalid failure type"):
            FailureEvent(
                node_id="HOSP_A",
                failure_type="invalid_type",
                timestamp=datetime.now(),
                error_message="Test error",
                severity="high"
            )
    
    def test_invalid_severity(self):
        """Test validation of severity"""
        with pytest.raises(ValueError, match="Invalid severity"):
            FailureEvent(
                node_id="HOSP_A",
                failure_type="timeout",
                timestamp=datetime.now(),
                error_message="Test error",
                severity="invalid_severity"
            )


class TestBackupSnapshot:
    """Test BackupSnapshot data class"""
    
    def test_valid_snapshot_creation(self):
        """Test creating a valid backup snapshot"""
        snapshot = BackupSnapshot(
            snapshot_id="snapshot_001",
            timestamp=datetime.now(),
            round_number=5,
            participating_nodes=["HOSP_A", "HOSP_B"],
            global_model_state={"layer1.weight": "tensor_data"},
            aggregation_metrics={"loss": 0.5, "accuracy": 0.85},
            checksum=""  # Will be generated
        )
        
        assert snapshot.snapshot_id == "snapshot_001"
        assert snapshot.round_number == 5
        assert len(snapshot.participating_nodes) == 2
        assert snapshot.checksum != ""  # Should be generated
    
    def test_checksum_generation(self):
        """Test that checksum is automatically generated"""
        snapshot1 = BackupSnapshot(
            snapshot_id="snapshot_001",
            timestamp=datetime.now(),
            round_number=5,
            participating_nodes=["HOSP_A", "HOSP_B"],
            global_model_state={"layer1.weight": "tensor_data"},
            aggregation_metrics={"loss": 0.5, "accuracy": 0.85},
            checksum=""
        )
        
        snapshot2 = BackupSnapshot(
            snapshot_id="snapshot_002",
            timestamp=datetime.now(),
            round_number=5,
            participating_nodes=["HOSP_A", "HOSP_B"],
            global_model_state={"layer1.weight": "tensor_data"},
            aggregation_metrics={"loss": 0.5, "accuracy": 0.85},
            checksum=""
        )
        
        # Same content should produce same checksum
        assert snapshot1.checksum == snapshot2.checksum
        
        # Different content should produce different checksum
        snapshot3 = BackupSnapshot(
            snapshot_id="snapshot_003",
            timestamp=datetime.now(),
            round_number=6,  # Different round number
            participating_nodes=["HOSP_A", "HOSP_B"],
            global_model_state={"layer1.weight": "tensor_data"},
            aggregation_metrics={"loss": 0.5, "accuracy": 0.85},
            checksum=""
        )
        
        assert snapshot1.checksum != snapshot3.checksum


class TestFaultToleranceManager:
    """Test FaultToleranceManager functionality"""
    
    @pytest.fixture
    def temp_backup_dir(self):
        """Create temporary backup directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def fault_manager(self, temp_backup_dir):
        """Create fault tolerance manager for testing"""
        return FaultToleranceManager(backup_directory=temp_backup_dir)
    
    @pytest.fixture
    def sample_health_metrics(self):
        """Create sample health metrics"""
        return NodeHealthMetrics(
            node_id="HOSP_A",
            last_heartbeat=datetime.now(),
            response_time_ms=150.0,
            success_rate=95.0,
            error_count=1,
            memory_usage_mb=2048.0,
            cpu_usage_percent=45.0,
            network_latency_ms=50.0,
            model_sync_status="synced"
        )
    
    def test_manager_initialization(self, fault_manager):
        """Test fault tolerance manager initialization"""
        assert len(fault_manager.node_health) == 0
        assert len(fault_manager.node_status) == 0
        assert len(fault_manager.failure_history) == 0
        assert len(fault_manager.backup_snapshots) == 0
        assert fault_manager.monitoring_active is False
        
        # Check configuration
        assert 'heartbeat_interval_seconds' in fault_manager.health_check_config
        assert 'max_recovery_attempts' in fault_manager.recovery_config
        assert 'backup_interval_rounds' in fault_manager.backup_config
    
    @pytest.mark.asyncio
    async def test_register_node(self, fault_manager, sample_health_metrics):
        """Test node registration"""
        await fault_manager.register_node("HOSP_A", sample_health_metrics)
        
        assert "HOSP_A" in fault_manager.node_health
        assert "HOSP_A" in fault_manager.node_status
        assert fault_manager.node_status["HOSP_A"] == NodeStatus.ACTIVE
        assert fault_manager.node_health["HOSP_A"].node_id == "HOSP_A"
    
    @pytest.mark.asyncio
    async def test_update_node_health(self, fault_manager, sample_health_metrics):
        """Test updating node health metrics"""
        # Register node first
        await fault_manager.register_node("HOSP_A", sample_health_metrics)
        
        # Update with new metrics
        updated_metrics = NodeHealthMetrics(
            node_id="HOSP_A",
            last_heartbeat=datetime.now(),
            response_time_ms=200.0,  # Higher response time
            success_rate=90.0,       # Lower success rate
            error_count=3,           # More errors
            memory_usage_mb=3072.0,  # Higher memory usage
            cpu_usage_percent=60.0,  # Higher CPU usage
            network_latency_ms=100.0, # Higher latency
            model_sync_status="syncing"
        )
        
        await fault_manager.update_node_health("HOSP_A", updated_metrics)
        
        assert fault_manager.node_health["HOSP_A"].response_time_ms == 200.0
        assert fault_manager.node_health["HOSP_A"].success_rate == 90.0
    
    def test_determine_failure_severity(self, fault_manager):
        """Test failure severity determination"""
        # Critical issues
        critical_issues = ["heartbeat timeout: 120s", "connection_lost"]
        assert fault_manager._determine_failure_severity(critical_issues) == 'critical'
        
        # High severity issues
        high_issues = ["high error_count: 15", "high cpu usage"]
        assert fault_manager._determine_failure_severity(high_issues) == 'high'
        
        # Medium severity (multiple issues)
        medium_issues = ["issue1", "issue2", "issue3"]
        assert fault_manager._determine_failure_severity(medium_issues) == 'medium'
        
        # Low severity (single issue)
        low_issues = ["minor issue"]
        assert fault_manager._determine_failure_severity(low_issues) == 'low'
    
    @pytest.mark.asyncio
    async def test_create_backup_snapshot(self, fault_manager):
        """Test creating backup snapshots"""
        model_state = {"layer1.weight": "tensor_data", "layer1.bias": "bias_data"}
        metrics = {"loss": 0.45, "accuracy": 0.88}
        nodes = ["HOSP_A", "HOSP_B", "CLINIC_C"]
        
        snapshot_id = await fault_manager.create_backup_snapshot(
            round_number=10,
            participating_nodes=nodes,
            global_model_state=model_state,
            aggregation_metrics=metrics
        )
        
        assert snapshot_id.startswith("snapshot_10_")
        assert len(fault_manager.backup_snapshots) == 1
        
        snapshot = fault_manager.backup_snapshots[0]
        assert snapshot.round_number == 10
        assert snapshot.participating_nodes == nodes
        assert snapshot.global_model_state == model_state
        assert snapshot.aggregation_metrics == metrics
        assert snapshot.file_path is not None
        assert Path(snapshot.file_path).exists()
    
    @pytest.mark.asyncio
    async def test_restore_from_snapshot(self, fault_manager):
        """Test restoring from backup snapshot"""
        # Create a snapshot first
        model_state = {"layer1.weight": "tensor_data"}
        metrics = {"loss": 0.45}
        nodes = ["HOSP_A", "HOSP_B"]
        
        snapshot_id = await fault_manager.create_backup_snapshot(
            round_number=5,
            participating_nodes=nodes,
            global_model_state=model_state,
            aggregation_metrics=metrics
        )
        
        # Restore from snapshot
        restored_data = await fault_manager.restore_from_snapshot(snapshot_id)
        
        assert restored_data['round_number'] == 5
        assert restored_data['participating_nodes'] == nodes
        assert restored_data['global_model_state'] == model_state
        assert restored_data['aggregation_metrics'] == metrics
        assert 'timestamp' in restored_data
    
    @pytest.mark.asyncio
    async def test_restore_nonexistent_snapshot(self, fault_manager):
        """Test restoring from non-existent snapshot"""
        with pytest.raises(ValueError, match="Snapshot nonexistent not found"):
            await fault_manager.restore_from_snapshot("nonexistent")
    
    def test_get_active_nodes(self, fault_manager):
        """Test getting list of active nodes"""
        # Set up some nodes with different statuses
        fault_manager.node_status = {
            "HOSP_A": NodeStatus.ACTIVE,
            "HOSP_B": NodeStatus.FAILED,
            "CLINIC_C": NodeStatus.ACTIVE,
            "HOSP_D": NodeStatus.RECOVERING
        }
        
        active_nodes = fault_manager.get_active_nodes()
        
        assert len(active_nodes) == 2
        assert "HOSP_A" in active_nodes
        assert "CLINIC_C" in active_nodes
        assert "HOSP_B" not in active_nodes
        assert "HOSP_D" not in active_nodes
    
    def test_get_failed_nodes(self, fault_manager):
        """Test getting list of failed nodes"""
        fault_manager.node_status = {
            "HOSP_A": NodeStatus.ACTIVE,
            "HOSP_B": NodeStatus.FAILED,
            "CLINIC_C": NodeStatus.ACTIVE,
            "HOSP_D": NodeStatus.FAILED
        }
        
        failed_nodes = fault_manager.get_failed_nodes()
        
        assert len(failed_nodes) == 2
        assert "HOSP_B" in failed_nodes
        assert "HOSP_D" in failed_nodes
        assert "HOSP_A" not in failed_nodes
        assert "CLINIC_C" not in failed_nodes
    
    def test_fault_tolerance_status(self, fault_manager):
        """Test getting comprehensive fault tolerance status"""
        # Set up test data
        fault_manager.node_status = {
            "HOSP_A": NodeStatus.ACTIVE,
            "HOSP_B": NodeStatus.FAILED,
            "CLINIC_C": NodeStatus.RECOVERING
        }
        
        # Add some failure history
        fault_manager.failure_history = [
            FailureEvent(
                node_id="HOSP_B",
                failure_type="timeout",
                timestamp=datetime.now() - timedelta(hours=1),
                error_message="Test failure",
                severity="high",
                resolved=True
            ),
            FailureEvent(
                node_id="CLINIC_C",
                failure_type="error",
                timestamp=datetime.now() - timedelta(minutes=30),
                error_message="Another test failure",
                severity="medium"
            )
        ]
        
        status = fault_manager.get_fault_tolerance_status()
        
        assert status['total_nodes'] == 3
        assert status['active_nodes'] == 1
        assert status['failed_nodes'] == 1
        assert status['recovering_nodes'] == 1
        assert status['total_failures'] == 2
        assert status['recovery_rate_percent'] == 50.0  # 1 out of 2 resolved
        assert 'node_status_summary' in status
    
    @pytest.mark.asyncio
    async def test_backup_cleanup(self, fault_manager):
        """Test automatic backup cleanup"""
        # Create multiple snapshots
        for i in range(5):
            await fault_manager.create_backup_snapshot(
                round_number=i,
                participating_nodes=["HOSP_A"],
                global_model_state={"data": f"round_{i}"},
                aggregation_metrics={"loss": 0.5 - i * 0.1}
            )
        
        assert len(fault_manager.backup_snapshots) == 5
        
        # Set max backup files to 3
        fault_manager.backup_config['max_backup_files'] = 3
        
        # Trigger cleanup
        await fault_manager._cleanup_old_backups()
        
        # Should keep only the 3 most recent snapshots
        assert len(fault_manager.backup_snapshots) == 3
        
        # Verify the most recent snapshots are kept
        round_numbers = [s.round_number for s in fault_manager.backup_snapshots]
        assert sorted(round_numbers) == [2, 3, 4]  # Most recent 3
    
    @pytest.mark.asyncio
    @patch('src.services.federated_learning.fault_tolerance.FaultToleranceManager._attempt_node_recovery')
    async def test_node_recovery_success(self, mock_recovery, fault_manager, sample_health_metrics):
        """Test successful node recovery"""
        # Mock successful recovery
        mock_recovery.return_value = True
        
        # Register node
        await fault_manager.register_node("HOSP_A", sample_health_metrics)
        
        # Create failure event
        failure_event = FailureEvent(
            node_id="HOSP_A",
            failure_type="timeout",
            timestamp=datetime.now(),
            error_message="Test failure",
            severity="high"
        )
        
        # Initiate recovery
        await fault_manager._initiate_node_recovery("HOSP_A", failure_event)
        
        # Wait for recovery to complete
        await asyncio.sleep(0.1)
        
        # Check that node is marked as active
        assert fault_manager.node_status["HOSP_A"] == NodeStatus.ACTIVE
        assert failure_event.resolved is True
        assert failure_event.recovery_attempts > 0
    
    @pytest.mark.asyncio
    @patch('src.services.federated_learning.fault_tolerance.FaultToleranceManager._attempt_node_recovery')
    async def test_node_recovery_failure(self, mock_recovery, fault_manager, sample_health_metrics):
        """Test failed node recovery leading to quarantine"""
        # Mock failed recovery
        mock_recovery.return_value = False
        
        # Set low max attempts for faster testing
        fault_manager.recovery_config['max_recovery_attempts'] = 2
        fault_manager.recovery_config['min_recovery_delay_seconds'] = 0.01
        
        # Register node
        await fault_manager.register_node("HOSP_A", sample_health_metrics)
        
        # Create failure event
        failure_event = FailureEvent(
            node_id="HOSP_A",
            failure_type="timeout",
            timestamp=datetime.now(),
            error_message="Test failure",
            severity="high"
        )
        
        # Initiate recovery
        await fault_manager._initiate_node_recovery("HOSP_A", failure_event)
        
        # Wait for recovery attempts to complete
        await asyncio.sleep(0.1)
        
        # Check that node is quarantined after failed recovery
        assert fault_manager.node_status["HOSP_A"] == NodeStatus.SUSPENDED
        assert failure_event.recovery_attempts == 2
        assert failure_event.resolved is False


@pytest.mark.asyncio
async def test_fault_tolerance_integration():
    """Integration test for fault tolerance workflow"""
    with tempfile.TemporaryDirectory() as temp_dir:
        fault_manager = FaultToleranceManager(backup_directory=temp_dir)
        
        # Register multiple nodes
        nodes = ["HOSP_A", "HOSP_B", "CLINIC_C"]
        for node_id in nodes:
            metrics = NodeHealthMetrics(
                node_id=node_id,
                last_heartbeat=datetime.now(),
                response_time_ms=100.0,
                success_rate=95.0,
                error_count=0,
                memory_usage_mb=1024.0,
                cpu_usage_percent=30.0,
                network_latency_ms=25.0,
                model_sync_status="synced"
            )
            await fault_manager.register_node(node_id, metrics)
        
        # Verify all nodes are active
        assert len(fault_manager.get_active_nodes()) == 3
        assert len(fault_manager.get_failed_nodes()) == 0
        
        # Create backup snapshot
        model_state = {"conv1.weight": "tensor_data", "fc.bias": "bias_data"}
        metrics = {"loss": 0.35, "accuracy": 0.92, "dice": 0.88}
        
        snapshot_id = await fault_manager.create_backup_snapshot(
            round_number=15,
            participating_nodes=nodes,
            global_model_state=model_state,
            aggregation_metrics=metrics
        )
        
        # Verify snapshot was created
        assert len(fault_manager.backup_snapshots) == 1
        assert Path(fault_manager.backup_snapshots[0].file_path).exists()
        
        # Simulate node failure
        failed_metrics = NodeHealthMetrics(
            node_id="HOSP_B",
            last_heartbeat=datetime.now() - timedelta(minutes=5),  # Old heartbeat
            response_time_ms=10000.0,  # Very high response time
            success_rate=30.0,         # Low success rate
            error_count=25,            # High error count
            memory_usage_mb=10240.0,   # High memory usage
            cpu_usage_percent=95.0,    # High CPU usage
            network_latency_ms=5000.0, # High latency
            model_sync_status="out_of_sync"
        )
        
        await fault_manager.update_node_health("HOSP_B", failed_metrics)
        
        # Verify node is marked as failed
        assert fault_manager.node_status["HOSP_B"] == NodeStatus.FAILED
        assert len(fault_manager.get_active_nodes()) == 2
        assert len(fault_manager.get_failed_nodes()) == 1
        
        # Verify failure was recorded
        assert len(fault_manager.failure_history) > 0
        latest_failure = fault_manager.failure_history[-1]
        assert latest_failure.node_id == "HOSP_B"
        assert latest_failure.severity in ['high', 'critical']
        
        # Test restore from backup
        restored_data = await fault_manager.restore_from_snapshot(snapshot_id)
        
        assert restored_data['round_number'] == 15
        assert restored_data['participating_nodes'] == nodes
        assert restored_data['global_model_state'] == model_state
        assert restored_data['aggregation_metrics'] == metrics
        
        # Get comprehensive status
        status = fault_manager.get_fault_tolerance_status()
        
        assert status['total_nodes'] == 3
        assert status['active_nodes'] == 2
        assert status['failed_nodes'] == 1
        assert status['total_failures'] >= 1
        assert status['backup_snapshots'] == 1