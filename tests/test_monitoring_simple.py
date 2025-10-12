"""
Simple tests for the monitoring system functionality.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch
from datetime import datetime
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_monitoring_system_basic():
    """Test basic monitoring system functionality without complex dependencies."""
    
    # Test that we can import the basic components
    from src.services.monitoring.system_monitor import AlertSeverity, ServiceStatus, SystemAlert, ServiceHealth
    
    # Test AlertSeverity enum
    assert AlertSeverity.INFO.value == "info"
    assert AlertSeverity.WARNING.value == "warning"
    assert AlertSeverity.CRITICAL.value == "critical"
    
    # Test ServiceStatus enum
    assert ServiceStatus.HEALTHY.value == "healthy"
    assert ServiceStatus.DEGRADED.value == "degraded"
    assert ServiceStatus.UNHEALTHY.value == "unhealthy"
    assert ServiceStatus.UNKNOWN.value == "unknown"
    
    # Test SystemAlert dataclass
    alert = SystemAlert(
        alert_id="test_alert",
        timestamp=datetime.utcnow(),
        severity=AlertSeverity.WARNING,
        service="test-service",
        metric="cpu_usage",
        value=85.0,
        threshold=80.0,
        message="Test alert message"
    )
    
    assert alert.alert_id == "test_alert"
    assert alert.severity == AlertSeverity.WARNING
    assert alert.service == "test-service"
    assert alert.resolved is False
    assert alert.resolved_at is None
    
    # Test ServiceHealth dataclass
    health = ServiceHealth(
        service_name="test-service",
        status=ServiceStatus.HEALTHY,
        response_time=0.1,
        last_check=datetime.utcnow(),
        details={"version": "1.0.0"}
    )
    
    assert health.service_name == "test-service"
    assert health.status == ServiceStatus.HEALTHY
    assert health.response_time == 0.1
    assert health.error_message is None


def test_system_monitor_creation():
    """Test SystemMonitor creation with mocked dependencies."""
    
    # Mock the settings dependency
    mock_settings = Mock()
    mock_settings.app_version = "1.0.0"
    
    with patch('src.config.settings.get_settings', return_value=mock_settings):
        from src.services.monitoring.system_monitor import SystemMonitor
        
        monitor = SystemMonitor()
        
        # Test initialization
        assert monitor.monitoring_enabled is True
        assert len(monitor.active_alerts) == 0
        assert len(monitor.alert_history) == 0
        assert len(monitor.service_health_cache) == 0
        
        # Test thresholds
        assert "cpu_usage" in monitor.thresholds
        assert "memory_usage" in monitor.thresholds
        assert monitor.thresholds["cpu_usage"]["warning"] == 80.0
        assert monitor.thresholds["cpu_usage"]["critical"] == 95.0


@pytest.mark.asyncio
async def test_system_metrics_collection():
    """Test system metrics collection with mocked psutil."""
    
    mock_settings = Mock()
    mock_settings.app_version = "1.0.0"
    
    with patch('src.config.settings.get_settings', return_value=mock_settings):
        from src.services.monitoring.system_monitor import SystemMonitor
        
        monitor = SystemMonitor()
        
        # Mock psutil functions
        with patch('psutil.cpu_percent', return_value=45.5), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_partitions', return_value=[]), \
             patch('psutil.net_io_counters') as mock_network, \
             patch.object(monitor, 'get_gpu_metrics', return_value=[]):
            
            # Mock memory info
            mock_memory.return_value = Mock(
                percent=65.2,
                total=16000000000
            )
            
            # Mock network info
            mock_network.return_value = Mock(
                bytes_sent=1000000,
                bytes_recv=2000000,
                packets_sent=5000,
                packets_recv=7000
            )
            
            metrics = await monitor.collect_system_metrics()
            
            assert metrics.cpu_usage == 45.5
            assert metrics.memory_usage == 65.2
            assert metrics.memory_total == 16000000000
            assert metrics.network_io["bytes_sent"] == 1000000
            assert isinstance(metrics.timestamp, datetime)


@pytest.mark.asyncio
async def test_alert_triggering():
    """Test alert triggering and resolution."""
    
    mock_settings = Mock()
    mock_settings.app_version = "1.0.0"
    
    with patch('src.config.settings.get_settings', return_value=mock_settings):
        from src.services.monitoring.system_monitor import SystemMonitor
        
        monitor = SystemMonitor()
        
        # Mock the notification sending
        with patch.object(monitor, 'send_alert_notification') as mock_send:
            # Trigger an alert
            await monitor.trigger_alert(
                "test_alert",
                monitor.AlertSeverity.WARNING,
                "test-service",
                "cpu_usage",
                85.5,
                80.0,
                "High CPU usage detected"
            )
            
            # Check alert was created
            assert "test_alert" in monitor.active_alerts
            alert = monitor.active_alerts["test_alert"]
            assert alert.severity.value == "warning"
            assert alert.service == "test-service"
            assert alert.value == 85.5
            assert not alert.resolved
            
            # Check notification was sent
            mock_send.assert_called_once()
            
            # Resolve the alert
            with patch.object(monitor, 'send_alert_resolution') as mock_resolve:
                await monitor.resolve_alert("test_alert")
                
                # Check alert was resolved
                assert "test_alert" not in monitor.active_alerts
                assert len(monitor.alert_history) == 1
                assert monitor.alert_history[0].resolved is True
                
                # Check resolution notification was sent
                mock_resolve.assert_called_once()


def test_gpu_metrics():
    """Test GPU metrics collection."""
    
    mock_settings = Mock()
    mock_settings.app_version = "1.0.0"
    
    with patch('src.config.settings.get_settings', return_value=mock_settings):
        from src.services.monitoring.system_monitor import SystemMonitor
        
        monitor = SystemMonitor()
        
        # Mock GPUtil
        with patch('GPUtil.getGPUs') as mock_gpus:
            # Mock GPU data
            mock_gpu = Mock()
            mock_gpu.id = 0
            mock_gpu.name = "NVIDIA RTX 4090"
            mock_gpu.load = 0.75
            mock_gpu.memoryTotal = 24000
            mock_gpu.memoryUsed = 18000
            mock_gpu.memoryFree = 6000
            mock_gpu.temperature = 65
            mock_gpu.uuid = "GPU-12345"
            
            mock_gpus.return_value = [mock_gpu]
            
            gpu_metrics = monitor.get_gpu_metrics()
            
            assert len(gpu_metrics) == 1
            assert gpu_metrics[0]["id"] == 0
            assert gpu_metrics[0]["name"] == "NVIDIA RTX 4090"
            assert gpu_metrics[0]["utilization"] == 75.0
            assert gpu_metrics[0]["memory_usage_percent"] == 75.0
            assert gpu_metrics[0]["temperature"] == 65


@pytest.mark.asyncio
async def test_http_service_health_check():
    """Test HTTP service health checking."""
    
    mock_settings = Mock()
    mock_settings.app_version = "1.0.0"
    
    with patch('src.config.settings.get_settings', return_value=mock_settings):
        from src.services.monitoring.system_monitor import SystemMonitor
        
        monitor = SystemMonitor()
        
        # Test successful health check
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {'content-type': 'application/json'}
            mock_response.json.return_value = {"status": "healthy"}
            mock_get.return_value = mock_response
            
            health = await monitor.check_http_service("test-service", "http://test:8000/health")
            
            assert health.service_name == "test-service"
            assert health.status.value == "healthy"
            assert health.response_time > 0
            assert health.details == {"status": "healthy"}
            assert health.error_message is None
        
        # Test failed health check
        with patch('requests.get', side_effect=ConnectionError("Connection failed")):
            health = await monitor.check_http_service("test-service", "http://test:8000/health")
            
            assert health.service_name == "test-service"
            assert health.status.value == "unhealthy"
            assert health.error_message == "Connection failed"


def test_monitoring_configuration():
    """Test monitoring configuration."""
    
    from src.config.monitoring_config import DEFAULT_MONITORING_CONFIG
    
    config = DEFAULT_MONITORING_CONFIG
    
    # Test basic configuration
    assert config.system_check_interval == 30
    assert config.health_check_interval == 300
    assert config.alert_history_limit == 1000
    assert config.enable_gpu_monitoring is True
    assert config.enable_model_monitoring is True
    
    # Test thresholds
    assert "cpu_usage" in config.thresholds
    assert config.thresholds["cpu_usage"].warning == 80.0
    assert config.thresholds["cpu_usage"].critical == 95.0
    
    # Test notifications
    assert len(config.notifications) > 0
    
    # Test service endpoints
    assert "neurodx-api" in config.service_endpoints
    assert "monai-label" in config.service_endpoints


if __name__ == "__main__":
    pytest.main([__file__, "-v"])