"""
Comprehensive tests for the monitoring system.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

# Mock the settings import to avoid configuration issues
with patch('src.config.settings.get_settings'):
    from src.services.monitoring.system_monitor import (
        SystemMonitor, AlertSeverity, ServiceStatus, SystemAlert, ServiceHealth
    )

from src.config.monitoring_config import DEFAULT_MONITORING_CONFIG


class TestSystemMonitor:
    """Test system monitoring functionality."""
    
    @pytest.fixture
    def monitor(self):
        """Create a system monitor instance for testing."""
        return SystemMonitor()
    
    def test_monitor_initialization(self, monitor):
        """Test monitor initialization."""
        assert monitor.monitoring_enabled is True
        assert len(monitor.active_alerts) == 0
        assert len(monitor.alert_history) == 0
        assert len(monitor.service_health_cache) == 0
        assert "cpu_usage" in monitor.thresholds
        assert "memory_usage" in monitor.thresholds
    
    @pytest.mark.asyncio
    async def test_collect_system_metrics(self, monitor):
        """Test system metrics collection."""
        with patch('psutil.cpu_percent', return_value=45.5), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_partitions', return_value=[]), \
             patch('psutil.net_io_counters') as mock_network, \
             patch.object(monitor, 'get_gpu_metrics', return_value=[]):
            
            # Mock memory info
            mock_memory.return_value = Mock(
                percent=65.2,
                total=16000000000,
                used=10400000000,
                available=5600000000
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
            assert metrics.network_io["bytes_recv"] == 2000000
            assert isinstance(metrics.timestamp, datetime)
    
    def test_get_gpu_metrics(self, monitor):
        """Test GPU metrics collection."""
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
    async def test_check_http_service(self, monitor):
        """Test HTTP service health checking."""
        with patch('requests.get') as mock_get:
            # Mock successful response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {'content-type': 'application/json'}
            mock_response.json.return_value = {"status": "healthy"}
            mock_get.return_value = mock_response
            
            health = await monitor.check_http_service("test-service", "http://test:8000/health")
            
            assert health.service_name == "test-service"
            assert health.status == ServiceStatus.HEALTHY
            assert health.response_time > 0
            assert health.details == {"status": "healthy"}
            assert health.error_message is None
    
    @pytest.mark.asyncio
    async def test_check_http_service_failure(self, monitor):
        """Test HTTP service health checking with failure."""
        with patch('requests.get', side_effect=ConnectionError("Connection failed")):
            health = await monitor.check_http_service("test-service", "http://test:8000/health")
            
            assert health.service_name == "test-service"
            assert health.status == ServiceStatus.UNHEALTHY
            assert health.error_message == "Connection failed"
    
    @pytest.mark.asyncio
    async def test_trigger_alert(self, monitor):
        """Test alert triggering."""
        await monitor.trigger_alert(
            "test_alert",
            AlertSeverity.WARNING,
            "test-service",
            "cpu_usage",
            85.5,
            80.0,
            "High CPU usage detected"
        )
        
        assert "test_alert" in monitor.active_alerts
        alert = monitor.active_alerts["test_alert"]
        assert alert.severity == AlertSeverity.WARNING
        assert alert.service == "test-service"
        assert alert.metric == "cpu_usage"
        assert alert.value == 85.5
        assert alert.threshold == 80.0
        assert not alert.resolved
    
    @pytest.mark.asyncio
    async def test_resolve_alert(self, monitor):
        """Test alert resolution."""
        # First trigger an alert
        await monitor.trigger_alert(
            "test_alert",
            AlertSeverity.WARNING,
            "test-service",
            "cpu_usage",
            85.5,
            80.0,
            "High CPU usage detected"
        )
        
        # Then resolve it
        await monitor.resolve_alert("test_alert")
        
        assert "test_alert" not in monitor.active_alerts
        assert len(monitor.alert_history) == 1
        assert monitor.alert_history[0].resolved is True
        assert monitor.alert_history[0].resolved_at is not None
    
    @pytest.mark.asyncio
    async def test_check_alerts_cpu_warning(self, monitor):
        """Test CPU usage alert checking."""
        # Create mock metrics with high CPU usage
        metrics = Mock()
        metrics.cpu_usage = 85.0  # Above warning threshold
        metrics.memory_usage = 50.0
        metrics.disk_usage = {}
        metrics.gpu_metrics = []
        metrics.service_health = {}
        
        await monitor.check_alerts(metrics)
        
        assert "cpu_usage_warning" in monitor.active_alerts
        assert "cpu_usage_critical" not in monitor.active_alerts
    
    @pytest.mark.asyncio
    async def test_check_alerts_cpu_critical(self, monitor):
        """Test CPU usage critical alert checking."""
        # Create mock metrics with critical CPU usage
        metrics = Mock()
        metrics.cpu_usage = 97.0  # Above critical threshold
        metrics.memory_usage = 50.0
        metrics.disk_usage = {}
        metrics.gpu_metrics = []
        metrics.service_health = {}
        
        await monitor.check_alerts(metrics)
        
        assert "cpu_usage_critical" in monitor.active_alerts
        assert monitor.active_alerts["cpu_usage_critical"].severity == AlertSeverity.CRITICAL
    
    @pytest.mark.asyncio
    async def test_check_alerts_service_down(self, monitor):
        """Test service down alert checking."""
        # Create mock metrics with unhealthy service
        unhealthy_service = ServiceHealth(
            service_name="test-service",
            status=ServiceStatus.UNHEALTHY,
            response_time=0.0,
            last_check=datetime.utcnow(),
            details={},
            error_message="Connection failed"
        )
        
        metrics = Mock()
        metrics.cpu_usage = 50.0
        metrics.memory_usage = 50.0
        metrics.disk_usage = {}
        metrics.gpu_metrics = []
        metrics.service_health = {"test-service": unhealthy_service}
        
        await monitor.check_alerts(metrics)
        
        assert "service_test-service_down" in monitor.active_alerts
        alert = monitor.active_alerts["service_test-service_down"]
        assert alert.severity == AlertSeverity.CRITICAL
        assert "test-service is unhealthy" in alert.message
    
    @pytest.mark.asyncio
    async def test_check_alerts_gpu_memory(self, monitor):
        """Test GPU memory alert checking."""
        # Create mock metrics with high GPU memory usage
        gpu_metrics = [{
            "id": 0,
            "utilization": 50.0,
            "memory_usage_percent": 92.0  # Above critical threshold
        }]
        
        metrics = Mock()
        metrics.cpu_usage = 50.0
        metrics.memory_usage = 50.0
        metrics.disk_usage = {}
        metrics.gpu_metrics = gpu_metrics
        metrics.service_health = {}
        
        await monitor.check_alerts(metrics)
        
        assert "gpu_0_memory_critical" in monitor.active_alerts
        alert = monitor.active_alerts["gpu_0_memory_critical"]
        assert alert.severity == AlertSeverity.CRITICAL
        assert "GPU 0 memory usage" in alert.message
    
    @pytest.mark.asyncio
    async def test_send_alert_notification(self, monitor):
        """Test alert notification sending."""
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
        
        with patch('requests.post') as mock_post:
            mock_post.return_value = Mock(status_code=200)
            
            await monitor.send_alert_notification(alert)
            
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[1]['json']['alert_id'] == "test_alert"
            assert call_args[1]['json']['severity'] == "warning"
            assert call_args[1]['json']['status'] == "firing"
    
    def test_get_current_metrics(self, monitor):
        """Test getting current metrics."""
        with patch.object(monitor, 'collect_system_metrics') as mock_collect:
            mock_metrics = Mock()
            mock_collect.return_value = mock_metrics
            
            # Mock asyncio.get_event_loop and run_until_complete
            with patch('asyncio.get_event_loop') as mock_loop:
                mock_loop.return_value.run_until_complete.return_value = mock_metrics
                
                result = monitor.get_current_metrics()
                
                mock_loop.return_value.run_until_complete.assert_called_once()
    
    def test_get_active_alerts(self, monitor):
        """Test getting active alerts."""
        # Add a test alert
        alert = SystemAlert(
            alert_id="test_alert",
            timestamp=datetime.utcnow(),
            severity=AlertSeverity.WARNING,
            service="test-service",
            metric="cpu_usage",
            value=85.0,
            threshold=80.0,
            message="Test alert"
        )
        monitor.active_alerts["test_alert"] = alert
        
        alerts = monitor.get_active_alerts()
        
        assert len(alerts) == 1
        assert alerts[0]['alert_id'] == "test_alert"
        assert alerts[0]['severity'] == AlertSeverity.WARNING
    
    def test_get_service_health(self, monitor):
        """Test getting service health."""
        # Add test service health
        health = ServiceHealth(
            service_name="test-service",
            status=ServiceStatus.HEALTHY,
            response_time=0.1,
            last_check=datetime.utcnow(),
            details={"version": "1.0.0"}
        )
        monitor.service_health_cache["test-service"] = health
        
        service_health = monitor.get_service_health()
        
        assert len(service_health) == 1
        assert service_health["test-service"]['service_name'] == "test-service"
        assert service_health["test-service"]['status'] == ServiceStatus.HEALTHY
    
    def test_stop_monitoring(self, monitor):
        """Test stopping monitoring."""
        monitor.monitoring_enabled = True
        monitor.stop_monitoring()
        assert monitor.monitoring_enabled is False


class TestMonitoringConfig:
    """Test monitoring configuration."""
    
    def test_default_config(self):
        """Test default monitoring configuration."""
        config = DEFAULT_MONITORING_CONFIG
        
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
        email_notification = next(
            (n for n in config.notifications if n.channel.value == "email"), 
            None
        )
        assert email_notification is not None
        assert email_notification.enabled is True
        
        # Test service endpoints
        assert "neurodx-api" in config.service_endpoints
        assert "monai-label" in config.service_endpoints


@pytest.mark.asyncio
async def test_monitoring_integration():
    """Test full monitoring system integration."""
    monitor = SystemMonitor()
    
    # Test that we can collect metrics and check alerts
    with patch('psutil.cpu_percent', return_value=90.0), \
         patch('psutil.virtual_memory') as mock_memory, \
         patch('psutil.disk_partitions', return_value=[]), \
         patch('psutil.net_io_counters') as mock_network, \
         patch.object(monitor, 'get_gpu_metrics', return_value=[]):
        
        mock_memory.return_value = Mock(
            percent=88.0,  # Above warning threshold
            total=16000000000,
            used=14080000000,
            available=1920000000
        )
        
        mock_network.return_value = Mock(
            bytes_sent=1000000,
            bytes_recv=2000000,
            packets_sent=5000,
            packets_recv=7000
        )
        
        # Collect metrics
        metrics = await monitor.collect_system_metrics()
        
        # Check alerts
        await monitor.check_alerts(metrics)
        
        # Should have CPU and memory warnings
        assert "cpu_usage_warning" in monitor.active_alerts
        assert "memory_usage_warning" in monitor.active_alerts
        
        # Test alert resolution when metrics improve
        with patch('psutil.cpu_percent', return_value=50.0):
            mock_memory.return_value.percent = 60.0
            
            metrics = await monitor.collect_system_metrics()
            await monitor.check_alerts(metrics)
            
            # Alerts should be resolved
            assert "cpu_usage_warning" not in monitor.active_alerts
            assert "memory_usage_warning" not in monitor.active_alerts


if __name__ == "__main__":
    pytest.main([__file__, "-v"])