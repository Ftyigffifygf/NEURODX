#!/usr/bin/env python3
"""
Demonstration of the monitoring system functionality.
This script shows that the monitoring system is working correctly.
"""

import asyncio
import sys
import os
from datetime import datetime
from unittest.mock import Mock, patch

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_monitoring_components():
    """Test that monitoring components can be imported and used."""
    
    print("=" * 60)
    print("NeuroDx-MultiModal Monitoring System Test")
    print("=" * 60)
    
    # Test 1: Import basic components
    print("\n1. Testing basic component imports...")
    try:
        # Mock the settings to avoid configuration issues
        mock_settings = Mock()
        mock_settings.app_version = "1.0.0"
        
        with patch('src.config.settings.get_settings', return_value=mock_settings):
            from src.services.monitoring.system_monitor import (
                AlertSeverity, ServiceStatus, SystemAlert, ServiceHealth, SystemMonitor
            )
            
        print("✓ Successfully imported AlertSeverity, ServiceStatus, SystemAlert, ServiceHealth, SystemMonitor")
        
        # Test AlertSeverity enum
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.CRITICAL.value == "critical"
        print("✓ AlertSeverity enum working correctly")
        
        # Test ServiceStatus enum
        assert ServiceStatus.HEALTHY.value == "healthy"
        assert ServiceStatus.DEGRADED.value == "degraded"
        assert ServiceStatus.UNHEALTHY.value == "unhealthy"
        print("✓ ServiceStatus enum working correctly")
        
    except Exception as e:
        print(f"✗ Failed to import basic components: {e}")
        return False
    
    # Test 2: Create SystemAlert
    print("\n2. Testing SystemAlert creation...")
    try:
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
        assert alert.resolved is False
        print("✓ SystemAlert creation and validation working correctly")
        
    except Exception as e:
        print(f"✗ Failed to create SystemAlert: {e}")
        return False
    
    # Test 3: Create ServiceHealth
    print("\n3. Testing ServiceHealth creation...")
    try:
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
        print("✓ ServiceHealth creation and validation working correctly")
        
    except Exception as e:
        print(f"✗ Failed to create ServiceHealth: {e}")
        return False
    
    # Test 4: Create SystemMonitor
    print("\n4. Testing SystemMonitor creation...")
    try:
        with patch('src.config.settings.get_settings', return_value=mock_settings):
            monitor = SystemMonitor()
            
        assert monitor.monitoring_enabled is True
        assert len(monitor.active_alerts) == 0
        assert "cpu_usage" in monitor.thresholds
        assert monitor.thresholds["cpu_usage"]["warning"] == 80.0
        print("✓ SystemMonitor creation and initialization working correctly")
        
    except Exception as e:
        print(f"✗ Failed to create SystemMonitor: {e}")
        return False
    
    # Test 5: Test GPU metrics collection
    print("\n5. Testing GPU metrics collection...")
    try:
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
            assert gpu_metrics[0]["utilization"] == 75.0
            assert gpu_metrics[0]["memory_usage_percent"] == 75.0
            print("✓ GPU metrics collection working correctly")
        
    except Exception as e:
        print(f"✗ Failed to test GPU metrics: {e}")
        return False
    
    return True

async def test_async_functionality():
    """Test async functionality of the monitoring system."""
    
    print("\n6. Testing async functionality...")
    
    try:
        mock_settings = Mock()
        mock_settings.app_version = "1.0.0"
        
        with patch('src.config.settings.get_settings', return_value=mock_settings):
            from src.services.monitoring.system_monitor import SystemMonitor, AlertSeverity
            
            monitor = SystemMonitor()
            
            # Test alert triggering
            with patch.object(monitor, 'send_alert_notification') as mock_send:
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
                assert monitor.active_alerts["test_alert"].value == 85.5
                print("✓ Alert triggering working correctly")
                
                # Test alert resolution
                with patch.object(monitor, 'send_alert_resolution') as mock_resolve:
                    await monitor.resolve_alert("test_alert")
                    
                    assert "test_alert" not in monitor.active_alerts
                    assert len(monitor.alert_history) == 1
                    print("✓ Alert resolution working correctly")
            
            # Test system metrics collection
            with patch('psutil.cpu_percent', return_value=45.5), \
                 patch('psutil.virtual_memory') as mock_memory, \
                 patch('psutil.disk_partitions', return_value=[]), \
                 patch('psutil.net_io_counters') as mock_network, \
                 patch.object(monitor, 'get_gpu_metrics', return_value=[]):
                
                mock_memory.return_value = Mock(percent=65.2, total=16000000000)
                mock_network.return_value = Mock(
                    bytes_sent=1000000, bytes_recv=2000000,
                    packets_sent=5000, packets_recv=7000
                )
                
                metrics = await monitor.collect_system_metrics()
                
                assert metrics.cpu_usage == 45.5
                assert metrics.memory_usage == 65.2
                assert isinstance(metrics.timestamp, datetime)
                print("✓ System metrics collection working correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed async functionality test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_monitoring_config():
    """Test monitoring configuration."""
    
    print("\n7. Testing monitoring configuration...")
    
    try:
        from src.config.monitoring_config import DEFAULT_MONITORING_CONFIG
        
        config = DEFAULT_MONITORING_CONFIG
        
        assert config.system_check_interval == 30
        assert config.enable_gpu_monitoring is True
        assert "cpu_usage" in config.thresholds
        assert config.thresholds["cpu_usage"].warning == 80.0
        assert len(config.notifications) > 0
        assert "neurodx-api" in config.service_endpoints
        
        print("✓ Monitoring configuration working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Failed monitoring configuration test: {e}")
        return False

def test_metrics_api():
    """Test metrics API components."""
    
    print("\n8. Testing metrics API components...")
    
    try:
        # Test that we can import the metrics API components
        from src.api.routes.metrics import (
            REQUEST_COUNT, REQUEST_DURATION, SYSTEM_CPU_USAGE,
            get_system_metrics, get_gpu_metrics
        )
        
        print("✓ Metrics API components imported successfully")
        
        # Test system metrics function
        with patch('psutil.cpu_percent', return_value=50.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_partitions', return_value=[]), \
             patch('psutil.net_io_counters') as mock_network:
            
            mock_memory.return_value = Mock(
                total=16000000000, available=8000000000, 
                used=8000000000, percent=50.0
            )
            mock_network.return_value = Mock(
                bytes_sent=1000000, bytes_recv=2000000,
                packets_sent=5000, packets_recv=7000
            )
            
            metrics = get_system_metrics()
            
            assert "cpu" in metrics
            assert "memory" in metrics
            assert metrics["cpu"]["usage_percent"] == 50.0
            print("✓ System metrics API function working correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed metrics API test: {e}")
        return False

async def main():
    """Main test function."""
    
    print("Testing NeuroDx-MultiModal Monitoring System Implementation")
    print(f"Test started at: {datetime.utcnow().isoformat()}")
    
    tests_passed = 0
    total_tests = 5
    
    # Run synchronous tests
    if test_monitoring_components():
        tests_passed += 1
    
    if test_monitoring_config():
        tests_passed += 1
    
    if test_metrics_api():
        tests_passed += 1
    
    # Run async tests
    if await test_async_functionality():
        tests_passed += 1
    
    # Final summary
    print("\n" + "=" * 60)
    print("MONITORING SYSTEM TEST SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("✓ ALL TESTS PASSED - Monitoring system is working correctly!")
        print("\nThe monitoring system includes:")
        print("- Comprehensive system metrics collection (CPU, memory, disk, network, GPU)")
        print("- Alert triggering and resolution with configurable thresholds")
        print("- Service health checking for all system components")
        print("- Prometheus metrics integration")
        print("- Grafana dashboard configuration")
        print("- Automated alerting via email, Slack, webhooks, and PagerDuty")
        print("- Systemd service for continuous monitoring")
        print("- Docker Compose integration for monitoring stack")
        
        print("\nNext steps:")
        print("1. Configure alert notification channels in monitoring/alertmanager.yml")
        print("2. Start the monitoring stack with: ./scripts/start_monitoring.sh")
        print("3. Access Grafana at http://localhost:3000 (admin/admin)")
        print("4. Monitor system status with: ./scripts/monitoring_status.sh")
        
        return True
    else:
        print("✗ Some tests failed - please check the implementation")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)