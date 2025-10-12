"""
Monitoring configuration for NeuroDx-MultiModal system.
"""

from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class NotificationChannel(Enum):
    """Notification channels."""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    PAGERDUTY = "pagerduty"


@dataclass
class AlertThreshold:
    """Alert threshold configuration."""
    warning: float
    critical: float
    duration: str = "5m"  # How long condition must persist


@dataclass
class NotificationConfig:
    """Notification configuration."""
    channel: NotificationChannel
    enabled: bool = True
    config: Dict[str, Any] = None


@dataclass
class MonitoringConfig:
    """Comprehensive monitoring configuration."""
    
    # Monitoring intervals
    system_check_interval: int = 30  # seconds
    health_check_interval: int = 300  # seconds
    metrics_cleanup_interval: int = 3600  # seconds
    
    # Alert thresholds
    thresholds: Dict[str, AlertThreshold] = None
    
    # Notification settings
    notifications: List[NotificationConfig] = None
    
    # Service endpoints
    service_endpoints: Dict[str, str] = None
    
    # Retention settings
    alert_history_limit: int = 1000
    metrics_retention_days: int = 30
    
    # Feature flags
    enable_gpu_monitoring: bool = True
    enable_model_monitoring: bool = True
    enable_security_monitoring: bool = True
    enable_business_monitoring: bool = True
    
    def __post_init__(self):
        """Initialize default values."""
        
        if self.thresholds is None:
            self.thresholds = {
                "cpu_usage": AlertThreshold(warning=80.0, critical=95.0),
                "memory_usage": AlertThreshold(warning=85.0, critical=95.0),
                "disk_usage": AlertThreshold(warning=85.0, critical=95.0),
                "gpu_utilization": AlertThreshold(warning=95.0, critical=99.0),
                "gpu_memory": AlertThreshold(warning=90.0, critical=95.0),
                "response_time": AlertThreshold(warning=2.0, critical=5.0),
                "error_rate": AlertThreshold(warning=0.05, critical=0.1),
                "database_connections": AlertThreshold(warning=80, critical=95),
                "cache_hit_rate": AlertThreshold(warning=70.0, critical=50.0),
                "model_inference_time": AlertThreshold(warning=10.0, critical=30.0),
                "model_error_rate": AlertThreshold(warning=0.1, critical=0.2),
                "request_rate": AlertThreshold(warning=100.0, critical=200.0),
                "queue_size": AlertThreshold(warning=100, critical=500),
                "temperature": AlertThreshold(warning=80.0, critical=90.0)
            }
        
        if self.notifications is None:
            self.notifications = [
                NotificationConfig(
                    channel=NotificationChannel.EMAIL,
                    enabled=True,
                    config={
                        "smtp_server": "localhost:587",
                        "from_email": "alerts@neurodx.example.com",
                        "to_emails": ["admin@neurodx.example.com"],
                        "username": "alerts@neurodx.example.com",
                        "password": "your_email_password"
                    }
                ),
                NotificationConfig(
                    channel=NotificationChannel.SLACK,
                    enabled=False,
                    config={
                        "webhook_url": "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
                        "channel": "#alerts",
                        "username": "NeuroDx Monitor"
                    }
                ),
                NotificationConfig(
                    channel=NotificationChannel.WEBHOOK,
                    enabled=True,
                    config={
                        "url": "http://neurodx-api:5000/api/alerts/webhook",
                        "timeout": 10,
                        "retry_count": 3
                    }
                ),
                NotificationConfig(
                    channel=NotificationChannel.PAGERDUTY,
                    enabled=False,
                    config={
                        "integration_key": "your_pagerduty_integration_key",
                        "severity_mapping": {
                            "critical": "critical",
                            "warning": "warning",
                            "info": "info"
                        }
                    }
                )
            ]
        
        if self.service_endpoints is None:
            self.service_endpoints = {
                "neurodx-api": "http://neurodx-api:5000/api/health",
                "monai-label": "http://monai-label:8000/info/",
                "postgres": "postgres://neurodx_user:neurodx_password@postgres:5432/neurodx",
                "redis": "redis://redis:6379",
                "influxdb": "http://influxdb:8086/ping",
                "minio": "http://minio:9000/minio/health/live",
                "prometheus": "http://prometheus:9090/-/healthy",
                "grafana": "http://grafana:3000/api/health"
            }


# Default monitoring configuration
DEFAULT_MONITORING_CONFIG = MonitoringConfig()


# Alert rule definitions for Prometheus
PROMETHEUS_ALERT_RULES = {
    "groups": [
        {
            "name": "neurodx-system-alerts",
            "rules": [
                {
                    "alert": "HighCPUUsage",
                    "expr": "system_cpu_usage_percent > 80",
                    "for": "5m",
                    "labels": {
                        "severity": "warning",
                        "service": "system"
                    },
                    "annotations": {
                        "summary": "High CPU usage detected",
                        "description": "CPU usage is {{ $value }}% on {{ $labels.instance }}"
                    }
                },
                {
                    "alert": "CriticalCPUUsage",
                    "expr": "system_cpu_usage_percent > 95",
                    "for": "2m",
                    "labels": {
                        "severity": "critical",
                        "service": "system"
                    },
                    "annotations": {
                        "summary": "Critical CPU usage detected",
                        "description": "CPU usage is {{ $value }}% on {{ $labels.instance }}"
                    }
                },
                {
                    "alert": "HighMemoryUsage",
                    "expr": "(system_memory_usage_bytes / (system_memory_usage_bytes + system_memory_available_bytes)) * 100 > 85",
                    "for": "5m",
                    "labels": {
                        "severity": "warning",
                        "service": "system"
                    },
                    "annotations": {
                        "summary": "High memory usage detected",
                        "description": "Memory usage is {{ $value }}% on {{ $labels.instance }}"
                    }
                },
                {
                    "alert": "CriticalMemoryUsage",
                    "expr": "(system_memory_usage_bytes / (system_memory_usage_bytes + system_memory_available_bytes)) * 100 > 95",
                    "for": "2m",
                    "labels": {
                        "severity": "critical",
                        "service": "system"
                    },
                    "annotations": {
                        "summary": "Critical memory usage detected",
                        "description": "Memory usage is {{ $value }}% on {{ $labels.instance }}"
                    }
                },
                {
                    "alert": "HighDiskUsage",
                    "expr": "(system_disk_usage_bytes / (system_disk_usage_bytes + system_disk_free_bytes)) * 100 > 85",
                    "for": "5m",
                    "labels": {
                        "severity": "warning",
                        "service": "storage"
                    },
                    "annotations": {
                        "summary": "High disk usage detected",
                        "description": "Disk usage is {{ $value }}% on {{ $labels.mount_point }}"
                    }
                },
                {
                    "alert": "CriticalDiskUsage",
                    "expr": "(system_disk_usage_bytes / (system_disk_usage_bytes + system_disk_free_bytes)) * 100 > 95",
                    "for": "1m",
                    "labels": {
                        "severity": "critical",
                        "service": "storage"
                    },
                    "annotations": {
                        "summary": "Critical disk usage detected",
                        "description": "Disk usage is {{ $value }}% on {{ $labels.mount_point }}"
                    }
                }
            ]
        },
        {
            "name": "neurodx-service-alerts",
            "rules": [
                {
                    "alert": "ServiceDown",
                    "expr": "up == 0",
                    "for": "1m",
                    "labels": {
                        "severity": "critical",
                        "service": "{{ $labels.job }}"
                    },
                    "annotations": {
                        "summary": "Service {{ $labels.job }} is down",
                        "description": "Service {{ $labels.job }} has been down for more than 1 minute"
                    }
                },
                {
                    "alert": "HighResponseTime",
                    "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2",
                    "for": "5m",
                    "labels": {
                        "severity": "warning",
                        "service": "{{ $labels.job }}"
                    },
                    "annotations": {
                        "summary": "High response time detected",
                        "description": "95th percentile response time is {{ $value }}s for {{ $labels.endpoint }}"
                    }
                },
                {
                    "alert": "HighErrorRate",
                    "expr": "rate(http_requests_total{status_code=~\"5..\"}[5m]) / rate(http_requests_total[5m]) > 0.05",
                    "for": "2m",
                    "labels": {
                        "severity": "critical",
                        "service": "{{ $labels.job }}"
                    },
                    "annotations": {
                        "summary": "High error rate detected",
                        "description": "Error rate is {{ $value | humanizePercentage }} for {{ $labels.endpoint }}"
                    }
                }
            ]
        },
        {
            "name": "neurodx-gpu-alerts",
            "rules": [
                {
                    "alert": "HighGPUUtilization",
                    "expr": "gpu_utilization_percent > 95",
                    "for": "5m",
                    "labels": {
                        "severity": "warning",
                        "service": "gpu"
                    },
                    "annotations": {
                        "summary": "High GPU utilization detected",
                        "description": "GPU {{ $labels.gpu_id }} utilization is {{ $value }}%"
                    }
                },
                {
                    "alert": "HighGPUMemoryUsage",
                    "expr": "(gpu_memory_usage_bytes / gpu_memory_total_bytes) * 100 > 90",
                    "for": "5m",
                    "labels": {
                        "severity": "critical",
                        "service": "gpu"
                    },
                    "annotations": {
                        "summary": "High GPU memory usage detected",
                        "description": "GPU {{ $labels.gpu_id }} memory usage is {{ $value }}%"
                    }
                },
                {
                    "alert": "GPUTemperatureHigh",
                    "expr": "gpu_temperature_celsius > 80",
                    "for": "3m",
                    "labels": {
                        "severity": "warning",
                        "service": "gpu"
                    },
                    "annotations": {
                        "summary": "High GPU temperature detected",
                        "description": "GPU {{ $labels.gpu_id }} temperature is {{ $value }}°C"
                    }
                }
            ]
        },
        {
            "name": "neurodx-ml-alerts",
            "rules": [
                {
                    "alert": "HighModelInferenceFailureRate",
                    "expr": "rate(model_inference_total{status=\"error\"}[5m]) / rate(model_inference_total[5m]) > 0.1",
                    "for": "3m",
                    "labels": {
                        "severity": "warning",
                        "service": "ml-inference"
                    },
                    "annotations": {
                        "summary": "High model inference failure rate",
                        "description": "Model inference failure rate is {{ $value | humanizePercentage }} for {{ $labels.model_type }}"
                    }
                },
                {
                    "alert": "SlowModelInference",
                    "expr": "histogram_quantile(0.95, rate(model_inference_duration_seconds_bucket[5m])) > 30",
                    "for": "5m",
                    "labels": {
                        "severity": "warning",
                        "service": "ml-inference"
                    },
                    "annotations": {
                        "summary": "Slow model inference detected",
                        "description": "95th percentile inference time is {{ $value }}s for {{ $labels.model_type }}"
                    }
                },
                {
                    "alert": "LowModelUsage",
                    "expr": "rate(model_inference_total[1h]) < 1",
                    "for": "30m",
                    "labels": {
                        "severity": "info",
                        "service": "business"
                    },
                    "annotations": {
                        "summary": "Low model usage detected",
                        "description": "Model inference rate is {{ $value }} inferences/hour"
                    }
                }
            ]
        }
    ]
}


# Grafana dashboard configuration
GRAFANA_DASHBOARD_CONFIG = {
    "dashboard": {
        "title": "NeuroDx-MultiModal System Monitoring",
        "refresh": "30s",
        "time": {"from": "now-1h", "to": "now"},
        "panels": [
            {
                "title": "System Health Overview",
                "type": "stat",
                "targets": [
                    {"expr": "up", "legendFormat": "{{job}}"}
                ]
            },
            {
                "title": "Resource Usage",
                "type": "graph",
                "targets": [
                    {"expr": "system_cpu_usage_percent", "legendFormat": "CPU %"},
                    {"expr": "(system_memory_usage_bytes / (system_memory_usage_bytes + system_memory_available_bytes)) * 100", "legendFormat": "Memory %"}
                ]
            },
            {
                "title": "GPU Metrics",
                "type": "graph",
                "targets": [
                    {"expr": "gpu_utilization_percent", "legendFormat": "GPU {{gpu_id}} Utilization"},
                    {"expr": "(gpu_memory_usage_bytes / gpu_memory_total_bytes) * 100", "legendFormat": "GPU {{gpu_id}} Memory"}
                ]
            },
            {
                "title": "Model Performance",
                "type": "graph",
                "targets": [
                    {"expr": "rate(model_inference_total[5m])", "legendFormat": "{{model_type}} inferences/sec"},
                    {"expr": "histogram_quantile(0.95, rate(model_inference_duration_seconds_bucket[5m]))", "legendFormat": "{{model_type}} 95th percentile"}
                ]
            }
        ]
    }
}