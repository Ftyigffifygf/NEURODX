"""
System monitoring service for NeuroDx-MultiModal.
"""

import asyncio
import time
import psutil
import GPUtil
import requests
import redis
import psycopg2
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import json
import uuid

from src.utils.logging_config import get_logger
from src.config.settings import get_settings

logger = get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class ServiceStatus(Enum):
    """Service status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class SystemAlert:
    """System alert data structure."""
    alert_id: str
    timestamp: datetime
    severity: AlertSeverity
    service: str
    metric: str
    value: float
    threshold: float
    message: str
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class ServiceHealth:
    """Service health data structure."""
    service_name: str
    status: ServiceStatus
    response_time: float
    last_check: datetime
    details: Dict[str, Any]
    error_message: Optional[str] = None


@dataclass
class SystemMetrics:
    """System metrics data structure."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    memory_total: float
    disk_usage: Dict[str, float]
    network_io: Dict[str, int]
    gpu_metrics: List[Dict[str, Any]]
    service_health: Dict[str, ServiceHealth]
    active_alerts: List[SystemAlert]


class SystemMonitor:
    """Comprehensive system monitoring service."""
    
    def __init__(self):
        self.settings = get_settings()
        self.active_alerts: Dict[str, SystemAlert] = {}
        self.alert_history: List[SystemAlert] = []
        self.service_health_cache: Dict[str, ServiceHealth] = {}
        self.monitoring_enabled = True
        
        # Alert thresholds
        self.thresholds = {
            "cpu_usage": {"warning": 80.0, "critical": 95.0},
            "memory_usage": {"warning": 85.0, "critical": 95.0},
            "disk_usage": {"warning": 85.0, "critical": 95.0},
            "gpu_utilization": {"warning": 95.0, "critical": 99.0},
            "gpu_memory": {"warning": 90.0, "critical": 95.0},
            "response_time": {"warning": 2.0, "critical": 5.0},
            "error_rate": {"warning": 0.05, "critical": 0.1},
            "database_connections": {"warning": 80, "critical": 95}
        }
    
    async def start_monitoring(self, interval: int = 30):
        """Start continuous system monitoring."""
        
        logger.info(f"Starting system monitoring with {interval}s interval")
        
        while self.monitoring_enabled:
            try:
                # Collect system metrics
                metrics = await self.collect_system_metrics()
                
                # Check for alerts
                await self.check_alerts(metrics)
                
                # Update service health
                await self.update_service_health()
                
                # Log metrics summary
                self.log_metrics_summary(metrics)
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(interval)
    
    async def collect_system_metrics(self) -> SystemMetrics:
        """Collect comprehensive system metrics."""
        
        timestamp = datetime.utcnow()
        
        # CPU metrics
        cpu_usage = psutil.cpu_percent(interval=1)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        memory_total = memory.total
        
        # Disk metrics
        disk_usage = {}
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disk_usage[partition.mountpoint] = (usage.used / usage.total) * 100
            except PermissionError:
                continue
        
        # Network metrics
        network = psutil.net_io_counters()
        network_io = {
            "bytes_sent": network.bytes_sent,
            "bytes_recv": network.bytes_recv,
            "packets_sent": network.packets_sent,
            "packets_recv": network.packets_recv
        }
        
        # GPU metrics
        gpu_metrics = self.get_gpu_metrics()
        
        # Service health
        service_health = dict(self.service_health_cache)
        
        # Active alerts
        active_alerts = list(self.active_alerts.values())
        
        return SystemMetrics(
            timestamp=timestamp,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            memory_total=memory_total,
            disk_usage=disk_usage,
            network_io=network_io,
            gpu_metrics=gpu_metrics,
            service_health=service_health,
            active_alerts=active_alerts
        )
    
    def get_gpu_metrics(self) -> List[Dict[str, Any]]:
        """Get GPU metrics."""
        
        try:
            gpus = GPUtil.getGPUs()
            gpu_metrics = []
            
            for gpu in gpus:
                gpu_info = {
                    "id": gpu.id,
                    "name": gpu.name,
                    "utilization": gpu.load * 100,
                    "memory_total": gpu.memoryTotal,
                    "memory_used": gpu.memoryUsed,
                    "memory_free": gpu.memoryFree,
                    "memory_usage_percent": (gpu.memoryUsed / gpu.memoryTotal) * 100,
                    "temperature": gpu.temperature,
                    "uuid": gpu.uuid
                }
                gpu_metrics.append(gpu_info)
                
            return gpu_metrics
            
        except Exception as e:
            logger.debug(f"GPU metrics not available: {e}")
            return []
    
    async def update_service_health(self):
        """Update health status of all services."""
        
        services = [
            ("neurodx-api", "http://neurodx-api:5000/api/health"),
            ("monai-label", "http://monai-label:8000/info/"),
            ("postgres", None),  # Special handling
            ("redis", None),     # Special handling
            ("influxdb", "http://influxdb:8086/ping"),
            ("minio", "http://minio:9000/minio/health/live"),
            ("prometheus", "http://prometheus:9090/-/healthy"),
            ("grafana", "http://grafana:3000/api/health")
        ]
        
        for service_name, endpoint in services:
            try:
                if endpoint:
                    health = await self.check_http_service(service_name, endpoint)
                elif service_name == "postgres":
                    health = await self.check_postgres_health()
                elif service_name == "redis":
                    health = await self.check_redis_health()
                else:
                    continue
                    
                self.service_health_cache[service_name] = health
                
            except Exception as e:
                logger.error(f"Error checking {service_name} health: {e}")
                self.service_health_cache[service_name] = ServiceHealth(
                    service_name=service_name,
                    status=ServiceStatus.UNKNOWN,
                    response_time=0.0,
                    last_check=datetime.utcnow(),
                    details={},
                    error_message=str(e)
                )
    
    async def check_http_service(self, service_name: str, endpoint: str) -> ServiceHealth:
        """Check HTTP service health."""
        
        start_time = time.time()
        
        try:
            response = requests.get(endpoint, timeout=10)
            response_time = time.time() - start_time
            
            if response.status_code == 200 or response.status_code == 204:
                status = ServiceStatus.HEALTHY
                details = {}
                
                # Try to parse JSON response
                try:
                    if response.headers.get('content-type') == 'application/json':
                        details = response.json()
                except:
                    pass
                    
                return ServiceHealth(
                    service_name=service_name,
                    status=status,
                    response_time=response_time,
                    last_check=datetime.utcnow(),
                    details=details
                )
            else:
                return ServiceHealth(
                    service_name=service_name,
                    status=ServiceStatus.UNHEALTHY,
                    response_time=response_time,
                    last_check=datetime.utcnow(),
                    details={"status_code": response.status_code},
                    error_message=f"HTTP {response.status_code}"
                )
                
        except Exception as e:
            response_time = time.time() - start_time
            return ServiceHealth(
                service_name=service_name,
                status=ServiceStatus.UNHEALTHY,
                response_time=response_time,
                last_check=datetime.utcnow(),
                details={},
                error_message=str(e)
            )
    
    async def check_postgres_health(self) -> ServiceHealth:
        """Check PostgreSQL health."""
        
        start_time = time.time()
        
        try:
            conn = psycopg2.connect(
                host="postgres",
                port=5432,
                database="neurodx",
                user="neurodx_user",
                password="neurodx_password",
                connect_timeout=10
            )
            
            cursor = conn.cursor()
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'neurodx';")
            table_count = cursor.fetchone()[0]
            
            cursor.close()
            conn.close()
            
            response_time = time.time() - start_time
            
            return ServiceHealth(
                service_name="postgres",
                status=ServiceStatus.HEALTHY,
                response_time=response_time,
                last_check=datetime.utcnow(),
                details={
                    "version": version,
                    "table_count": table_count
                }
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            return ServiceHealth(
                service_name="postgres",
                status=ServiceStatus.UNHEALTHY,
                response_time=response_time,
                last_check=datetime.utcnow(),
                details={},
                error_message=str(e)
            )
    
    async def check_redis_health(self) -> ServiceHealth:
        """Check Redis health."""
        
        start_time = time.time()
        
        try:
            r = redis.Redis(host="redis", port=6379, decode_responses=True, socket_timeout=10)
            
            # Test basic operations
            r.ping()
            r.set("health_check", "test", ex=60)
            value = r.get("health_check")
            r.delete("health_check")
            
            info = r.info()
            
            response_time = time.time() - start_time
            
            return ServiceHealth(
                service_name="redis",
                status=ServiceStatus.HEALTHY,
                response_time=response_time,
                last_check=datetime.utcnow(),
                details={
                    "version": info.get("redis_version"),
                    "connected_clients": info.get("connected_clients"),
                    "used_memory_human": info.get("used_memory_human")
                }
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            return ServiceHealth(
                service_name="redis",
                status=ServiceStatus.UNHEALTHY,
                response_time=response_time,
                last_check=datetime.utcnow(),
                details={},
                error_message=str(e)
            )
    
    async def check_alerts(self, metrics: SystemMetrics):
        """Check for alert conditions and trigger alerts."""
        
        # CPU usage alerts
        if metrics.cpu_usage > self.thresholds["cpu_usage"]["critical"]:
            await self.trigger_alert(
                "cpu_usage_critical",
                AlertSeverity.CRITICAL,
                "system",
                "cpu_usage",
                metrics.cpu_usage,
                self.thresholds["cpu_usage"]["critical"],
                f"Critical CPU usage: {metrics.cpu_usage:.1f}%"
            )
        elif metrics.cpu_usage > self.thresholds["cpu_usage"]["warning"]:
            await self.trigger_alert(
                "cpu_usage_warning",
                AlertSeverity.WARNING,
                "system",
                "cpu_usage",
                metrics.cpu_usage,
                self.thresholds["cpu_usage"]["warning"],
                f"High CPU usage: {metrics.cpu_usage:.1f}%"
            )
        else:
            await self.resolve_alert("cpu_usage_critical")
            await self.resolve_alert("cpu_usage_warning")
        
        # Memory usage alerts
        if metrics.memory_usage > self.thresholds["memory_usage"]["critical"]:
            await self.trigger_alert(
                "memory_usage_critical",
                AlertSeverity.CRITICAL,
                "system",
                "memory_usage",
                metrics.memory_usage,
                self.thresholds["memory_usage"]["critical"],
                f"Critical memory usage: {metrics.memory_usage:.1f}%"
            )
        elif metrics.memory_usage > self.thresholds["memory_usage"]["warning"]:
            await self.trigger_alert(
                "memory_usage_warning",
                AlertSeverity.WARNING,
                "system",
                "memory_usage",
                metrics.memory_usage,
                self.thresholds["memory_usage"]["warning"],
                f"High memory usage: {metrics.memory_usage:.1f}%"
            )
        else:
            await self.resolve_alert("memory_usage_critical")
            await self.resolve_alert("memory_usage_warning")
        
        # Disk usage alerts
        for mount_point, usage in metrics.disk_usage.items():
            alert_id = f"disk_usage_{mount_point.replace('/', '_')}"
            
            if usage > self.thresholds["disk_usage"]["critical"]:
                await self.trigger_alert(
                    f"{alert_id}_critical",
                    AlertSeverity.CRITICAL,
                    "storage",
                    "disk_usage",
                    usage,
                    self.thresholds["disk_usage"]["critical"],
                    f"Critical disk usage on {mount_point}: {usage:.1f}%"
                )
            elif usage > self.thresholds["disk_usage"]["warning"]:
                await self.trigger_alert(
                    f"{alert_id}_warning",
                    AlertSeverity.WARNING,
                    "storage",
                    "disk_usage",
                    usage,
                    self.thresholds["disk_usage"]["warning"],
                    f"High disk usage on {mount_point}: {usage:.1f}%"
                )
            else:
                await self.resolve_alert(f"{alert_id}_critical")
                await self.resolve_alert(f"{alert_id}_warning")
        
        # GPU alerts
        for gpu in metrics.gpu_metrics:
            gpu_id = gpu["id"]
            utilization = gpu["utilization"]
            memory_usage = gpu["memory_usage_percent"]
            
            # GPU utilization alerts
            if utilization > self.thresholds["gpu_utilization"]["critical"]:
                await self.trigger_alert(
                    f"gpu_{gpu_id}_utilization_critical",
                    AlertSeverity.CRITICAL,
                    "gpu",
                    "gpu_utilization",
                    utilization,
                    self.thresholds["gpu_utilization"]["critical"],
                    f"Critical GPU {gpu_id} utilization: {utilization:.1f}%"
                )
            elif utilization > self.thresholds["gpu_utilization"]["warning"]:
                await self.trigger_alert(
                    f"gpu_{gpu_id}_utilization_warning",
                    AlertSeverity.WARNING,
                    "gpu",
                    "gpu_utilization",
                    utilization,
                    self.thresholds["gpu_utilization"]["warning"],
                    f"High GPU {gpu_id} utilization: {utilization:.1f}%"
                )
            else:
                await self.resolve_alert(f"gpu_{gpu_id}_utilization_critical")
                await self.resolve_alert(f"gpu_{gpu_id}_utilization_warning")
            
            # GPU memory alerts
            if memory_usage > self.thresholds["gpu_memory"]["critical"]:
                await self.trigger_alert(
                    f"gpu_{gpu_id}_memory_critical",
                    AlertSeverity.CRITICAL,
                    "gpu",
                    "gpu_memory",
                    memory_usage,
                    self.thresholds["gpu_memory"]["critical"],
                    f"Critical GPU {gpu_id} memory usage: {memory_usage:.1f}%"
                )
            elif memory_usage > self.thresholds["gpu_memory"]["warning"]:
                await self.trigger_alert(
                    f"gpu_{gpu_id}_memory_warning",
                    AlertSeverity.WARNING,
                    "gpu",
                    "gpu_memory",
                    memory_usage,
                    self.thresholds["gpu_memory"]["warning"],
                    f"High GPU {gpu_id} memory usage: {memory_usage:.1f}%"
                )
            else:
                await self.resolve_alert(f"gpu_{gpu_id}_memory_critical")
                await self.resolve_alert(f"gpu_{gpu_id}_memory_warning")
        
        # Service health alerts
        for service_name, health in metrics.service_health.items():
            if health.status == ServiceStatus.UNHEALTHY:
                await self.trigger_alert(
                    f"service_{service_name}_down",
                    AlertSeverity.CRITICAL,
                    service_name,
                    "service_status",
                    0,
                    1,
                    f"Service {service_name} is unhealthy: {health.error_message or 'Unknown error'}"
                )
            elif health.status == ServiceStatus.DEGRADED:
                await self.trigger_alert(
                    f"service_{service_name}_degraded",
                    AlertSeverity.WARNING,
                    service_name,
                    "service_status",
                    0.5,
                    1,
                    f"Service {service_name} is degraded"
                )
            else:
                await self.resolve_alert(f"service_{service_name}_down")
                await self.resolve_alert(f"service_{service_name}_degraded")
            
            # Response time alerts
            if health.response_time > self.thresholds["response_time"]["critical"]:
                await self.trigger_alert(
                    f"service_{service_name}_slow_critical",
                    AlertSeverity.CRITICAL,
                    service_name,
                    "response_time",
                    health.response_time,
                    self.thresholds["response_time"]["critical"],
                    f"Critical response time for {service_name}: {health.response_time:.2f}s"
                )
            elif health.response_time > self.thresholds["response_time"]["warning"]:
                await self.trigger_alert(
                    f"service_{service_name}_slow_warning",
                    AlertSeverity.WARNING,
                    service_name,
                    "response_time",
                    health.response_time,
                    self.thresholds["response_time"]["warning"],
                    f"Slow response time for {service_name}: {health.response_time:.2f}s"
                )
            else:
                await self.resolve_alert(f"service_{service_name}_slow_critical")
                await self.resolve_alert(f"service_{service_name}_slow_warning")
    
    async def trigger_alert(self, alert_id: str, severity: AlertSeverity, service: str, 
                          metric: str, value: float, threshold: float, message: str):
        """Trigger a system alert."""
        
        # Check if alert already exists
        if alert_id in self.active_alerts:
            # Update existing alert
            self.active_alerts[alert_id].value = value
            self.active_alerts[alert_id].timestamp = datetime.utcnow()
            return
        
        # Create new alert
        alert = SystemAlert(
            alert_id=alert_id,
            timestamp=datetime.utcnow(),
            severity=severity,
            service=service,
            metric=metric,
            value=value,
            threshold=threshold,
            message=message
        )
        
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        logger.warning(f"ALERT [{severity.value.upper()}] {service}: {message}")
        
        # Send alert to external systems
        await self.send_alert_notification(alert)
    
    async def resolve_alert(self, alert_id: str):
        """Resolve a system alert."""
        
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.utcnow()
            
            del self.active_alerts[alert_id]
            
            logger.info(f"RESOLVED [{alert.severity.value.upper()}] {alert.service}: {alert.message}")
            
            # Send resolution notification
            await self.send_alert_resolution(alert)
    
    async def send_alert_notification(self, alert: SystemAlert):
        """Send alert notification to external systems."""
        
        try:
            # Send to webhook endpoint
            webhook_url = "http://neurodx-api:5000/api/alerts/webhook"
            
            payload = {
                "alert_id": alert.alert_id,
                "timestamp": alert.timestamp.isoformat(),
                "severity": alert.severity.value,
                "service": alert.service,
                "metric": alert.metric,
                "value": alert.value,
                "threshold": alert.threshold,
                "message": alert.message,
                "status": "firing"
            }
            
            requests.post(webhook_url, json=payload, timeout=5)
            
        except Exception as e:
            logger.error(f"Error sending alert notification: {e}")
    
    async def send_alert_resolution(self, alert: SystemAlert):
        """Send alert resolution notification."""
        
        try:
            # Send to webhook endpoint
            webhook_url = "http://neurodx-api:5000/api/alerts/webhook"
            
            payload = {
                "alert_id": alert.alert_id,
                "timestamp": alert.resolved_at.isoformat(),
                "severity": alert.severity.value,
                "service": alert.service,
                "metric": alert.metric,
                "value": alert.value,
                "threshold": alert.threshold,
                "message": alert.message,
                "status": "resolved"
            }
            
            requests.post(webhook_url, json=payload, timeout=5)
            
        except Exception as e:
            logger.error(f"Error sending alert resolution: {e}")
    
    def log_metrics_summary(self, metrics: SystemMetrics):
        """Log a summary of system metrics."""
        
        logger.info(
            f"System Metrics - CPU: {metrics.cpu_usage:.1f}%, "
            f"Memory: {metrics.memory_usage:.1f}%, "
            f"Active Alerts: {len(metrics.active_alerts)}, "
            f"Healthy Services: {sum(1 for h in metrics.service_health.values() if h.status == ServiceStatus.HEALTHY)}/{len(metrics.service_health)}"
        )
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics as dictionary."""
        
        try:
            loop = asyncio.get_event_loop()
            metrics = loop.run_until_complete(self.collect_system_metrics())
            return asdict(metrics)
        except Exception as e:
            logger.error(f"Error getting current metrics: {e}")
            return {"error": str(e)}
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active alerts as list of dictionaries."""
        
        return [asdict(alert) for alert in self.active_alerts.values()]
    
    def get_service_health(self) -> Dict[str, Dict[str, Any]]:
        """Get service health status as dictionary."""
        
        return {name: asdict(health) for name, health in self.service_health_cache.items()}
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        
        self.monitoring_enabled = False
        logger.info("System monitoring stopped")


# Global system monitor instance
system_monitor = SystemMonitor()