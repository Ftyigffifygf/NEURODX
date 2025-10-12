"""
Metrics and monitoring endpoints for NeuroDx-MultiModal API.
"""

import time
import psutil
import GPUtil
from datetime import datetime
from typing import Dict, Any, List
from flask import Blueprint, jsonify, request
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

from src.utils.logging_config import get_logger
from src.config.settings import get_settings

logger = get_logger(__name__)
metrics_bp = Blueprint('metrics', __name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint']
)

ACTIVE_CONNECTIONS = Gauge(
    'active_connections',
    'Number of active connections'
)

MODEL_INFERENCE_COUNT = Counter(
    'model_inference_total',
    'Total model inference requests',
    ['model_type', 'status']
)

MODEL_INFERENCE_DURATION = Histogram(
    'model_inference_duration_seconds',
    'Model inference duration in seconds',
    ['model_type']
)

GPU_UTILIZATION = Gauge(
    'gpu_utilization_percent',
    'GPU utilization percentage',
    ['gpu_id']
)

GPU_MEMORY_USAGE = Gauge(
    'gpu_memory_usage_bytes',
    'GPU memory usage in bytes',
    ['gpu_id']
)

SYSTEM_CPU_USAGE = Gauge(
    'system_cpu_usage_percent',
    'System CPU usage percentage'
)

SYSTEM_MEMORY_USAGE = Gauge(
    'system_memory_usage_bytes',
    'System memory usage in bytes'
)

SYSTEM_DISK_USAGE = Gauge(
    'system_disk_usage_bytes',
    'System disk usage in bytes',
    ['mount_point']
)

DATABASE_CONNECTIONS = Gauge(
    'database_connections_active',
    'Active database connections',
    ['database']
)

CACHE_HIT_RATE = Gauge(
    'cache_hit_rate_percent',
    'Cache hit rate percentage',
    ['cache_type']
)


@metrics_bp.route('/metrics')
def prometheus_metrics():
    """Prometheus metrics endpoint."""
    
    try:
        # Update system metrics before serving
        update_system_metrics()
        
        # Return Prometheus formatted metrics
        return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}
        
    except Exception as e:
        logger.error(f"Error generating metrics: {e}")
        return "Error generating metrics", 500


@metrics_bp.route('/health/detailed')
def detailed_health():
    """Detailed health check with system metrics."""
    
    try:
        health_data = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": get_settings().app_version,
            "system": get_system_metrics(),
            "gpu": get_gpu_metrics(),
            "services": get_service_health(),
            "performance": get_performance_metrics()
        }
        
        # Determine overall health status
        if any(service["status"] != "healthy" for service in health_data["services"].values()):
            health_data["status"] = "degraded"
            
        return jsonify(health_data)
        
    except Exception as e:
        logger.error(f"Error in detailed health check: {e}")
        return jsonify({
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }), 500


def update_system_metrics():
    """Update system metrics for Prometheus."""
    
    try:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        SYSTEM_CPU_USAGE.set(cpu_percent)
        
        # Memory usage
        memory = psutil.virtual_memory()
        SYSTEM_MEMORY_USAGE.set(memory.used)
        
        # Disk usage
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                SYSTEM_DISK_USAGE.labels(mount_point=partition.mountpoint).set(usage.used)
            except PermissionError:
                continue
        
        # GPU metrics
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                GPU_UTILIZATION.labels(gpu_id=str(gpu.id)).set(gpu.load * 100)
                GPU_MEMORY_USAGE.labels(gpu_id=str(gpu.id)).set(gpu.memoryUsed * 1024 * 1024)  # Convert to bytes
        except Exception as e:
            logger.debug(f"GPU metrics not available: {e}")
            
    except Exception as e:
        logger.error(f"Error updating system metrics: {e}")


def get_system_metrics() -> Dict[str, Any]:
    """Get current system metrics."""
    
    try:
        # CPU information
        cpu_info = {
            "usage_percent": psutil.cpu_percent(interval=1),
            "count": psutil.cpu_count(),
            "count_logical": psutil.cpu_count(logical=True),
            "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
        }
        
        # Memory information
        memory = psutil.virtual_memory()
        memory_info = {
            "total": memory.total,
            "available": memory.available,
            "used": memory.used,
            "usage_percent": memory.percent
        }
        
        # Disk information
        disk_info = {}
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disk_info[partition.mountpoint] = {
                    "total": usage.total,
                    "used": usage.used,
                    "free": usage.free,
                    "usage_percent": (usage.used / usage.total) * 100
                }
            except PermissionError:
                continue
        
        # Network information
        network = psutil.net_io_counters()
        network_info = {
            "bytes_sent": network.bytes_sent,
            "bytes_recv": network.bytes_recv,
            "packets_sent": network.packets_sent,
            "packets_recv": network.packets_recv
        }
        
        return {
            "cpu": cpu_info,
            "memory": memory_info,
            "disk": disk_info,
            "network": network_info,
            "boot_time": psutil.boot_time(),
            "uptime": time.time() - psutil.boot_time()
        }
        
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        return {"error": str(e)}


def get_gpu_metrics() -> List[Dict[str, Any]]:
    """Get GPU metrics."""
    
    try:
        gpus = GPUtil.getGPUs()
        gpu_metrics = []
        
        for gpu in gpus:
            gpu_info = {
                "id": gpu.id,
                "name": gpu.name,
                "load": gpu.load,
                "memory_total": gpu.memoryTotal,
                "memory_used": gpu.memoryUsed,
                "memory_free": gpu.memoryFree,
                "temperature": gpu.temperature,
                "uuid": gpu.uuid
            }
            gpu_metrics.append(gpu_info)
            
        return gpu_metrics
        
    except Exception as e:
        logger.debug(f"GPU metrics not available: {e}")
        return []


def get_service_health() -> Dict[str, Dict[str, Any]]:
    """Get health status of dependent services."""
    
    services = {}
    
    # Database health
    try:
        # This would typically check actual database connection
        services["database"] = {
            "status": "healthy",
            "response_time": 0.05,
            "connections": 5  # Mock data
        }
    except Exception as e:
        services["database"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    # Cache health
    try:
        # This would typically check Redis connection
        services["cache"] = {
            "status": "healthy",
            "response_time": 0.01,
            "hit_rate": 85.5  # Mock data
        }
    except Exception as e:
        services["cache"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    # MONAI Label health
    try:
        # This would typically check MONAI Label service
        services["monai_label"] = {
            "status": "healthy",
            "response_time": 0.1,
            "active_tasks": 2  # Mock data
        }
    except Exception as e:
        services["monai_label"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    return services


def get_performance_metrics() -> Dict[str, Any]:
    """Get performance metrics."""
    
    return {
        "requests_per_second": 45.2,  # Mock data
        "average_response_time": 0.125,
        "error_rate": 0.02,
        "active_users": 12,
        "model_inference_rate": 8.5,
        "cache_hit_rate": 85.5
    }


def record_request_metrics(method: str, endpoint: str, status_code: int, duration: float):
    """Record request metrics."""
    
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=str(status_code)).inc()
    REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)


def record_model_inference_metrics(model_type: str, status: str, duration: float):
    """Record model inference metrics."""
    
    MODEL_INFERENCE_COUNT.labels(model_type=model_type, status=status).inc()
    MODEL_INFERENCE_DURATION.labels(model_type=model_type).observe(duration)


# Middleware to automatically record request metrics
@metrics_bp.before_app_request
def before_request():
    """Record request start time."""
    request.start_time = time.time()


@metrics_bp.route('/alerts/webhook', methods=['POST'])
def alert_webhook():
    """Webhook endpoint for receiving alerts from monitoring systems."""
    
    try:
        alert_data = request.get_json()
        
        if not alert_data:
            return jsonify({"error": "No alert data provided"}), 400
        
        # Log the alert
        severity = alert_data.get('severity', 'unknown')
        service = alert_data.get('service', 'unknown')
        message = alert_data.get('message', 'No message')
        status = alert_data.get('status', 'firing')
        
        if status == 'firing':
            logger.warning(f"Alert received - {severity.upper()} [{service}]: {message}")
        else:
            logger.info(f"Alert resolved - {severity.upper()} [{service}]: {message}")
        
        # Here you could add additional alert processing logic:
        # - Store in database
        # - Send to external notification systems
        # - Update dashboards
        # - Trigger automated responses
        
        return jsonify({"status": "received", "alert_id": alert_data.get('alert_id')}), 200
        
    except Exception as e:
        logger.error(f"Error processing alert webhook: {e}")
        return jsonify({"error": "Failed to process alert"}), 500


@metrics_bp.route('/system/status')
def system_status():
    """Get comprehensive system status."""
    
    try:
        from src.services.monitoring.system_monitor import system_monitor
        
        # Get current metrics from system monitor
        metrics = system_monitor.get_current_metrics()
        active_alerts = system_monitor.get_active_alerts()
        service_health = system_monitor.get_service_health()
        
        # Determine overall system status
        critical_alerts = [a for a in active_alerts if a.get('severity') == 'critical']
        unhealthy_services = [s for s in service_health.values() if s.get('status') == 'unhealthy']
        
        if critical_alerts or unhealthy_services:
            overall_status = "critical"
        elif active_alerts or any(s.get('status') == 'degraded' for s in service_health.values()):
            overall_status = "warning"
        else:
            overall_status = "healthy"
        
        return jsonify({
            "overall_status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics,
            "active_alerts": active_alerts,
            "service_health": service_health,
            "summary": {
                "total_services": len(service_health),
                "healthy_services": sum(1 for s in service_health.values() if s.get('status') == 'healthy'),
                "total_alerts": len(active_alerts),
                "critical_alerts": len(critical_alerts)
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return jsonify({
            "overall_status": "unknown",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }), 500


@metrics_bp.route('/alerts')
def get_alerts():
    """Get active system alerts."""
    
    try:
        from src.services.monitoring.system_monitor import system_monitor
        
        active_alerts = system_monitor.get_active_alerts()
        
        return jsonify({
            "alerts": active_alerts,
            "count": len(active_alerts),
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        return jsonify({"error": str(e)}), 500


@metrics_bp.route('/performance')
def performance_metrics():
    """Get detailed performance metrics."""
    
    try:
        # Get system performance data
        performance_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "system": get_system_metrics(),
            "gpu": get_gpu_metrics(),
            "services": get_service_health(),
            "application": get_performance_metrics()
        }
        
        return jsonify(performance_data)
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        return jsonify({"error": str(e)}), 500


@metrics_bp.after_app_request
def after_request(response):
    """Record request metrics after each request."""
    
    try:
        if hasattr(request, 'start_time'):
            duration = time.time() - request.start_time
            
            # Skip metrics endpoint to avoid recursion
            if request.endpoint != 'metrics.prometheus_metrics':
                record_request_metrics(
                    method=request.method,
                    endpoint=request.endpoint or 'unknown',
                    status_code=response.status_code,
                    duration=duration
                )
                
    except Exception as e:
        logger.error(f"Error recording request metrics: {e}")
    
    return response