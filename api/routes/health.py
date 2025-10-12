"""
Health check endpoints for system monitoring.
"""

from flask import Blueprint, jsonify
from datetime import datetime
import psutil
import torch

from src.config.settings import get_settings
from src.utils.logging_config import get_logger

logger = get_logger(__name__)
health_bp = Blueprint('health', __name__)


@health_bp.route('/', methods=['GET'])
def health_check():
    """Basic health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'NeuroDx-MultiModal API'
    })


@health_bp.route('/detailed', methods=['GET'])
def detailed_health_check():
    """Detailed health check with system information."""
    settings = get_settings()
    
    # System metrics
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    # GPU information
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            'available': True,
            'device_count': torch.cuda.device_count(),
            'current_device': torch.cuda.current_device(),
            'device_name': torch.cuda.get_device_name(),
            'memory_allocated': torch.cuda.memory_allocated(),
            'memory_reserved': torch.cuda.memory_reserved()
        }
    else:
        gpu_info = {'available': False}
    
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'NeuroDx-MultiModal API',
        'version': settings.app_version,
        'environment': settings.environment,
        'system': {
            'cpu_percent': cpu_percent,
            'memory': {
                'total': memory.total,
                'available': memory.available,
                'percent': memory.percent
            },
            'disk': {
                'total': disk.total,
                'free': disk.free,
                'percent': (disk.used / disk.total) * 100
            }
        },
        'gpu': gpu_info
    })