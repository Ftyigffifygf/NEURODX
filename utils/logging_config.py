"""
Logging configuration for NeuroDx-MultiModal system.
"""

import logging
import logging.config
import sys
from pathlib import Path
from typing import Dict, Any
import structlog
import colorlog
from src.config.settings import get_settings

settings = get_settings()


def setup_logging() -> None:
    """Configure structured logging for the application."""
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if settings.logging.log_format == "json" 
            else structlog.dev.ConsoleRenderer(colors=True)
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    logging_config = get_logging_config()
    logging.config.dictConfig(logging_config)
    
    # Set up HIPAA audit logging
    setup_audit_logging()


def get_logging_config() -> Dict[str, Any]:
    """Get logging configuration dictionary."""
    
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            },
            "detailed": {
                "format": "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s"
            },
            "json": {
                "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "format": "%(asctime)s %(name)s %(levelname)s %(message)s"
            },
            "colored": {
                "()": "colorlog.ColoredFormatter",
                "format": "%(log_color)s%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                "log_colors": {
                    "DEBUG": "cyan",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "red,bg_white",
                }
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": settings.logging.log_level,
                "formatter": "colored" if settings.logging.log_format != "json" else "json",
                "stream": sys.stdout
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": settings.logging.log_level,
                "formatter": "json" if settings.logging.log_format == "json" else "detailed",
                "filename": "logs/neurodx.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5
            },
            "error_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "ERROR",
                "formatter": "detailed",
                "filename": "logs/errors.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5
            }
        },
        "loggers": {
            "": {  # Root logger
                "handlers": ["console", "file"],
                "level": settings.logging.log_level,
                "propagate": False
            },
            "neurodx": {
                "handlers": ["console", "file", "error_file"],
                "level": settings.logging.log_level,
                "propagate": False
            },
            "monai": {
                "handlers": ["console", "file"],
                "level": settings.monai.log_level,
                "propagate": False
            },
            "nvidia": {
                "handlers": ["console", "file"],
                "level": "INFO",
                "propagate": False
            },
            "flask": {
                "handlers": ["console", "file"],
                "level": "INFO",
                "propagate": False
            },
            "werkzeug": {
                "handlers": ["file"],
                "level": "WARNING",
                "propagate": False
            }
        }
    }
    
    return config


def setup_audit_logging() -> None:
    """Set up HIPAA-compliant audit logging."""
    
    audit_logger = logging.getLogger("neurodx.audit")
    
    # Create audit log handler
    audit_handler = logging.handlers.RotatingFileHandler(
        filename=settings.security.hipaa_audit_log_path,
        maxBytes=50485760,  # 50MB
        backupCount=10
    )
    
    # Audit log format (structured for compliance)
    audit_formatter = logging.Formatter(
        '%(asctime)s|%(levelname)s|%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    audit_handler.setFormatter(audit_formatter)
    
    # Set audit logger configuration
    audit_logger.addHandler(audit_handler)
    audit_logger.setLevel(logging.INFO)
    audit_logger.propagate = False


class AuditLogger:
    """HIPAA-compliant audit logger for medical data access."""
    
    def __init__(self):
        self.logger = logging.getLogger("neurodx.audit")
    
    def log_data_access(
        self, 
        user_id: str, 
        patient_id: str, 
        action: str, 
        resource: str,
        ip_address: str = None,
        success: bool = True
    ) -> None:
        """Log data access events for HIPAA compliance."""
        
        audit_entry = {
            "event_type": "DATA_ACCESS",
            "user_id": user_id,
            "patient_id": patient_id,
            "action": action,
            "resource": resource,
            "ip_address": ip_address,
            "success": success,
            "timestamp": "%(asctime)s"
        }
        
        message = "|".join([
            f"USER:{user_id}",
            f"PATIENT:{patient_id}",
            f"ACTION:{action}",
            f"RESOURCE:{resource}",
            f"IP:{ip_address or 'unknown'}",
            f"SUCCESS:{success}"
        ])
        
        if success:
            self.logger.info(message)
        else:
            self.logger.warning(message)
    
    def log_model_inference(
        self, 
        user_id: str, 
        patient_id: str, 
        model_type: str,
        confidence_score: float = None
    ) -> None:
        """Log AI model inference events."""
        
        message = "|".join([
            f"USER:{user_id}",
            f"PATIENT:{patient_id}",
            f"MODEL:{model_type}",
            f"CONFIDENCE:{confidence_score or 'N/A'}",
            "ACTION:MODEL_INFERENCE"
        ])
        
        self.logger.info(message)
    
    def log_annotation_activity(
        self, 
        user_id: str, 
        patient_id: str, 
        annotation_type: str,
        action: str
    ) -> None:
        """Log annotation activities for MONAI Label."""
        
        message = "|".join([
            f"USER:{user_id}",
            f"PATIENT:{patient_id}",
            f"ANNOTATION_TYPE:{annotation_type}",
            f"ACTION:{action}"
        ])
        
        self.logger.info(message)


class PerformanceLogger:
    """Logger for performance metrics and monitoring."""
    
    def __init__(self):
        self.logger = logging.getLogger("neurodx.performance")
    
    def log_inference_time(
        self, 
        model_type: str, 
        inference_time: float, 
        input_size: tuple,
        gpu_memory_used: float = None
    ) -> None:
        """Log model inference performance metrics."""
        
        self.logger.info(
            f"INFERENCE|MODEL:{model_type}|TIME:{inference_time:.3f}s|"
            f"INPUT_SIZE:{input_size}|GPU_MEMORY:{gpu_memory_used or 'N/A'}MB"
        )
    
    def log_preprocessing_time(
        self, 
        operation: str, 
        processing_time: float, 
        data_size: int
    ) -> None:
        """Log data preprocessing performance metrics."""
        
        self.logger.info(
            f"PREPROCESSING|OPERATION:{operation}|TIME:{processing_time:.3f}s|"
            f"DATA_SIZE:{data_size}"
        )
    
    def log_api_request(
        self, 
        endpoint: str, 
        method: str, 
        response_time: float,
        status_code: int,
        user_id: str = None
    ) -> None:
        """Log API request performance metrics."""
        
        self.logger.info(
            f"API_REQUEST|ENDPOINT:{endpoint}|METHOD:{method}|"
            f"TIME:{response_time:.3f}s|STATUS:{status_code}|USER:{user_id or 'anonymous'}"
        )


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance."""
    return logging.getLogger(f"neurodx.{name}")


def get_audit_logger() -> AuditLogger:
    """Get audit logger instance."""
    return AuditLogger()


def get_performance_logger() -> PerformanceLogger:
    """Get performance logger instance."""
    return PerformanceLogger()


# Initialize logging on module import
setup_logging()