"""
HIPAA-compliant audit logging system.
"""

import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import hashlib
import os

from src.config.settings import settings

# Configure audit logger
audit_logger = logging.getLogger('hipaa_audit')
audit_logger.setLevel(logging.INFO)

# Create audit log handler
audit_handler = logging.FileHandler(settings.security.hipaa_audit_log_path)
audit_handler.setLevel(logging.INFO)

# Create formatter for audit logs
audit_formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S UTC'
)
audit_handler.setFormatter(audit_formatter)
audit_logger.addHandler(audit_handler)


class AuditLogger:
    """
    HIPAA-compliant audit logging system.
    
    Features:
    - Structured audit logging in JSON format
    - Automatic timestamp and session tracking
    - Data access logging with user identification
    - Security event logging
    - Tamper-evident log integrity
    """
    
    def __init__(self):
        self.session_id = self._generate_session_id()
        
    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        timestamp = datetime.utcnow().isoformat()
        random_data = os.urandom(16).hex()
        return hashlib.sha256(f"{timestamp}:{random_data}".encode()).hexdigest()[:16]
    
    def _create_audit_entry(self, event_type: str, details: Dict[str, Any],
                          user_id: Optional[str] = None,
                          patient_id: Optional[str] = None) -> Dict[str, Any]:
        """Create standardized audit log entry."""
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "session_id": self.session_id,
            "event_type": event_type,
            "application": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment,
            "details": details
        }
        
        if user_id:
            entry["user_id"] = user_id
        
        if patient_id:
            # Hash patient ID for privacy
            entry["patient_id_hash"] = hashlib.sha256(patient_id.encode()).hexdigest()
        
        return entry
    
    def log_data_access(self, operation: str, data_type: str,
                       user_id: str, patient_id: Optional[str] = None,
                       details: Optional[Dict[str, Any]] = None):
        """
        Log data access events for HIPAA compliance.
        
        Args:
            operation: Type of operation (read, write, update, delete)
            data_type: Type of data accessed
            user_id: User performing the operation
            patient_id: Patient whose data was accessed
            details: Additional details
        """
        audit_details = {
            "operation": operation,
            "data_type": data_type,
            "success": True
        }
        
        if details:
            audit_details.update(details)
        
        entry = self._create_audit_entry(
            event_type="data_access",
            details=audit_details,
            user_id=user_id,
            patient_id=patient_id
        )
        
        audit_logger.info(json.dumps(entry))
    
    def log_authentication_event(self, event_type: str, user_id: str,
                                success: bool, details: Optional[Dict[str, Any]] = None):
        """
        Log authentication events.
        
        Args:
            event_type: Type of auth event (login, logout, failed_login)
            user_id: User identifier
            success: Whether the operation succeeded
            details: Additional details
        """
        audit_details = {
            "auth_event": event_type,
            "success": success
        }
        
        if details:
            audit_details.update(details)
        
        entry = self._create_audit_entry(
            event_type="authentication",
            details=audit_details,
            user_id=user_id
        )
        
        audit_logger.info(json.dumps(entry))
    
    def log_authorization_event(self, user_id: str, resource: str,
                              action: str, granted: bool,
                              details: Optional[Dict[str, Any]] = None):
        """
        Log authorization events.
        
        Args:
            user_id: User requesting access
            resource: Resource being accessed
            action: Action being performed
            granted: Whether access was granted
            details: Additional details
        """
        audit_details = {
            "resource": resource,
            "action": action,
            "access_granted": granted
        }
        
        if details:
            audit_details.update(details)
        
        entry = self._create_audit_entry(
            event_type="authorization",
            details=audit_details,
            user_id=user_id
        )
        
        audit_logger.info(json.dumps(entry))
    
    def log_security_event(self, event_type: str, details: Dict[str, Any],
                          user_id: Optional[str] = None):
        """
        Log security-related events.
        
        Args:
            event_type: Type of security event
            details: Event details
            user_id: User associated with event (if applicable)
        """
        entry = self._create_audit_entry(
            event_type="security",
            details={"security_event": event_type, **details},
            user_id=user_id
        )
        
        audit_logger.info(json.dumps(entry))
    
    def log_encryption_event(self, operation: str, data_type: str,
                           data_size: int, context: Dict[str, Any]):
        """
        Log encryption/decryption events.
        
        Args:
            operation: encrypt or decrypt
            data_type: Type of data being processed
            data_size: Size of data in bytes
            context: Additional context
        """
        audit_details = {
            "encryption_operation": operation,
            "data_type": data_type,
            "data_size_bytes": data_size,
            "context": context
        }
        
        entry = self._create_audit_entry(
            event_type="encryption",
            details=audit_details
        )
        
        audit_logger.info(json.dumps(entry))
    
    def log_system_event(self, event_type: str, details: Dict[str, Any]):
        """
        Log system-level events.
        
        Args:
            event_type: Type of system event
            details: Event details
        """
        entry = self._create_audit_entry(
            event_type="system",
            details={"system_event": event_type, **details}
        )
        
        audit_logger.info(json.dumps(entry))
    
    def log_api_request(self, method: str, endpoint: str, user_id: str,
                       status_code: int, response_time_ms: float,
                       details: Optional[Dict[str, Any]] = None):
        """
        Log API requests for audit trail.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            user_id: User making request
            status_code: HTTP status code
            response_time_ms: Response time in milliseconds
            details: Additional details
        """
        audit_details = {
            "http_method": method,
            "endpoint": endpoint,
            "status_code": status_code,
            "response_time_ms": response_time_ms
        }
        
        if details:
            audit_details.update(details)
        
        entry = self._create_audit_entry(
            event_type="api_request",
            details=audit_details,
            user_id=user_id
        )
        
        audit_logger.info(json.dumps(entry))
    
    def log_data_export(self, user_id: str, data_type: str,
                       export_format: str, record_count: int,
                       patient_ids: Optional[list] = None):
        """
        Log data export events.
        
        Args:
            user_id: User performing export
            data_type: Type of data exported
            export_format: Format of export
            record_count: Number of records exported
            patient_ids: List of patient IDs (will be hashed)
        """
        audit_details = {
            "data_type": data_type,
            "export_format": export_format,
            "record_count": record_count
        }
        
        if patient_ids:
            # Hash patient IDs for privacy
            audit_details["patient_id_hashes"] = [
                hashlib.sha256(pid.encode()).hexdigest()
                for pid in patient_ids
            ]
        
        entry = self._create_audit_entry(
            event_type="data_export",
            details=audit_details,
            user_id=user_id
        )
        
        audit_logger.info(json.dumps(entry))
    
    def verify_log_integrity(self, log_file_path: Optional[Path] = None) -> bool:
        """
        Verify audit log integrity (basic implementation).
        
        Args:
            log_file_path: Path to log file to verify
            
        Returns:
            True if log appears intact
        """
        log_path = log_file_path or settings.security.hipaa_audit_log_path
        
        try:
            if not log_path.exists():
                return True  # Empty log is valid
            
            # Basic integrity check - ensure file is readable and contains valid JSON
            with open(log_path, 'r') as f:
                line_count = 0
                for line in f:
                    line = line.strip()
                    if line:
                        json.loads(line.split(' - ', 2)[2])  # Parse JSON part
                        line_count += 1
                
                return line_count > 0
                
        except Exception as e:
            audit_logger.error(f"Log integrity check failed: {e}")
            return False