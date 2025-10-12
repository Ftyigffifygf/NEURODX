"""
Multi-Factor Authentication (MFA) service for enhanced security.
"""

import pyotp
import qrcode
import io
import base64
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import secrets
import logging

from .audit_logger import AuditLogger

logger = logging.getLogger(__name__)


@dataclass
class MFADevice:
    """MFA device data model."""
    device_id: str
    user_id: str
    device_type: str  # 'totp', 'sms', 'email'
    device_name: str
    secret_key: Optional[str] = None
    phone_number: Optional[str] = None
    email: Optional[str] = None
    is_active: bool = True
    created_at: datetime = None
    last_used: Optional[datetime] = None
    backup_codes: Optional[list] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


class MFAService:
    """
    Multi-Factor Authentication service.
    
    Features:
    - TOTP (Time-based One-Time Password) support
    - SMS and email backup methods
    - Backup codes for recovery
    - Device management and registration
    - Audit logging for all MFA events
    """
    
    def __init__(self, audit_logger: Optional[AuditLogger] = None):
        self.audit_logger = audit_logger or AuditLogger()
        
        # In-memory storage for demo (replace with database in production)
        self.mfa_devices: Dict[str, MFADevice] = {}
        self.pending_registrations: Dict[str, Dict[str, Any]] = {}
    
    def setup_totp_device(self, user_id: str, username: str, 
                         device_name: str = "NeuroDx App") -> Dict[str, Any]:
        """
        Set up TOTP device for user.
        
        Args:
            user_id: User ID
            username: Username for QR code
            device_name: Name for the device
            
        Returns:
            Setup information including QR code and secret
        """
        try:
            # Generate secret key
            secret_key = pyotp.random_base32()
            
            # Create TOTP instance
            totp = pyotp.TOTP(secret_key)
            
            # Generate provisioning URI for QR code
            provisioning_uri = totp.provisioning_uri(
                name=username,
                issuer_name="NeuroDx-MultiModal"
            )
            
            # Generate QR code
            qr = qrcode.QRCode(version=1, box_size=10, border=5)
            qr.add_data(provisioning_uri)
            qr.make(fit=True)
            
            # Convert QR code to base64 image
            img = qr.make_image(fill_color="black", back_color="white")
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            qr_code_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            
            # Generate backup codes
            backup_codes = [secrets.token_hex(4).upper() for _ in range(10)]
            
            # Store pending registration
            device_id = f"MFA_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(4)}"
            
            self.pending_registrations[device_id] = {
                "user_id": user_id,
                "device_type": "totp",
                "device_name": device_name,
                "secret_key": secret_key,
                "backup_codes": backup_codes,
                "created_at": datetime.utcnow()
            }
            
            self.audit_logger.log_security_event(
                event_type="mfa_setup_initiated",
                user_id=user_id,
                details={
                    "device_type": "totp",
                    "device_name": device_name
                }
            )
            
            logger.info(f"TOTP setup initiated for user: {user_id}")
            
            return {
                "device_id": device_id,
                "secret_key": secret_key,
                "qr_code": f"data:image/png;base64,{qr_code_base64}",
                "provisioning_uri": provisioning_uri,
                "backup_codes": backup_codes
            }
            
        except Exception as e:
            logger.error(f"TOTP setup failed for user {user_id}: {e}")
            raise
    
    def verify_totp_setup(self, device_id: str, verification_code: str) -> bool:
        """
        Verify TOTP setup with verification code.
        
        Args:
            device_id: Device ID from setup
            verification_code: 6-digit TOTP code
            
        Returns:
            True if verification successful
        """
        try:
            pending = self.pending_registrations.get(device_id)
            if not pending:
                return False
            
            # Check if setup has expired (30 minutes)
            if datetime.utcnow() - pending["created_at"] > timedelta(minutes=30):
                del self.pending_registrations[device_id]
                return False
            
            # Verify TOTP code
            totp = pyotp.TOTP(pending["secret_key"])
            if not totp.verify(verification_code, valid_window=1):
                return False
            
            # Create MFA device
            mfa_device = MFADevice(
                device_id=device_id,
                user_id=pending["user_id"],
                device_type=pending["device_type"],
                device_name=pending["device_name"],
                secret_key=pending["secret_key"],
                backup_codes=pending["backup_codes"]
            )
            
            self.mfa_devices[device_id] = mfa_device
            
            # Remove pending registration
            del self.pending_registrations[device_id]
            
            self.audit_logger.log_security_event(
                event_type="mfa_device_registered",
                user_id=pending["user_id"],
                details={
                    "device_type": "totp",
                    "device_name": pending["device_name"],
                    "device_id": device_id
                }
            )
            
            logger.info(f"TOTP device registered for user: {pending['user_id']}")
            return True
            
        except Exception as e:
            logger.error(f"TOTP verification failed for device {device_id}: {e}")
            return False
    
    def verify_totp_code(self, user_id: str, verification_code: str) -> bool:
        """
        Verify TOTP code for authentication.
        
        Args:
            user_id: User ID
            verification_code: 6-digit TOTP code
            
        Returns:
            True if code is valid
        """
        try:
            # Find active TOTP device for user
            totp_device = None
            for device in self.mfa_devices.values():
                if (device.user_id == user_id and 
                    device.device_type == "totp" and 
                    device.is_active):
                    totp_device = device
                    break
            
            if not totp_device:
                return False
            
            # Verify TOTP code
            totp = pyotp.TOTP(totp_device.secret_key)
            is_valid = totp.verify(verification_code, valid_window=1)
            
            if is_valid:
                totp_device.last_used = datetime.utcnow()
                
                self.audit_logger.log_authentication_event(
                    event_type="mfa_verification_success",
                    user_id=user_id,
                    success=True,
                    details={
                        "device_type": "totp",
                        "device_id": totp_device.device_id
                    }
                )
            else:
                self.audit_logger.log_authentication_event(
                    event_type="mfa_verification_failed",
                    user_id=user_id,
                    success=False,
                    details={
                        "device_type": "totp",
                        "reason": "invalid_code"
                    }
                )
            
            return is_valid
            
        except Exception as e:
            logger.error(f"TOTP verification failed for user {user_id}: {e}")
            return False
    
    def verify_backup_code(self, user_id: str, backup_code: str) -> bool:
        """
        Verify backup code for authentication.
        
        Args:
            user_id: User ID
            backup_code: Backup recovery code
            
        Returns:
            True if backup code is valid
        """
        try:
            # Find MFA device for user
            user_device = None
            for device in self.mfa_devices.values():
                if device.user_id == user_id and device.is_active:
                    user_device = device
                    break
            
            if not user_device or not user_device.backup_codes:
                return False
            
            # Check if backup code exists and remove it (one-time use)
            backup_code_upper = backup_code.upper().strip()
            if backup_code_upper in user_device.backup_codes:
                user_device.backup_codes.remove(backup_code_upper)
                user_device.last_used = datetime.utcnow()
                
                self.audit_logger.log_authentication_event(
                    event_type="mfa_backup_code_used",
                    user_id=user_id,
                    success=True,
                    details={
                        "device_id": user_device.device_id,
                        "remaining_codes": len(user_device.backup_codes)
                    }
                )
                
                logger.info(f"Backup code used for user: {user_id}")
                return True
            
            self.audit_logger.log_authentication_event(
                event_type="mfa_backup_code_failed",
                user_id=user_id,
                success=False,
                details={"reason": "invalid_code"}
            )
            
            return False
            
        except Exception as e:
            logger.error(f"Backup code verification failed for user {user_id}: {e}")
            return False
    
    def is_mfa_enabled(self, user_id: str) -> bool:
        """Check if MFA is enabled for user."""
        return any(
            device.user_id == user_id and device.is_active
            for device in self.mfa_devices.values()
        )
    
    def get_user_mfa_devices(self, user_id: str) -> list:
        """Get all MFA devices for user."""
        devices = []
        for device in self.mfa_devices.values():
            if device.user_id == user_id and device.is_active:
                devices.append({
                    "device_id": device.device_id,
                    "device_type": device.device_type,
                    "device_name": device.device_name,
                    "created_at": device.created_at.isoformat(),
                    "last_used": device.last_used.isoformat() if device.last_used else None,
                    "backup_codes_remaining": len(device.backup_codes) if device.backup_codes else 0
                })
        return devices
    
    def disable_mfa_device(self, user_id: str, device_id: str) -> bool:
        """
        Disable MFA device for user.
        
        Args:
            user_id: User ID
            device_id: Device ID to disable
            
        Returns:
            True if device disabled successfully
        """
        try:
            device = self.mfa_devices.get(device_id)
            if not device or device.user_id != user_id:
                return False
            
            device.is_active = False
            
            self.audit_logger.log_security_event(
                event_type="mfa_device_disabled",
                user_id=user_id,
                details={
                    "device_type": device.device_type,
                    "device_name": device.device_name,
                    "device_id": device_id
                }
            )
            
            logger.info(f"MFA device disabled for user: {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to disable MFA device {device_id}: {e}")
            return False
    
    def generate_new_backup_codes(self, user_id: str, device_id: str) -> Optional[list]:
        """
        Generate new backup codes for user.
        
        Args:
            user_id: User ID
            device_id: Device ID
            
        Returns:
            List of new backup codes
        """
        try:
            device = self.mfa_devices.get(device_id)
            if not device or device.user_id != user_id or not device.is_active:
                return None
            
            # Generate new backup codes
            new_backup_codes = [secrets.token_hex(4).upper() for _ in range(10)]
            device.backup_codes = new_backup_codes
            
            self.audit_logger.log_security_event(
                event_type="mfa_backup_codes_regenerated",
                user_id=user_id,
                details={
                    "device_id": device_id,
                    "codes_count": len(new_backup_codes)
                }
            )
            
            logger.info(f"New backup codes generated for user: {user_id}")
            return new_backup_codes
            
        except Exception as e:
            logger.error(f"Failed to generate backup codes for user {user_id}: {e}")
            return None