"""
Tests for Multi-Factor Authentication service.
"""

import pytest
import pyotp
from datetime import datetime, timedelta

from src.services.security.mfa_service import MFAService, MFADevice


class TestMFAService:
    """Test MFA service functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mfa_service = MFAService()
        self.test_user_id = "TEST_USER_001"
        self.test_username = "testuser"
    
    def test_setup_totp_device(self):
        """Test TOTP device setup."""
        # Set up TOTP device
        setup_result = self.mfa_service.setup_totp_device(
            user_id=self.test_user_id,
            username=self.test_username
        )
        
        # Verify setup result
        assert "device_id" in setup_result
        assert "secret_key" in setup_result
        assert "qr_code" in setup_result
        assert "backup_codes" in setup_result
        assert len(setup_result["backup_codes"]) == 10
        
        # Verify pending registration
        device_id = setup_result["device_id"]
        assert device_id in self.mfa_service.pending_registrations
        
        pending = self.mfa_service.pending_registrations[device_id]
        assert pending["user_id"] == self.test_user_id
        assert pending["device_type"] == "totp"
    
    def test_verify_totp_setup_success(self):
        """Test successful TOTP setup verification."""
        # Set up device
        setup_result = self.mfa_service.setup_totp_device(
            user_id=self.test_user_id,
            username=self.test_username
        )
        
        device_id = setup_result["device_id"]
        secret_key = setup_result["secret_key"]
        
        # Generate valid TOTP code
        totp = pyotp.TOTP(secret_key)
        verification_code = totp.now()
        
        # Verify setup
        success = self.mfa_service.verify_totp_setup(device_id, verification_code)
        assert success is True
        
        # Verify device is registered
        assert device_id in self.mfa_service.mfa_devices
        device = self.mfa_service.mfa_devices[device_id]
        assert device.user_id == self.test_user_id
        assert device.is_active is True
        
        # Verify pending registration is removed
        assert device_id not in self.mfa_service.pending_registrations
    
    def test_verify_totp_setup_invalid_code(self):
        """Test TOTP setup verification with invalid code."""
        # Set up device
        setup_result = self.mfa_service.setup_totp_device(
            user_id=self.test_user_id,
            username=self.test_username
        )
        
        device_id = setup_result["device_id"]
        
        # Try to verify with invalid code
        success = self.mfa_service.verify_totp_setup(device_id, "000000")
        assert success is False
        
        # Verify device is not registered
        assert device_id not in self.mfa_service.mfa_devices
        
        # Verify pending registration still exists
        assert device_id in self.mfa_service.pending_registrations
    
    def test_verify_totp_code_success(self):
        """Test successful TOTP code verification."""
        # Set up and register device
        setup_result = self.mfa_service.setup_totp_device(
            user_id=self.test_user_id,
            username=self.test_username
        )
        
        device_id = setup_result["device_id"]
        secret_key = setup_result["secret_key"]
        
        totp = pyotp.TOTP(secret_key)
        verification_code = totp.now()
        
        # Complete setup
        self.mfa_service.verify_totp_setup(device_id, verification_code)
        
        # Generate new code for authentication
        auth_code = totp.now()
        
        # Verify code
        success = self.mfa_service.verify_totp_code(self.test_user_id, auth_code)
        assert success is True
        
        # Verify last_used is updated
        device = self.mfa_service.mfa_devices[device_id]
        assert device.last_used is not None
    
    def test_verify_totp_code_invalid(self):
        """Test TOTP code verification with invalid code."""
        # Set up and register device
        setup_result = self.mfa_service.setup_totp_device(
            user_id=self.test_user_id,
            username=self.test_username
        )
        
        device_id = setup_result["device_id"]
        secret_key = setup_result["secret_key"]
        
        totp = pyotp.TOTP(secret_key)
        verification_code = totp.now()
        
        # Complete setup
        self.mfa_service.verify_totp_setup(device_id, verification_code)
        
        # Try invalid code
        success = self.mfa_service.verify_totp_code(self.test_user_id, "000000")
        assert success is False
    
    def test_verify_backup_code_success(self):
        """Test successful backup code verification."""
        # Set up and register device
        setup_result = self.mfa_service.setup_totp_device(
            user_id=self.test_user_id,
            username=self.test_username
        )
        
        device_id = setup_result["device_id"]
        secret_key = setup_result["secret_key"]
        backup_codes = setup_result["backup_codes"]
        
        totp = pyotp.TOTP(secret_key)
        verification_code = totp.now()
        
        # Complete setup
        self.mfa_service.verify_totp_setup(device_id, verification_code)
        
        # Use backup code
        backup_code = backup_codes[0]
        success = self.mfa_service.verify_backup_code(self.test_user_id, backup_code)
        assert success is True
        
        # Verify backup code is removed (one-time use)
        device = self.mfa_service.mfa_devices[device_id]
        assert backup_code not in device.backup_codes
        assert len(device.backup_codes) == 9
    
    def test_verify_backup_code_invalid(self):
        """Test backup code verification with invalid code."""
        # Set up and register device
        setup_result = self.mfa_service.setup_totp_device(
            user_id=self.test_user_id,
            username=self.test_username
        )
        
        device_id = setup_result["device_id"]
        secret_key = setup_result["secret_key"]
        
        totp = pyotp.TOTP(secret_key)
        verification_code = totp.now()
        
        # Complete setup
        self.mfa_service.verify_totp_setup(device_id, verification_code)
        
        # Try invalid backup code
        success = self.mfa_service.verify_backup_code(self.test_user_id, "INVALID")
        assert success is False
    
    def test_is_mfa_enabled(self):
        """Test MFA enabled check."""
        # Initially not enabled
        assert self.mfa_service.is_mfa_enabled(self.test_user_id) is False
        
        # Set up and register device
        setup_result = self.mfa_service.setup_totp_device(
            user_id=self.test_user_id,
            username=self.test_username
        )
        
        device_id = setup_result["device_id"]
        secret_key = setup_result["secret_key"]
        
        totp = pyotp.TOTP(secret_key)
        verification_code = totp.now()
        
        # Complete setup
        self.mfa_service.verify_totp_setup(device_id, verification_code)
        
        # Now enabled
        assert self.mfa_service.is_mfa_enabled(self.test_user_id) is True
    
    def test_get_user_mfa_devices(self):
        """Test getting user MFA devices."""
        # Set up and register device
        setup_result = self.mfa_service.setup_totp_device(
            user_id=self.test_user_id,
            username=self.test_username
        )
        
        device_id = setup_result["device_id"]
        secret_key = setup_result["secret_key"]
        
        totp = pyotp.TOTP(secret_key)
        verification_code = totp.now()
        
        # Complete setup
        self.mfa_service.verify_totp_setup(device_id, verification_code)
        
        # Get devices
        devices = self.mfa_service.get_user_mfa_devices(self.test_user_id)
        assert len(devices) == 1
        
        device_info = devices[0]
        assert device_info["device_id"] == device_id
        assert device_info["device_type"] == "totp"
        assert device_info["backup_codes_remaining"] == 10
    
    def test_disable_mfa_device(self):
        """Test disabling MFA device."""
        # Set up and register device
        setup_result = self.mfa_service.setup_totp_device(
            user_id=self.test_user_id,
            username=self.test_username
        )
        
        device_id = setup_result["device_id"]
        secret_key = setup_result["secret_key"]
        
        totp = pyotp.TOTP(secret_key)
        verification_code = totp.now()
        
        # Complete setup
        self.mfa_service.verify_totp_setup(device_id, verification_code)
        
        # Disable device
        success = self.mfa_service.disable_mfa_device(self.test_user_id, device_id)
        assert success is True
        
        # Verify device is disabled
        device = self.mfa_service.mfa_devices[device_id]
        assert device.is_active is False
        
        # MFA should no longer be enabled
        assert self.mfa_service.is_mfa_enabled(self.test_user_id) is False
    
    def test_generate_new_backup_codes(self):
        """Test generating new backup codes."""
        # Set up and register device
        setup_result = self.mfa_service.setup_totp_device(
            user_id=self.test_user_id,
            username=self.test_username
        )
        
        device_id = setup_result["device_id"]
        secret_key = setup_result["secret_key"]
        original_codes = setup_result["backup_codes"]
        
        totp = pyotp.TOTP(secret_key)
        verification_code = totp.now()
        
        # Complete setup
        self.mfa_service.verify_totp_setup(device_id, verification_code)
        
        # Generate new backup codes
        new_codes = self.mfa_service.generate_new_backup_codes(self.test_user_id, device_id)
        assert new_codes is not None
        assert len(new_codes) == 10
        assert new_codes != original_codes
        
        # Verify device has new codes
        device = self.mfa_service.mfa_devices[device_id]
        assert device.backup_codes == new_codes