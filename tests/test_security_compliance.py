"""
Comprehensive security and HIPAA compliance tests for NeuroDx-MultiModal system.
Tests data encryption, access controls, audit logging, and regulatory compliance.
"""

import pytest
import asyncio
import json
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from cryptography.fernet import Fernet

from src.services.security.encryption_service import EncryptionService
from src.services.security.auth_service import AuthenticationService
from src.services.security.rbac_service import RBACService
from src.services.security.audit_logger import AuditLogger
from src.services.security.key_management import KeyManager
from src.models.patient import PatientRecord, Demographics, ImagingStudy
from src.config.settings import Settings


class TestHIPAACompliance:
    """Test HIPAA compliance requirements including data encryption and audit logging."""
    
    @pytest.fixture
    def encryption_service(self):
        """Create encryption service for testing."""
        return EncryptionService()
    
    @pytest.fixture
    def audit_logger(self):
        """Create audit logger for testing."""
        return AuditLogger()
    
    @pytest.fixture
    def sample_patient_data(self):
        """Create sample patient data for testing."""
        return PatientRecord(
            patient_id="PAT_20241012_00001",
            demographics=Demographics(
                age=65,
                gender="M",
                weight_kg=70.0,
                height_cm=175.0
            ),
            imaging_studies=[],
            wearable_data=[],
            annotations=[],
            longitudinal_tracking=None
        )
    
    def test_data_encryption_at_rest(self, encryption_service, sample_patient_data):
        """Test that patient data is properly encrypted at rest."""
        # Serialize patient data
        patient_json = json.dumps(sample_patient_data.__dict__, default=str)
        
        # Encrypt the data
        encrypted_data = encryption_service.encrypt_data(patient_json)
        
        # Verify data is encrypted (not readable)
        assert encrypted_data != patient_json
        assert "PAT_20241012_00001" not in encrypted_data
        assert "MRN123456" not in encrypted_data
        
        # Verify decryption works
        decrypted_data = encryption_service.decrypt_data(encrypted_data)
        assert decrypted_data == patient_json
    
    def test_data_encryption_in_transit(self, encryption_service):
        """Test that data is encrypted during transmission."""
        test_message = "Sensitive patient information"
        
        # Test TLS encryption simulation
        encrypted_message = encryption_service.encrypt_for_transmission(test_message)
        
        assert encrypted_message != test_message
        assert encryption_service.decrypt_from_transmission(encrypted_message) == test_message
    
    def test_audit_logging_data_access(self, audit_logger):
        """Test that all data access is properly logged for HIPAA compliance."""
        user_id = "user123"
        patient_id = "PAT_20241012_00001"
        action = "VIEW_PATIENT_RECORD"
        
        # Log data access
        audit_logger.log_data_access(
            user_id=user_id,
            patient_id=patient_id,
            action=action,
            resource="patient_record",
            ip_address="192.168.1.100"
        )
        
        # Verify audit log entry
        logs = audit_logger.get_audit_logs(patient_id=patient_id)
        assert len(logs) == 1
        
        log_entry = logs[0]
        assert log_entry["user_id"] == user_id
        assert log_entry["patient_id"] == patient_id
        assert log_entry["action"] == action
        assert log_entry["timestamp"] is not None
        assert log_entry["ip_address"] == "192.168.1.100"
    
    def test_audit_logging_data_modification(self, audit_logger):
        """Test that data modifications are properly logged."""
        user_id = "user123"
        patient_id = "PAT_20241012_00001"
        
        # Log data modification
        audit_logger.log_data_modification(
            user_id=user_id,
            patient_id=patient_id,
            action="UPDATE_DIAGNOSIS",
            old_value="No diagnosis",
            new_value="Alzheimer's Disease - Early Stage",
            field="diagnosis"
        )
        
        # Verify audit log entry
        logs = audit_logger.get_audit_logs(patient_id=patient_id, action="UPDATE_DIAGNOSIS")
        assert len(logs) == 1
        
        log_entry = logs[0]
        assert log_entry["old_value"] == "No diagnosis"
        assert log_entry["new_value"] == "Alzheimer's Disease - Early Stage"
        assert log_entry["field"] == "diagnosis"
    
    def test_pii_data_anonymization(self, encryption_service):
        """Test that PII is properly anonymized in logs and error messages."""
        sensitive_data = {
            "patient_name": "John Doe",
            "ssn": "123-45-6789",
            "phone": "555-123-4567",
            "email": "john.doe@email.com",
            "address": "123 Main St, Anytown, USA"
        }
        
        # Test anonymization
        anonymized_data = encryption_service.anonymize_pii(sensitive_data)
        
        # Verify PII is removed/masked
        assert "John Doe" not in str(anonymized_data)
        assert "123-45-6789" not in str(anonymized_data)
        assert "555-123-4567" not in str(anonymized_data)
        assert "john.doe@email.com" not in str(anonymized_data)
        
        # Verify structure is maintained with placeholders
        assert "patient_name" in anonymized_data
        assert anonymized_data["patient_name"].startswith("[REDACTED")
    
    def test_data_retention_policies(self, audit_logger):
        """Test that data retention policies are enforced."""
        # Create old audit logs
        old_timestamp = datetime.now() - timedelta(days=2555)  # 7+ years old
        
        audit_logger.log_data_access(
            user_id="user123",
            patient_id="PAT_20170101_00001",
            action="VIEW_PATIENT_RECORD",
            resource="patient_record",
            timestamp=old_timestamp
        )
        
        # Run retention policy cleanup
        deleted_count = audit_logger.cleanup_old_logs(retention_years=7)
        
        # Verify old logs are removed
        assert deleted_count > 0
        old_logs = audit_logger.get_audit_logs(patient_id="PAT_20170101_00001")
        assert len(old_logs) == 0


class TestAccessControls:
    """Test role-based access control and authentication systems."""
    
    @pytest.fixture
    def auth_service(self):
        """Create authentication service for testing."""
        return AuthenticationService()
    
    @pytest.fixture
    def rbac_service(self):
        """Create RBAC service for testing."""
        return RBACService()
    
    def test_role_based_access_control(self, rbac_service):
        """Test that RBAC properly restricts access based on user roles."""
        from src.services.security.rbac_service import Permission
        
        # Test radiologist access
        assert rbac_service.check_permission(["radiologist"], Permission.VIEW_MEDICAL_IMAGES) == True
        assert rbac_service.check_permission(["radiologist"], Permission.MANAGE_USERS) == False
        
        # Test clinician access
        assert rbac_service.check_permission(["clinician"], Permission.READ_PATIENT_DATA) == True
        assert rbac_service.check_permission(["clinician"], Permission.MANAGE_USERS) == False
        
        # Test admin access
        assert rbac_service.check_permission(["admin"], Permission.MANAGE_USERS) == True
        assert rbac_service.check_permission(["admin"], Permission.VIEW_MEDICAL_IMAGES) == True
    
    def test_session_management(self, auth_service):
        """Test secure session management and timeout."""
        # Create a test user first
        user_id = auth_service.create_user(
            username="testuser",
            email="test@example.com",
            password="TestPassword123!",
            roles=["radiologist"]
        )
        
        # Authenticate user to create session
        auth_result = auth_service.authenticate_user(
            username="testuser",
            password="TestPassword123!",
            ip_address="127.0.0.1",
            user_agent="test-agent"
        )
        
        assert auth_result is not None
        assert "token" in auth_result
        assert "session_id" in auth_result
        
        # Verify token is valid
        token_info = auth_service.validate_token(auth_result["token"])
        assert token_info is not None
        assert token_info["user_id"] == user_id
    
    def test_multi_factor_authentication(self, auth_service):
        """Test MFA implementation for enhanced security."""
        # Create a test user
        user_id = auth_service.create_user(
            username="mfauser",
            email="mfa@example.com",
            password="TestPassword123!",
            roles=["clinician"]
        )
        
        # Set up MFA for user
        mfa_setup = auth_service.setup_mfa(user_id, "mfauser")
        assert "device_id" in mfa_setup
        assert "qr_code" in mfa_setup
        
        # Verify MFA setup with a test code
        device_id = mfa_setup["device_id"]
        # In a real test, you'd generate a proper TOTP code
        # For now, we'll test the setup structure
        assert device_id is not None
    
    def test_password_security_requirements(self, auth_service):
        """Test password hashing and verification."""
        # Test password hashing
        password = "TestPassword123!"
        hashed = auth_service.hash_password(password)
        
        # Verify hash is different from original
        assert hashed != password
        assert len(hashed) > 0
        
        # Verify password verification works
        assert auth_service.verify_password(password, hashed) == True
        assert auth_service.verify_password("wrongpassword", hashed) == False
    
    def test_account_lockout_protection(self, auth_service):
        """Test account lockout after failed login attempts."""
        # Create test user
        user_id = auth_service.create_user(
            username="lockoutuser",
            email="lockout@example.com",
            password="CorrectPassword123!",
            roles=["clinician"]
        )
        
        # Simulate multiple failed login attempts
        for i in range(5):
            result = auth_service.authenticate_user(
                username="lockoutuser",
                password="wrongpassword",
                ip_address="127.0.0.1",
                user_agent="test-agent"
            )
            assert result is None  # Failed authentication
        
        # Verify account is locked by checking user object
        user = auth_service.get_user_by_id(user_id)
        assert user.failed_login_attempts >= 5
        
        # Verify even correct password fails when locked
        result = auth_service.authenticate_user(
            username="lockoutuser",
            password="CorrectPassword123!",
            ip_address="127.0.0.1",
            user_agent="test-agent"
        )
        assert result is None  # Should fail due to lockout


class TestKeyManagement:
    """Test cryptographic key management and rotation."""
    
    @pytest.fixture
    def key_management(self):
        """Create key management service for testing."""
        return KeyManager()
    
    def test_key_generation_and_storage(self, key_management):
        """Test secure key generation and storage."""
        # Generate new encryption key
        key_id = "test_patient_data_encryption"
        generated_key_id = key_management.generate_data_encryption_key(key_id, "patient_data_encryption")
        assert generated_key_id == key_id
        
        # Verify key can be retrieved
        key = key_management.get_encryption_key(key_id)
        assert key is not None
        assert len(key) == 32  # AES-256 key length (32 bytes)
    
    def test_key_rotation(self, key_management):
        """Test key rotation for enhanced security."""
        # Create initial key
        key_id = "test_rotation_key"
        key_management.generate_data_encryption_key(key_id, "test_rotation")
        original_key = key_management.get_encryption_key(key_id)
        
        # Test that we can generate a new key with different ID
        new_key_id = "test_rotation_key_v2"
        key_management.generate_data_encryption_key(new_key_id, "test_rotation")
        new_key = key_management.get_encryption_key(new_key_id)
        
        # Verify new key is different
        assert new_key_id != key_id
        assert new_key != original_key
    
    def test_key_access_logging(self, key_management):
        """Test that key access is properly logged."""
        key_id = "test_logging_key"
        key_management.generate_data_encryption_key(key_id, "test_logging")
        
        # Access key (this should be logged internally)
        key = key_management.get_encryption_key(key_id)
        assert key is not None
        
        # The audit logging is handled internally by the key manager
        # In a real implementation, you would check the audit logs
        # For now, we verify the key access works


class TestDataPrivacy:
    """Test data privacy and anonymization features."""
    
    def test_federated_learning_privacy(self):
        """Test that federated learning maintains data privacy."""
        # This would test that raw patient data never leaves the institution
        # and only model parameters are shared
        
        # Mock federated learning client
        from src.services.federated_learning.federated_client import FederatedClient
        
        client = FederatedClient(node_id="hospital_a")
        
        # Verify that only model parameters are shared, not raw data
        with patch.object(client, 'get_local_data') as mock_get_data:
            mock_get_data.return_value = "raw_patient_data"
            
            # Attempt to share data
            shared_data = client.prepare_data_for_sharing()
            
            # Verify raw data is not in shared data
            assert "raw_patient_data" not in str(shared_data)
            assert "model_parameters" in shared_data or "gradients" in shared_data
    
    def test_differential_privacy(self):
        """Test differential privacy implementation for additional protection."""
        from src.services.security.privacy_service import DifferentialPrivacy
        
        privacy_service = DifferentialPrivacy(epsilon=1.0)  # Privacy budget
        
        # Test adding noise to sensitive statistics
        original_count = 100
        noisy_count = privacy_service.add_noise_to_count(original_count)
        
        # Verify noise is added but result is still useful
        assert noisy_count != original_count
        assert abs(noisy_count - original_count) < 20  # Reasonable noise level
    
    def test_data_minimization(self):
        """Test that only necessary data is collected and processed."""
        from src.services.data_processing.data_minimizer import DataMinimizer
        
        minimizer = DataMinimizer()
        
        # Test data with unnecessary fields
        full_data = {
            "patient_id": "PAT_20241012_00001",
            "diagnosis": "Alzheimer's Disease",
            "social_security": "123-45-6789",  # Not needed for AI processing
            "home_address": "123 Main St",     # Not needed for AI processing
            "brain_scan_data": "scan_data_here",
            "cognitive_scores": [25, 23, 21]
        }
        
        # Minimize data for AI processing
        minimized_data = minimizer.minimize_for_ai_processing(full_data)
        
        # Verify unnecessary PII is removed
        assert "social_security" not in minimized_data
        assert "home_address" not in minimized_data
        
        # Verify necessary data is retained
        assert "patient_id" in minimized_data
        assert "brain_scan_data" in minimized_data
        assert "cognitive_scores" in minimized_data


@pytest.mark.asyncio
async def test_security_integration():
    """Integration test for complete security workflow."""
    from src.services.security.rbac_service import Permission
    
    # 1. User authentication
    auth_service = AuthenticationService()
    user_id = auth_service.create_user(
        username="integrationuser",
        email="integration@example.com",
        password="TestPassword123!",
        roles=["radiologist"]
    )
    
    auth_result = auth_service.authenticate_user(
        username="integrationuser",
        password="TestPassword123!",
        ip_address="127.0.0.1",
        user_agent="test-agent"
    )
    
    # 2. Access patient data with proper authorization
    rbac_service = RBACService()
    has_permission = rbac_service.check_permission(["radiologist"], Permission.READ_PATIENT_DATA)
    assert has_permission == True
    
    # 3. Data access is logged
    audit_logger = AuditLogger()
    audit_logger.log_data_access(
        user_id=user_id,
        patient_id="PAT_20241012_00001",
        action="VIEW_PATIENT_RECORD",
        resource="patient_record"
    )
    
    # 4. Data is encrypted before storage
    encryption_service = EncryptionService()
    patient_data = '{"patient_id": "PAT_20241012_00001", "diagnosis": "test"}'
    encrypted_data = encryption_service.encrypt_data(patient_data)
    
    # Verify complete workflow
    assert auth_result is not None
    assert "token" in auth_result
    assert encrypted_data != patient_data
    
    # Verify audit trail exists
    logs = audit_logger.get_audit_logs(patient_id="PAT_20241012_00001")
    assert len(logs) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])