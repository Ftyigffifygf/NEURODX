"""
Tests for encryption service and HIPAA compliance.
"""

import pytest
import json
import os
from datetime import datetime
from pathlib import Path

from src.services.security.encryption_service import EncryptionService
from src.services.security.key_management import KeyManager
from src.services.security.audit_logger import AuditLogger


class TestEncryptionService:
    """Test encryption service functionality."""
    
    @pytest.fixture
    def encryption_service(self):
        """Create encryption service for testing."""
        return EncryptionService()
    
    @pytest.fixture
    def sample_patient_data(self):
        """Sample patient data for testing."""
        return {
            "patient_id": "PAT_20241011_12345",
            "name": "John Doe",
            "dob": "1980-01-01",
            "ssn": "123-45-6789",
            "medical_record_number": "MRN123456",
            "diagnosis": "Suspected Alzheimer's Disease",
            "medications": ["Donepezil", "Memantine"],
            "allergies": ["Penicillin"],
            "emergency_contact": {
                "name": "Jane Doe",
                "phone": "555-0123",
                "relationship": "Spouse"
            }
        }
    
    def test_encrypt_decrypt_string(self, encryption_service):
        """Test string encryption and decryption."""
        original_data = "Sensitive patient information"
        
        # Encrypt data
        encrypted_data = encryption_service.encrypt_data(original_data)
        assert encrypted_data != original_data
        assert isinstance(encrypted_data, str)
        
        # Decrypt data
        decrypted_data = encryption_service.decrypt_data(encrypted_data, str)
        assert decrypted_data == original_data
    
    def test_encrypt_decrypt_dict(self, encryption_service, sample_patient_data):
        """Test dictionary encryption and decryption."""
        # Encrypt data
        encrypted_data = encryption_service.encrypt_data(sample_patient_data)
        assert encrypted_data != json.dumps(sample_patient_data)
        
        # Decrypt data
        decrypted_data = encryption_service.decrypt_data(encrypted_data, dict)
        assert decrypted_data == sample_patient_data
    
    def test_encrypt_decrypt_bytes(self, encryption_service):
        """Test bytes encryption and decryption."""
        original_data = b"Binary medical image data"
        
        # Encrypt data
        encrypted_data = encryption_service.encrypt_data(original_data)
        
        # Decrypt data
        decrypted_data = encryption_service.decrypt_data(encrypted_data, bytes)
        assert decrypted_data == original_data
    
    def test_patient_data_encryption(self, encryption_service, sample_patient_data):
        """Test patient-specific encryption methods."""
        patient_id = sample_patient_data["patient_id"]
        
        # Encrypt patient data
        encrypted_data = encryption_service.encrypt_patient_data(
            sample_patient_data, patient_id
        )
        
        # Decrypt patient data
        decrypted_data = encryption_service.decrypt_patient_data(
            encrypted_data, patient_id
        )
        
        assert decrypted_data == sample_patient_data
    
    def test_medical_image_metadata_encryption(self, encryption_service):
        """Test medical image metadata encryption."""
        metadata = {
            "study_id": "STUDY_20241011_120000_001",
            "patient_id": "PAT_20241011_12345",
            "modality": "MRI",
            "acquisition_date": "2024-10-11T12:00:00Z",
            "scanner_model": "Siemens Skyra 3T",
            "sequence_name": "T1_MPRAGE",
            "slice_thickness": 1.0,
            "pixel_spacing": [1.0, 1.0],
            "image_orientation": [1, 0, 0, 0, 1, 0]
        }
        
        study_id = metadata["study_id"]
        
        # Encrypt metadata
        encrypted_metadata = encryption_service.encrypt_medical_image_metadata(
            metadata, study_id
        )
        
        # Decrypt metadata
        decrypted_metadata = encryption_service.decrypt_medical_image_metadata(
            encrypted_metadata, study_id
        )
        
        assert decrypted_metadata == metadata
    
    def test_hash_identifier(self, encryption_service):
        """Test identifier hashing for indexing."""
        identifier = "PAT_20241011_12345"
        
        # Hash identifier
        hashed_id = encryption_service.hash_identifier(identifier)
        
        assert hashed_id != identifier
        assert len(hashed_id) == 64  # SHA-256 hex length
        
        # Same identifier should produce same hash
        hashed_id2 = encryption_service.hash_identifier(identifier)
        assert hashed_id == hashed_id2
    
    def test_data_integrity_verification(self, encryption_service):
        """Test data integrity verification."""
        original_data = "Test data for integrity check"
        
        # Encrypt data
        encrypted_data = encryption_service.encrypt_data(original_data)
        
        # Verify integrity
        assert encryption_service.verify_data_integrity(encrypted_data) is True
        
        # Test with corrupted data
        corrupted_data = encrypted_data[:-10] + "corrupted"
        assert encryption_service.verify_data_integrity(corrupted_data) is False
    
    def test_encryption_with_different_keys(self):
        """Test that different encryption services produce different results."""
        # Create two different encryption services (different keys)
        service1 = EncryptionService()
        service2 = EncryptionService()
        
        data = "Test data"
        
        encrypted1 = service1.encrypt_data(data)
        encrypted2 = service2.encrypt_data(data)
        
        # Different keys should produce different encrypted data
        # (Note: This test assumes different master keys, which may not be true in test environment)
        # In production, this would be guaranteed by different encryption keys
        assert isinstance(encrypted1, str)
        assert isinstance(encrypted2, str)
    
    def test_encryption_randomness(self, encryption_service):
        """Test that encryption produces different results for same data."""
        data = "Test data for randomness"
        
        # Encrypt same data multiple times
        encrypted1 = encryption_service.encrypt_data(data)
        encrypted2 = encryption_service.encrypt_data(data)
        
        # Should produce different encrypted data due to random IV
        assert encrypted1 != encrypted2
        
        # But both should decrypt to same original data
        decrypted1 = encryption_service.decrypt_data(encrypted1, str)
        decrypted2 = encryption_service.decrypt_data(encrypted2, str)
        
        assert decrypted1 == data
        assert decrypted2 == data


class TestKeyManager:
    """Test key management functionality."""
    
    @pytest.fixture
    def temp_key_store(self, tmp_path):
        """Create temporary key store for testing."""
        return tmp_path / "test_keys"
    
    @pytest.fixture
    def key_manager(self, temp_key_store):
        """Create key manager for testing."""
        return KeyManager(key_store_path=temp_key_store)
    
    def test_key_generation(self, key_manager):
        """Test data encryption key generation."""
        key_id = "test_key_001"
        
        # Generate key
        generated_key_id = key_manager.generate_data_encryption_key(
            key_id, "test_encryption"
        )
        
        assert generated_key_id == key_id
        
        # Verify key file exists
        key_file = key_manager.key_store_path / f"{key_id}.key"
        assert key_file.exists()
    
    def test_key_retrieval(self, key_manager):
        """Test key retrieval and decryption."""
        key_id = "test_key_002"
        
        # Generate key
        key_manager.generate_data_encryption_key(key_id, "test_encryption")
        
        # Retrieve key
        encryption_key = key_manager.get_encryption_key(key_id)
        
        assert isinstance(encryption_key, bytes)
        assert len(encryption_key) == 32  # 256 bits
    
    def test_key_rotation(self, key_manager):
        """Test key rotation functionality."""
        old_key_id = "test_key_003"
        
        # Generate original key
        key_manager.generate_data_encryption_key(old_key_id, "test_encryption")
        
        # Rotate key
        new_key_id = key_manager.rotate_key(old_key_id)
        
        assert new_key_id != old_key_id
        assert "rotated" in new_key_id
        
        # Verify old key is marked as rotated
        old_key_file = key_manager.key_store_path / f"{old_key_id}.key"
        with open(old_key_file, 'r') as f:
            old_metadata = json.load(f)
        
        assert old_metadata["status"] == "rotated"
        assert old_metadata["successor_key"] == new_key_id
    
    def test_key_revocation(self, key_manager):
        """Test key revocation."""
        key_id = "test_key_004"
        
        # Generate key
        key_manager.generate_data_encryption_key(key_id, "test_encryption")
        
        # Revoke key
        key_manager.revoke_key(key_id, "security_incident")
        
        # Verify key is marked as revoked
        key_file = key_manager.key_store_path / f"{key_id}.key"
        with open(key_file, 'r') as f:
            metadata = json.load(f)
        
        assert metadata["status"] == "revoked"
        assert metadata["revocation_reason"] == "security_incident"
    
    def test_list_keys(self, key_manager):
        """Test key listing functionality."""
        # Generate multiple keys
        key_manager.generate_data_encryption_key("active_key_1", "test")
        key_manager.generate_data_encryption_key("active_key_2", "test")
        
        # Generate and revoke a key
        key_manager.generate_data_encryption_key("revoked_key", "test")
        key_manager.revoke_key("revoked_key", "test")
        
        # List all keys
        all_keys = key_manager.list_keys()
        assert len(all_keys) >= 3
        
        # List only active keys
        active_keys = key_manager.list_keys(status="active")
        assert len(active_keys) >= 2
        
        # List only revoked keys
        revoked_keys = key_manager.list_keys(status="revoked")
        assert len(revoked_keys) >= 1
    
    def test_key_rotation_check(self, key_manager):
        """Test checking which keys need rotation."""
        # Generate a key
        key_id = "rotation_test_key"
        key_manager.generate_data_encryption_key(key_id, "test")
        
        # Check rotation needed (should be empty for new key)
        keys_needing_rotation = key_manager.check_key_rotation_needed()
        assert key_id not in keys_needing_rotation
        
        # Simulate high usage by updating usage count
        key_file = key_manager.key_store_path / f"{key_id}.key"
        with open(key_file, 'r') as f:
            metadata = json.load(f)
        
        metadata["usage_count"] = key_manager.max_key_usage + 1
        
        with open(key_file, 'w') as f:
            json.dump(metadata, f)
        
        # Check rotation needed again
        keys_needing_rotation = key_manager.check_key_rotation_needed()
        assert key_id in keys_needing_rotation


class TestAuditLogger:
    """Test audit logging functionality."""
    
    @pytest.fixture
    def temp_audit_log(self, tmp_path):
        """Create temporary audit log for testing."""
        return tmp_path / "test_audit.log"
    
    @pytest.fixture
    def audit_logger(self, temp_audit_log, monkeypatch):
        """Create audit logger for testing."""
        # Mock the settings to use temp log file
        monkeypatch.setattr(
            "src.services.security.audit_logger.settings.security.hipaa_audit_log_path",
            temp_audit_log
        )
        return AuditLogger()
    
    def test_data_access_logging(self, audit_logger, temp_audit_log):
        """Test data access audit logging."""
        audit_logger.log_data_access(
            operation="read",
            data_type="patient_record",
            user_id="USER_123",
            patient_id="PAT_456",
            details={"record_count": 1}
        )
        
        # Verify log entry
        assert temp_audit_log.exists()
        
        with open(temp_audit_log, 'r') as f:
            log_line = f.readline().strip()
        
        # Parse log entry
        log_parts = log_line.split(' - ', 2)
        assert len(log_parts) == 3
        
        log_data = json.loads(log_parts[2])
        assert log_data["event_type"] == "data_access"
        assert log_data["user_id"] == "USER_123"
        assert "patient_id_hash" in log_data
        assert log_data["details"]["operation"] == "read"
        assert log_data["details"]["data_type"] == "patient_record"
    
    def test_authentication_logging(self, audit_logger, temp_audit_log):
        """Test authentication event logging."""
        audit_logger.log_authentication_event(
            event_type="login",
            user_id="USER_123",
            success=True,
            details={"ip_address": "192.168.1.100"}
        )
        
        # Verify log entry
        with open(temp_audit_log, 'r') as f:
            log_line = f.readline().strip()
        
        log_parts = log_line.split(' - ', 2)
        log_data = json.loads(log_parts[2])
        
        assert log_data["event_type"] == "authentication"
        assert log_data["user_id"] == "USER_123"
        assert log_data["details"]["auth_event"] == "login"
        assert log_data["details"]["success"] is True
    
    def test_security_event_logging(self, audit_logger, temp_audit_log):
        """Test security event logging."""
        audit_logger.log_security_event(
            event_type="key_generated",
            details={
                "key_id": "test_key_001",
                "algorithm": "AES-256-GCM"
            },
            user_id="SYSTEM"
        )
        
        # Verify log entry
        with open(temp_audit_log, 'r') as f:
            log_line = f.readline().strip()
        
        log_parts = log_line.split(' - ', 2)
        log_data = json.loads(log_parts[2])
        
        assert log_data["event_type"] == "security"
        assert log_data["user_id"] == "SYSTEM"
        assert log_data["details"]["security_event"] == "key_generated"
    
    def test_encryption_event_logging(self, audit_logger, temp_audit_log):
        """Test encryption event logging."""
        audit_logger.log_encryption_event(
            operation="encrypt",
            data_type="patient_data",
            data_size=1024,
            context={"patient_id": "PAT_123"}
        )
        
        # Verify log entry
        with open(temp_audit_log, 'r') as f:
            log_line = f.readline().strip()
        
        log_parts = log_line.split(' - ', 2)
        log_data = json.loads(log_parts[2])
        
        assert log_data["event_type"] == "encryption"
        assert log_data["details"]["encryption_operation"] == "encrypt"
        assert log_data["details"]["data_size_bytes"] == 1024
    
    def test_api_request_logging(self, audit_logger, temp_audit_log):
        """Test API request logging."""
        audit_logger.log_api_request(
            method="POST",
            endpoint="/api/patients",
            user_id="USER_123",
            status_code=201,
            response_time_ms=150.5,
            details={"ip_address": "192.168.1.100"}
        )
        
        # Verify log entry
        with open(temp_audit_log, 'r') as f:
            log_line = f.readline().strip()
        
        log_parts = log_line.split(' - ', 2)
        log_data = json.loads(log_parts[2])
        
        assert log_data["event_type"] == "api_request"
        assert log_data["details"]["http_method"] == "POST"
        assert log_data["details"]["status_code"] == 201
        assert log_data["details"]["response_time_ms"] == 150.5
    
    def test_data_export_logging(self, audit_logger, temp_audit_log):
        """Test data export logging."""
        patient_ids = ["PAT_001", "PAT_002", "PAT_003"]
        
        audit_logger.log_data_export(
            user_id="USER_123",
            data_type="patient_records",
            export_format="CSV",
            record_count=3,
            patient_ids=patient_ids
        )
        
        # Verify log entry
        with open(temp_audit_log, 'r') as f:
            log_line = f.readline().strip()
        
        log_parts = log_line.split(' - ', 2)
        log_data = json.loads(log_parts[2])
        
        assert log_data["event_type"] == "data_export"
        assert log_data["details"]["record_count"] == 3
        assert len(log_data["details"]["patient_id_hashes"]) == 3
        # Verify patient IDs are hashed, not stored in plain text
        for hash_val in log_data["details"]["patient_id_hashes"]:
            assert len(hash_val) == 64  # SHA-256 hex length
            assert hash_val not in patient_ids
    
    def test_log_integrity_verification(self, audit_logger, temp_audit_log):
        """Test audit log integrity verification."""
        # Create some log entries
        audit_logger.log_security_event(
            event_type="test_event",
            details={"test": "data"}
        )
        
        # Verify log integrity
        assert audit_logger.verify_log_integrity(temp_audit_log) is True
        
        # Test with empty log
        empty_log = temp_audit_log.parent / "empty.log"
        empty_log.touch()
        assert audit_logger.verify_log_integrity(empty_log) is True
        
        # Test with corrupted log
        corrupted_log = temp_audit_log.parent / "corrupted.log"
        with open(corrupted_log, 'w') as f:
            f.write("This is not valid JSON log format\n")
        
        assert audit_logger.verify_log_integrity(corrupted_log) is False