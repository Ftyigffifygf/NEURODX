"""
HIPAA-compliant encryption service for patient data protection.
Implements AES-256-GCM encryption with secure key management.
"""

import os
import base64
import hashlib
from typing import Union, Dict, Any, Optional
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import json
import logging
from datetime import datetime

from src.config.settings import settings
from .audit_logger import AuditLogger

logger = logging.getLogger(__name__)


class EncryptionService:
    """
    HIPAA-compliant encryption service using AES-256-GCM.
    
    Features:
    - AES-256-GCM encryption for authenticated encryption
    - Secure key derivation using PBKDF2
    - Automatic IV generation for each encryption
    - Metadata encryption for additional security
    - Audit logging for all encryption/decryption operations
    """
    
    def __init__(self, audit_logger: Optional[AuditLogger] = None):
        self.audit_logger = audit_logger or AuditLogger()
        self._master_key = self._derive_master_key()
        
    def _derive_master_key(self) -> bytes:
        """Derive master encryption key from configuration."""
        try:
            # Use configured encryption key as password
            password = settings.security.encryption_key.encode('utf-8')
            
            # Generate salt from app name and version for consistency
            salt_source = f"{settings.app_name}:{settings.app_version}".encode('utf-8')
            salt = hashlib.sha256(salt_source).digest()[:16]  # 16 bytes salt
            
            # Derive key using PBKDF2
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,  # 256 bits
                salt=salt,
                iterations=100000,  # NIST recommended minimum
                backend=default_backend()
            )
            
            key = kdf.derive(password)
            logger.info("Master encryption key derived successfully")
            return key
            
        except Exception as e:
            logger.error(f"Failed to derive master key: {e}")
            raise
    
    def encrypt_data(self, data: Union[str, bytes, Dict[str, Any]], 
                    context: Optional[Dict[str, str]] = None) -> str:
        """
        Encrypt data with AES-256-GCM.
        
        Args:
            data: Data to encrypt (string, bytes, or dict)
            context: Additional context for audit logging
            
        Returns:
            Base64-encoded encrypted data with IV
        """
        try:
            # Convert data to bytes
            if isinstance(data, dict):
                data_bytes = json.dumps(data, sort_keys=True).encode('utf-8')
            elif isinstance(data, str):
                data_bytes = data.encode('utf-8')
            else:
                data_bytes = data
            
            # Generate random IV (12 bytes for GCM)
            iv = os.urandom(12)
            
            # Create cipher
            cipher = AESGCM(self._master_key)
            
            # Encrypt data
            ciphertext = cipher.encrypt(iv, data_bytes, None)
            
            # Combine IV and ciphertext
            encrypted_data = iv + ciphertext
            
            # Encode as base64 for storage
            encoded_data = base64.b64encode(encrypted_data).decode('utf-8')
            
            # Audit log
            self.audit_logger.log_encryption_event(
                operation="encrypt",
                data_type=type(data).__name__,
                data_size=len(data_bytes),
                context=context or {}
            )
            
            return encoded_data
            
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            self.audit_logger.log_security_event(
                event_type="encryption_failure",
                details={"error": str(e), "context": context or {}}
            )
            raise
    
    def decrypt_data(self, encrypted_data: str, 
                    expected_type: type = str,
                    context: Optional[Dict[str, str]] = None) -> Union[str, bytes, Dict[str, Any]]:
        """
        Decrypt data encrypted with encrypt_data.
        
        Args:
            encrypted_data: Base64-encoded encrypted data
            expected_type: Expected type of decrypted data
            context: Additional context for audit logging
            
        Returns:
            Decrypted data in expected format
        """
        try:
            # Decode from base64
            encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
            
            # Extract IV and ciphertext
            iv = encrypted_bytes[:12]  # First 12 bytes
            ciphertext = encrypted_bytes[12:]  # Rest is ciphertext
            
            # Create cipher
            cipher = AESGCM(self._master_key)
            
            # Decrypt data
            decrypted_bytes = cipher.decrypt(iv, ciphertext, None)
            
            # Convert to expected type
            if expected_type == dict:
                result = json.loads(decrypted_bytes.decode('utf-8'))
            elif expected_type == str:
                result = decrypted_bytes.decode('utf-8')
            else:
                result = decrypted_bytes
            
            # Audit log
            self.audit_logger.log_encryption_event(
                operation="decrypt",
                data_type=expected_type.__name__,
                data_size=len(decrypted_bytes),
                context=context or {}
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            self.audit_logger.log_security_event(
                event_type="decryption_failure",
                details={"error": str(e), "context": context or {}}
            )
            raise
    
    def encrypt_patient_data(self, patient_data: Dict[str, Any], 
                           patient_id: str) -> str:
        """
        Encrypt patient data with specific context.
        
        Args:
            patient_data: Patient data dictionary
            patient_id: Patient identifier for audit trail
            
        Returns:
            Encrypted patient data
        """
        context = {
            "patient_id": patient_id,
            "data_type": "patient_record",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return self.encrypt_data(patient_data, context)
    
    def decrypt_patient_data(self, encrypted_data: str, 
                           patient_id: str) -> Dict[str, Any]:
        """
        Decrypt patient data with specific context.
        
        Args:
            encrypted_data: Encrypted patient data
            patient_id: Patient identifier for audit trail
            
        Returns:
            Decrypted patient data dictionary
        """
        context = {
            "patient_id": patient_id,
            "data_type": "patient_record",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return self.decrypt_data(encrypted_data, dict, context)
    
    def encrypt_medical_image_metadata(self, metadata: Dict[str, Any], 
                                     study_id: str) -> str:
        """
        Encrypt medical image metadata.
        
        Args:
            metadata: Image metadata dictionary
            study_id: Study identifier for audit trail
            
        Returns:
            Encrypted metadata
        """
        context = {
            "study_id": study_id,
            "data_type": "image_metadata",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return self.encrypt_data(metadata, context)
    
    def decrypt_medical_image_metadata(self, encrypted_metadata: str, 
                                     study_id: str) -> Dict[str, Any]:
        """
        Decrypt medical image metadata.
        
        Args:
            encrypted_metadata: Encrypted metadata
            study_id: Study identifier for audit trail
            
        Returns:
            Decrypted metadata dictionary
        """
        context = {
            "study_id": study_id,
            "data_type": "image_metadata",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return self.decrypt_data(encrypted_metadata, dict, context)
    
    def hash_identifier(self, identifier: str) -> str:
        """
        Create a secure hash of an identifier for indexing.
        
        Args:
            identifier: Original identifier
            
        Returns:
            SHA-256 hash of identifier
        """
        return hashlib.sha256(identifier.encode('utf-8')).hexdigest()
    
    def verify_data_integrity(self, data: str) -> bool:
        """
        Verify data integrity by attempting decryption.
        
        Args:
            data: Encrypted data to verify
            
        Returns:
            True if data is valid and can be decrypted
        """
        try:
            self.decrypt_data(data, bytes)
            return True
        except Exception:
            return False