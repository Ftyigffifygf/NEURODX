"""
Secure key management and rotation for HIPAA compliance.
"""

import os
import json
import base64
import hashlib
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.backends import default_backend
import logging

from src.config.settings import settings
from .audit_logger import AuditLogger

logger = logging.getLogger(__name__)


class KeyManager:
    """
    Secure key management system with automatic rotation.
    
    Features:
    - Automatic key rotation based on time or usage
    - Secure key storage with encryption
    - Key versioning for backward compatibility
    - Audit logging for all key operations
    - Emergency key revocation
    """
    
    def __init__(self, key_store_path: Optional[Path] = None, 
                 audit_logger: Optional[AuditLogger] = None):
        self.key_store_path = key_store_path or Path("./keys")
        self.key_store_path.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.audit_logger = audit_logger or AuditLogger()
        
        # Key rotation settings
        self.rotation_interval = timedelta(days=90)  # HIPAA recommended
        self.max_key_usage = 1000000  # Maximum encryptions per key
        
        # Initialize key store
        self._initialize_key_store()
    
    def _initialize_key_store(self):
        """Initialize the key store with master key if not exists."""
        master_key_path = self.key_store_path / "master.key"
        
        if not master_key_path.exists():
            logger.info("Initializing new key store")
            self._generate_master_key()
            self.audit_logger.log_security_event(
                event_type="key_store_initialized",
                details={"key_store_path": str(self.key_store_path)}
            )
    
    def _generate_master_key(self) -> bytes:
        """Generate a new master key for key encryption."""
        master_key = AESGCM.generate_key(bit_length=256)
        
        # Encrypt master key with password from settings
        password = settings.security.encryption_key.encode('utf-8')
        salt = os.urandom(16)
        
        # Simple key derivation for master key encryption
        key_hash = hashlib.pbkdf2_hmac('sha256', password, salt, 100000)
        
        # Encrypt master key
        cipher = AESGCM(key_hash)
        iv = os.urandom(12)
        encrypted_master = cipher.encrypt(iv, master_key, None)
        
        # Store encrypted master key
        master_key_data = {
            "salt": base64.b64encode(salt).decode('utf-8'),
            "iv": base64.b64encode(iv).decode('utf-8'),
            "encrypted_key": base64.b64encode(encrypted_master).decode('utf-8'),
            "created_at": datetime.utcnow().isoformat()
        }
        
        master_key_path = self.key_store_path / "master.key"
        with open(master_key_path, 'w') as f:
            json.dump(master_key_data, f)
        
        # Set file permissions after creation (Windows compatible)
        try:
            import stat
            master_key_path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # Owner read/write only
        except (OSError, AttributeError):
            # Windows may not support chmod, continue anyway
            pass
        
        return master_key
    
    def _load_master_key(self) -> bytes:
        """Load and decrypt the master key."""
        master_key_path = self.key_store_path / "master.key"
        
        if not master_key_path.exists():
            raise ValueError("Master key not found")
        
        with open(master_key_path, 'r') as f:
            master_key_data = json.load(f)
        
        # Decrypt master key
        password = settings.security.encryption_key.encode('utf-8')
        salt = base64.b64decode(master_key_data["salt"])
        iv = base64.b64decode(master_key_data["iv"])
        encrypted_key = base64.b64decode(master_key_data["encrypted_key"])
        
        # Derive decryption key
        key_hash = hashlib.pbkdf2_hmac('sha256', password, salt, 100000)
        
        # Decrypt master key
        cipher = AESGCM(key_hash)
        master_key = cipher.decrypt(iv, encrypted_key, None)
        
        return master_key
    
    def generate_data_encryption_key(self, key_id: str, 
                                   purpose: str = "data_encryption") -> str:
        """
        Generate a new data encryption key.
        
        Args:
            key_id: Unique identifier for the key
            purpose: Purpose of the key (for audit trail)
            
        Returns:
            Key ID for the generated key
        """
        try:
            # Generate new encryption key
            data_key = AESGCM.generate_key(bit_length=256)
            
            # Load master key for encryption
            master_key = self._load_master_key()
            
            # Encrypt data key with master key
            cipher = AESGCM(master_key)
            iv = os.urandom(12)
            encrypted_data_key = cipher.encrypt(iv, data_key, None)
            
            # Create key metadata
            key_metadata = {
                "key_id": key_id,
                "purpose": purpose,
                "created_at": datetime.utcnow().isoformat(),
                "usage_count": 0,
                "status": "active",
                "iv": base64.b64encode(iv).decode('utf-8'),
                "encrypted_key": base64.b64encode(encrypted_data_key).decode('utf-8')
            }
            
            # Store key
            key_path = self.key_store_path / f"{key_id}.key"
            with open(key_path, 'w') as f:
                json.dump(key_metadata, f)
            
            # Set file permissions after creation (Windows compatible)
            try:
                import stat
                key_path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # Owner read/write only
            except (OSError, AttributeError):
                # Windows may not support chmod, continue anyway
                pass
            
            # Audit log
            self.audit_logger.log_security_event(
                event_type="key_generated",
                details={
                    "key_id": key_id,
                    "purpose": purpose,
                    "algorithm": "AES-256-GCM"
                }
            )
            
            logger.info(f"Generated new data encryption key: {key_id}")
            return key_id
            
        except Exception as e:
            logger.error(f"Failed to generate key {key_id}: {e}")
            self.audit_logger.log_security_event(
                event_type="key_generation_failure",
                details={"key_id": key_id, "error": str(e)}
            )
            raise
    
    def get_encryption_key(self, key_id: str) -> bytes:
        """
        Retrieve and decrypt a data encryption key.
        
        Args:
            key_id: Key identifier
            
        Returns:
            Decrypted encryption key
        """
        try:
            key_path = self.key_store_path / f"{key_id}.key"
            
            if not key_path.exists():
                raise ValueError(f"Key {key_id} not found")
            
            # Load key metadata
            with open(key_path, 'r') as f:
                key_metadata = json.load(f)
            
            # Check key status
            if key_metadata["status"] != "active":
                raise ValueError(f"Key {key_id} is not active")
            
            # Check if key needs rotation
            created_at = datetime.fromisoformat(key_metadata["created_at"])
            if datetime.utcnow() - created_at > self.rotation_interval:
                logger.warning(f"Key {key_id} is due for rotation")
            
            if key_metadata["usage_count"] >= self.max_key_usage:
                logger.warning(f"Key {key_id} has exceeded usage limit")
            
            # Decrypt key
            master_key = self._load_master_key()
            cipher = AESGCM(master_key)
            iv = base64.b64decode(key_metadata["iv"])
            encrypted_key = base64.b64decode(key_metadata["encrypted_key"])
            
            data_key = cipher.decrypt(iv, encrypted_key, None)
            
            # Update usage count
            key_metadata["usage_count"] += 1
            key_metadata["last_used"] = datetime.utcnow().isoformat()
            
            with open(key_path, 'w') as f:
                json.dump(key_metadata, f)
            
            return data_key
            
        except Exception as e:
            logger.error(f"Failed to retrieve key {key_id}: {e}")
            self.audit_logger.log_security_event(
                event_type="key_retrieval_failure",
                details={"key_id": key_id, "error": str(e)}
            )
            raise
    
    def rotate_key(self, old_key_id: str) -> str:
        """
        Rotate an encryption key by generating a new one.
        
        Args:
            old_key_id: ID of key to rotate
            
        Returns:
            ID of new key
        """
        try:
            # Generate new key ID
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            new_key_id = f"{old_key_id}_rotated_{timestamp}"
            
            # Get old key metadata for purpose
            old_key_path = self.key_store_path / f"{old_key_id}.key"
            if old_key_path.exists():
                with open(old_key_path, 'r') as f:
                    old_metadata = json.load(f)
                purpose = old_metadata.get("purpose", "data_encryption")
            else:
                purpose = "data_encryption"
            
            # Generate new key
            self.generate_data_encryption_key(new_key_id, purpose)
            
            # Mark old key as rotated
            if old_key_path.exists():
                old_metadata["status"] = "rotated"
                old_metadata["rotated_at"] = datetime.utcnow().isoformat()
                old_metadata["successor_key"] = new_key_id
                
                with open(old_key_path, 'w') as f:
                    json.dump(old_metadata, f)
            
            # Audit log
            self.audit_logger.log_security_event(
                event_type="key_rotated",
                details={
                    "old_key_id": old_key_id,
                    "new_key_id": new_key_id
                }
            )
            
            logger.info(f"Rotated key {old_key_id} to {new_key_id}")
            return new_key_id
            
        except Exception as e:
            logger.error(f"Failed to rotate key {old_key_id}: {e}")
            self.audit_logger.log_security_event(
                event_type="key_rotation_failure",
                details={"key_id": old_key_id, "error": str(e)}
            )
            raise
    
    def revoke_key(self, key_id: str, reason: str = "security_incident"):
        """
        Revoke a key immediately.
        
        Args:
            key_id: Key to revoke
            reason: Reason for revocation
        """
        try:
            key_path = self.key_store_path / f"{key_id}.key"
            
            if key_path.exists():
                with open(key_path, 'r') as f:
                    key_metadata = json.load(f)
                
                key_metadata["status"] = "revoked"
                key_metadata["revoked_at"] = datetime.utcnow().isoformat()
                key_metadata["revocation_reason"] = reason
                
                with open(key_path, 'w') as f:
                    json.dump(key_metadata, f)
            
            # Audit log
            self.audit_logger.log_security_event(
                event_type="key_revoked",
                details={
                    "key_id": key_id,
                    "reason": reason
                }
            )
            
            logger.warning(f"Revoked key {key_id}: {reason}")
            
        except Exception as e:
            logger.error(f"Failed to revoke key {key_id}: {e}")
            raise
    
    def list_keys(self, status: Optional[str] = None) -> List[Dict[str, str]]:
        """
        List all keys with their metadata.
        
        Args:
            status: Filter by key status (active, rotated, revoked)
            
        Returns:
            List of key metadata
        """
        keys = []
        
        for key_file in self.key_store_path.glob("*.key"):
            if key_file.name == "master.key":
                continue
                
            try:
                with open(key_file, 'r') as f:
                    metadata = json.load(f)
                
                if status is None or metadata.get("status") == status:
                    # Remove sensitive data
                    safe_metadata = {
                        "key_id": metadata["key_id"],
                        "purpose": metadata["purpose"],
                        "created_at": metadata["created_at"],
                        "status": metadata["status"],
                        "usage_count": metadata["usage_count"]
                    }
                    keys.append(safe_metadata)
                    
            except Exception as e:
                logger.error(f"Failed to read key file {key_file}: {e}")
        
        return keys
    
    def check_key_rotation_needed(self) -> List[str]:
        """
        Check which keys need rotation.
        
        Returns:
            List of key IDs that need rotation
        """
        keys_needing_rotation = []
        
        for key_metadata in self.list_keys(status="active"):
            key_id = key_metadata["key_id"]
            created_at = datetime.fromisoformat(key_metadata["created_at"])
            usage_count = key_metadata["usage_count"]
            
            # Check time-based rotation
            if datetime.utcnow() - created_at > self.rotation_interval:
                keys_needing_rotation.append(key_id)
            
            # Check usage-based rotation
            elif usage_count >= self.max_key_usage:
                keys_needing_rotation.append(key_id)
        
        return keys_needing_rotation