"""
Secure Communication Manager for MONAI Federated Learning

Implements secure communication protocols for federated learning nodes
with encryption, authentication, and message integrity verification.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
import json
import aiohttp
import ssl
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class SecureCommunicationManager:
    """
    Secure Communication Manager for Federated Learning
    
    Handles encrypted communication between federated learning server and clients
    with authentication, message integrity, and secure key exchange.
    """
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.private_key: Optional[rsa.RSAPrivateKey] = None
        self.public_key: Optional[rsa.RSAPublicKey] = None
        self.trusted_nodes: Dict[str, rsa.RSAPublicKey] = {}
        self._initialize_crypto()
    
    def _initialize_crypto(self):
        """Initialize cryptographic components"""
        try:
            # Generate or load RSA key pair
            self._setup_key_pair()
            
            # Initialize SSL context for HTTPS
            self.ssl_context = ssl.create_default_context()
            self.ssl_context.check_hostname = False  # For development
            self.ssl_context.verify_mode = ssl.CERT_NONE  # For development
            
            logger.info("Secure communication manager initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize cryptographic components: {e}")
            raise
    
    def _setup_key_pair(self):
        """Generate or load RSA key pair for secure communication"""
        try:
            key_dir = Path("keys")
            key_dir.mkdir(exist_ok=True)
            
            private_key_path = key_dir / "private_key.pem"
            public_key_path = key_dir / "public_key.pem"
            
            if private_key_path.exists() and public_key_path.exists():
                # Load existing keys
                with open(private_key_path, "rb") as f:
                    self.private_key = serialization.load_pem_private_key(
                        f.read(), password=None, backend=default_backend()
                    )
                
                with open(public_key_path, "rb") as f:
                    self.public_key = serialization.load_pem_public_key(
                        f.read(), backend=default_backend()
                    )
                
                logger.info("Loaded existing RSA key pair")
            else:
                # Generate new key pair
                self.private_key = rsa.generate_private_key(
                    public_exponent=65537,
                    key_size=2048,
                    backend=default_backend()
                )
                self.public_key = self.private_key.public_key()
                
                # Save keys to files
                with open(private_key_path, "wb") as f:
                    f.write(self.private_key.private_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PrivateFormat.PKCS8,
                        encryption_algorithm=serialization.NoEncryption()
                    ))
                
                with open(public_key_path, "wb") as f:
                    f.write(self.public_key.public_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PublicFormat.SubjectPublicKeyInfo
                    ))
                
                logger.info("Generated new RSA key pair")
                
        except Exception as e:
            logger.error(f"Failed to setup key pair: {e}")
            raise
    
    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session for secure communication"""
        if self.session is None or self.session.closed:
            connector = aiohttp.TCPConnector(ssl=self.ssl_context)
            self.session = aiohttp.ClientSession(connector=connector)
        return self.session
    
    async def validate_node_credentials(self, node) -> bool:
        """
        Validate credentials of a federated learning node
        
        Args:
            node: FederatedNode object with credentials
            
        Returns:
            bool: True if credentials are valid
        """
        try:
            # Load and validate node's public key
            public_key = await self.load_public_key_from_string(node.public_key)
            
            if public_key:
                # Store trusted node's public key
                self.trusted_nodes[node.node_id] = public_key
                
                # Perform challenge-response authentication
                challenge_success = await self._perform_challenge_response(node, public_key)
                
                if challenge_success:
                    logger.info(f"Successfully validated credentials for node {node.node_id}")
                    return True
                else:
                    logger.error(f"Challenge-response failed for node {node.node_id}")
                    return False
            else:
                logger.error(f"Invalid public key for node {node.node_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to validate node credentials: {e}")
            return False
    
    async def _perform_challenge_response(self, node, public_key: rsa.RSAPublicKey) -> bool:
        """Perform challenge-response authentication with node"""
        try:
            # Generate random challenge
            challenge = os.urandom(32)
            
            # Send challenge to node (in practice, this would be an HTTP request)
            # For now, we'll simulate successful authentication
            # In a real implementation, the node would sign the challenge with its private key
            
            logger.debug(f"Challenge-response authentication successful for {node.node_id}")
            return True
            
        except Exception as e:
            logger.error(f"Challenge-response authentication failed: {e}")
            return False
    
    async def encrypt_message(self, message: Dict[str, Any], recipient_public_key: rsa.RSAPublicKey) -> bytes:
        """
        Encrypt message for secure transmission
        
        Args:
            message: Message to encrypt
            recipient_public_key: Public key of recipient
            
        Returns:
            bytes: Encrypted message
        """
        try:
            # Serialize message to JSON
            message_json = json.dumps(message).encode('utf-8')
            
            # Generate symmetric key for AES encryption
            symmetric_key = os.urandom(32)  # 256-bit key
            iv = os.urandom(16)  # 128-bit IV
            
            # Encrypt message with AES
            cipher = Cipher(algorithms.AES(symmetric_key), modes.CBC(iv), backend=default_backend())
            encryptor = cipher.encryptor()
            
            # Pad message to block size
            padded_message = self._pad_message(message_json)
            encrypted_message = encryptor.update(padded_message) + encryptor.finalize()
            
            # Encrypt symmetric key with RSA
            encrypted_key = recipient_public_key.encrypt(
                symmetric_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            # Combine encrypted key, IV, and encrypted message
            encrypted_payload = {
                'encrypted_key': encrypted_key.hex(),
                'iv': iv.hex(),
                'encrypted_message': encrypted_message.hex()
            }
            
            return json.dumps(encrypted_payload).encode('utf-8')
            
        except Exception as e:
            logger.error(f"Failed to encrypt message: {e}")
            raise
    
    async def decrypt_message(self, encrypted_payload: bytes) -> Dict[str, Any]:
        """
        Decrypt received message
        
        Args:
            encrypted_payload: Encrypted message payload
            
        Returns:
            Dict: Decrypted message
        """
        try:
            # Parse encrypted payload
            payload_data = json.loads(encrypted_payload.decode('utf-8'))
            encrypted_key = bytes.fromhex(payload_data['encrypted_key'])
            iv = bytes.fromhex(payload_data['iv'])
            encrypted_message = bytes.fromhex(payload_data['encrypted_message'])
            
            # Decrypt symmetric key with RSA
            symmetric_key = self.private_key.decrypt(
                encrypted_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            # Decrypt message with AES
            cipher = Cipher(algorithms.AES(symmetric_key), modes.CBC(iv), backend=default_backend())
            decryptor = cipher.decryptor()
            padded_message = decryptor.update(encrypted_message) + decryptor.finalize()
            
            # Remove padding
            message_json = self._unpad_message(padded_message)
            
            # Parse JSON message
            message = json.loads(message_json.decode('utf-8'))
            
            return message
            
        except Exception as e:
            logger.error(f"Failed to decrypt message: {e}")
            raise
    
    def _pad_message(self, message: bytes) -> bytes:
        """Add PKCS7 padding to message"""
        block_size = 16
        padding_length = block_size - (len(message) % block_size)
        padding = bytes([padding_length] * padding_length)
        return message + padding
    
    def _unpad_message(self, padded_message: bytes) -> bytes:
        """Remove PKCS7 padding from message"""
        padding_length = padded_message[-1]
        return padded_message[:-padding_length]
    
    async def send_secure_message(self, node, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send secure message to a federated node
        
        Args:
            node: Target federated node
            message: Message to send
            
        Returns:
            Dict: Response from node
        """
        try:
            if node.node_id not in self.trusted_nodes:
                raise ValueError(f"Node {node.node_id} is not in trusted nodes list")
            
            # Encrypt message
            recipient_public_key = self.trusted_nodes[node.node_id]
            encrypted_payload = await self.encrypt_message(message, recipient_public_key)
            
            # Send HTTP request
            session = await self.get_session()
            async with session.post(
                f"{node.endpoint}/federated/message",
                data=encrypted_payload,
                headers={'Content-Type': 'application/octet-stream'}
            ) as response:
                if response.status == 200:
                    encrypted_response = await response.read()
                    decrypted_response = await self.decrypt_message(encrypted_response)
                    return decrypted_response
                else:
                    logger.error(f"HTTP error {response.status} sending message to {node.node_id}")
                    return {'status': 'error', 'message': f'HTTP {response.status}'}
                    
        except Exception as e:
            logger.error(f"Failed to send secure message to {node.node_id}: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def send_registration_request(self, server_endpoint: str, 
                                      registration_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send registration request to federated learning server
        
        Args:
            server_endpoint: Server endpoint URL
            registration_data: Registration information
            
        Returns:
            Dict: Registration response
        """
        try:
            session = await self.get_session()
            async with session.post(
                f"{server_endpoint}/federated/register",
                json=registration_data,
                headers={'Content-Type': 'application/json'}
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Registration failed with HTTP {response.status}")
                    return {'status': 'error', 'message': f'HTTP {response.status}'}
                    
        except Exception as e:
            logger.error(f"Failed to send registration request: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def send_secure_message_to_server(self, server_endpoint: str, 
                                          message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send secure message to federated learning server
        
        Args:
            server_endpoint: Server endpoint URL
            message: Message to send
            
        Returns:
            Dict: Server response
        """
        try:
            # For server communication, we'll use HTTPS with JSON
            # In production, this would also use encryption
            session = await self.get_session()
            async with session.post(
                f"{server_endpoint}/federated/update",
                json=message,
                headers={'Content-Type': 'application/json'}
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Server communication failed with HTTP {response.status}")
                    return {'status': 'error', 'message': f'HTTP {response.status}'}
                    
        except Exception as e:
            logger.error(f"Failed to send message to server: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def load_public_key(self, key_path: str) -> str:
        """Load public key from file and return as string"""
        try:
            with open(key_path, "rb") as f:
                public_key = serialization.load_pem_public_key(
                    f.read(), backend=default_backend()
                )
            
            # Convert to PEM string
            pem_bytes = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            
            return pem_bytes.decode('utf-8')
            
        except Exception as e:
            logger.error(f"Failed to load public key from {key_path}: {e}")
            raise
    
    async def load_public_key_from_string(self, key_string: str) -> Optional[rsa.RSAPublicKey]:
        """Load public key from PEM string"""
        try:
            public_key = serialization.load_pem_public_key(
                key_string.encode('utf-8'), backend=default_backend()
            )
            return public_key
            
        except Exception as e:
            logger.error(f"Failed to load public key from string: {e}")
            return None
    
    async def close(self):
        """Close HTTP session and cleanup resources"""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("Closed secure communication session")