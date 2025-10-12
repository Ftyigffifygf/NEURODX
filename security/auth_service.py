"""
Authentication service with JWT tokens and session management.
"""

import jwt
import bcrypt
import secrets
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

from src.config.settings import settings
from .audit_logger import AuditLogger
from .mfa_service import MFAService

logger = logging.getLogger(__name__)


@dataclass
class User:
    """User data model."""
    user_id: str
    username: str
    email: str
    password_hash: str
    roles: List[str]
    is_active: bool = True
    created_at: datetime = None
    last_login: datetime = None
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class Session:
    """User session data model."""
    session_id: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    ip_address: str
    user_agent: str
    is_active: bool = True


class AuthenticationService:
    """
    Authentication service with JWT tokens and session management.
    
    Features:
    - Password hashing with bcrypt
    - JWT token generation and validation
    - Session management with timeout
    - Account lockout after failed attempts
    - Audit logging for all auth events
    """
    
    def __init__(self, audit_logger: Optional[AuditLogger] = None):
        self.audit_logger = audit_logger or AuditLogger()
        self.mfa_service = MFAService(audit_logger)
        self.jwt_secret = settings.security.jwt_secret_key
        self.session_timeout = settings.security.session_timeout
        self.max_login_attempts = settings.security.max_login_attempts
        
        # In-memory storage for demo (replace with database in production)
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Session] = {}
        
        # Create default admin user for testing
        self._create_default_users()
    
    def _create_default_users(self):
        """Create default users for testing."""
        admin_user = self.create_user(
            username="admin",
            email="admin@neurodx.com",
            password="admin123",
            roles=["admin", "clinician", "researcher"]
        )
        
        clinician_user = self.create_user(
            username="clinician",
            email="clinician@neurodx.com", 
            password="clinician123",
            roles=["clinician"]
        )
        
        researcher_user = self.create_user(
            username="researcher",
            email="researcher@neurodx.com",
            password="researcher123", 
            roles=["researcher"]
        )
        
        logger.info("Created default users for testing")
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        salt = bcrypt.gensalt()
        password_hash = bcrypt.hashpw(password.encode('utf-8'), salt)
        return password_hash.decode('utf-8')
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash."""
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    
    def create_user(self, username: str, email: str, password: str,
                   roles: List[str]) -> str:
        """
        Create a new user account.
        
        Args:
            username: Unique username
            email: User email
            password: Plain text password
            roles: List of user roles
            
        Returns:
            User ID
        """
        try:
            # Check if user already exists
            for user in self.users.values():
                if user.username == username or user.email == email:
                    raise ValueError("User already exists")
            
            # Generate user ID
            user_id = f"USER_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(4)}"
            
            # Hash password
            password_hash = self.hash_password(password)
            
            # Create user
            user = User(
                user_id=user_id,
                username=username,
                email=email,
                password_hash=password_hash,
                roles=roles
            )
            
            self.users[user_id] = user
            
            # Audit log
            self.audit_logger.log_security_event(
                event_type="user_created",
                details={
                    "username": username,
                    "email": email,
                    "roles": roles
                }
            )
            
            logger.info(f"Created user: {username}")
            return user_id
            
        except Exception as e:
            logger.error(f"Failed to create user {username}: {e}")
            raise
    
    def authenticate_user(self, username: str, password: str,
                         ip_address: str, user_agent: str, 
                         mfa_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Authenticate user with username and password.
        
        Args:
            username: Username or email
            password: Plain text password
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            Authentication result with token and user info
        """
        try:
            # Find user
            user = None
            for u in self.users.values():
                if u.username == username or u.email == username:
                    user = u
                    break
            
            if not user:
                self.audit_logger.log_authentication_event(
                    event_type="login_failed",
                    user_id=username,
                    success=False,
                    details={"reason": "user_not_found", "ip_address": ip_address}
                )
                return None
            
            # Check if account is locked
            if user.locked_until and datetime.utcnow() < user.locked_until:
                self.audit_logger.log_authentication_event(
                    event_type="login_failed",
                    user_id=user.user_id,
                    success=False,
                    details={"reason": "account_locked", "ip_address": ip_address}
                )
                return None
            
            # Check if account is active
            if not user.is_active:
                self.audit_logger.log_authentication_event(
                    event_type="login_failed",
                    user_id=user.user_id,
                    success=False,
                    details={"reason": "account_inactive", "ip_address": ip_address}
                )
                return None
            
            # Verify password
            if not self.verify_password(password, user.password_hash):
                # Increment failed attempts
                user.failed_login_attempts += 1
                
                # Lock account if too many failures
                if user.failed_login_attempts >= self.max_login_attempts:
                    user.locked_until = datetime.utcnow() + timedelta(minutes=30)
                    logger.warning(f"Account locked for user {user.username}")
                
                self.audit_logger.log_authentication_event(
                    event_type="login_failed",
                    user_id=user.user_id,
                    success=False,
                    details={
                        "reason": "invalid_password",
                        "failed_attempts": user.failed_login_attempts,
                        "ip_address": ip_address
                    }
                )
                return None
            
            # Check if MFA is required
            if self.mfa_service.is_mfa_enabled(user.user_id):
                if not mfa_code:
                    # Return partial success - MFA required
                    return {
                        "mfa_required": True,
                        "user_id": user.user_id,
                        "message": "MFA verification required"
                    }
                
                # Verify MFA code
                mfa_valid = (self.mfa_service.verify_totp_code(user.user_id, mfa_code) or
                           self.mfa_service.verify_backup_code(user.user_id, mfa_code))
                
                if not mfa_valid:
                    self.audit_logger.log_authentication_event(
                        event_type="login_failed",
                        user_id=user.user_id,
                        success=False,
                        details={
                            "reason": "invalid_mfa_code",
                            "ip_address": ip_address
                        }
                    )
                    return None
            
            # Reset failed attempts on successful login
            user.failed_login_attempts = 0
            user.locked_until = None
            user.last_login = datetime.utcnow()
            
            # Create session
            session = self._create_session(user.user_id, ip_address, user_agent)
            
            # Generate JWT token
            token = self._generate_jwt_token(user, session.session_id)
            
            # Audit log
            self.audit_logger.log_authentication_event(
                event_type="login_success",
                user_id=user.user_id,
                success=True,
                details={"ip_address": ip_address, "session_id": session.session_id}
            )
            
            logger.info(f"User authenticated: {user.username}")
            
            return {
                "token": token,
                "user": {
                    "user_id": user.user_id,
                    "username": user.username,
                    "email": user.email,
                    "roles": user.roles
                },
                "session_id": session.session_id,
                "expires_at": session.expires_at.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Authentication failed for {username}: {e}")
            self.audit_logger.log_authentication_event(
                event_type="login_error",
                user_id=username,
                success=False,
                details={"error": str(e), "ip_address": ip_address}
            )
            return None
    
    def _create_session(self, user_id: str, ip_address: str, user_agent: str) -> Session:
        """Create a new user session."""
        session_id = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow() + timedelta(seconds=self.session_timeout)
        
        session = Session(
            session_id=session_id,
            user_id=user_id,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.sessions[session_id] = session
        return session
    
    def _generate_jwt_token(self, user: User, session_id: str) -> str:
        """Generate JWT token for user."""
        payload = {
            "user_id": user.user_id,
            "username": user.username,
            "roles": user.roles,
            "session_id": session_id,
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(seconds=self.session_timeout)
        }
        
        token = jwt.encode(payload, self.jwt_secret, algorithm="HS256")
        return token
    
    def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Validate JWT token and return user info.
        
        Args:
            token: JWT token
            
        Returns:
            User info if token is valid
        """
        try:
            # Decode token
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            
            # Check if session exists and is active
            session_id = payload.get("session_id")
            session = self.sessions.get(session_id)
            
            if not session or not session.is_active:
                return None
            
            # Check if session has expired
            if datetime.utcnow() > session.expires_at:
                session.is_active = False
                return None
            
            # Get user
            user_id = payload.get("user_id")
            user = self.users.get(user_id)
            
            if not user or not user.is_active:
                return None
            
            return {
                "user_id": user.user_id,
                "username": user.username,
                "email": user.email,
                "roles": user.roles,
                "session_id": session_id
            }
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
        except Exception as e:
            logger.error(f"Token validation error: {e}")
            return None
    
    def logout_user(self, session_id: str, user_id: str):
        """
        Logout user by invalidating session.
        
        Args:
            session_id: Session to invalidate
            user_id: User ID for audit logging
        """
        try:
            session = self.sessions.get(session_id)
            if session:
                session.is_active = False
            
            self.audit_logger.log_authentication_event(
                event_type="logout",
                user_id=user_id,
                success=True,
                details={"session_id": session_id}
            )
            
            logger.info(f"User logged out: {user_id}")
            
        except Exception as e:
            logger.error(f"Logout error for user {user_id}: {e}")
    
    def change_password(self, user_id: str, old_password: str, new_password: str) -> bool:
        """
        Change user password.
        
        Args:
            user_id: User ID
            old_password: Current password
            new_password: New password
            
        Returns:
            True if password changed successfully
        """
        try:
            user = self.users.get(user_id)
            if not user:
                return False
            
            # Verify old password
            if not self.verify_password(old_password, user.password_hash):
                self.audit_logger.log_security_event(
                    event_type="password_change_failed",
                    details={"reason": "invalid_old_password"},
                    user_id=user_id
                )
                return False
            
            # Hash new password
            user.password_hash = self.hash_password(new_password)
            
            # Invalidate all sessions for security
            for session in self.sessions.values():
                if session.user_id == user_id:
                    session.is_active = False
            
            self.audit_logger.log_security_event(
                event_type="password_changed",
                details={"username": user.username},
                user_id=user_id
            )
            
            logger.info(f"Password changed for user: {user.username}")
            return True
            
        except Exception as e:
            logger.error(f"Password change failed for user {user_id}: {e}")
            return False
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self.users.get(user_id)
    
    def get_active_sessions(self, user_id: str) -> List[Session]:
        """Get active sessions for user."""
        return [
            session for session in self.sessions.values()
            if session.user_id == user_id and session.is_active
            and datetime.utcnow() <= session.expires_at
        ]
    
    def setup_mfa(self, user_id: str, username: str) -> Dict[str, Any]:
        """Set up MFA for user."""
        return self.mfa_service.setup_totp_device(user_id, username)
    
    def verify_mfa_setup(self, device_id: str, verification_code: str) -> bool:
        """Verify MFA setup."""
        return self.mfa_service.verify_totp_setup(device_id, verification_code)
    
    def get_user_mfa_devices(self, user_id: str) -> list:
        """Get user's MFA devices."""
        return self.mfa_service.get_user_mfa_devices(user_id)
    
    def disable_mfa_device(self, user_id: str, device_id: str) -> bool:
        """Disable MFA device."""
        return self.mfa_service.disable_mfa_device(user_id, device_id)
    
    def generate_backup_codes(self, user_id: str, device_id: str) -> Optional[list]:
        """Generate new backup codes."""
        return self.mfa_service.generate_new_backup_codes(user_id, device_id)