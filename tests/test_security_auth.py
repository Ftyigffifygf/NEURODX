"""
Tests for authentication and authorization services.
"""

import pytest
import jwt
from datetime import datetime, timedelta
from unittest.mock import patch

from src.services.security.auth_service import AuthenticationService, User, Session
from src.services.security.rbac_service import RBACService, Permission, Role


class TestAuthenticationService:
    """Test authentication service functionality."""
    
    @pytest.fixture
    def auth_service(self):
        """Create authentication service for testing."""
        return AuthenticationService()
    
    def test_password_hashing(self, auth_service):
        """Test password hashing and verification."""
        password = "test_password_123"
        
        # Hash password
        password_hash = auth_service.hash_password(password)
        
        assert password_hash != password
        assert isinstance(password_hash, str)
        
        # Verify correct password
        assert auth_service.verify_password(password, password_hash) is True
        
        # Verify incorrect password
        assert auth_service.verify_password("wrong_password", password_hash) is False
    
    def test_create_user(self, auth_service):
        """Test user creation."""
        username = "test_user"
        email = "test@example.com"
        password = "test_password_123"
        roles = ["clinician"]
        
        # Create user
        user_id = auth_service.create_user(username, email, password, roles)
        
        assert user_id.startswith("USER_")
        assert user_id in auth_service.users
        
        user = auth_service.users[user_id]
        assert user.username == username
        assert user.email == email
        assert user.roles == roles
        assert user.is_active is True
        
        # Verify password is hashed
        assert user.password_hash != password
        assert auth_service.verify_password(password, user.password_hash)
    
    def test_create_duplicate_user(self, auth_service):
        """Test creating duplicate user fails."""
        username = "duplicate_user"
        email = "duplicate@example.com"
        password = "password123"
        roles = ["clinician"]
        
        # Create first user
        auth_service.create_user(username, email, password, roles)
        
        # Try to create duplicate username
        with pytest.raises(ValueError, match="User already exists"):
            auth_service.create_user(username, "different@example.com", password, roles)
        
        # Try to create duplicate email
        with pytest.raises(ValueError, match="User already exists"):
            auth_service.create_user("different_user", email, password, roles)
    
    def test_successful_authentication(self, auth_service):
        """Test successful user authentication."""
        username = "auth_test_user"
        email = "auth_test@example.com"
        password = "auth_password_123"
        roles = ["clinician"]
        
        # Create user
        user_id = auth_service.create_user(username, email, password, roles)
        
        # Authenticate with username
        result = auth_service.authenticate_user(
            username=username,
            password=password,
            ip_address="192.168.1.100",
            user_agent="Test Agent"
        )
        
        assert result is not None
        assert "token" in result
        assert "user" in result
        assert "session_id" in result
        assert "expires_at" in result
        
        assert result["user"]["user_id"] == user_id
        assert result["user"]["username"] == username
        assert result["user"]["roles"] == roles
        
        # Authenticate with email
        result_email = auth_service.authenticate_user(
            username=email,
            password=password,
            ip_address="192.168.1.100",
            user_agent="Test Agent"
        )
        
        assert result_email is not None
        assert result_email["user"]["user_id"] == user_id
    
    def test_failed_authentication(self, auth_service):
        """Test failed authentication scenarios."""
        username = "fail_test_user"
        email = "fail_test@example.com"
        password = "correct_password"
        roles = ["clinician"]
        
        # Create user
        auth_service.create_user(username, email, password, roles)
        
        # Test wrong password
        result = auth_service.authenticate_user(
            username=username,
            password="wrong_password",
            ip_address="192.168.1.100",
            user_agent="Test Agent"
        )
        assert result is None
        
        # Test non-existent user
        result = auth_service.authenticate_user(
            username="non_existent_user",
            password=password,
            ip_address="192.168.1.100",
            user_agent="Test Agent"
        )
        assert result is None
    
    def test_account_lockout(self, auth_service):
        """Test account lockout after failed attempts."""
        username = "lockout_test_user"
        email = "lockout_test@example.com"
        password = "correct_password"
        roles = ["clinician"]
        
        # Create user
        user_id = auth_service.create_user(username, email, password, roles)
        user = auth_service.users[user_id]
        
        # Attempt login with wrong password multiple times
        for i in range(auth_service.max_login_attempts):
            result = auth_service.authenticate_user(
                username=username,
                password="wrong_password",
                ip_address="192.168.1.100",
                user_agent="Test Agent"
            )
            assert result is None
        
        # Check that account is locked
        assert user.locked_until is not None
        assert user.locked_until > datetime.utcnow()
        
        # Try to login with correct password (should fail due to lockout)
        result = auth_service.authenticate_user(
            username=username,
            password=password,
            ip_address="192.168.1.100",
            user_agent="Test Agent"
        )
        assert result is None
    
    def test_inactive_user_authentication(self, auth_service):
        """Test authentication fails for inactive users."""
        username = "inactive_user"
        email = "inactive@example.com"
        password = "password123"
        roles = ["clinician"]
        
        # Create user
        user_id = auth_service.create_user(username, email, password, roles)
        user = auth_service.users[user_id]
        
        # Deactivate user
        user.is_active = False
        
        # Try to authenticate
        result = auth_service.authenticate_user(
            username=username,
            password=password,
            ip_address="192.168.1.100",
            user_agent="Test Agent"
        )
        assert result is None
    
    def test_jwt_token_validation(self, auth_service):
        """Test JWT token generation and validation."""
        username = "token_test_user"
        email = "token_test@example.com"
        password = "password123"
        roles = ["clinician"]
        
        # Create and authenticate user
        auth_service.create_user(username, email, password, roles)
        auth_result = auth_service.authenticate_user(
            username=username,
            password=password,
            ip_address="192.168.1.100",
            user_agent="Test Agent"
        )
        
        token = auth_result["token"]
        
        # Validate token
        user_info = auth_service.validate_token(token)
        
        assert user_info is not None
        assert user_info["username"] == username
        assert user_info["roles"] == roles
        assert "session_id" in user_info
    
    def test_invalid_token_validation(self, auth_service):
        """Test validation of invalid tokens."""
        # Test with invalid token
        assert auth_service.validate_token("invalid_token") is None
        
        # Test with expired token
        expired_payload = {
            "user_id": "USER_123",
            "username": "test_user",
            "roles": ["clinician"],
            "session_id": "session_123",
            "iat": datetime.utcnow() - timedelta(hours=2),
            "exp": datetime.utcnow() - timedelta(hours=1)  # Expired
        }
        
        expired_token = jwt.encode(expired_payload, auth_service.jwt_secret, algorithm="HS256")
        assert auth_service.validate_token(expired_token) is None
    
    def test_logout_user(self, auth_service):
        """Test user logout functionality."""
        username = "logout_test_user"
        email = "logout_test@example.com"
        password = "password123"
        roles = ["clinician"]
        
        # Create and authenticate user
        auth_service.create_user(username, email, password, roles)
        auth_result = auth_service.authenticate_user(
            username=username,
            password=password,
            ip_address="192.168.1.100",
            user_agent="Test Agent"
        )
        
        session_id = auth_result["session_id"]
        user_id = auth_result["user"]["user_id"]
        
        # Verify session is active
        session = auth_service.sessions[session_id]
        assert session.is_active is True
        
        # Logout user
        auth_service.logout_user(session_id, user_id)
        
        # Verify session is inactive
        assert session.is_active is False
        
        # Verify token is no longer valid
        token = auth_result["token"]
        assert auth_service.validate_token(token) is None
    
    def test_change_password(self, auth_service):
        """Test password change functionality."""
        username = "password_change_user"
        email = "password_change@example.com"
        old_password = "old_password_123"
        new_password = "new_password_456"
        roles = ["clinician"]
        
        # Create user
        user_id = auth_service.create_user(username, email, old_password, roles)
        
        # Change password
        success = auth_service.change_password(user_id, old_password, new_password)
        assert success is True
        
        # Verify old password no longer works
        result = auth_service.authenticate_user(
            username=username,
            password=old_password,
            ip_address="192.168.1.100",
            user_agent="Test Agent"
        )
        assert result is None
        
        # Verify new password works
        result = auth_service.authenticate_user(
            username=username,
            password=new_password,
            ip_address="192.168.1.100",
            user_agent="Test Agent"
        )
        assert result is not None
    
    def test_change_password_invalid_old_password(self, auth_service):
        """Test password change with invalid old password."""
        username = "invalid_old_password_user"
        email = "invalid_old@example.com"
        password = "correct_password"
        roles = ["clinician"]
        
        # Create user
        user_id = auth_service.create_user(username, email, password, roles)
        
        # Try to change password with wrong old password
        success = auth_service.change_password(user_id, "wrong_old_password", "new_password")
        assert success is False


class TestRBACService:
    """Test role-based access control service."""
    
    @pytest.fixture
    def rbac_service(self):
        """Create RBAC service for testing."""
        return RBACService()
    
    def test_default_roles_initialization(self, rbac_service):
        """Test that default roles are properly initialized."""
        expected_roles = ["clinician", "researcher", "radiologist", "data_scientist", "admin", "viewer"]
        
        for role_name in expected_roles:
            role = rbac_service.get_role(role_name)
            assert role is not None
            assert role.is_active is True
            assert len(role.permissions) > 0
    
    def test_clinician_permissions(self, rbac_service):
        """Test clinician role permissions."""
        clinician_role = rbac_service.get_role("clinician")
        
        # Should have patient data access
        assert Permission.READ_PATIENT_DATA in clinician_role.permissions
        assert Permission.WRITE_PATIENT_DATA in clinician_role.permissions
        
        # Should have medical imaging access
        assert Permission.VIEW_MEDICAL_IMAGES in clinician_role.permissions
        assert Permission.UPLOAD_MEDICAL_IMAGES in clinician_role.permissions
        
        # Should have diagnostic capabilities
        assert Permission.VIEW_DIAGNOSTIC_RESULTS in clinician_role.permissions
        assert Permission.CREATE_DIAGNOSTIC_RESULTS in clinician_role.permissions
        
        # Should NOT have admin permissions
        assert Permission.MANAGE_USERS not in clinician_role.permissions
        assert Permission.MANAGE_SYSTEM_CONFIG not in clinician_role.permissions
    
    def test_admin_permissions(self, rbac_service):
        """Test admin role has all permissions."""
        admin_role = rbac_service.get_role("admin")
        
        # Admin should have all permissions
        all_permissions = set(Permission)
        assert admin_role.permissions == all_permissions
    
    def test_viewer_permissions(self, rbac_service):
        """Test viewer role has only read permissions."""
        viewer_role = rbac_service.get_role("viewer")
        
        # Should have read permissions
        assert Permission.READ_PATIENT_DATA in viewer_role.permissions
        assert Permission.VIEW_MEDICAL_IMAGES in viewer_role.permissions
        assert Permission.VIEW_DIAGNOSTIC_RESULTS in viewer_role.permissions
        
        # Should NOT have write permissions
        assert Permission.WRITE_PATIENT_DATA not in viewer_role.permissions
        assert Permission.UPLOAD_MEDICAL_IMAGES not in viewer_role.permissions
        assert Permission.CREATE_DIAGNOSTIC_RESULTS not in viewer_role.permissions
    
    def test_check_permission_single_role(self, rbac_service):
        """Test permission checking with single role."""
        user_roles = ["clinician"]
        
        # Should have clinician permissions
        assert rbac_service.check_permission(
            user_roles, Permission.READ_PATIENT_DATA, user_id="test_user"
        ) is True
        
        assert rbac_service.check_permission(
            user_roles, Permission.VIEW_MEDICAL_IMAGES, user_id="test_user"
        ) is True
        
        # Should NOT have admin permissions
        assert rbac_service.check_permission(
            user_roles, Permission.MANAGE_USERS, user_id="test_user"
        ) is False
    
    def test_check_permission_multiple_roles(self, rbac_service):
        """Test permission checking with multiple roles."""
        user_roles = ["clinician", "researcher"]
        
        # Should have permissions from both roles
        assert rbac_service.check_permission(
            user_roles, Permission.READ_PATIENT_DATA, user_id="test_user"
        ) is True
        
        assert rbac_service.check_permission(
            user_roles, Permission.ACCESS_RESEARCH_DATA, user_id="test_user"
        ) is True
        
        assert rbac_service.check_permission(
            user_roles, Permission.TRAIN_MODELS, user_id="test_user"
        ) is True
    
    def test_check_multiple_permissions_require_all(self, rbac_service):
        """Test checking multiple permissions with require_all=True."""
        user_roles = ["clinician"]
        
        # Test permissions that clinician should have
        clinician_permissions = [
            Permission.READ_PATIENT_DATA,
            Permission.VIEW_MEDICAL_IMAGES,
            Permission.CREATE_DIAGNOSTIC_RESULTS
        ]
        
        assert rbac_service.check_multiple_permissions(
            user_roles, clinician_permissions, require_all=True, user_id="test_user"
        ) is True
        
        # Test with one permission clinician doesn't have
        mixed_permissions = [
            Permission.READ_PATIENT_DATA,
            Permission.MANAGE_USERS  # Admin only
        ]
        
        assert rbac_service.check_multiple_permissions(
            user_roles, mixed_permissions, require_all=True, user_id="test_user"
        ) is False
    
    def test_check_multiple_permissions_require_any(self, rbac_service):
        """Test checking multiple permissions with require_all=False."""
        user_roles = ["clinician"]
        
        # Test with one permission clinician has and one they don't
        mixed_permissions = [
            Permission.READ_PATIENT_DATA,  # Clinician has this
            Permission.MANAGE_USERS  # Admin only
        ]
        
        assert rbac_service.check_multiple_permissions(
            user_roles, mixed_permissions, require_all=False, user_id="test_user"
        ) is True
        
        # Test with permissions clinician doesn't have
        admin_permissions = [
            Permission.MANAGE_USERS,
            Permission.MANAGE_SYSTEM_CONFIG
        ]
        
        assert rbac_service.check_multiple_permissions(
            user_roles, admin_permissions, require_all=False, user_id="test_user"
        ) is False
    
    def test_get_user_permissions(self, rbac_service):
        """Test getting all permissions for user roles."""
        user_roles = ["clinician", "researcher"]
        
        permissions = rbac_service.get_user_permissions(user_roles)
        
        # Should include permissions from both roles
        assert Permission.READ_PATIENT_DATA in permissions
        assert Permission.ACCESS_RESEARCH_DATA in permissions
        assert Permission.TRAIN_MODELS in permissions
        
        # Should not include admin-only permissions
        assert Permission.MANAGE_USERS not in permissions
    
    def test_create_custom_role(self, rbac_service):
        """Test creating custom role."""
        role_name = "custom_test_role"
        description = "Custom role for testing"
        permissions = {
            Permission.READ_PATIENT_DATA,
            Permission.VIEW_MEDICAL_IMAGES
        }
        
        # Create role
        success = rbac_service.create_role(role_name, description, permissions)
        assert success is True
        
        # Verify role exists
        role = rbac_service.get_role(role_name)
        assert role is not None
        assert role.name == role_name
        assert role.description == description
        assert role.permissions == permissions
        assert role.is_active is True
    
    def test_create_duplicate_role(self, rbac_service):
        """Test creating duplicate role fails."""
        role_name = "duplicate_role"
        description = "Test role"
        permissions = {Permission.READ_PATIENT_DATA}
        
        # Create first role
        success1 = rbac_service.create_role(role_name, description, permissions)
        assert success1 is True
        
        # Try to create duplicate
        success2 = rbac_service.create_role(role_name, description, permissions)
        assert success2 is False
    
    def test_update_role_permissions(self, rbac_service):
        """Test updating role permissions."""
        role_name = "update_test_role"
        original_permissions = {Permission.READ_PATIENT_DATA}
        updated_permissions = {
            Permission.READ_PATIENT_DATA,
            Permission.VIEW_MEDICAL_IMAGES,
            Permission.CREATE_ANNOTATIONS
        }
        
        # Create role
        rbac_service.create_role(role_name, "Test role", original_permissions)
        
        # Update permissions
        success = rbac_service.update_role_permissions(role_name, updated_permissions)
        assert success is True
        
        # Verify permissions updated
        role = rbac_service.get_role(role_name)
        assert role.permissions == updated_permissions
    
    def test_deactivate_role(self, rbac_service):
        """Test role deactivation."""
        role_name = "deactivate_test_role"
        permissions = {Permission.READ_PATIENT_DATA}
        
        # Create role
        rbac_service.create_role(role_name, "Test role", permissions)
        
        # Verify role is active
        role = rbac_service.get_role(role_name)
        assert role.is_active is True
        
        # Deactivate role
        success = rbac_service.deactivate_role(role_name)
        assert success is True
        
        # Verify role is inactive
        assert role.is_active is False
    
    def test_list_roles(self, rbac_service):
        """Test listing roles."""
        # List all roles
        all_roles = rbac_service.list_roles(active_only=False)
        assert len(all_roles) >= 6  # Default roles
        
        # List only active roles
        active_roles = rbac_service.list_roles(active_only=True)
        assert len(active_roles) >= 6
        
        # Create and deactivate a role
        rbac_service.create_role("inactive_test_role", "Test", {Permission.READ_PATIENT_DATA})
        rbac_service.deactivate_role("inactive_test_role")
        
        # Verify active count doesn't include inactive role
        new_active_count = len(rbac_service.list_roles(active_only=True))
        assert new_active_count == len(active_roles)
    
    def test_get_permissions_for_resource(self, rbac_service):
        """Test getting permissions for specific resource types."""
        # Test patient data permissions
        patient_permissions = rbac_service.get_permissions_for_resource("patient_data")
        assert Permission.READ_PATIENT_DATA in patient_permissions
        assert Permission.WRITE_PATIENT_DATA in patient_permissions
        assert Permission.DELETE_PATIENT_DATA in patient_permissions
        
        # Test medical images permissions
        image_permissions = rbac_service.get_permissions_for_resource("medical_images")
        assert Permission.VIEW_MEDICAL_IMAGES in image_permissions
        assert Permission.UPLOAD_MEDICAL_IMAGES in image_permissions
        
        # Test system permissions
        system_permissions = rbac_service.get_permissions_for_resource("system")
        assert Permission.MANAGE_USERS in system_permissions
        assert Permission.MANAGE_ROLES in system_permissions
        
        # Test unknown resource type
        unknown_permissions = rbac_service.get_permissions_for_resource("unknown_resource")
        assert unknown_permissions == []