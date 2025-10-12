"""
Authentication API routes.
"""

from flask import Blueprint, request, jsonify, g
from typing import Dict, Any
import logging

from src.services.security.auth_service import AuthenticationService
from src.services.security.rbac_service import RBACService, Permission
from src.api.middleware import require_auth, require_permission

logger = logging.getLogger(__name__)

# Create blueprint
auth_bp = Blueprint('auth', __name__, url_prefix='/api/auth')

# Initialize services
auth_service = AuthenticationService()
rbac_service = RBACService()


@auth_bp.route('/login', methods=['POST'])
def login():
    """
    User login endpoint.
    
    Expected JSON payload:
    {
        "username": "user@example.com",
        "password": "password123"
    }
    """
    try:
        data = request.get_json()
        
        if not data or not data.get('username') or not data.get('password'):
            return jsonify({
                'error': 'Username and password are required'
            }), 400
        
        username = data['username']
        password = data['password']
        ip_address = request.remote_addr
        user_agent = request.headers.get('User-Agent', 'unknown')
        
        # Get MFA code if provided
        mfa_code = data.get('mfa_code')
        
        # Authenticate user
        auth_result = auth_service.authenticate_user(
            username=username,
            password=password,
            ip_address=ip_address,
            user_agent=user_agent,
            mfa_code=mfa_code
        )
        
        if not auth_result:
            return jsonify({
                'error': 'Invalid credentials or account locked'
            }), 401
        
        # Check if MFA is required
        if auth_result.get('mfa_required'):
            return jsonify({
                'mfa_required': True,
                'user_id': auth_result['user_id'],
                'message': auth_result['message']
            }), 200
        
        return jsonify({
            'message': 'Login successful',
            'token': auth_result['token'],
            'user': auth_result['user'],
            'session_id': auth_result['session_id'],
            'expires_at': auth_result['expires_at']
        }), 200
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({
            'error': 'Internal server error'
        }), 500


@auth_bp.route('/logout', methods=['POST'])
@require_auth
def logout():
    """User logout endpoint."""
    try:
        user_info = g.current_user
        session_id = user_info.get('session_id')
        user_id = user_info.get('user_id')
        
        # Logout user
        auth_service.logout_user(session_id, user_id)
        
        return jsonify({
            'message': 'Logout successful'
        }), 200
        
    except Exception as e:
        logger.error(f"Logout error: {e}")
        return jsonify({
            'error': 'Internal server error'
        }), 500


@auth_bp.route('/me', methods=['GET'])
@require_auth
def get_current_user():
    """Get current user information."""
    try:
        user_info = g.current_user
        user_id = user_info.get('user_id')
        
        # Get full user details
        user = auth_service.get_user_by_id(user_id)
        if not user:
            return jsonify({
                'error': 'User not found'
            }), 404
        
        # Get user permissions
        user_permissions = rbac_service.get_user_permissions(user.roles)
        
        # Get active sessions
        active_sessions = auth_service.get_active_sessions(user_id)
        
        return jsonify({
            'user': {
                'user_id': user.user_id,
                'username': user.username,
                'email': user.email,
                'roles': user.roles,
                'created_at': user.created_at.isoformat(),
                'last_login': user.last_login.isoformat() if user.last_login else None
            },
            'permissions': [p.value for p in user_permissions],
            'active_sessions': len(active_sessions)
        }), 200
        
    except Exception as e:
        logger.error(f"Get current user error: {e}")
        return jsonify({
            'error': 'Internal server error'
        }), 500


@auth_bp.route('/change-password', methods=['POST'])
@require_auth
def change_password():
    """
    Change user password.
    
    Expected JSON payload:
    {
        "old_password": "current_password",
        "new_password": "new_password"
    }
    """
    try:
        data = request.get_json()
        
        if not data or not data.get('old_password') or not data.get('new_password'):
            return jsonify({
                'error': 'Old password and new password are required'
            }), 400
        
        user_info = g.current_user
        user_id = user_info.get('user_id')
        old_password = data['old_password']
        new_password = data['new_password']
        
        # Validate new password strength (basic validation)
        if len(new_password) < 8:
            return jsonify({
                'error': 'New password must be at least 8 characters long'
            }), 400
        
        # Change password
        success = auth_service.change_password(user_id, old_password, new_password)
        
        if not success:
            return jsonify({
                'error': 'Failed to change password. Check your current password.'
            }), 400
        
        return jsonify({
            'message': 'Password changed successfully. Please log in again.'
        }), 200
        
    except Exception as e:
        logger.error(f"Change password error: {e}")
        return jsonify({
            'error': 'Internal server error'
        }), 500


@auth_bp.route('/permissions', methods=['GET'])
@require_auth
def get_user_permissions():
    """Get current user's permissions."""
    try:
        user_info = g.current_user
        user_roles = user_info.get('roles', [])
        
        # Get all permissions for user
        permissions = rbac_service.get_user_permissions(user_roles)
        
        # Group permissions by category
        permission_groups = {
            'patient_data': [],
            'medical_images': [],
            'diagnostic_results': [],
            'annotations': [],
            'models': [],
            'research': [],
            'system': []
        }
        
        for permission in permissions:
            perm_value = permission.value
            
            if 'patient_data' in perm_value:
                permission_groups['patient_data'].append(perm_value)
            elif 'medical_image' in perm_value:
                permission_groups['medical_images'].append(perm_value)
            elif 'diagnostic' in perm_value:
                permission_groups['diagnostic_results'].append(perm_value)
            elif 'annotation' in perm_value:
                permission_groups['annotations'].append(perm_value)
            elif any(x in perm_value for x in ['model', 'inference', 'train']):
                permission_groups['models'].append(perm_value)
            elif 'research' in perm_value:
                permission_groups['research'].append(perm_value)
            else:
                permission_groups['system'].append(perm_value)
        
        return jsonify({
            'user_roles': user_roles,
            'permissions': {
                'all': [p.value for p in permissions],
                'grouped': permission_groups
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Get permissions error: {e}")
        return jsonify({
            'error': 'Internal server error'
        }), 500


@auth_bp.route('/roles', methods=['GET'])
@require_permission(Permission.MANAGE_ROLES)
def list_roles():
    """List all system roles (admin only)."""
    try:
        roles = rbac_service.list_roles()
        
        role_data = []
        for role in roles:
            role_data.append({
                'name': role.name,
                'description': role.description,
                'permissions': [p.value for p in role.permissions],
                'is_active': role.is_active
            })
        
        return jsonify({
            'roles': role_data
        }), 200
        
    except Exception as e:
        logger.error(f"List roles error: {e}")
        return jsonify({
            'error': 'Internal server error'
        }), 500


@auth_bp.route('/users', methods=['POST'])
@require_permission(Permission.MANAGE_USERS)
def create_user():
    """
    Create new user (admin only).
    
    Expected JSON payload:
    {
        "username": "newuser",
        "email": "user@example.com",
        "password": "password123",
        "roles": ["clinician"]
    }
    """
    try:
        data = request.get_json()
        
        required_fields = ['username', 'email', 'password', 'roles']
        if not data or not all(field in data for field in required_fields):
            return jsonify({
                'error': 'Username, email, password, and roles are required'
            }), 400
        
        username = data['username']
        email = data['email']
        password = data['password']
        roles = data['roles']
        
        # Validate roles exist
        for role_name in roles:
            if not rbac_service.get_role(role_name):
                return jsonify({
                    'error': f'Role {role_name} does not exist'
                }), 400
        
        # Create user
        user_id = auth_service.create_user(
            username=username,
            email=email,
            password=password,
            roles=roles
        )
        
        return jsonify({
            'message': 'User created successfully',
            'user_id': user_id
        }), 201
        
    except ValueError as e:
        return jsonify({
            'error': str(e)
        }), 400
    except Exception as e:
        logger.error(f"Create user error: {e}")
        return jsonify({
            'error': 'Internal server error'
        }), 500


@auth_bp.route('/validate', methods=['POST'])
def validate_token():
    """
    Validate JWT token.
    
    Expected JSON payload:
    {
        "token": "jwt_token_here"
    }
    """
    try:
        data = request.get_json()
        
        if not data or not data.get('token'):
            return jsonify({
                'error': 'Token is required'
            }), 400
        
        token = data['token']
        
        # Validate token
        user_info = auth_service.validate_token(token)
        
        if not user_info:
            return jsonify({
                'valid': False,
                'error': 'Invalid or expired token'
            }), 401
        
        return jsonify({
            'valid': True,
            'user': {
                'user_id': user_info['user_id'],
                'username': user_info['username'],
                'email': user_info['email'],
                'roles': user_info['roles']
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Token validation error: {e}")
        return jsonify({
            'error': 'Internal server error'
        }), 500


@auth_bp.route('/mfa/setup', methods=['POST'])
@require_auth
def setup_mfa():
    """Set up MFA for current user."""
    try:
        user_info = g.current_user
        user_id = user_info.get('user_id')
        username = user_info.get('username')
        
        # Set up MFA
        mfa_setup = auth_service.setup_mfa(user_id, username)
        
        return jsonify({
            'message': 'MFA setup initiated',
            'device_id': mfa_setup['device_id'],
            'qr_code': mfa_setup['qr_code'],
            'secret_key': mfa_setup['secret_key'],
            'backup_codes': mfa_setup['backup_codes']
        }), 200
        
    except Exception as e:
        logger.error(f"MFA setup error: {e}")
        return jsonify({
            'error': 'Internal server error'
        }), 500


@auth_bp.route('/mfa/verify-setup', methods=['POST'])
@require_auth
def verify_mfa_setup():
    """
    Verify MFA setup with verification code.
    
    Expected JSON payload:
    {
        "device_id": "device_id_from_setup",
        "verification_code": "123456"
    }
    """
    try:
        data = request.get_json()
        
        if not data or not data.get('device_id') or not data.get('verification_code'):
            return jsonify({
                'error': 'Device ID and verification code are required'
            }), 400
        
        device_id = data['device_id']
        verification_code = data['verification_code']
        
        # Verify setup
        success = auth_service.verify_mfa_setup(device_id, verification_code)
        
        if not success:
            return jsonify({
                'error': 'Invalid verification code or setup expired'
            }), 400
        
        return jsonify({
            'message': 'MFA setup completed successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"MFA verification error: {e}")
        return jsonify({
            'error': 'Internal server error'
        }), 500


@auth_bp.route('/mfa/devices', methods=['GET'])
@require_auth
def get_mfa_devices():
    """Get user's MFA devices."""
    try:
        user_info = g.current_user
        user_id = user_info.get('user_id')
        
        devices = auth_service.get_user_mfa_devices(user_id)
        
        return jsonify({
            'devices': devices
        }), 200
        
    except Exception as e:
        logger.error(f"Get MFA devices error: {e}")
        return jsonify({
            'error': 'Internal server error'
        }), 500


@auth_bp.route('/mfa/devices/<device_id>', methods=['DELETE'])
@require_auth
def disable_mfa_device(device_id):
    """Disable MFA device."""
    try:
        user_info = g.current_user
        user_id = user_info.get('user_id')
        
        success = auth_service.disable_mfa_device(user_id, device_id)
        
        if not success:
            return jsonify({
                'error': 'Device not found or access denied'
            }), 404
        
        return jsonify({
            'message': 'MFA device disabled successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Disable MFA device error: {e}")
        return jsonify({
            'error': 'Internal server error'
        }), 500


@auth_bp.route('/mfa/backup-codes/<device_id>', methods=['POST'])
@require_auth
def generate_backup_codes(device_id):
    """Generate new backup codes for MFA device."""
    try:
        user_info = g.current_user
        user_id = user_info.get('user_id')
        
        backup_codes = auth_service.generate_backup_codes(user_id, device_id)
        
        if not backup_codes:
            return jsonify({
                'error': 'Device not found or access denied'
            }), 404
        
        return jsonify({
            'message': 'New backup codes generated',
            'backup_codes': backup_codes
        }), 200
        
    except Exception as e:
        logger.error(f"Generate backup codes error: {e}")
        return jsonify({
            'error': 'Internal server error'
        }), 500