"""
Authentication middleware for Flask API.
"""

from functools import wraps
from typing import List, Optional
from flask import request, jsonify, g
import time
import logging

from src.services.security.auth_service import AuthenticationService
from src.services.security.rbac_service import RBACService, Permission
from src.services.security.audit_logger import AuditLogger

logger = logging.getLogger(__name__)


class AuthMiddleware:
    """Authentication middleware for Flask."""
    
    def __init__(self):
        self.auth_service = AuthenticationService()
        self.rbac_service = RBACService()
        self.audit_logger = AuditLogger()
    
    def require_auth(self, f):
        """Decorator to require authentication."""
        @wraps(f)
        def decorated_function(*args, **kwargs):
            start_time = time.time()
            
            # Get token from Authorization header
            auth_header = request.headers.get('Authorization')
            if not auth_header or not auth_header.startswith('Bearer '):
                response = jsonify({'error': 'Missing or invalid authorization header'})
                response.status_code = 401
                
                self._log_api_request(
                    method=request.method,
                    endpoint=request.endpoint,
                    user_id="anonymous",
                    status_code=401,
                    response_time_ms=(time.time() - start_time) * 1000
                )
                
                return response
            
            token = auth_header.split(' ')[1]
            
            # Validate token
            user_info = self.auth_service.validate_token(token)
            if not user_info:
                response = jsonify({'error': 'Invalid or expired token'})
                response.status_code = 401
                
                self._log_api_request(
                    method=request.method,
                    endpoint=request.endpoint,
                    user_id="unknown",
                    status_code=401,
                    response_time_ms=(time.time() - start_time) * 1000
                )
                
                return response
            
            # Store user info in Flask g object
            g.current_user = user_info
            g.start_time = start_time
            
            # Call the original function
            result = f(*args, **kwargs)
            
            # Log successful API request
            status_code = getattr(result, 'status_code', 200)
            self._log_api_request(
                method=request.method,
                endpoint=request.endpoint,
                user_id=user_info['user_id'],
                status_code=status_code,
                response_time_ms=(time.time() - start_time) * 1000
            )
            
            return result
        
        return decorated_function
    
    def require_permission(self, permission: Permission, resource: Optional[str] = None):
        """Decorator to require specific permission."""
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                # Check if user is authenticated
                if not hasattr(g, 'current_user') or not g.current_user:
                    return jsonify({'error': 'Authentication required'}), 401
                
                user_info = g.current_user
                user_roles = user_info.get('roles', [])
                user_id = user_info.get('user_id')
                
                # Check permission
                has_permission = self.rbac_service.check_permission(
                    user_roles=user_roles,
                    permission=permission,
                    resource=resource,
                    user_id=user_id
                )
                
                if not has_permission:
                    return jsonify({
                        'error': 'Insufficient permissions',
                        'required_permission': permission.value
                    }), 403
                
                return f(*args, **kwargs)
            
            return decorated_function
        return decorator
    
    def require_any_permission(self, permissions: List[Permission], resource: Optional[str] = None):
        """Decorator to require any of the specified permissions."""
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                if not hasattr(g, 'current_user') or not g.current_user:
                    return jsonify({'error': 'Authentication required'}), 401
                
                user_info = g.current_user
                user_roles = user_info.get('roles', [])
                user_id = user_info.get('user_id')
                
                # Check if user has any of the required permissions
                has_permission = self.rbac_service.check_multiple_permissions(
                    user_roles=user_roles,
                    permissions=permissions,
                    require_all=False,
                    user_id=user_id
                )
                
                if not has_permission:
                    return jsonify({
                        'error': 'Insufficient permissions',
                        'required_permissions': [p.value for p in permissions]
                    }), 403
                
                return f(*args, **kwargs)
            
            return decorated_function
        return decorator
    
    def require_all_permissions(self, permissions: List[Permission], resource: Optional[str] = None):
        """Decorator to require all of the specified permissions."""
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                if not hasattr(g, 'current_user') or not g.current_user:
                    return jsonify({'error': 'Authentication required'}), 401
                
                user_info = g.current_user
                user_roles = user_info.get('roles', [])
                user_id = user_info.get('user_id')
                
                # Check if user has all required permissions
                has_permission = self.rbac_service.check_multiple_permissions(
                    user_roles=user_roles,
                    permissions=permissions,
                    require_all=True,
                    user_id=user_id
                )
                
                if not has_permission:
                    return jsonify({
                        'error': 'Insufficient permissions',
                        'required_permissions': [p.value for p in permissions]
                    }), 403
                
                return f(*args, **kwargs)
            
            return decorated_function
        return decorator
    
    def require_role(self, required_roles: List[str]):
        """Decorator to require specific roles."""
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                if not hasattr(g, 'current_user') or not g.current_user:
                    return jsonify({'error': 'Authentication required'}), 401
                
                user_info = g.current_user
                user_roles = user_info.get('roles', [])
                
                # Check if user has any of the required roles
                has_role = any(role in user_roles for role in required_roles)
                
                if not has_role:
                    return jsonify({
                        'error': 'Insufficient role privileges',
                        'required_roles': required_roles,
                        'user_roles': user_roles
                    }), 403
                
                return f(*args, **kwargs)
            
            return decorated_function
        return decorator
    
    def _log_api_request(self, method: str, endpoint: str, user_id: str,
                        status_code: int, response_time_ms: float):
        """Log API request for audit trail."""
        try:
            self.audit_logger.log_api_request(
                method=method,
                endpoint=endpoint or "unknown",
                user_id=user_id,
                status_code=status_code,
                response_time_ms=response_time_ms,
                details={
                    "ip_address": request.remote_addr,
                    "user_agent": request.headers.get('User-Agent', 'unknown')
                }
            )
        except Exception as e:
            logger.error(f"Failed to log API request: {e}")


# Global middleware instance
auth_middleware = AuthMiddleware()

# Convenience decorators
require_auth = auth_middleware.require_auth
require_permission = auth_middleware.require_permission
require_any_permission = auth_middleware.require_any_permission
require_all_permissions = auth_middleware.require_all_permissions
require_role = auth_middleware.require_role