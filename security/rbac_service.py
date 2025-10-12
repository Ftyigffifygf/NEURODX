"""
Role-Based Access Control (RBAC) service for healthcare permissions.
"""

from typing import Dict, List, Set, Optional
from dataclasses import dataclass
from enum import Enum
import logging

from .audit_logger import AuditLogger

logger = logging.getLogger(__name__)


class Permission(Enum):
    """System permissions."""
    # Patient data permissions
    READ_PATIENT_DATA = "read_patient_data"
    WRITE_PATIENT_DATA = "write_patient_data"
    DELETE_PATIENT_DATA = "delete_patient_data"
    EXPORT_PATIENT_DATA = "export_patient_data"
    
    # Medical imaging permissions
    VIEW_MEDICAL_IMAGES = "view_medical_images"
    UPLOAD_MEDICAL_IMAGES = "upload_medical_images"
    PROCESS_MEDICAL_IMAGES = "process_medical_images"
    DELETE_MEDICAL_IMAGES = "delete_medical_images"
    
    # Diagnostic permissions
    VIEW_DIAGNOSTIC_RESULTS = "view_diagnostic_results"
    CREATE_DIAGNOSTIC_RESULTS = "create_diagnostic_results"
    APPROVE_DIAGNOSTIC_RESULTS = "approve_diagnostic_results"
    
    # Annotation permissions
    CREATE_ANNOTATIONS = "create_annotations"
    REVIEW_ANNOTATIONS = "review_annotations"
    APPROVE_ANNOTATIONS = "approve_annotations"
    
    # Model and AI permissions
    RUN_AI_INFERENCE = "run_ai_inference"
    TRAIN_MODELS = "train_models"
    DEPLOY_MODELS = "deploy_models"
    VIEW_MODEL_METRICS = "view_model_metrics"
    
    # Research permissions
    ACCESS_RESEARCH_DATA = "access_research_data"
    EXPORT_RESEARCH_DATA = "export_research_data"
    MANAGE_RESEARCH_STUDIES = "manage_research_studies"
    
    # Federated learning permissions
    PARTICIPATE_FEDERATED_LEARNING = "participate_federated_learning"
    MANAGE_FEDERATED_NODES = "manage_federated_nodes"
    
    # System administration
    MANAGE_USERS = "manage_users"
    MANAGE_ROLES = "manage_roles"
    VIEW_AUDIT_LOGS = "view_audit_logs"
    MANAGE_SYSTEM_CONFIG = "manage_system_config"
    
    # Monitoring and alerts
    VIEW_SYSTEM_METRICS = "view_system_metrics"
    MANAGE_ALERTS = "manage_alerts"


@dataclass
class Role:
    """Role definition with permissions."""
    name: str
    description: str
    permissions: Set[Permission]
    is_active: bool = True


class RBACService:
    """
    Role-Based Access Control service for healthcare permissions.
    
    Features:
    - Predefined healthcare roles (clinician, researcher, admin)
    - Fine-grained permissions for medical data access
    - Hierarchical role inheritance
    - Audit logging for all authorization decisions
    - Dynamic permission checking
    """
    
    def __init__(self, audit_logger: Optional[AuditLogger] = None):
        self.audit_logger = audit_logger or AuditLogger()
        self.roles: Dict[str, Role] = {}
        self._initialize_default_roles()
    
    def _initialize_default_roles(self):
        """Initialize default healthcare roles."""
        
        # Clinician role - can view and create patient data and diagnostics
        clinician_permissions = {
            Permission.READ_PATIENT_DATA,
            Permission.WRITE_PATIENT_DATA,
            Permission.VIEW_MEDICAL_IMAGES,
            Permission.UPLOAD_MEDICAL_IMAGES,
            Permission.PROCESS_MEDICAL_IMAGES,
            Permission.VIEW_DIAGNOSTIC_RESULTS,
            Permission.CREATE_DIAGNOSTIC_RESULTS,
            Permission.RUN_AI_INFERENCE,
            Permission.CREATE_ANNOTATIONS,
            Permission.VIEW_MODEL_METRICS
        }
        
        self.roles["clinician"] = Role(
            name="clinician",
            description="Healthcare clinician with patient care responsibilities",
            permissions=clinician_permissions
        )
        
        # Researcher role - can access research data and train models
        researcher_permissions = {
            Permission.READ_PATIENT_DATA,
            Permission.VIEW_MEDICAL_IMAGES,
            Permission.VIEW_DIAGNOSTIC_RESULTS,
            Permission.ACCESS_RESEARCH_DATA,
            Permission.EXPORT_RESEARCH_DATA,
            Permission.MANAGE_RESEARCH_STUDIES,
            Permission.RUN_AI_INFERENCE,
            Permission.TRAIN_MODELS,
            Permission.VIEW_MODEL_METRICS,
            Permission.CREATE_ANNOTATIONS,
            Permission.REVIEW_ANNOTATIONS,
            Permission.PARTICIPATE_FEDERATED_LEARNING
        }
        
        self.roles["researcher"] = Role(
            name="researcher",
            description="Medical researcher with data analysis capabilities",
            permissions=researcher_permissions
        )
        
        # Radiologist role - specialized imaging permissions
        radiologist_permissions = {
            Permission.READ_PATIENT_DATA,
            Permission.VIEW_MEDICAL_IMAGES,
            Permission.UPLOAD_MEDICAL_IMAGES,
            Permission.PROCESS_MEDICAL_IMAGES,
            Permission.VIEW_DIAGNOSTIC_RESULTS,
            Permission.CREATE_DIAGNOSTIC_RESULTS,
            Permission.APPROVE_DIAGNOSTIC_RESULTS,
            Permission.RUN_AI_INFERENCE,
            Permission.CREATE_ANNOTATIONS,
            Permission.REVIEW_ANNOTATIONS,
            Permission.APPROVE_ANNOTATIONS,
            Permission.VIEW_MODEL_METRICS
        }
        
        self.roles["radiologist"] = Role(
            name="radiologist",
            description="Medical imaging specialist",
            permissions=radiologist_permissions
        )
        
        # Data Scientist role - ML and model development
        data_scientist_permissions = {
            Permission.ACCESS_RESEARCH_DATA,
            Permission.EXPORT_RESEARCH_DATA,
            Permission.VIEW_MEDICAL_IMAGES,
            Permission.VIEW_DIAGNOSTIC_RESULTS,
            Permission.RUN_AI_INFERENCE,
            Permission.TRAIN_MODELS,
            Permission.DEPLOY_MODELS,
            Permission.VIEW_MODEL_METRICS,
            Permission.PARTICIPATE_FEDERATED_LEARNING,
            Permission.MANAGE_FEDERATED_NODES
        }
        
        self.roles["data_scientist"] = Role(
            name="data_scientist",
            description="AI/ML specialist for model development",
            permissions=data_scientist_permissions
        )
        
        # System Administrator role - full system access
        admin_permissions = set(Permission)  # All permissions
        
        self.roles["admin"] = Role(
            name="admin",
            description="System administrator with full access",
            permissions=admin_permissions
        )
        
        # Viewer role - read-only access
        viewer_permissions = {
            Permission.READ_PATIENT_DATA,
            Permission.VIEW_MEDICAL_IMAGES,
            Permission.VIEW_DIAGNOSTIC_RESULTS,
            Permission.VIEW_MODEL_METRICS,
            Permission.VIEW_SYSTEM_METRICS
        }
        
        self.roles["viewer"] = Role(
            name="viewer",
            description="Read-only access to system data",
            permissions=viewer_permissions
        )
        
        logger.info("Initialized default RBAC roles")
    
    def check_permission(self, user_roles: List[str], permission: Permission,
                        resource: Optional[str] = None, user_id: Optional[str] = None) -> bool:
        """
        Check if user has required permission.
        
        Args:
            user_roles: List of user's roles
            permission: Permission to check
            resource: Optional resource identifier
            user_id: User ID for audit logging
            
        Returns:
            True if user has permission
        """
        try:
            # Check if any of the user's roles has the required permission
            has_permission = False
            
            for role_name in user_roles:
                role = self.roles.get(role_name)
                if role and role.is_active and permission in role.permissions:
                    has_permission = True
                    break
            
            # Audit log the authorization decision
            self.audit_logger.log_authorization_event(
                user_id=user_id or "unknown",
                resource=resource or permission.value,
                action="access",
                granted=has_permission,
                details={
                    "permission": permission.value,
                    "user_roles": user_roles,
                    "resource": resource
                }
            )
            
            return has_permission
            
        except Exception as e:
            logger.error(f"Permission check failed: {e}")
            # Deny access on error for security
            return False
    
    def check_multiple_permissions(self, user_roles: List[str], 
                                 permissions: List[Permission],
                                 require_all: bool = True,
                                 user_id: Optional[str] = None) -> bool:
        """
        Check multiple permissions.
        
        Args:
            user_roles: List of user's roles
            permissions: List of permissions to check
            require_all: If True, user must have all permissions. If False, any permission is sufficient.
            user_id: User ID for audit logging
            
        Returns:
            True if permission check passes
        """
        if require_all:
            return all(
                self.check_permission(user_roles, perm, user_id=user_id)
                for perm in permissions
            )
        else:
            return any(
                self.check_permission(user_roles, perm, user_id=user_id)
                for perm in permissions
            )
    
    def get_user_permissions(self, user_roles: List[str]) -> Set[Permission]:
        """
        Get all permissions for user based on their roles.
        
        Args:
            user_roles: List of user's roles
            
        Returns:
            Set of all permissions
        """
        all_permissions = set()
        
        for role_name in user_roles:
            role = self.roles.get(role_name)
            if role and role.is_active:
                all_permissions.update(role.permissions)
        
        return all_permissions
    
    def create_role(self, name: str, description: str, 
                   permissions: Set[Permission]) -> bool:
        """
        Create a new role.
        
        Args:
            name: Role name
            description: Role description
            permissions: Set of permissions for the role
            
        Returns:
            True if role created successfully
        """
        try:
            if name in self.roles:
                logger.warning(f"Role {name} already exists")
                return False
            
            role = Role(
                name=name,
                description=description,
                permissions=permissions
            )
            
            self.roles[name] = role
            
            self.audit_logger.log_security_event(
                event_type="role_created",
                details={
                    "role_name": name,
                    "description": description,
                    "permissions": [p.value for p in permissions]
                }
            )
            
            logger.info(f"Created role: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create role {name}: {e}")
            return False
    
    def update_role_permissions(self, role_name: str, 
                              permissions: Set[Permission]) -> bool:
        """
        Update permissions for an existing role.
        
        Args:
            role_name: Name of role to update
            permissions: New set of permissions
            
        Returns:
            True if updated successfully
        """
        try:
            role = self.roles.get(role_name)
            if not role:
                logger.warning(f"Role {role_name} not found")
                return False
            
            old_permissions = role.permissions.copy()
            role.permissions = permissions
            
            self.audit_logger.log_security_event(
                event_type="role_permissions_updated",
                details={
                    "role_name": role_name,
                    "old_permissions": [p.value for p in old_permissions],
                    "new_permissions": [p.value for p in permissions]
                }
            )
            
            logger.info(f"Updated permissions for role: {role_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update role {role_name}: {e}")
            return False
    
    def deactivate_role(self, role_name: str) -> bool:
        """
        Deactivate a role.
        
        Args:
            role_name: Name of role to deactivate
            
        Returns:
            True if deactivated successfully
        """
        try:
            role = self.roles.get(role_name)
            if not role:
                return False
            
            role.is_active = False
            
            self.audit_logger.log_security_event(
                event_type="role_deactivated",
                details={"role_name": role_name}
            )
            
            logger.info(f"Deactivated role: {role_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deactivate role {role_name}: {e}")
            return False
    
    def get_role(self, role_name: str) -> Optional[Role]:
        """Get role by name."""
        return self.roles.get(role_name)
    
    def list_roles(self, active_only: bool = True) -> List[Role]:
        """
        List all roles.
        
        Args:
            active_only: If True, only return active roles
            
        Returns:
            List of roles
        """
        if active_only:
            return [role for role in self.roles.values() if role.is_active]
        else:
            return list(self.roles.values())
    
    def get_permissions_for_resource(self, resource_type: str) -> List[Permission]:
        """
        Get relevant permissions for a resource type.
        
        Args:
            resource_type: Type of resource (patient_data, medical_images, etc.)
            
        Returns:
            List of relevant permissions
        """
        resource_permissions = {
            "patient_data": [
                Permission.READ_PATIENT_DATA,
                Permission.WRITE_PATIENT_DATA,
                Permission.DELETE_PATIENT_DATA,
                Permission.EXPORT_PATIENT_DATA
            ],
            "medical_images": [
                Permission.VIEW_MEDICAL_IMAGES,
                Permission.UPLOAD_MEDICAL_IMAGES,
                Permission.PROCESS_MEDICAL_IMAGES,
                Permission.DELETE_MEDICAL_IMAGES
            ],
            "diagnostic_results": [
                Permission.VIEW_DIAGNOSTIC_RESULTS,
                Permission.CREATE_DIAGNOSTIC_RESULTS,
                Permission.APPROVE_DIAGNOSTIC_RESULTS
            ],
            "annotations": [
                Permission.CREATE_ANNOTATIONS,
                Permission.REVIEW_ANNOTATIONS,
                Permission.APPROVE_ANNOTATIONS
            ],
            "models": [
                Permission.RUN_AI_INFERENCE,
                Permission.TRAIN_MODELS,
                Permission.DEPLOY_MODELS,
                Permission.VIEW_MODEL_METRICS
            ],
            "research": [
                Permission.ACCESS_RESEARCH_DATA,
                Permission.EXPORT_RESEARCH_DATA,
                Permission.MANAGE_RESEARCH_STUDIES
            ],
            "system": [
                Permission.MANAGE_USERS,
                Permission.MANAGE_ROLES,
                Permission.VIEW_AUDIT_LOGS,
                Permission.MANAGE_SYSTEM_CONFIG,
                Permission.VIEW_SYSTEM_METRICS,
                Permission.MANAGE_ALERTS
            ]
        }
        
        return resource_permissions.get(resource_type, [])