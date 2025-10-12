"""
MONAI Federated Learning Service

This module provides federated learning capabilities using MONAI's federated learning framework
for privacy-preserving multi-institutional model training.
"""

# Import only the components that don't have heavy dependencies
from .aggregation_engine import ModelAggregationEngine
from .longitudinal_tracker import LongitudinalTracker
from .fault_tolerance import FaultToleranceManager

__all__ = [
    'ModelAggregationEngine',
    'LongitudinalTracker',
    'FaultToleranceManager'
]

# Lazy imports to avoid dependency issues
def get_federated_server():
    """Get FederatedLearningServer with lazy import"""
    from .federated_server import FederatedLearningServer
    return FederatedLearningServer

def get_federated_client():
    """Get FederatedLearningClient with lazy import"""
    from .federated_client import FederatedLearningClient
    return FederatedLearningClient

def get_secure_communication():
    """Get SecureCommunicationManager with lazy import"""
    from .secure_communication import SecureCommunicationManager
    return SecureCommunicationManager