"""
Data minimizer for HIPAA compliance and privacy protection.
Implements data minimization principles to ensure only necessary data is processed.
"""

from src.services.security.privacy_service import DataMinimizer

# Re-export the DataMinimizer class for backward compatibility
__all__ = ['DataMinimizer']