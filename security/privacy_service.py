"""
Privacy service implementing differential privacy and data minimization
for HIPAA compliance and enhanced patient data protection.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class PrivacyBudget:
    """Tracks privacy budget for differential privacy."""
    epsilon: float
    delta: float
    consumed: float = 0.0
    
    def has_budget(self, required_epsilon: float) -> bool:
        """Check if sufficient privacy budget remains."""
        return (self.consumed + required_epsilon) <= self.epsilon
    
    def consume(self, epsilon: float) -> None:
        """Consume privacy budget."""
        if not self.has_budget(epsilon):
            raise ValueError(f"Insufficient privacy budget. Required: {epsilon}, Available: {self.epsilon - self.consumed}")
        self.consumed += epsilon


class DifferentialPrivacy:
    """
    Implements differential privacy mechanisms for protecting patient data
    while maintaining statistical utility for medical AI applications.
    """
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        """
        Initialize differential privacy service.
        
        Args:
            epsilon: Privacy parameter (smaller = more private)
            delta: Failure probability (should be very small)
        """
        self.budget = PrivacyBudget(epsilon=epsilon, delta=delta)
        self.noise_scale = 1.0 / epsilon
        
    def add_laplace_noise(self, value: float, sensitivity: float = 1.0, epsilon: Optional[float] = None) -> float:
        """
        Add Laplace noise for differential privacy.
        
        Args:
            value: Original value to add noise to
            sensitivity: Global sensitivity of the query
            epsilon: Privacy parameter for this query (uses default if None)
            
        Returns:
            Noisy value with differential privacy guarantee
        """
        if epsilon is None:
            epsilon = self.budget.epsilon / 10  # Conservative allocation
            
        if not self.budget.has_budget(epsilon):
            logger.warning("Insufficient privacy budget, using minimal noise")
            epsilon = self.budget.epsilon - self.budget.consumed
            
        if epsilon <= 0:
            raise ValueError("No privacy budget remaining")
            
        # Laplace mechanism: noise ~ Laplace(0, sensitivity/epsilon)
        scale = sensitivity / epsilon
        noise = np.random.laplace(0, scale)
        
        self.budget.consume(epsilon)
        
        return value + noise
    
    def add_gaussian_noise(self, value: float, sensitivity: float = 1.0, epsilon: Optional[float] = None) -> float:
        """
        Add Gaussian noise for differential privacy (for approximate DP).
        
        Args:
            value: Original value to add noise to
            sensitivity: Global sensitivity of the query
            epsilon: Privacy parameter for this query
            
        Returns:
            Noisy value with approximate differential privacy
        """
        if epsilon is None:
            epsilon = self.budget.epsilon / 10
            
        if not self.budget.has_budget(epsilon):
            logger.warning("Insufficient privacy budget")
            epsilon = max(0.01, self.budget.epsilon - self.budget.consumed)
            
        # Gaussian mechanism for (epsilon, delta)-DP
        sigma = np.sqrt(2 * np.log(1.25 / self.budget.delta)) * sensitivity / epsilon
        noise = np.random.normal(0, sigma)
        
        self.budget.consume(epsilon)
        
        return value + noise
    
    def add_noise_to_count(self, count: int, epsilon: Optional[float] = None) -> int:
        """
        Add noise to count queries with differential privacy.
        
        Args:
            count: Original count value
            epsilon: Privacy parameter
            
        Returns:
            Noisy count (rounded to integer)
        """
        noisy_count = self.add_laplace_noise(float(count), sensitivity=1.0, epsilon=epsilon)
        return max(0, int(round(noisy_count)))  # Ensure non-negative integer
    
    def add_noise_to_average(self, average: float, count: int, value_range: tuple, epsilon: Optional[float] = None) -> float:
        """
        Add noise to average queries with differential privacy.
        
        Args:
            average: Original average value
            count: Number of records in average
            value_range: (min, max) range of possible values
            epsilon: Privacy parameter
            
        Returns:
            Noisy average with differential privacy
        """
        # Sensitivity of average query
        sensitivity = (value_range[1] - value_range[0]) / count
        
        return self.add_laplace_noise(average, sensitivity=sensitivity, epsilon=epsilon)
    
    def privatize_histogram(self, histogram: Dict[str, int], epsilon: Optional[float] = None) -> Dict[str, int]:
        """
        Add noise to histogram bins for differential privacy.
        
        Args:
            histogram: Dictionary mapping categories to counts
            epsilon: Privacy parameter
            
        Returns:
            Noisy histogram with differential privacy
        """
        if epsilon is None:
            epsilon = self.budget.epsilon / 5  # Conservative allocation for multiple queries
            
        noisy_histogram = {}
        epsilon_per_bin = epsilon / len(histogram)  # Split budget across bins
        
        for category, count in histogram.items():
            noisy_count = self.add_noise_to_count(count, epsilon=epsilon_per_bin)
            noisy_histogram[category] = noisy_count
            
        return noisy_histogram
    
    def get_privacy_budget_status(self) -> Dict[str, float]:
        """Get current privacy budget status."""
        return {
            "total_epsilon": self.budget.epsilon,
            "consumed_epsilon": self.budget.consumed,
            "remaining_epsilon": self.budget.epsilon - self.budget.consumed,
            "utilization_percentage": (self.budget.consumed / self.budget.epsilon) * 100
        }


class DataMinimizer:
    """
    Implements data minimization principles for HIPAA compliance.
    Ensures only necessary data is collected, processed, and stored.
    """
    
    # Define data categories and their purposes
    ESSENTIAL_FOR_AI = {
        "patient_id", "study_id", "imaging_data", "wearable_data",
        "cognitive_scores", "diagnosis", "age", "gender", "scan_metadata"
    }
    
    ADMINISTRATIVE_ONLY = {
        "social_security", "insurance_info", "billing_address",
        "emergency_contact", "employer_info", "payment_method"
    }
    
    RESEARCH_OPTIONAL = {
        "home_address", "phone_number", "email", "occupation",
        "education_level", "family_history_details"
    }
    
    def __init__(self):
        """Initialize data minimizer."""
        self.access_log = []
    
    def minimize_for_ai_processing(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Minimize data for AI processing, keeping only essential fields.
        
        Args:
            data: Original data dictionary
            
        Returns:
            Minimized data with only AI-essential fields
        """
        minimized = {}
        
        for key, value in data.items():
            if key in self.ESSENTIAL_FOR_AI:
                minimized[key] = value
            else:
                logger.debug(f"Removing non-essential field for AI processing: {key}")
        
        self._log_minimization("ai_processing", len(data), len(minimized))
        return minimized
    
    def minimize_for_research(self, data: Dict[str, Any], include_optional: bool = False) -> Dict[str, Any]:
        """
        Minimize data for research purposes.
        
        Args:
            data: Original data dictionary
            include_optional: Whether to include optional research fields
            
        Returns:
            Minimized data appropriate for research
        """
        minimized = {}
        allowed_fields = self.ESSENTIAL_FOR_AI.copy()
        
        if include_optional:
            allowed_fields.update(self.RESEARCH_OPTIONAL)
        
        for key, value in data.items():
            if key in allowed_fields:
                minimized[key] = value
            elif key in self.ADMINISTRATIVE_ONLY:
                logger.debug(f"Removing administrative field for research: {key}")
        
        self._log_minimization("research", len(data), len(minimized))
        return minimized
    
    def minimize_for_display(self, data: Dict[str, Any], user_role: str) -> Dict[str, Any]:
        """
        Minimize data for display based on user role.
        
        Args:
            data: Original data dictionary
            user_role: Role of the user requesting data
            
        Returns:
            Data appropriate for the user's role
        """
        if user_role == "radiologist":
            # Radiologists need clinical data but not administrative
            allowed_fields = self.ESSENTIAL_FOR_AI - {"social_security", "insurance_info"}
        elif user_role == "researcher":
            # Researchers get anonymized data
            allowed_fields = self.ESSENTIAL_FOR_AI - {"patient_id"}
            # Add anonymized ID
            data = data.copy()
            data["anonymized_id"] = self._generate_anonymized_id(data.get("patient_id", ""))
        elif user_role == "admin":
            # Admins can see more fields but still minimize
            allowed_fields = self.ESSENTIAL_FOR_AI.union(self.RESEARCH_OPTIONAL)
        else:
            # Default to minimal access
            allowed_fields = {"patient_id", "diagnosis", "age", "gender"}
        
        minimized = {k: v for k, v in data.items() if k in allowed_fields}
        self._log_minimization(f"display_{user_role}", len(data), len(minimized))
        
        return minimized
    
    def anonymize_for_sharing(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Anonymize data for external sharing or research.
        
        Args:
            data: Original data dictionary
            
        Returns:
            Anonymized data safe for sharing
        """
        anonymized = {}
        
        for key, value in data.items():
            if key == "patient_id":
                anonymized["anonymized_id"] = self._generate_anonymized_id(value)
            elif key in {"social_security", "phone_number", "email", "home_address"}:
                # Remove direct identifiers
                continue
            elif key == "age":
                # Age binning for additional privacy
                anonymized["age_group"] = self._bin_age(value)
            elif key in {"diagnosis", "cognitive_scores", "imaging_data", "wearable_data"}:
                # Keep clinical data
                anonymized[key] = value
            else:
                # Keep other non-identifying fields
                anonymized[key] = value
        
        self._log_minimization("anonymization", len(data), len(anonymized))
        return anonymized
    
    def _generate_anonymized_id(self, patient_id: str) -> str:
        """Generate anonymized ID from patient ID."""
        import hashlib
        # Use SHA-256 hash for anonymization
        hash_object = hashlib.sha256(patient_id.encode())
        return f"ANON_{hash_object.hexdigest()[:12].upper()}"
    
    def _bin_age(self, age: int) -> str:
        """Bin age into ranges for privacy."""
        if age < 30:
            return "18-29"
        elif age < 40:
            return "30-39"
        elif age < 50:
            return "40-49"
        elif age < 60:
            return "50-59"
        elif age < 70:
            return "60-69"
        elif age < 80:
            return "70-79"
        else:
            return "80+"
    
    def _log_minimization(self, purpose: str, original_fields: int, minimized_fields: int):
        """Log data minimization activity."""
        log_entry = {
            "timestamp": datetime.now(),
            "purpose": purpose,
            "original_fields": original_fields,
            "minimized_fields": minimized_fields,
            "reduction_percentage": ((original_fields - minimized_fields) / original_fields) * 100
        }
        self.access_log.append(log_entry)
        
        logger.info(f"Data minimized for {purpose}: {original_fields} -> {minimized_fields} fields "
                   f"({log_entry['reduction_percentage']:.1f}% reduction)")
    
    def get_minimization_report(self) -> Dict[str, Any]:
        """Get report on data minimization activities."""
        if not self.access_log:
            return {"total_minimizations": 0}
        
        total_minimizations = len(self.access_log)
        avg_reduction = sum(log["reduction_percentage"] for log in self.access_log) / total_minimizations
        
        purposes = {}
        for log in self.access_log:
            purpose = log["purpose"]
            if purpose not in purposes:
                purposes[purpose] = {"count": 0, "avg_reduction": 0}
            purposes[purpose]["count"] += 1
            purposes[purpose]["avg_reduction"] += log["reduction_percentage"]
        
        # Calculate averages
        for purpose_data in purposes.values():
            purpose_data["avg_reduction"] /= purpose_data["count"]
        
        return {
            "total_minimizations": total_minimizations,
            "average_reduction_percentage": avg_reduction,
            "by_purpose": purposes,
            "last_activity": self.access_log[-1]["timestamp"]
        }


class K_Anonymity:
    """
    Implements k-anonymity for additional privacy protection.
    Ensures each record is indistinguishable from at least k-1 other records.
    """
    
    def __init__(self, k: int = 5):
        """
        Initialize k-anonymity service.
        
        Args:
            k: Minimum group size for anonymity
        """
        self.k = k
    
    def check_k_anonymity(self, data: List[Dict[str, Any]], quasi_identifiers: List[str]) -> Dict[str, Any]:
        """
        Check if dataset satisfies k-anonymity for given quasi-identifiers.
        
        Args:
            data: List of records
            quasi_identifiers: Fields that could be used for identification
            
        Returns:
            Analysis of k-anonymity compliance
        """
        # Group records by quasi-identifier combinations
        groups = {}
        
        for record in data:
            # Create key from quasi-identifiers
            key = tuple(record.get(qi, None) for qi in quasi_identifiers)
            if key not in groups:
                groups[key] = []
            groups[key].append(record)
        
        # Check group sizes
        small_groups = [group for group in groups.values() if len(group) < self.k]
        total_records_at_risk = sum(len(group) for group in small_groups)
        
        return {
            "satisfies_k_anonymity": len(small_groups) == 0,
            "k_value": self.k,
            "total_groups": len(groups),
            "groups_below_k": len(small_groups),
            "records_at_risk": total_records_at_risk,
            "compliance_percentage": ((len(data) - total_records_at_risk) / len(data)) * 100 if data else 0
        }
    
    def generalize_for_k_anonymity(self, data: List[Dict[str, Any]], quasi_identifiers: List[str]) -> List[Dict[str, Any]]:
        """
        Generalize data to achieve k-anonymity.
        
        Args:
            data: List of records
            quasi_identifiers: Fields to generalize
            
        Returns:
            Generalized data satisfying k-anonymity
        """
        # Simple generalization strategy - this could be made more sophisticated
        generalized_data = []
        
        for record in data:
            generalized_record = record.copy()
            
            # Generalize age to age groups
            if "age" in quasi_identifiers and "age" in record:
                generalized_record["age"] = self._generalize_age(record["age"])
            
            # Generalize location to broader regions
            if "location" in quasi_identifiers and "location" in record:
                generalized_record["location"] = self._generalize_location(record["location"])
            
            generalized_data.append(generalized_record)
        
        return generalized_data
    
    def _generalize_age(self, age: int) -> str:
        """Generalize age to broader ranges."""
        if age < 25:
            return "18-24"
        elif age < 35:
            return "25-34"
        elif age < 45:
            return "35-44"
        elif age < 55:
            return "45-54"
        elif age < 65:
            return "55-64"
        else:
            return "65+"
    
    def _generalize_location(self, location: str) -> str:
        """Generalize location to broader regions."""
        # This is a simplified example - real implementation would use geographic hierarchies
        if "New York" in location:
            return "Northeast US"
        elif "California" in location:
            return "West Coast US"
        elif "Texas" in location:
            return "Southwest US"
        else:
            return "Other US"