"""
FHIR API client for patient data exchange and diagnostic result formatting.

This module provides FHIR R4 compliant client for integrating with healthcare systems,
supporting patient data retrieval and diagnostic result submission.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import requests
from requests.auth import HTTPBasicAuth
import base64

from src.models.patient import PatientRecord, ImagingStudy
from src.models.diagnostics import DiagnosticResult


@dataclass
class FHIRConfig:
    """Configuration for FHIR server connection."""
    base_url: str
    auth_type: str  # "basic", "bearer", "oauth2"
    username: Optional[str] = None
    password: Optional[str] = None
    token: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    timeout: int = 30
    verify_ssl: bool = True


class FHIRAuthenticationError(Exception):
    """Raised when FHIR authentication fails."""
    pass


class FHIRValidationError(Exception):
    """Raised when FHIR resource validation fails."""
    pass


class FHIRClient:
    """
    FHIR R4 compliant client for healthcare system integration.
    
    Provides methods for:
    - Patient data retrieval and exchange
    - Diagnostic result formatting and submission
    - Authentication and authorization handling
    """
    
    def __init__(self, config: FHIRConfig):
        """
        Initialize FHIR client with configuration.
        
        Args:
            config: FHIR server configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.session = requests.Session()
        self.session.verify = config.verify_ssl
        self.session.timeout = config.timeout
        
        # Set up authentication
        self._setup_authentication()
        
        # Set FHIR headers
        self.session.headers.update({
            'Accept': 'application/fhir+json',
            'Content-Type': 'application/fhir+json',
            'User-Agent': 'NeuroDx-MultiModal/1.0'
        })
    
    def _setup_authentication(self) -> None:
        """Set up authentication based on configuration."""
        if self.config.auth_type == "basic":
            if not self.config.username or not self.config.password:
                raise FHIRAuthenticationError("Username and password required for basic auth")
            self.session.auth = HTTPBasicAuth(self.config.username, self.config.password)
            
        elif self.config.auth_type == "bearer":
            if not self.config.token:
                raise FHIRAuthenticationError("Token required for bearer auth")
            self.session.headers['Authorization'] = f'Bearer {self.config.token}'
            
        elif self.config.auth_type == "oauth2":
            if not self.config.client_id or not self.config.client_secret:
                raise FHIRAuthenticationError("Client ID and secret required for OAuth2")
            self._obtain_oauth2_token()
    
    def _obtain_oauth2_token(self) -> None:
        """Obtain OAuth2 access token."""
        token_url = f"{self.config.base_url}/oauth2/token"
        
        # Prepare credentials
        credentials = base64.b64encode(
            f"{self.config.client_id}:{self.config.client_secret}".encode()
        ).decode()
        
        headers = {
            'Authorization': f'Basic {credentials}',
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        data = {
            'grant_type': 'client_credentials',
            'scope': 'system/Patient.read system/DiagnosticReport.write'
        }
        
        try:
            response = requests.post(token_url, headers=headers, data=data, timeout=self.config.timeout)
            response.raise_for_status()
            
            token_data = response.json()
            access_token = token_data.get('access_token')
            
            if not access_token:
                raise FHIRAuthenticationError("No access token in OAuth2 response")
            
            self.session.headers['Authorization'] = f'Bearer {access_token}'
            self.logger.info("OAuth2 token obtained successfully")
            
        except requests.RequestException as e:
            raise FHIRAuthenticationError(f"Failed to obtain OAuth2 token: {e}")
    
    def get_patient(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve patient data from FHIR server.
        
        Args:
            patient_id: FHIR patient identifier
            
        Returns:
            Patient FHIR resource or None if not found
        """
        url = f"{self.config.base_url}/Patient/{patient_id}"
        
        try:
            response = self.session.get(url)
            
            if response.status_code == 404:
                self.logger.warning(f"Patient {patient_id} not found")
                return None
            
            response.raise_for_status()
            patient_data = response.json()
            
            self.logger.info(f"Retrieved patient data for {patient_id}")
            return patient_data
            
        except requests.RequestException as e:
            self.logger.error(f"Failed to retrieve patient {patient_id}: {e}")
            raise
    
    def search_patients(self, search_params: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Search for patients using FHIR search parameters.
        
        Args:
            search_params: FHIR search parameters (e.g., {'name': 'Smith', 'birthdate': '1980-01-01'})
            
        Returns:
            List of patient FHIR resources
        """
        url = f"{self.config.base_url}/Patient"
        
        try:
            response = self.session.get(url, params=search_params)
            response.raise_for_status()
            
            bundle = response.json()
            patients = []
            
            if bundle.get('resourceType') == 'Bundle':
                for entry in bundle.get('entry', []):
                    if entry.get('resource', {}).get('resourceType') == 'Patient':
                        patients.append(entry['resource'])
            
            self.logger.info(f"Found {len(patients)} patients matching search criteria")
            return patients
            
        except requests.RequestException as e:
            self.logger.error(f"Failed to search patients: {e}")
            raise
    
    def create_diagnostic_report(self, diagnostic_result: DiagnosticResult, patient_id: str) -> str:
        """
        Create FHIR DiagnosticReport from diagnostic result.
        
        Args:
            diagnostic_result: NeuroDx diagnostic result
            patient_id: FHIR patient identifier
            
        Returns:
            Created DiagnosticReport ID
        """
        # Convert diagnostic result to FHIR DiagnosticReport
        fhir_report = self._convert_to_fhir_diagnostic_report(diagnostic_result, patient_id)
        
        # Validate FHIR resource
        self._validate_fhir_resource(fhir_report, 'DiagnosticReport')
        
        url = f"{self.config.base_url}/DiagnosticReport"
        
        try:
            response = self.session.post(url, json=fhir_report)
            response.raise_for_status()
            
            created_report = response.json()
            report_id = created_report.get('id')
            
            self.logger.info(f"Created DiagnosticReport {report_id} for patient {patient_id}")
            return report_id
            
        except requests.RequestException as e:
            self.logger.error(f"Failed to create DiagnosticReport: {e}")
            raise
    
    def _convert_to_fhir_diagnostic_report(self, diagnostic_result: DiagnosticResult, patient_id: str) -> Dict[str, Any]:
        """Convert NeuroDx diagnostic result to FHIR DiagnosticReport."""
        # Generate unique identifier
        report_id = f"neurodx-{diagnostic_result.patient_id}-{int(diagnostic_result.timestamp.timestamp())}"
        
        # Build classification observations
        observations = []
        for condition, probability in diagnostic_result.classification_probabilities.items():
            observation = {
                "resourceType": "Observation",
                "id": f"{report_id}-{condition.lower().replace(' ', '-')}",
                "status": "final",
                "category": [{
                    "coding": [{
                        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                        "code": "imaging",
                        "display": "Imaging"
                    }]
                }],
                "code": {
                    "coding": [{
                        "system": "http://snomed.info/sct",
                        "code": "386053000",  # Evaluation procedure
                        "display": f"AI Classification: {condition}"
                    }]
                },
                "subject": {
                    "reference": f"Patient/{patient_id}"
                },
                "effectiveDateTime": diagnostic_result.timestamp.isoformat(),
                "valueQuantity": {
                    "value": probability,
                    "unit": "probability",
                    "system": "http://unitsofmeasure.org",
                    "code": "1"
                }
            }
            observations.append(observation)
        
        # Build main DiagnosticReport
        diagnostic_report = {
            "resourceType": "DiagnosticReport",
            "id": report_id,
            "status": "final",
            "category": [{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                    "code": "RAD",
                    "display": "Radiology"
                }]
            }],
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "code": "18748-4",
                    "display": "Diagnostic imaging study"
                }],
                "text": "NeuroDx Multi-Modal Neurodegenerative Disease Analysis"
            },
            "subject": {
                "reference": f"Patient/{patient_id}"
            },
            "effectiveDateTime": diagnostic_result.timestamp.isoformat(),
            "issued": datetime.now().isoformat(),
            "performer": [{
                "reference": "Organization/neurodx-system",
                "display": "NeuroDx-MultiModal AI System"
            }],
            "result": [{"reference": f"Observation/{obs['id']}"} for obs in observations],
            "conclusion": self._generate_diagnostic_conclusion(diagnostic_result),
            "conclusionCode": [{
                "coding": [{
                    "system": "http://snomed.info/sct",
                    "code": "386053000",
                    "display": "Evaluation procedure"
                }]
            }]
        }
        
        # Add confidence scores as extensions
        if diagnostic_result.confidence_scores:
            diagnostic_report["extension"] = []
            for metric, score in diagnostic_result.confidence_scores.items():
                diagnostic_report["extension"].append({
                    "url": f"http://neurodx.ai/fhir/StructureDefinition/confidence-{metric}",
                    "valueDecimal": score
                })
        
        return diagnostic_report
    
    def _generate_diagnostic_conclusion(self, diagnostic_result: DiagnosticResult) -> str:
        """Generate human-readable diagnostic conclusion."""
        conclusions = []
        
        # Find highest probability classification
        if diagnostic_result.classification_probabilities:
            max_condition = max(
                diagnostic_result.classification_probabilities.items(),
                key=lambda x: x[1]
            )
            condition, probability = max_condition
            
            if probability > 0.7:
                conclusions.append(f"High probability ({probability:.2%}) of {condition}")
            elif probability > 0.5:
                conclusions.append(f"Moderate probability ({probability:.2%}) of {condition}")
            else:
                conclusions.append(f"Low probability ({probability:.2%}) of {condition}")
        
        # Add confidence information
        if diagnostic_result.confidence_scores:
            avg_confidence = sum(diagnostic_result.confidence_scores.values()) / len(diagnostic_result.confidence_scores)
            conclusions.append(f"Overall model confidence: {avg_confidence:.2%}")
        
        return ". ".join(conclusions) if conclusions else "Analysis completed with no significant findings."
    
    def _validate_fhir_resource(self, resource: Dict[str, Any], resource_type: str) -> None:
        """Validate FHIR resource structure."""
        if not isinstance(resource, dict):
            raise FHIRValidationError("FHIR resource must be a dictionary")
        
        if resource.get('resourceType') != resource_type:
            raise FHIRValidationError(f"Expected resourceType '{resource_type}', got '{resource.get('resourceType')}'")
        
        # Basic required fields validation
        required_fields = {
            'DiagnosticReport': ['status', 'code', 'subject'],
            'Observation': ['status', 'code', 'subject']
        }
        
        if resource_type in required_fields:
            for field in required_fields[resource_type]:
                if field not in resource:
                    raise FHIRValidationError(f"Required field '{field}' missing from {resource_type}")
    
    def get_diagnostic_reports(self, patient_id: str, date_range: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """
        Retrieve diagnostic reports for a patient.
        
        Args:
            patient_id: FHIR patient identifier
            date_range: Optional tuple of (start_date, end_date) as ISO strings
            
        Returns:
            List of DiagnosticReport FHIR resources
        """
        url = f"{self.config.base_url}/DiagnosticReport"
        params = {'subject': f"Patient/{patient_id}"}
        
        if date_range:
            start_date, end_date = date_range
            params['date'] = f"ge{start_date}&date=le{end_date}"
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            bundle = response.json()
            reports = []
            
            if bundle.get('resourceType') == 'Bundle':
                for entry in bundle.get('entry', []):
                    if entry.get('resource', {}).get('resourceType') == 'DiagnosticReport':
                        reports.append(entry['resource'])
            
            self.logger.info(f"Retrieved {len(reports)} diagnostic reports for patient {patient_id}")
            return reports
            
        except requests.RequestException as e:
            self.logger.error(f"Failed to retrieve diagnostic reports: {e}")
            raise
    
    def test_connection(self) -> bool:
        """
        Test FHIR server connection and authentication.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Test with a simple metadata request
            url = f"{self.config.base_url}/metadata"
            response = self.session.get(url)
            response.raise_for_status()
            
            metadata = response.json()
            if metadata.get('resourceType') == 'CapabilityStatement':
                self.logger.info("FHIR server connection test successful")
                return True
            else:
                self.logger.error("Invalid response from FHIR server metadata endpoint")
                return False
                
        except Exception as e:
            self.logger.error(f"FHIR server connection test failed: {e}")
            return False