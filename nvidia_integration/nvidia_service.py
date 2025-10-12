"""
Unified NVIDIA service interface for multi-API integration strategy.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Union, AsyncGenerator
from dataclasses import dataclass
from enum import Enum
import time
import random

from .palmyra_client import PalmyraClient, create_palmyra_client
from .genomics_client import GenomicsClient, create_genomics_client
from src.utils.logging_config import get_logger, get_performance_logger

logger = get_logger("nvidia_service")
perf_logger = get_performance_logger()


class ServiceType(Enum):
    """NVIDIA service types."""
    PALMYRA = "palmyra"
    GENOMICS = "genomics"


@dataclass
class APIKeyConfig:
    """Configuration for API key rotation and load balancing."""
    service_type: ServiceType
    api_keys: List[str]
    endpoints: List[str]
    current_key_index: int = 0
    failure_count: int = 0
    last_failure_time: Optional[float] = None
    max_failures: int = 3
    cooldown_period: int = 300  # 5 minutes


class NVIDIAService:
    """Unified NVIDIA service with API key rotation and failover capabilities."""
    
    def __init__(self):
        self.palmyra_configs: List[APIKeyConfig] = []
        self.genomics_configs: List[APIKeyConfig] = []
        self.clients: Dict[str, Union[PalmyraClient, GenomicsClient]] = {}
        self._initialize_configurations()
        logger.info("Initialized unified NVIDIA service")
    
    def _initialize_configurations(self) -> None:
        """Initialize API configurations from environment variables."""
        
        # Initialize Palmyra configurations
        palmyra_keys = self._get_api_keys("NVIDIA_PALMYRA_API_KEY")
        palmyra_endpoints = self._get_endpoints("NVIDIA_PALMYRA_BASE_URL")
        
        if palmyra_keys:
            self.palmyra_configs = [
                APIKeyConfig(
                    service_type=ServiceType.PALMYRA,
                    api_keys=palmyra_keys,
                    endpoints=palmyra_endpoints
                )
            ]
        
        # Initialize genomics configurations (if needed for API-based genomics)
        genomics_keys = self._get_api_keys("NVIDIA_GENOMICS_API_KEY")
        genomics_endpoints = self._get_endpoints("NVIDIA_GENOMICS_BASE_URL")
        
        if genomics_keys:
            self.genomics_configs = [
                APIKeyConfig(
                    service_type=ServiceType.GENOMICS,
                    api_keys=genomics_keys,
                    endpoints=genomics_endpoints
                )
            ]
    
    def _get_api_keys(self, env_prefix: str) -> List[str]:
        """Extract multiple API keys from environment variables."""
        import os
        
        keys = []
        
        # Try single key first
        single_key = os.getenv(env_prefix)
        if single_key and single_key != "your_nvidia_palmyra_api_key_here":
            keys.append(single_key)
        
        # Try multiple keys (KEY_1, KEY_2, etc.)
        index = 1
        while True:
            key = os.getenv(f"{env_prefix}_{index}")
            if key:
                keys.append(key)
                index += 1
            else:
                break
        
        return keys
    
    def _get_endpoints(self, env_prefix: str) -> List[str]:
        """Extract multiple endpoints from environment variables."""
        import os
        
        endpoints = []
        
        # Try single endpoint first
        single_endpoint = os.getenv(env_prefix)
        if single_endpoint:
            endpoints.append(single_endpoint)
        
        # Try multiple endpoints
        index = 1
        while True:
            endpoint = os.getenv(f"{env_prefix}_{index}")
            if endpoint:
                endpoints.append(endpoint)
                index += 1
            else:
                break
        
        return endpoints or ["https://integrate.api.nvidia.com/v1"]
    
    def get_palmyra_client(self) -> Optional[PalmyraClient]:
        """Get available Palmyra client with failover support."""
        
        for config in self.palmyra_configs:
            if self._is_config_available(config):
                try:
                    client_key = f"palmyra_{config.current_key_index}"
                    
                    if client_key not in self.clients:
                        # Create new client with current configuration
                        from .palmyra_client import PalmyraConfig
                        
                        palmyra_config = PalmyraConfig(
                            api_key=config.api_keys[config.current_key_index],
                            base_url=config.endpoints[config.current_key_index % len(config.endpoints)]
                        )
                        
                        self.clients[client_key] = PalmyraClient(palmyra_config)
                    
                    return self.clients[client_key]
                    
                except Exception as e:
                    logger.error(f"Failed to create Palmyra client: {e}")
                    self._handle_client_failure(config)
        
        logger.error("No available Palmyra clients")
        return None
    
    def get_genomics_client(self) -> Optional[GenomicsClient]:
        """Get available genomics client."""
        
        try:
            if "genomics" not in self.clients:
                self.clients["genomics"] = create_genomics_client()
            
            return self.clients["genomics"]
            
        except Exception as e:
            logger.error(f"Failed to create genomics client: {e}")
            return None
    
    async def generate_diagnostic_report_with_failover(
        self, 
        imaging_findings: Dict, 
        wearable_data: Dict,
        patient_context: Optional[Dict] = None,
        max_retries: int = 3
    ) -> Optional[str]:
        """Generate diagnostic report with automatic failover."""
        
        start_time = time.time()
        
        for attempt in range(max_retries):
            client = self.get_palmyra_client()
            if not client:
                logger.error("No available Palmyra clients for diagnostic report generation")
                return None
            
            try:
                report = client.generate_diagnostic_report(
                    imaging_findings, wearable_data, patient_context
                )
                
                # Log successful request
                processing_time = time.time() - start_time
                perf_logger.log_inference_time(
                    "palmyra_diagnostic_report", processing_time, 
                    (len(str(imaging_findings)), len(str(wearable_data)))
                )
                
                logger.info(f"Generated diagnostic report successfully (attempt {attempt + 1})")
                return report
                
            except Exception as e:
                logger.warning(f"Diagnostic report generation failed (attempt {attempt + 1}): {e}")
                
                # Handle failure and rotate to next client
                for config in self.palmyra_configs:
                    if config.api_keys[config.current_key_index] == client.config.api_key:
                        self._handle_client_failure(config)
                        break
                
                if attempt < max_retries - 1:
                    # Wait before retry with exponential backoff
                    await asyncio.sleep(2 ** attempt + random.uniform(0, 1))
        
        logger.error("All attempts to generate diagnostic report failed")
        return None
    
    async def stream_diagnostic_insights_with_failover(
        self, 
        imaging_findings: Dict, 
        wearable_data: Dict
    ) -> AsyncGenerator[str, None]:
        """Stream diagnostic insights with failover support."""
        
        client = self.get_palmyra_client()
        if not client:
            logger.error("No available Palmyra clients for streaming")
            return
        
        try:
            async for chunk in client.stream_diagnostic_insights(imaging_findings, wearable_data):
                yield chunk
                
        except Exception as e:
            logger.error(f"Streaming diagnostic insights failed: {e}")
            
            # Handle failure
            for config in self.palmyra_configs:
                if config.api_keys[config.current_key_index] == client.config.api_key:
                    self._handle_client_failure(config)
                    break
    
    def analyze_genomic_variants_with_monitoring(
        self, 
        fastq_files: List[str], 
        patient_id: str,
        analysis_params: Optional[Dict] = None
    ) -> Optional[Dict]:
        """Analyze genomic variants with performance monitoring."""
        
        start_time = time.time()
        
        client = self.get_genomics_client()
        if not client:
            logger.error("Genomics client not available")
            return None
        
        try:
            results = client.analyze_genomic_variants(fastq_files, patient_id, analysis_params)
            
            # Log performance metrics
            processing_time = time.time() - start_time
            perf_logger.log_inference_time(
                "genomics_variant_analysis", processing_time, 
                (len(fastq_files), sum(len(f) for f in fastq_files))
            )
            
            logger.info(f"Completed genomic analysis for patient {patient_id}")
            return results
            
        except Exception as e:
            logger.error(f"Genomic analysis failed for patient {patient_id}: {e}")
            return None
    
    def integrate_multimodal_analysis(
        self, 
        imaging_results: Dict, 
        wearable_data: Dict,
        genomics_results: Optional[Dict] = None
    ) -> Dict:
        """Integrate results from multiple NVIDIA services."""
        
        start_time = time.time()
        
        try:
            integration_results = {
                "timestamp": time.time(),
                "imaging_analysis": imaging_results,
                "wearable_analysis": wearable_data,
                "genomics_analysis": genomics_results,
                "integrated_insights": {}
            }
            
            # Enhanced multi-modal risk assessment
            risk_factors = []
            confidence_scores = []
            
            # Extract imaging risk factors
            if imaging_results.get("classification"):
                for condition, probability in imaging_results["classification"].items():
                    if probability > 0.6:
                        risk_factors.append(f"imaging_{condition}")
                        confidence_scores.append(probability)
            
            # Extract wearable risk factors
            if wearable_data.get("anomalies"):
                for anomaly, severity in wearable_data["anomalies"].items():
                    if severity > 0.5:
                        risk_factors.append(f"wearable_{anomaly}")
                        confidence_scores.append(severity)
            
            # Extract genomic risk factors
            if genomics_results and genomics_results.get("risk_factors"):
                for gene, risk_level in genomics_results["risk_factors"].items():
                    if risk_level in ["HIGH", "MODERATE"]:
                        risk_factors.append(f"genomic_{gene}")
                        confidence_scores.append(0.8 if risk_level == "HIGH" else 0.6)
            
            # Calculate overall risk assessment
            if confidence_scores:
                overall_confidence = sum(confidence_scores) / len(confidence_scores)
                risk_level = "HIGH" if overall_confidence > 0.7 else "MODERATE" if overall_confidence > 0.4 else "LOW"
            else:
                overall_confidence = 0.0
                risk_level = "LOW"
            
            integration_results["integrated_insights"] = {
                "overall_risk_level": risk_level,
                "confidence_score": overall_confidence,
                "contributing_factors": risk_factors,
                "recommendation": self._generate_recommendation(risk_level, risk_factors)
            }
            
            # Log integration performance
            processing_time = time.time() - start_time
            perf_logger.log_preprocessing_time(
                "multimodal_integration", processing_time, 
                len(str(integration_results))
            )
            
            logger.info(f"Completed multimodal integration: {risk_level} risk")
            return integration_results
            
        except Exception as e:
            logger.error(f"Multimodal integration failed: {e}")
            raise
    
    def _is_config_available(self, config: APIKeyConfig) -> bool:
        """Check if API configuration is available for use."""
        
        current_time = time.time()
        
        # Check if in cooldown period
        if (config.last_failure_time and 
            current_time - config.last_failure_time < config.cooldown_period):
            return False
        
        # Check failure count
        if config.failure_count >= config.max_failures:
            return False
        
        return True
    
    def _handle_client_failure(self, config: APIKeyConfig) -> None:
        """Handle client failure and rotate to next available key."""
        
        config.failure_count += 1
        config.last_failure_time = time.time()
        
        # Rotate to next API key
        if len(config.api_keys) > 1:
            config.current_key_index = (config.current_key_index + 1) % len(config.api_keys)
            logger.info(f"Rotated to next API key for {config.service_type.value}")
        
        # Remove failed client from cache
        client_keys_to_remove = [
            key for key in self.clients.keys() 
            if key.startswith(config.service_type.value)
        ]
        for key in client_keys_to_remove:
            self.clients.pop(key, None)
    
    def _generate_recommendation(self, risk_level: str, risk_factors: List[str]) -> str:
        """Generate clinical recommendation based on integrated analysis."""
        
        if risk_level == "HIGH":
            return "Immediate clinical evaluation recommended. Multiple high-risk factors identified."
        elif risk_level == "MODERATE":
            return "Regular monitoring and follow-up recommended. Some risk factors present."
        else:
            return "Continue standard screening schedule. Low risk factors identified."
    
    def get_service_health(self) -> Dict:
        """Get health status of all NVIDIA services."""
        
        health_status = {
            "palmyra_service": {
                "available_clients": len([c for c in self.palmyra_configs if self._is_config_available(c)]),
                "total_clients": len(self.palmyra_configs),
                "status": "healthy" if any(self._is_config_available(c) for c in self.palmyra_configs) else "degraded"
            },
            "genomics_service": {
                "status": "healthy" if self.get_genomics_client() else "unavailable"
            },
            "overall_status": "healthy"
        }
        
        # Determine overall status
        if (health_status["palmyra_service"]["status"] == "degraded" or 
            health_status["genomics_service"]["status"] == "unavailable"):
            health_status["overall_status"] = "degraded"
        
        return health_status


# Global service instance
nvidia_service = NVIDIAService()


def get_nvidia_service() -> NVIDIAService:
    """Get the global NVIDIA service instance."""
    return nvidia_service