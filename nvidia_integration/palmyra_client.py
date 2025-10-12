"""
NVIDIA Palmyra-Med-70B API client for medical text analysis and report generation.
"""

import os
import logging
from typing import Dict, List, Optional, AsyncGenerator
from openai import OpenAI
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PalmyraConfig:
    """Configuration for Palmyra-Med-70B API."""
    api_key: str
    base_url: str = "https://integrate.api.nvidia.com/v1"
    model_name: str = "nvidia/palmyra-med-70b"
    max_tokens: int = 1024
    temperature: float = 0.1
    timeout: int = 30


class PalmyraClient:
    """Client for NVIDIA Palmyra-Med-70B medical language model."""
    
    def __init__(self, config: PalmyraConfig):
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout
        )
        logger.info("Initialized Palmyra-Med-70B client")
    
    def generate_diagnostic_report(
        self, 
        imaging_findings: Dict, 
        wearable_data: Dict,
        patient_context: Optional[Dict] = None
    ) -> str:
        """Generate comprehensive diagnostic report from multi-modal data."""
        
        prompt = self._build_diagnostic_prompt(imaging_findings, wearable_data, patient_context)
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a medical AI assistant specializing in neurodegenerative disease analysis. Provide accurate, evidence-based diagnostic insights."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            report = response.choices[0].message.content
            logger.info("Generated diagnostic report successfully")
            return report
            
        except Exception as e:
            logger.error(f"Error generating diagnostic report: {e}")
            raise
    
    async def stream_diagnostic_insights(
        self, 
        imaging_findings: Dict, 
        wearable_data: Dict
    ) -> AsyncGenerator[str, None]:
        """Stream real-time diagnostic insights."""
        
        prompt = self._build_streaming_prompt(imaging_findings, wearable_data)
        
        try:
            stream = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "Provide real-time diagnostic insights for neurodegenerative disease analysis."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Error streaming diagnostic insights: {e}")
            raise
    
    def explain_findings(self, findings: Dict, complexity_level: str = "clinical") -> str:
        """Generate natural language explanations for complex medical findings."""
        
        prompt = f"""
        Explain the following medical findings in {complexity_level} language:
        
        Findings: {findings}
        
        Provide clear, accurate explanations suitable for the specified audience.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a medical educator. Explain complex medical findings clearly and accurately."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=512,
                temperature=0.2
            )
            
            explanation = response.choices[0].message.content
            logger.info("Generated finding explanation successfully")
            return explanation
            
        except Exception as e:
            logger.error(f"Error explaining findings: {e}")
            raise
    
    def _build_diagnostic_prompt(
        self, 
        imaging_findings: Dict, 
        wearable_data: Dict, 
        patient_context: Optional[Dict]
    ) -> str:
        """Build comprehensive diagnostic prompt."""
        
        prompt = f"""
        Analyze the following multi-modal medical data for neurodegenerative disease assessment:
        
        IMAGING FINDINGS:
        - Segmentation Results: {imaging_findings.get('segmentation', {})}
        - Classification Probabilities: {imaging_findings.get('classification', {})}
        - Confidence Scores: {imaging_findings.get('confidence', {})}
        
        WEARABLE SENSOR DATA:
        - EEG Patterns: {wearable_data.get('eeg', {})}
        - Heart Rate Variability: {wearable_data.get('heart_rate', {})}
        - Sleep Metrics: {wearable_data.get('sleep', {})}
        - Gait Analysis: {wearable_data.get('gait', {})}
        """
        
        if patient_context:
            prompt += f"\nPATIENT CONTEXT:\n{patient_context}"
        
        prompt += """
        
        Please provide:
        1. Comprehensive diagnostic assessment
        2. Risk stratification
        3. Recommended follow-up actions
        4. Confidence levels for each finding
        """
        
        return prompt
    
    def _build_streaming_prompt(self, imaging_findings: Dict, wearable_data: Dict) -> str:
        """Build prompt for streaming insights."""
        
        return f"""
        Provide real-time analysis of:
        Imaging: {imaging_findings}
        Wearable: {wearable_data}
        
        Stream key insights as they emerge.
        """


def create_palmyra_client() -> PalmyraClient:
    """Factory function to create Palmyra client from environment variables."""
    
    api_key = os.getenv("NVIDIA_PALMYRA_API_KEY")
    if not api_key:
        raise ValueError("NVIDIA_PALMYRA_API_KEY environment variable not set")
    
    config = PalmyraConfig(
        api_key=api_key,
        base_url=os.getenv("NVIDIA_PALMYRA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        model_name=os.getenv("NVIDIA_PALMYRA_MODEL", "nvidia/palmyra-med-70b"),
        max_tokens=int(os.getenv("NVIDIA_PALMYRA_MAX_TOKENS", "1024")),
        temperature=float(os.getenv("NVIDIA_PALMYRA_TEMPERATURE", "0.1"))
    )
    
    return PalmyraClient(config)