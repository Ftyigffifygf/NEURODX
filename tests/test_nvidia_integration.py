"""
Tests for NVIDIA AI integration services.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from src.services.nvidia_integration.nvidia_enhanced_service import NVIDIAEnhancedService
from src.services.nvidia_integration.palmyra_enhanced_client import PalmyraMedEnhancedClient, MedicalInsight
from src.services.nvidia_integration.genomics_enhanced_client import GenomicsEnhancedClient, GenomicRiskProfile


class TestNVIDIAEnhancedService:
    """Test enhanced NVIDIA service integration."""
    
    @pytest.fixture
    def nvidia_service(self):
        """Create NVIDIA enhanced service for testing."""
        return NVIDIAEnhancedService()
    
    @pytest.fixture
    def sample_patient_data(self):
        """Create sample patient data."""
        return {
            "patient_id": "PAT_20241012_00001",
            "age": 65,
            "gender": "M",
            "medical_history": ["hypertension", "diabetes"],
            "lifestyle_factors": {
                "smoking": False,
                "exercise": "moderate",
                "diet": "mediterranean"
            }
        }
    
    @pytest.fixture
    def sample_imaging_results(self):
        """Create sample imaging results."""
        return {
            "modality": "MRI",
            "classification_probabilities": {
                "healthy": 0.2,
                "mild_cognitive_impairment": 0.3,
                "alzheimers_disease": 0.5
            },
            "segmentation_summary": {
                "hippocampus_volume": 3200,
                "ventricle_volume": 15000,
                "cortical_thickness": 2.8
            },
            "biomarkers": {
                "amyloid_beta": 0.75,
                "tau_protein": 0.68
            }
        }
    
    @pytest.fixture
    def sample_wearable_data(self):
        """Create sample wearable data."""
        return {
            "eeg_features": {
                "alpha_power": 0.45,
                "beta_power": 0.32,
                "theta_power": 0.23
            },
            "cognitive_metrics": {
                "reaction_time": 450,
                "accuracy": 0.85,
                "attention_score": 0.72
            },
            "sleep_data": {
                "sleep_efficiency": 0.78,
                "rem_percentage": 0.22,
                "deep_sleep_percentage": 0.18
            },
            "sleep_quality": 0.75,
            "cognitive_decline_risk": 0.4
        }
    
    def test_service_initialization(self, nvidia_service):
        """Test that NVIDIA enhanced service initializes correctly."""
        assert nvidia_service is not None
        assert nvidia_service.use_enhanced_clients == True
        assert nvidia_service.enable_streaming == True
        assert nvidia_service.max_retries == 3
    
    @pytest.mark.asyncio
    async def test_comprehensive_patient_analysis(self, nvidia_service, sample_patient_data, 
                                                 sample_imaging_results, sample_wearable_data):
        """Test comprehensive patient data analysis."""
        
        # Mock the enhanced Palmyra client methods
        with patch.object(nvidia_service.palmyra_enhanced, 'analyze_diagnostic_data') as mock_analyze, \
             patch.object(nvidia_service.palmyra_enhanced, 'generate_comprehensive_report') as mock_report, \
             patch.object(nvidia_service.palmyra_enhanced, 'assess_disease_risk') as mock_risk:
            
            # Set up mock returns
            mock_analyze.return_value = MedicalInsight(
                insight_type="diagnostic_analysis",
                content="Analysis shows moderate risk for cognitive decline",
                confidence=0.85,
                supporting_evidence=["Hippocampal atrophy", "Elevated tau levels"],
                clinical_relevance="High",
                timestamp=datetime.now()
            )
            
            mock_report.return_value = Mock(
                summary="Comprehensive neurological assessment completed",
                findings=["Hippocampal volume reduction", "Elevated biomarkers"],
                recommendations=["Follow-up in 6 months", "Cognitive assessment"]
            )
            
            mock_risk.return_value = {
                "alzheimers_disease": 0.65,
                "parkinsons_disease": 0.25,
                "mild_cognitive_impairment": 0.75
            }
            
            # Run analysis
            result = await nvidia_service.analyze_patient_data_comprehensive(
                sample_patient_data, sample_imaging_results, sample_wearable_data
            )
            
            # Verify results
            assert result is not None
            assert "medical_insights" in result
            assert "diagnostic_report" in result
            assert "risk_assessment" in result
            assert result["model_version"] == "palmyra-med-70b-enhanced"
            assert "confidence_scores" in result
            
            # Verify mocks were called
            mock_analyze.assert_called_once()
            mock_report.assert_called_once()
            mock_risk.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_genomics_analysis(self, nvidia_service):
        """Test comprehensive genomics analysis."""
        
        patient_id = "PAT_20241012_00001"
        fastq_files = ["/path/to/sample_R1.fastq.gz", "/path/to/sample_R2.fastq.gz"]
        sample_id = "SAMPLE_001"
        
        # Mock the genomics analysis
        with patch.object(nvidia_service.genomics_enhanced, 'run_comprehensive_analysis') as mock_genomics, \
             patch.object(nvidia_service.genomics_enhanced, 'generate_genomics_report') as mock_report:
            
            # Create mock genomics result
            mock_risk_profile = Mock()
            mock_risk_profile.polygenic_scores = {"alzheimers_prs": 0.72}
            mock_risk_profile.pharmacogenomic_insights = {"CYP2D6": "Normal metabolizer"}
            
            mock_genomics_result = Mock()
            mock_genomics_result.risk_profile = mock_risk_profile
            
            mock_genomics.return_value = mock_genomics_result
            mock_report.return_value = {"executive_summary": "Genomics analysis completed"}
            
            # Run analysis
            result = await nvidia_service.analyze_genomics_comprehensive(
                patient_id, fastq_files, sample_id
            )
            
            # Verify results
            assert result is not None
            assert "genomics_analysis" in result
            assert "genomics_report" in result
            assert "risk_profile" in result
            assert "polygenic_scores" in result
            assert "pharmacogenomic_insights" in result
            assert result["pipeline_version"] == "clara-parabricks-4.0-enhanced"
            
            # Verify mocks were called
            mock_genomics.assert_called_once_with(
                fastq_files=fastq_files,
                patient_id=patient_id,
                sample_id=sample_id
            )
    
    @pytest.mark.asyncio
    async def test_multi_modal_report_generation(self, nvidia_service, sample_patient_data,
                                                sample_imaging_results, sample_wearable_data):
        """Test multi-modal report generation."""
        
        # Mock genomics results
        genomics_results = {
            "risk_profile": {
                "risk_scores": {"alzheimers": 0.8},
                "polygenic_scores": {"alzheimers_prs": 0.75}
            }
        }
        
        with patch.object(nvidia_service.palmyra_enhanced, 'generate_comprehensive_report') as mock_report:
            
            mock_report.return_value = Mock(
                summary="Multi-modal analysis shows elevated risk",
                findings=["Imaging abnormalities", "Genetic risk factors"],
                recommendations=["Genetic counseling", "Enhanced monitoring"]
            )
            
            # Generate report
            result = await nvidia_service.generate_multi_modal_report(
                sample_patient_data, sample_imaging_results, 
                sample_wearable_data, genomics_results
            )
            
            # Verify results
            assert result is not None
            assert "comprehensive_report" in result
            assert "multi_modal_insights" in result
            assert "clinical_recommendations" in result
            assert result["report_version"] == "enhanced-v2.0"
            
            # Check multi-modal insights
            insights = result["multi_modal_insights"]
            assert "data_consistency" in insights
            assert "cross_modal_correlations" in insights
            assert "integrated_risk_score" in insights
            assert "confidence_assessment" in insights
    
    @pytest.mark.asyncio
    async def test_streaming_diagnostic_insights(self, nvidia_service, sample_patient_data):
        """Test streaming diagnostic insights."""
        
        diagnostic_data = {
            "imaging": {"classification": "mild_impairment"},
            "wearable": {"cognitive_score": 0.65}
        }
        
        # Mock streaming response
        async def mock_stream():
            yield "Analyzing patient data..."
            yield "Imaging shows mild abnormalities..."
            yield "Wearable data indicates cognitive changes..."
            yield "Overall assessment: moderate risk"
        
        with patch.object(nvidia_service.palmyra_enhanced, 'stream_diagnostic_insights', 
                         return_value=mock_stream()):
            
            # Collect streaming results
            insights = []
            async for insight in nvidia_service.stream_diagnostic_insights(
                sample_patient_data, diagnostic_data
            ):
                insights.append(insight)
            
            # Verify streaming worked
            assert len(insights) == 4
            assert "Analyzing patient data..." in insights[0]
            assert "Overall assessment: moderate risk" in insights[-1]
    
    def test_multi_modal_insights_generation(self, nvidia_service):
        """Test multi-modal insights generation."""
        
        combined_data = {
            "imaging": {
                "classification_probabilities": {"alzheimers_disease": 0.7}
            },
            "genomics": {
                "risk_profile": {"risk_scores": {"alzheimers": 0.8}}
            },
            "wearable": {
                "cognitive_decline_risk": 0.6,
                "sleep_quality": 0.4
            }
        }
        
        insights = nvidia_service._generate_multi_modal_insights(combined_data)
        
        assert "data_consistency" in insights
        assert "cross_modal_correlations" in insights
        assert "integrated_risk_score" in insights
        assert "confidence_assessment" in insights
        
        # Check integrated risk calculation
        integrated_risk = nvidia_service._calculate_integrated_risk(combined_data)
        assert 0.0 <= integrated_risk <= 1.0
        assert integrated_risk > 0.5  # Should be elevated given the input data
    
    def test_clinical_recommendations_generation(self, nvidia_service):
        """Test clinical recommendations generation."""
        
        combined_data = {
            "imaging": {
                "classification_probabilities": {"alzheimers_disease": 0.8}
            },
            "genomics": {
                "risk_profile": {"risk_scores": {"alzheimers": 0.9}}
            },
            "wearable": {
                "sleep_quality": 0.3
            }
        }
        
        recommendations = nvidia_service._generate_clinical_recommendations(combined_data)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Check for expected recommendations based on high risk data
        recommendation_text = " ".join(recommendations).lower()
        assert any(keyword in recommendation_text for keyword in 
                  ["amyloid", "genetic", "counseling", "sleep"])
    
    @pytest.mark.asyncio
    async def test_health_check(self, nvidia_service):
        """Test health check functionality."""
        
        # Mock health check responses
        with patch.object(nvidia_service.palmyra_enhanced, 'get_model_health') as mock_palmyra_health, \
             patch.object(nvidia_service.basic_service, 'get_service_health') as mock_basic_health:
            
            mock_palmyra_health.return_value = {"available_clients": 2}
            mock_basic_health.return_value = {"overall_status": "healthy"}
            
            health_status = await nvidia_service.health_check()
            
            assert health_status is not None
            assert "palmyra_enhanced" in health_status
            assert "genomics_enhanced" in health_status
            assert "basic_service" in health_status
            assert "timestamp" in health_status
            
            assert health_status["palmyra_enhanced"] == "healthy"
            assert health_status["genomics_enhanced"] == "available"
            assert health_status["basic_service"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_fallback_to_basic_service(self, nvidia_service, sample_patient_data,
                                           sample_imaging_results, sample_wearable_data):
        """Test fallback to basic service when enhanced clients fail."""
        
        # Mock enhanced client to raise exception
        with patch.object(nvidia_service.palmyra_enhanced, 'analyze_diagnostic_data', 
                         side_effect=Exception("Enhanced client failed")), \
             patch.object(nvidia_service, '_analyze_patient_data_basic') as mock_basic:
            
            mock_basic.return_value = {
                "analysis_type": "basic_fallback",
                "timestamp": datetime.now()
            }
            
            # Should fallback to basic analysis
            result = await nvidia_service.analyze_patient_data_comprehensive(
                sample_patient_data, sample_imaging_results, sample_wearable_data
            )
            
            assert result is not None
            assert result["analysis_type"] == "basic_fallback"
            mock_basic.assert_called_once()


class TestPalmyraMedEnhancedClient:
    """Test enhanced Palmyra client functionality."""
    
    def test_client_initialization(self):
        """Test Palmyra enhanced client initialization."""
        with patch('src.services.nvidia_integration.palmyra_enhanced_client.settings') as mock_settings:
            mock_settings.nvidia.palmyra_api_key = "test_key"
            mock_settings.nvidia.palmyra_api_key_2 = "test_key_2"
            mock_settings.nvidia.palmyra_api_key_3 = None
            mock_settings.nvidia.palmyra_base_url = "https://test.nvidia.com/v1"
            
            client = PalmyraMedEnhancedClient()
            
            assert client is not None
            assert len(client.clients) == 2  # Two valid API keys
            assert client.model_name == "nvidia/palmyra-med-70b"
            assert client.temperature == 0.1  # Low temperature for medical accuracy
    
    def test_diagnostic_prompt_creation(self):
        """Test diagnostic prompt creation."""
        with patch('src.services.nvidia_integration.palmyra_enhanced_client.settings') as mock_settings:
            mock_settings.nvidia.palmyra_api_key = "test_key"
            mock_settings.nvidia.palmyra_base_url = "https://test.nvidia.com/v1"
            
            client = PalmyraMedEnhancedClient()
            
            patient_data = {"age": 65, "gender": "M"}
            imaging_results = {"modality": "MRI", "classification_probabilities": {"alzheimers": 0.7}}
            wearable_data = {"eeg_features": {"alpha_power": 0.45}}
            
            prompt = client._create_diagnostic_prompt(patient_data, imaging_results, wearable_data)
            
            assert "PATIENT INFORMATION" in prompt
            assert "BRAIN IMAGING RESULTS" in prompt
            assert "WEARABLE SENSOR DATA" in prompt
            assert "Age: 65" in prompt
            assert "MRI" in prompt


class TestGenomicsEnhancedClient:
    """Test enhanced genomics client functionality."""
    
    def test_client_initialization(self):
        """Test genomics enhanced client initialization."""
        with patch('src.services.nvidia_integration.genomics_enhanced_client.settings') as mock_settings:
            mock_settings.nvidia.parabricks_path = "/opt/parabricks"
            mock_settings.nvidia.reference_genome_path = "/ref/genome.fa"
            mock_settings.nvidia.gpu_devices = "0,1,2,3"
            
            client = GenomicsEnhancedClient()
            
            assert client is not None
            assert client.parabricks_path == "/opt/parabricks"
            assert "alzheimers" in client.neurodegeneration_genes
            assert "parkinsons" in client.neurodegeneration_genes
            assert len(client.neurodegeneration_genes["alzheimers"]) > 0
    
    def test_neurodegeneration_gene_panels(self):
        """Test neurodegeneration gene panels are properly defined."""
        client = GenomicsEnhancedClient()
        
        # Check key genes are included
        assert "APOE" in client.neurodegeneration_genes["alzheimers"]
        assert "APP" in client.neurodegeneration_genes["alzheimers"]
        assert "PSEN1" in client.neurodegeneration_genes["alzheimers"]
        
        assert "SNCA" in client.neurodegeneration_genes["parkinsons"]
        assert "LRRK2" in client.neurodegeneration_genes["parkinsons"]
        
        assert "HTT" in client.neurodegeneration_genes["huntingtons"]
    
    def test_risk_score_calculation(self):
        """Test disease risk score calculation."""
        client = GenomicsEnhancedClient()
        
        # Mock variants
        from src.services.nvidia_integration.genomics_enhanced_client import GenomicVariant
        
        risk_variants = [
            GenomicVariant(
                chromosome="19", position=45411941, reference="T", alternate="C",
                gene="APOE", variant_type="SNV", clinical_significance="Pathogenic",
                allele_frequency=0.15, quality_score=99.0, genotype="1/1"
            )
        ]
        
        protective_variants = []
        
        risk_score = client._calculate_disease_risk_score(risk_variants, protective_variants)
        
        assert 0.0 <= risk_score <= 1.0
        assert risk_score > 0.5  # Should be elevated due to pathogenic variant


if __name__ == "__main__":
    pytest.main([__file__, "-v"])