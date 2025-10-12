"""
Enhanced NVIDIA Genomics client for neurodegenerative disease analysis.
Integrates with NVIDIA Clara Parabricks for genomic variant analysis.
"""

import asyncio
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
import pandas as pd

from src.config.settings import settings
from src.services.security.audit_logger import AuditLogger

logger = logging.getLogger(__name__)


@dataclass
class GenomicVariant:
    """Genomic variant information."""
    chromosome: str
    position: int
    reference: str
    alternate: str
    gene: str
    variant_type: str
    clinical_significance: str
    allele_frequency: float
    quality_score: float
    genotype: str


@dataclass
class GenomicRiskProfile:
    """Genomic risk profile for neurodegenerative diseases."""
    patient_id: str
    risk_variants: List[GenomicVariant]
    protective_variants: List[GenomicVariant]
    risk_scores: Dict[str, float]
    polygenic_scores: Dict[str, float]
    pharmacogenomic_insights: Dict[str, str]
    ancestry_analysis: Dict[str, float]
    generated_at: datetime


@dataclass
class GenomicAnalysisResult:
    """Complete genomic analysis result."""
    patient_id: str
    analysis_id: str
    variant_calling_results: Dict[str, Any]
    annotation_results: Dict[str, Any]
    risk_profile: GenomicRiskProfile
    quality_metrics: Dict[str, float]
    processing_time: float
    pipeline_version: str


class GenomicsEnhancedClient:
    """
    Enhanced NVIDIA Genomics client for neurodegenerative disease analysis.
    
    Features:
    - Variant calling with Clara Parabricks
    - Annotation with neurological disease databases
    - Polygenic risk score calculation
    - Pharmacogenomic analysis
    - Ancestry inference
    - Multi-sample family analysis
    """
    
    def __init__(self, audit_logger: Optional[AuditLogger] = None):
        """Initialize enhanced genomics client."""
        self.audit_logger = audit_logger or AuditLogger()
        
        # Clara Parabricks configuration
        self.parabricks_path = getattr(settings.nvidia, 'parabricks_path', "/opt/parabricks")
        self.reference_genome = getattr(settings.nvidia, 'reference_genome_path', "/ref/genome.fa")
        self.gpu_devices = getattr(settings.nvidia, 'gpu_devices', "0,1,2,3")
        
        # Neurodegenerative disease gene panels
        self.neurodegeneration_genes = {
            "alzheimers": [
                "APP", "PSEN1", "PSEN2", "APOE", "TREM2", "SORL1", 
                "ABCA7", "CLU", "CR1", "PICALM", "BIN1", "CD2AP"
            ],
            "parkinsons": [
                "SNCA", "LRRK2", "PARK2", "PINK1", "PARK7", "VPS35",
                "EIF4G1", "DNAJC13", "CHCHD2", "GBA", "SMPD1"
            ],
            "frontotemporal": [
                "MAPT", "GRN", "C9orf72", "VCP", "CHMP2B", "TARDBP",
                "FUS", "CHCHD10", "TBK1", "SQSTM1", "OPTN"
            ],
            "huntingtons": ["HTT"],
            "als": [
                "SOD1", "TARDBP", "FUS", "C9orf72", "OPTN", "VCP",
                "UBQLN2", "PFN1", "ERBB4", "CHCHD10", "TBK1"
            ]
        }
        
        # Reference databases
        self.databases = {
            "clinvar": getattr(settings.nvidia, 'clinvar_db_path', "/db/clinvar.vcf"),
            "gnomad": getattr(settings.nvidia, 'gnomad_db_path', "/db/gnomad.vcf"),
            "neurodegeneration_db": getattr(settings.nvidia, 'neurodegeneration_db_path', "/db/neurodegeneration.vcf")
        }
        
        # Polygenic risk score models
        self.prs_models = {
            "alzheimers_prs": "models/alzheimers_prs.model",
            "parkinsons_prs": "models/parkinsons_prs.model",
            "cognitive_decline_prs": "models/cognitive_decline_prs.model"
        }
    
    async def run_comprehensive_analysis(self,
                                       fastq_files: List[str],
                                       patient_id: str,
                                       sample_id: str) -> GenomicAnalysisResult:
        """
        Run comprehensive genomic analysis pipeline.
        
        Args:
            fastq_files: List of FASTQ file paths
            patient_id: Patient identifier
            sample_id: Sample identifier
            
        Returns:
            Complete genomic analysis result
        """
        start_time = datetime.now()
        analysis_id = f"GEN_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{sample_id}"
        
        try:
            logger.info(f"Starting genomic analysis for {patient_id}")
            
            # Step 1: Quality control and preprocessing
            qc_results = await self._run_quality_control(fastq_files)
            
            # Step 2: Alignment with BWA-MEM
            aligned_bam = await self._run_alignment(fastq_files, sample_id)
            
            # Step 3: Variant calling with HaplotypeCaller
            variant_vcf = await self._run_variant_calling(aligned_bam, sample_id)
            
            # Step 4: Variant annotation
            annotated_variants = await self._annotate_variants(variant_vcf, sample_id)
            
            # Step 5: Neurodegeneration-specific analysis
            risk_profile = await self._analyze_neurodegeneration_risk(
                annotated_variants, patient_id
            )
            
            # Step 6: Polygenic risk scores
            prs_scores = await self._calculate_polygenic_scores(
                annotated_variants, patient_id
            )
            risk_profile.polygenic_scores = prs_scores
            
            # Step 7: Pharmacogenomic analysis
            pgx_insights = await self._analyze_pharmacogenomics(
                annotated_variants, patient_id
            )
            risk_profile.pharmacogenomic_insights = pgx_insights
            
            # Step 8: Ancestry analysis
            ancestry = await self._analyze_ancestry(annotated_variants)
            risk_profile.ancestry_analysis = ancestry
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = GenomicAnalysisResult(
                patient_id=patient_id,
                analysis_id=analysis_id,
                variant_calling_results={"vcf_path": variant_vcf, "total_variants": len(annotated_variants)},
                annotation_results={"annotated_variants": len(annotated_variants)},
                risk_profile=risk_profile,
                quality_metrics=qc_results,
                processing_time=processing_time,
                pipeline_version="parabricks-4.0"
            )
            
            # Audit log
            self.audit_logger.log_genomics_analysis(
                patient_id=patient_id,
                analysis_id=analysis_id,
                processing_time=processing_time,
                variant_count=len(annotated_variants),
                pipeline_version="parabricks-4.0"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Genomic analysis failed for {patient_id}: {e}")
            self.audit_logger.log_genomics_error(
                patient_id=patient_id,
                analysis_id=analysis_id,
                error=str(e)
            )
            raise
    
    async def _run_quality_control(self, fastq_files: List[str]) -> Dict[str, float]:
        """Run quality control on FASTQ files."""
        try:
            # Use Parabricks fq2bam for QC
            cmd = [
                f"{self.parabricks_path}/pbrun", "fq2bam",
                "--ref", self.reference_genome,
                "--in-fq", " ".join(fastq_files),
                "--out-bam", "/tmp/qc_output.bam",
                "--tmp-dir", "/tmp",
                "--num-gpus", str(len(self.gpu_devices.split(","))),
                "--qc-only"
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise RuntimeError(f"QC failed: {stderr.decode()}")
            
            # Parse QC metrics
            qc_metrics = self._parse_qc_output(stdout.decode())
            
            return qc_metrics
            
        except Exception as e:
            logger.error(f"Quality control failed: {e}")
            raise
    
    async def _run_alignment(self, fastq_files: List[str], sample_id: str) -> str:
        """Run alignment with BWA-MEM using Parabricks."""
        try:
            output_bam = f"/tmp/{sample_id}_aligned.bam"
            
            cmd = [
                f"{self.parabricks_path}/pbrun", "fq2bam",
                "--ref", self.reference_genome,
                "--in-fq", " ".join(fastq_files),
                "--out-bam", output_bam,
                "--tmp-dir", "/tmp",
                "--num-gpus", str(len(self.gpu_devices.split(","))),
                "--read-group", f"@RG\\tID:{sample_id}\\tSM:{sample_id}\\tPL:ILLUMINA"
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise RuntimeError(f"Alignment failed: {stderr.decode()}")
            
            logger.info(f"Alignment completed: {output_bam}")
            return output_bam
            
        except Exception as e:
            logger.error(f"Alignment failed: {e}")
            raise
    
    async def _run_variant_calling(self, bam_file: str, sample_id: str) -> str:
        """Run variant calling with HaplotypeCaller."""
        try:
            output_vcf = f"/tmp/{sample_id}_variants.vcf"
            
            cmd = [
                f"{self.parabricks_path}/pbrun", "haplotypecaller",
                "--ref", self.reference_genome,
                "--in-bam", bam_file,
                "--out-variants", output_vcf,
                "--tmp-dir", "/tmp",
                "--num-gpus", str(len(self.gpu_devices.split(","))),
                "--gvcf"  # Generate GVCF for better variant calling
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise RuntimeError(f"Variant calling failed: {stderr.decode()}")
            
            logger.info(f"Variant calling completed: {output_vcf}")
            return output_vcf
            
        except Exception as e:
            logger.error(f"Variant calling failed: {e}")
            raise
    
    async def _annotate_variants(self, vcf_file: str, sample_id: str) -> List[GenomicVariant]:
        """Annotate variants with functional and clinical information."""
        try:
            # Use VEP or similar annotation tool
            annotated_vcf = f"/tmp/{sample_id}_annotated.vcf"
            
            # Run annotation (simplified - would use actual annotation tools)
            variants = self._parse_vcf_file(vcf_file)
            
            # Annotate with neurodegeneration databases
            annotated_variants = []
            for variant in variants:
                annotated_variant = await self._annotate_single_variant(variant)
                annotated_variants.append(annotated_variant)
            
            return annotated_variants
            
        except Exception as e:
            logger.error(f"Variant annotation failed: {e}")
            raise
    
    async def _annotate_single_variant(self, variant: Dict[str, Any]) -> GenomicVariant:
        """Annotate a single variant with clinical significance."""
        # Mock annotation - in real implementation, would query databases
        return GenomicVariant(
            chromosome=variant["chromosome"],
            position=variant["position"],
            reference=variant["reference"],
            alternate=variant["alternate"],
            gene=variant.get("gene", "Unknown"),
            variant_type=variant.get("type", "SNV"),
            clinical_significance=self._determine_clinical_significance(variant),
            allele_frequency=variant.get("af", 0.0),
            quality_score=variant.get("qual", 0.0),
            genotype=variant.get("genotype", "0/1")
        )
    
    async def _analyze_neurodegeneration_risk(self,
                                           variants: List[GenomicVariant],
                                           patient_id: str) -> GenomicRiskProfile:
        """Analyze risk for neurodegenerative diseases."""
        try:
            risk_variants = []
            protective_variants = []
            risk_scores = {}
            
            # Analyze variants in neurodegeneration genes
            for disease, genes in self.neurodegeneration_genes.items():
                disease_risk_variants = []
                disease_protective_variants = []
                
                for variant in variants:
                    if variant.gene in genes:
                        if variant.clinical_significance in ["Pathogenic", "Likely pathogenic"]:
                            risk_variants.append(variant)
                            disease_risk_variants.append(variant)
                        elif variant.clinical_significance in ["Protective", "Likely protective"]:
                            protective_variants.append(variant)
                            disease_protective_variants.append(variant)
                
                # Calculate disease-specific risk score
                risk_scores[disease] = self._calculate_disease_risk_score(
                    disease_risk_variants, disease_protective_variants
                )
            
            return GenomicRiskProfile(
                patient_id=patient_id,
                risk_variants=risk_variants,
                protective_variants=protective_variants,
                risk_scores=risk_scores,
                polygenic_scores={},  # Will be filled later
                pharmacogenomic_insights={},  # Will be filled later
                ancestry_analysis={},  # Will be filled later
                generated_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Risk analysis failed: {e}")
            raise
    
    async def _calculate_polygenic_scores(self,
                                        variants: List[GenomicVariant],
                                        patient_id: str) -> Dict[str, float]:
        """Calculate polygenic risk scores."""
        try:
            prs_scores = {}
            
            for disease, model_path in self.prs_models.items():
                # Load PRS model weights
                prs_weights = self._load_prs_model(model_path)
                
                # Calculate weighted sum of risk alleles
                score = 0.0
                total_variants = 0
                
                for variant in variants:
                    variant_key = f"{variant.chromosome}:{variant.position}:{variant.reference}:{variant.alternate}"
                    if variant_key in prs_weights:
                        # Get dosage (0, 1, or 2 copies of risk allele)
                        dosage = self._get_allele_dosage(variant.genotype)
                        score += prs_weights[variant_key] * dosage
                        total_variants += 1
                
                # Normalize score
                if total_variants > 0:
                    prs_scores[disease] = score / total_variants
                else:
                    prs_scores[disease] = 0.0
            
            return prs_scores
            
        except Exception as e:
            logger.error(f"PRS calculation failed: {e}")
            return {}
    
    async def _analyze_pharmacogenomics(self,
                                      variants: List[GenomicVariant],
                                      patient_id: str) -> Dict[str, str]:
        """Analyze pharmacogenomic variants."""
        try:
            pgx_insights = {}
            
            # Key pharmacogenomic genes for neurological drugs
            pgx_genes = {
                "CYP2D6": "Metabolizes many psychiatric medications",
                "CYP2C19": "Metabolizes clopidogrel, antidepressants",
                "COMT": "Affects dopamine metabolism, relevant for Parkinson's",
                "APOE": "Affects drug response in Alzheimer's disease",
                "CACNA1C": "Affects response to calcium channel blockers"
            }
            
            for variant in variants:
                if variant.gene in pgx_genes:
                    # Determine drug response implications
                    if variant.clinical_significance in ["Pathogenic", "Likely pathogenic"]:
                        pgx_insights[variant.gene] = f"Altered drug metabolism: {pgx_genes[variant.gene]}"
                    elif variant.allele_frequency < 0.05:  # Rare variant
                        pgx_insights[variant.gene] = f"Rare variant may affect: {pgx_genes[variant.gene]}"
            
            return pgx_insights
            
        except Exception as e:
            logger.error(f"Pharmacogenomic analysis failed: {e}")
            return {}
    
    async def _analyze_ancestry(self, variants: List[GenomicVariant]) -> Dict[str, float]:
        """Analyze genetic ancestry."""
        try:
            # Simplified ancestry analysis
            # In real implementation, would use ancestry informative markers
            ancestry_scores = {
                "European": 0.0,
                "African": 0.0,
                "Asian": 0.0,
                "Native American": 0.0,
                "Oceanian": 0.0
            }
            
            # Mock calculation based on allele frequencies
            for variant in variants:
                # Use allele frequency to infer ancestry (simplified)
                if variant.allele_frequency > 0.1:
                    ancestry_scores["European"] += 0.1
                elif variant.allele_frequency > 0.05:
                    ancestry_scores["Asian"] += 0.1
                else:
                    ancestry_scores["African"] += 0.1
            
            # Normalize scores
            total = sum(ancestry_scores.values())
            if total > 0:
                for ancestry in ancestry_scores:
                    ancestry_scores[ancestry] /= total
            
            return ancestry_scores
            
        except Exception as e:
            logger.error(f"Ancestry analysis failed: {e}")
            return {}
    
    def _parse_qc_output(self, qc_output: str) -> Dict[str, float]:
        """Parse quality control output."""
        # Mock QC metrics
        return {
            "total_reads": 50000000,
            "mapped_reads_percentage": 95.5,
            "duplicate_percentage": 12.3,
            "mean_coverage": 30.2,
            "gc_content": 41.5
        }
    
    def _parse_vcf_file(self, vcf_file: str) -> List[Dict[str, Any]]:
        """Parse VCF file and extract variants."""
        variants = []
        
        try:
            with open(vcf_file, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    
                    fields = line.strip().split('\t')
                    if len(fields) >= 8:
                        variant = {
                            "chromosome": fields[0],
                            "position": int(fields[1]),
                            "reference": fields[3],
                            "alternate": fields[4],
                            "qual": float(fields[5]) if fields[5] != '.' else 0.0,
                            "info": fields[7],
                            "genotype": fields[9].split(':')[0] if len(fields) > 9 else "0/1"
                        }
                        variants.append(variant)
            
            return variants
            
        except Exception as e:
            logger.error(f"VCF parsing failed: {e}")
            return []
    
    def _determine_clinical_significance(self, variant: Dict[str, Any]) -> str:
        """Determine clinical significance of variant."""
        # Mock clinical significance determination
        # In real implementation, would query ClinVar and other databases
        
        gene = variant.get("gene", "")
        if gene in ["APP", "PSEN1", "PSEN2"]:
            return "Pathogenic"
        elif gene == "APOE":
            return "Risk factor"
        else:
            return "Uncertain significance"
    
    def _calculate_disease_risk_score(self,
                                    risk_variants: List[GenomicVariant],
                                    protective_variants: List[GenomicVariant]) -> float:
        """Calculate disease risk score based on variants."""
        risk_score = 0.0
        
        # Add risk from pathogenic variants
        for variant in risk_variants:
            if variant.clinical_significance == "Pathogenic":
                risk_score += 0.8
            elif variant.clinical_significance == "Likely pathogenic":
                risk_score += 0.4
        
        # Subtract protection from protective variants
        for variant in protective_variants:
            if variant.clinical_significance == "Protective":
                risk_score -= 0.3
            elif variant.clinical_significance == "Likely protective":
                risk_score -= 0.1
        
        # Normalize to 0-1 range
        return max(0.0, min(1.0, risk_score))
    
    def _load_prs_model(self, model_path: str) -> Dict[str, float]:
        """Load polygenic risk score model weights."""
        # Mock PRS weights
        return {
            "19:45411941:T:C": 0.15,  # APOE e4
            "11:47557871:C:T": 0.08,  # SORL1
            "2:127892810:G:A": 0.06,  # BIN1
            # ... more variants
        }
    
    def _get_allele_dosage(self, genotype: str) -> int:
        """Get allele dosage from genotype."""
        if genotype in ["0/0", "0|0"]:
            return 0
        elif genotype in ["0/1", "1/0", "0|1", "1|0"]:
            return 1
        elif genotype in ["1/1", "1|1"]:
            return 2
        else:
            return 0
    
    async def run_family_analysis(self,
                                family_samples: Dict[str, str],
                                proband_id: str) -> Dict[str, Any]:
        """Run family-based genomic analysis."""
        try:
            family_results = {}
            
            # Analyze each family member
            for member_id, fastq_path in family_samples.items():
                result = await self.run_comprehensive_analysis(
                    [fastq_path], member_id, member_id
                )
                family_results[member_id] = result
            
            # Perform inheritance analysis
            inheritance_analysis = self._analyze_inheritance_patterns(
                family_results, proband_id
            )
            
            return {
                "family_results": family_results,
                "inheritance_analysis": inheritance_analysis,
                "segregation_analysis": self._analyze_variant_segregation(family_results),
                "de_novo_variants": self._identify_de_novo_variants(family_results, proband_id)
            }
            
        except Exception as e:
            logger.error(f"Family analysis failed: {e}")
            raise
    
    def _analyze_inheritance_patterns(self,
                                    family_results: Dict[str, GenomicAnalysisResult],
                                    proband_id: str) -> Dict[str, Any]:
        """Analyze inheritance patterns in family."""
        # Mock inheritance analysis
        return {
            "autosomal_dominant": [],
            "autosomal_recessive": [],
            "x_linked": [],
            "compound_heterozygous": []
        }
    
    def _analyze_variant_segregation(self,
                                   family_results: Dict[str, GenomicAnalysisResult]) -> Dict[str, Any]:
        """Analyze how variants segregate in the family."""
        return {
            "segregating_variants": [],
            "non_segregating_variants": [],
            "incomplete_penetrance": []
        }
    
    def _identify_de_novo_variants(self,
                                 family_results: Dict[str, GenomicAnalysisResult],
                                 proband_id: str) -> List[GenomicVariant]:
        """Identify de novo variants in proband."""
        # Mock de novo variant identification
        return []
    
    async def generate_genomics_report(self,
                                     analysis_result: GenomicAnalysisResult) -> Dict[str, Any]:
        """Generate comprehensive genomics report."""
        try:
            report = {
                "patient_id": analysis_result.patient_id,
                "analysis_id": analysis_result.analysis_id,
                "executive_summary": self._generate_executive_summary(analysis_result),
                "variant_summary": self._generate_variant_summary(analysis_result),
                "risk_assessment": self._generate_risk_assessment(analysis_result),
                "pharmacogenomic_recommendations": self._generate_pgx_recommendations(analysis_result),
                "follow_up_recommendations": self._generate_followup_recommendations(analysis_result),
                "technical_details": {
                    "pipeline_version": analysis_result.pipeline_version,
                    "processing_time": analysis_result.processing_time,
                    "quality_metrics": analysis_result.quality_metrics
                },
                "generated_at": datetime.now()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            raise
    
    def _generate_executive_summary(self, result: GenomicAnalysisResult) -> str:
        """Generate executive summary of genomic analysis."""
        risk_profile = result.risk_profile
        high_risk_diseases = [
            disease for disease, score in risk_profile.risk_scores.items()
            if score > 0.7
        ]
        
        if high_risk_diseases:
            return f"High genetic risk identified for: {', '.join(high_risk_diseases)}. " \
                   f"Found {len(risk_profile.risk_variants)} pathogenic variants. " \
                   f"Recommend genetic counseling and enhanced monitoring."
        else:
            return "No high-risk genetic variants identified. " \
                   f"Standard monitoring recommended based on {len(risk_profile.risk_variants)} variants of uncertain significance."
    
    def _generate_variant_summary(self, result: GenomicAnalysisResult) -> Dict[str, Any]:
        """Generate variant summary."""
        risk_profile = result.risk_profile
        
        return {
            "total_variants": len(risk_profile.risk_variants) + len(risk_profile.protective_variants),
            "pathogenic_variants": len([v for v in risk_profile.risk_variants if v.clinical_significance == "Pathogenic"]),
            "likely_pathogenic_variants": len([v for v in risk_profile.risk_variants if v.clinical_significance == "Likely pathogenic"]),
            "protective_variants": len(risk_profile.protective_variants),
            "variants_by_gene": self._group_variants_by_gene(risk_profile.risk_variants)
        }
    
    def _generate_risk_assessment(self, result: GenomicAnalysisResult) -> Dict[str, Any]:
        """Generate risk assessment summary."""
        return {
            "genetic_risk_scores": result.risk_profile.risk_scores,
            "polygenic_risk_scores": result.risk_profile.polygenic_scores,
            "overall_risk_category": self._determine_overall_risk(result.risk_profile),
            "risk_factors": [v.gene for v in result.risk_profile.risk_variants],
            "protective_factors": [v.gene for v in result.risk_profile.protective_variants]
        }
    
    def _generate_pgx_recommendations(self, result: GenomicAnalysisResult) -> Dict[str, str]:
        """Generate pharmacogenomic recommendations."""
        return result.risk_profile.pharmacogenomic_insights
    
    def _generate_followup_recommendations(self, result: GenomicAnalysisResult) -> List[str]:
        """Generate follow-up recommendations."""
        recommendations = []
        
        risk_profile = result.risk_profile
        high_risk_diseases = [
            disease for disease, score in risk_profile.risk_scores.items()
            if score > 0.7
        ]
        
        if high_risk_diseases:
            recommendations.append("Genetic counseling recommended")
            recommendations.append("Enhanced neurological monitoring")
            recommendations.append("Consider preventive interventions")
        
        if risk_profile.pharmacogenomic_insights:
            recommendations.append("Pharmacogenomic-guided medication selection")
        
        return recommendations
    
    def _group_variants_by_gene(self, variants: List[GenomicVariant]) -> Dict[str, int]:
        """Group variants by gene."""
        gene_counts = {}
        for variant in variants:
            gene_counts[variant.gene] = gene_counts.get(variant.gene, 0) + 1
        return gene_counts
    
    def _determine_overall_risk(self, risk_profile: GenomicRiskProfile) -> str:
        """Determine overall risk category."""
        max_risk = max(risk_profile.risk_scores.values()) if risk_profile.risk_scores else 0.0
        
        if max_risk > 0.8:
            return "High"
        elif max_risk > 0.5:
            return "Moderate"
        elif max_risk > 0.2:
            return "Low-Moderate"
        else:
            return "Low"