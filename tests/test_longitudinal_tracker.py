"""
Tests for Longitudinal Tracking in MONAI Federated Learning

Tests patient progression monitoring, trend detection, and alert mechanisms.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import numpy as np

from src.services.federated_learning.longitudinal_tracker import (
    LongitudinalTracker, LongitudinalDataPoint, ProgressionTrend, ProgressionAlert
)


class TestLongitudinalDataPoint:
    """Test LongitudinalDataPoint data class"""
    
    def test_valid_data_point_creation(self):
        """Test creating a valid longitudinal data point"""
        data_point = LongitudinalDataPoint(
            patient_id="PAT_20240101_00001",
            timestamp=datetime.now(),
            study_id="STUDY_20240101_120000_001",
            modality="MRI",
            biomarkers={"amyloid_beta": 0.75, "tau_protein": 0.45},
            diagnostic_scores={"cognitive_score": 0.8, "memory_score": 0.7},
            clinical_metrics={"mmse": 28, "cdr": 0.5},
            node_id="HOSP_A"
        )
        
        assert data_point.patient_id == "PAT_20240101_00001"
        assert data_point.modality == "MRI"
        assert data_point.node_id == "HOSP_A"
        assert "amyloid_beta" in data_point.biomarkers
    
    def test_invalid_patient_id(self):
        """Test validation of patient ID format"""
        with pytest.raises(ValueError, match="Patient ID must start with 'PAT_'"):
            LongitudinalDataPoint(
                patient_id="INVALID_ID",
                timestamp=datetime.now(),
                study_id="STUDY_20240101_120000_001",
                modality="MRI",
                biomarkers={},
                diagnostic_scores={},
                clinical_metrics={},
                node_id="HOSP_A"
            )
    
    def test_invalid_study_id(self):
        """Test validation of study ID format"""
        with pytest.raises(ValueError, match="Study ID must start with 'STUDY_'"):
            LongitudinalDataPoint(
                patient_id="PAT_20240101_00001",
                timestamp=datetime.now(),
                study_id="INVALID_STUDY",
                modality="MRI",
                biomarkers={},
                diagnostic_scores={},
                clinical_metrics={},
                node_id="HOSP_A"
            )
    
    def test_invalid_modality(self):
        """Test validation of modality"""
        with pytest.raises(ValueError, match="Invalid modality"):
            LongitudinalDataPoint(
                patient_id="PAT_20240101_00001",
                timestamp=datetime.now(),
                study_id="STUDY_20240101_120000_001",
                modality="INVALID_MODALITY",
                biomarkers={},
                diagnostic_scores={},
                clinical_metrics={},
                node_id="HOSP_A"
            )


class TestProgressionTrend:
    """Test ProgressionTrend data class"""
    
    def test_valid_trend_creation(self):
        """Test creating a valid progression trend"""
        trend = ProgressionTrend(
            patient_id="PAT_20240101_00001",
            metric_name="cognitive_score",
            trend_type="declining",
            slope=-0.05,
            confidence=0.85,
            start_date=datetime.now() - timedelta(days=90),
            end_date=datetime.now(),
            data_points=5,
            significance=0.02
        )
        
        assert trend.patient_id == "PAT_20240101_00001"
        assert trend.trend_type == "declining"
        assert trend.confidence == 0.85
        assert trend.data_points == 5
    
    def test_invalid_trend_type(self):
        """Test validation of trend type"""
        with pytest.raises(ValueError, match="Invalid trend type"):
            ProgressionTrend(
                patient_id="PAT_20240101_00001",
                metric_name="cognitive_score",
                trend_type="invalid_trend",
                slope=-0.05,
                confidence=0.85,
                start_date=datetime.now() - timedelta(days=90),
                end_date=datetime.now(),
                data_points=5,
                significance=0.02
            )
    
    def test_invalid_confidence(self):
        """Test validation of confidence range"""
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            ProgressionTrend(
                patient_id="PAT_20240101_00001",
                metric_name="cognitive_score",
                trend_type="declining",
                slope=-0.05,
                confidence=1.5,  # Invalid confidence > 1
                start_date=datetime.now() - timedelta(days=90),
                end_date=datetime.now(),
                data_points=5,
                significance=0.02
            )


class TestProgressionAlert:
    """Test ProgressionAlert data class"""
    
    def test_valid_alert_creation(self):
        """Test creating a valid progression alert"""
        alert = ProgressionAlert(
            patient_id="PAT_20240101_00001",
            alert_type="rapid_decline",
            severity="high",
            message="Rapid decline in cognitive function detected",
            triggered_at=datetime.now(),
            metrics_involved=["cognitive_score", "memory_score"],
            threshold_exceeded={"cognitive_score": -0.15},
            recommended_actions=["Schedule urgent clinical review"]
        )
        
        assert alert.patient_id == "PAT_20240101_00001"
        assert alert.alert_type == "rapid_decline"
        assert alert.severity == "high"
        assert len(alert.metrics_involved) == 2
    
    def test_invalid_alert_type(self):
        """Test validation of alert type"""
        with pytest.raises(ValueError, match="Invalid alert type"):
            ProgressionAlert(
                patient_id="PAT_20240101_00001",
                alert_type="invalid_alert",
                severity="high",
                message="Test message",
                triggered_at=datetime.now(),
                metrics_involved=[],
                threshold_exceeded={},
                recommended_actions=[]
            )
    
    def test_invalid_severity(self):
        """Test validation of severity"""
        with pytest.raises(ValueError, match="Invalid severity"):
            ProgressionAlert(
                patient_id="PAT_20240101_00001",
                alert_type="rapid_decline",
                severity="invalid_severity",
                message="Test message",
                triggered_at=datetime.now(),
                metrics_involved=[],
                threshold_exceeded={},
                recommended_actions=[]
            )


class TestLongitudinalTracker:
    """Test LongitudinalTracker functionality"""
    
    @pytest.fixture
    def tracker(self):
        """Create longitudinal tracker for testing"""
        return LongitudinalTracker()
    
    @pytest.fixture
    def sample_data_point(self):
        """Create sample longitudinal data point"""
        return LongitudinalDataPoint(
            patient_id="PAT_20240101_00001",
            timestamp=datetime.now(),
            study_id="STUDY_20240101_120000_001",
            modality="MRI",
            biomarkers={"amyloid_beta": 0.75, "tau_protein": 0.45},
            diagnostic_scores={"cognitive_score": 0.8, "memory_score": 0.7},
            clinical_metrics={"mmse": 28, "cdr": 0.5},
            node_id="HOSP_A"
        )
    
    def test_tracker_initialization(self, tracker):
        """Test tracker initialization"""
        assert len(tracker.patient_timelines) == 0
        assert len(tracker.detected_trends) == 0
        assert len(tracker.active_alerts) == 0
        assert 'min_data_points' in tracker.trend_detection_config
        assert 'cognitive_decline' in tracker.alert_thresholds
    
    @pytest.mark.asyncio
    async def test_add_longitudinal_data(self, tracker, sample_data_point):
        """Test adding longitudinal data point"""
        success = await tracker.add_longitudinal_data(sample_data_point)
        
        assert success is True
        assert sample_data_point.patient_id in tracker.patient_timelines
        assert len(tracker.patient_timelines[sample_data_point.patient_id]) == 1
    
    @pytest.mark.asyncio
    async def test_timeline_sorting(self, tracker):
        """Test that timeline is sorted by timestamp"""
        patient_id = "PAT_20240101_00001"
        base_time = datetime.now()
        
        # Add data points in reverse chronological order
        for i in range(3):
            data_point = LongitudinalDataPoint(
                patient_id=patient_id,
                timestamp=base_time - timedelta(days=i),
                study_id=f"STUDY_20240101_12000{i}_001",
                modality="MRI",
                biomarkers={"test_marker": 0.5 + i * 0.1},
                diagnostic_scores={},
                clinical_metrics={},
                node_id="HOSP_A"
            )
            await tracker.add_longitudinal_data(data_point)
        
        timeline = tracker.patient_timelines[patient_id]
        
        # Verify timeline is sorted (earliest first)
        for i in range(len(timeline) - 1):
            assert timeline[i].timestamp <= timeline[i + 1].timestamp
    
    @pytest.mark.asyncio
    async def test_trend_detection_insufficient_data(self, tracker):
        """Test that trend detection requires minimum data points"""
        patient_id = "PAT_20240101_00001"
        
        # Add only 2 data points (less than minimum of 3)
        for i in range(2):
            data_point = LongitudinalDataPoint(
                patient_id=patient_id,
                timestamp=datetime.now() + timedelta(days=i * 30),
                study_id=f"STUDY_20240101_12000{i}_001",
                modality="MRI",
                biomarkers={"test_marker": 0.5 + i * 0.1},
                diagnostic_scores={"cognitive_score": 0.8 - i * 0.1},
                clinical_metrics={},
                node_id="HOSP_A"
            )
            await tracker.add_longitudinal_data(data_point)
        
        # Should not have detected any trends yet
        assert len(tracker.detected_trends[patient_id]) == 0
    
    @pytest.mark.asyncio
    @patch('src.services.federated_learning.longitudinal_tracker.LongitudinalTracker._linear_regression')
    async def test_trend_detection_with_sufficient_data(self, mock_regression, tracker):
        """Test trend detection with sufficient data points"""
        # Mock linear regression to return significant declining trend
        mock_regression.return_value = (-0.05, 0.8, -0.9, 0.01, 0.005)  # slope, intercept, r, p, stderr
        
        patient_id = "PAT_20240101_00001"
        base_time = datetime.now()
        
        # Add sufficient data points with declining trend
        for i in range(5):
            data_point = LongitudinalDataPoint(
                patient_id=patient_id,
                timestamp=base_time + timedelta(days=i * 30),
                study_id=f"STUDY_20240101_12000{i}_001",
                modality="MRI",
                biomarkers={"test_marker": 0.8 - i * 0.05},
                diagnostic_scores={"cognitive_score": 0.9 - i * 0.05},
                clinical_metrics={},
                node_id="HOSP_A"
            )
            await tracker.add_longitudinal_data(data_point)
        
        # Should have detected trends
        trends = tracker.detected_trends[patient_id]
        assert len(trends) > 0
        
        # Check for declining trends
        declining_trends = [t for t in trends if t.trend_type == 'declining']
        assert len(declining_trends) > 0
    
    def test_z_score_calculation(self, tracker):
        """Test z-score calculation for anomaly detection"""
        historical_values = [0.5, 0.6, 0.55, 0.58, 0.52]
        current_value = 0.9  # Anomalous value
        
        z_score = tracker._calculate_z_score(current_value, historical_values)
        
        # Should be a high z-score indicating anomaly
        assert abs(z_score) > 2.0
    
    def test_severity_determination(self, tracker):
        """Test alert severity determination"""
        threshold = 0.1
        
        # Test different severity levels
        assert tracker._determine_severity(0.15, threshold) == 'medium'  # 1.5x threshold
        assert tracker._determine_severity(0.25, threshold) == 'high'    # 2.5x threshold
        assert tracker._determine_severity(0.35, threshold) == 'critical' # 3.5x threshold
        assert tracker._determine_severity(0.08, threshold) == 'low'     # Below threshold
    
    def test_recommended_actions(self, tracker):
        """Test recommended actions generation"""
        # Test cognitive metric
        actions = tracker._get_recommended_actions("cognitive_score", -0.15)
        assert any("cognitive" in action.lower() for action in actions)
        
        # Test motor metric
        actions = tracker._get_recommended_actions("motor_function", -0.2)
        assert any("physical therapy" in action.lower() for action in actions)
        
        # Test significant change
        actions = tracker._get_recommended_actions("test_metric", -3.0)
        assert any("urgent" in action.lower() for action in actions)
    
    @pytest.mark.asyncio
    async def test_progression_summary(self, tracker):
        """Test patient progression summary generation"""
        patient_id = "PAT_20240101_00001"
        base_time = datetime.now()
        
        # Add some data points
        for i in range(3):
            data_point = LongitudinalDataPoint(
                patient_id=patient_id,
                timestamp=base_time + timedelta(days=i * 30),
                study_id=f"STUDY_20240101_12000{i}_001",
                modality="MRI",
                biomarkers={"test_marker": 0.5 + i * 0.1},
                diagnostic_scores={"cognitive_score": 0.8 - i * 0.05},
                clinical_metrics={},
                node_id="HOSP_A"
            )
            await tracker.add_longitudinal_data(data_point)
        
        summary = tracker.get_patient_progression_summary(patient_id)
        
        assert 'patient_id' in summary
        assert 'timeline_summary' in summary
        assert 'trend_summary' in summary
        assert 'alert_summary' in summary
        assert summary['timeline_summary']['data_points'] == 3
        assert summary['timeline_summary']['modalities'] == ['MRI']
    
    def test_progression_summary_no_data(self, tracker):
        """Test progression summary with no data"""
        summary = tracker.get_patient_progression_summary("NONEXISTENT_PATIENT")
        
        assert 'error' in summary
        assert 'No longitudinal data available' in summary['error']
    
    @pytest.mark.asyncio
    async def test_federated_insights(self, tracker):
        """Test federated progression insights"""
        # Add data for multiple patients from different nodes
        patients = ["PAT_20240101_00001", "PAT_20240101_00002"]
        nodes = ["HOSP_A", "HOSP_B"]
        
        for i, patient_id in enumerate(patients):
            for j in range(2):
                data_point = LongitudinalDataPoint(
                    patient_id=patient_id,
                    timestamp=datetime.now() + timedelta(days=j * 30),
                    study_id=f"STUDY_20240101_12000{j}_00{i+1}",
                    modality="MRI",
                    biomarkers={"test_marker": 0.5 + j * 0.1},
                    diagnostic_scores={},
                    clinical_metrics={},
                    node_id=nodes[i]
                )
                await tracker.add_longitudinal_data(data_point)
        
        insights = tracker.get_federated_progression_insights()
        
        assert 'network_summary' in insights
        assert 'trend_distribution' in insights
        assert 'alert_distribution' in insights
        assert insights['network_summary']['total_patients'] == 2
        assert insights['network_summary']['contributing_nodes'] == 2
        assert 'HOSP_A' in insights['network_summary']['node_contributions']
        assert 'HOSP_B' in insights['network_summary']['node_contributions']
    
    def test_smoothing_function(self, tracker):
        """Test moving average smoothing"""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        window_size = 3
        
        smoothed = tracker._apply_smoothing(values, window_size)
        
        assert len(smoothed) == len(values)
        # Middle value should be average of surrounding values
        assert abs(smoothed[2] - 3.0) < 0.1  # Should be close to 3.0
    
    def test_outlier_removal(self, tracker):
        """Test outlier removal using z-score"""
        values = [1.0, 1.1, 1.2, 10.0, 1.3, 1.4]  # 10.0 is an outlier
        time_days = list(range(len(values)))
        
        filtered_values, filtered_time = tracker._remove_outliers(values, time_days)
        
        # Outlier should be removed
        assert len(filtered_values) < len(values)
        assert 10.0 not in filtered_values
    
    def test_trend_classification(self, tracker):
        """Test trend classification based on slope and significance"""
        # Test declining trend
        assert tracker._classify_trend(-0.05, 0.01) == 'declining'
        
        # Test improving trend
        assert tracker._classify_trend(0.05, 0.01) == 'improving'
        
        # Test stable trend (non-significant)
        assert tracker._classify_trend(-0.05, 0.1) == 'stable'
        
        # Test stable trend (near-zero slope)
        assert tracker._classify_trend(1e-7, 0.01) == 'stable'


@pytest.mark.asyncio
async def test_longitudinal_tracking_integration():
    """Integration test for longitudinal tracking workflow"""
    tracker = LongitudinalTracker()
    patient_id = "PAT_20240101_00001"
    base_time = datetime.now()
    
    # Simulate patient progression over 6 months with declining cognitive function
    cognitive_scores = [0.9, 0.85, 0.8, 0.7, 0.65, 0.6]  # Declining trend
    biomarker_values = [0.3, 0.35, 0.4, 0.5, 0.55, 0.6]  # Increasing (worsening)
    
    for i, (cog_score, biomarker) in enumerate(zip(cognitive_scores, biomarker_values)):
        data_point = LongitudinalDataPoint(
            patient_id=patient_id,
            timestamp=base_time + timedelta(days=i * 30),
            study_id=f"STUDY_20240101_12000{i}_001",
            modality="MRI",
            biomarkers={"amyloid_beta": biomarker, "tau_protein": biomarker * 0.8},
            diagnostic_scores={"cognitive_score": cog_score, "memory_score": cog_score * 0.9},
            clinical_metrics={"mmse": int(30 * cog_score), "cdr": (1 - cog_score) * 2},
            node_id="HOSP_A"
        )
        
        success = await tracker.add_longitudinal_data(data_point)
        assert success is True
    
    # Verify timeline was created
    assert patient_id in tracker.patient_timelines
    assert len(tracker.patient_timelines[patient_id]) == 6
    
    # Get progression summary
    summary = tracker.get_patient_progression_summary(patient_id)
    
    assert summary['patient_id'] == patient_id
    assert summary['timeline_summary']['data_points'] == 6
    assert summary['timeline_summary']['total_timespan_days'] >= 150  # ~5 months
    
    # Get federated insights
    insights = tracker.get_federated_progression_insights()
    
    assert insights['network_summary']['total_patients'] == 1
    assert insights['network_summary']['total_data_points'] == 6
    assert 'HOSP_A' in insights['network_summary']['node_contributions']