"""
Longitudinal Tracking for MONAI Federated Learning

Implements patient progression monitoring over time using MONAI's temporal analysis capabilities.
Tracks disease progression, treatment response, and biomarker changes across federated nodes.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import torch
from collections import defaultdict
import json

from src.models.patient import PatientRecord, ImagingStudy
from src.models.diagnostics import DiagnosticResult

logger = logging.getLogger(__name__)


@dataclass
class LongitudinalDataPoint:
    """Represents a single data point in a patient's longitudinal timeline"""
    patient_id: str
    timestamp: datetime
    study_id: str
    modality: str
    biomarkers: Dict[str, float]
    diagnostic_scores: Dict[str, float]
    clinical_metrics: Dict[str, float]
    node_id: str  # Which federated node contributed this data
    
    def __post_init__(self):
        """Validate longitudinal data point"""
        if not self.patient_id or not self.patient_id.startswith('PAT_'):
            raise ValueError("Patient ID must start with 'PAT_'")
        if not self.study_id or not self.study_id.startswith('STUDY_'):
            raise ValueError("Study ID must start with 'STUDY_'")
        if self.modality not in ['MRI', 'CT', 'Ultrasound', 'EEG', 'HeartRate', 'Sleep', 'Gait']:
            raise ValueError(f"Invalid modality: {self.modality}")


@dataclass
class ProgressionTrend:
    """Represents a detected trend in patient progression"""
    patient_id: str
    metric_name: str
    trend_type: str  # 'improving', 'declining', 'stable', 'fluctuating'
    slope: float  # Rate of change
    confidence: float  # Confidence in trend detection (0-1)
    start_date: datetime
    end_date: datetime
    data_points: int
    significance: float  # Statistical significance (p-value)
    
    def __post_init__(self):
        """Validate progression trend"""
        if self.trend_type not in ['improving', 'declining', 'stable', 'fluctuating']:
            raise ValueError(f"Invalid trend type: {self.trend_type}")
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        if self.data_points < 2:
            raise ValueError("At least 2 data points required for trend analysis")


@dataclass
class ProgressionAlert:
    """Represents an alert for significant changes in patient progression"""
    patient_id: str
    alert_type: str  # 'rapid_decline', 'unexpected_improvement', 'anomaly'
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    triggered_at: datetime
    metrics_involved: List[str]
    threshold_exceeded: Dict[str, float]
    recommended_actions: List[str]
    
    def __post_init__(self):
        """Validate progression alert"""
        if self.alert_type not in ['rapid_decline', 'unexpected_improvement', 'anomaly']:
            raise ValueError(f"Invalid alert type: {self.alert_type}")
        if self.severity not in ['low', 'medium', 'high', 'critical']:
            raise ValueError(f"Invalid severity: {self.severity}")


class LongitudinalTracker:
    """
    MONAI Longitudinal Tracking System
    
    Monitors patient progression over time using temporal analysis features
    and provides trend detection and alert mechanisms for federated learning.
    """
    
    def __init__(self):
        self.patient_timelines: Dict[str, List[LongitudinalDataPoint]] = defaultdict(list)
        self.detected_trends: Dict[str, List[ProgressionTrend]] = defaultdict(list)
        self.active_alerts: List[ProgressionAlert] = []
        self.trend_detection_config = self._get_default_trend_config()
        self.alert_thresholds = self._get_default_alert_thresholds()
    
    def _get_default_trend_config(self) -> Dict[str, Any]:
        """Get default configuration for trend detection"""
        return {
            'min_data_points': 3,
            'min_time_span_days': 30,
            'significance_threshold': 0.05,
            'confidence_threshold': 0.7,
            'smoothing_window': 3,
            'outlier_threshold': 2.5  # Standard deviations
        }
    
    def _get_default_alert_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Get default thresholds for progression alerts"""
        return {
            'cognitive_decline': {
                'rapid_decline_rate': -0.1,  # Per month
                'critical_threshold': 0.3,   # Absolute change
                'anomaly_zscore': 3.0
            },
            'motor_function': {
                'rapid_decline_rate': -0.15,
                'critical_threshold': 0.4,
                'anomaly_zscore': 2.5
            },
            'biomarkers': {
                'rapid_change_rate': 0.2,
                'critical_threshold': 0.5,
                'anomaly_zscore': 3.0
            }
        }
    
    async def add_longitudinal_data(self, data_point: LongitudinalDataPoint) -> bool:
        """
        Add a new longitudinal data point for a patient
        
        Args:
            data_point: Longitudinal data point to add
            
        Returns:
            bool: True if data point added successfully
        """
        try:
            patient_id = data_point.patient_id
            
            # Add to patient timeline (sorted by timestamp)
            self.patient_timelines[patient_id].append(data_point)
            self.patient_timelines[patient_id].sort(key=lambda x: x.timestamp)
            
            # Trigger trend analysis if we have enough data points
            if len(self.patient_timelines[patient_id]) >= self.trend_detection_config['min_data_points']:
                await self._analyze_patient_trends(patient_id)
            
            # Check for alerts
            await self._check_progression_alerts(patient_id, data_point)
            
            logger.info(f"Added longitudinal data point for patient {patient_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add longitudinal data point: {e}")
            return False
    
    async def _analyze_patient_trends(self, patient_id: str):
        """Analyze trends for a specific patient"""
        try:
            timeline = self.patient_timelines[patient_id]
            
            if len(timeline) < self.trend_detection_config['min_data_points']:
                return
            
            # Check time span
            time_span = (timeline[-1].timestamp - timeline[0].timestamp).days
            if time_span < self.trend_detection_config['min_time_span_days']:
                return
            
            # Analyze trends for each metric type
            await self._analyze_biomarker_trends(patient_id, timeline)
            await self._analyze_diagnostic_trends(patient_id, timeline)
            await self._analyze_clinical_trends(patient_id, timeline)
            
        except Exception as e:
            logger.error(f"Failed to analyze trends for patient {patient_id}: {e}")
    
    async def _analyze_biomarker_trends(self, patient_id: str, timeline: List[LongitudinalDataPoint]):
        """Analyze biomarker trends using temporal analysis"""
        try:
            # Collect all biomarker names
            all_biomarkers = set()
            for point in timeline:
                all_biomarkers.update(point.biomarkers.keys())
            
            for biomarker in all_biomarkers:
                # Extract time series data
                timestamps = []
                values = []
                
                for point in timeline:
                    if biomarker in point.biomarkers:
                        timestamps.append(point.timestamp)
                        values.append(point.biomarkers[biomarker])
                
                if len(values) >= self.trend_detection_config['min_data_points']:
                    trend = await self._detect_trend(
                        patient_id, f"biomarker_{biomarker}", timestamps, values
                    )
                    if trend:
                        self.detected_trends[patient_id].append(trend)
                        
        except Exception as e:
            logger.error(f"Failed to analyze biomarker trends: {e}")
    
    async def _analyze_diagnostic_trends(self, patient_id: str, timeline: List[LongitudinalDataPoint]):
        """Analyze diagnostic score trends"""
        try:
            # Collect all diagnostic score names
            all_scores = set()
            for point in timeline:
                all_scores.update(point.diagnostic_scores.keys())
            
            for score_name in all_scores:
                timestamps = []
                values = []
                
                for point in timeline:
                    if score_name in point.diagnostic_scores:
                        timestamps.append(point.timestamp)
                        values.append(point.diagnostic_scores[score_name])
                
                if len(values) >= self.trend_detection_config['min_data_points']:
                    trend = await self._detect_trend(
                        patient_id, f"diagnostic_{score_name}", timestamps, values
                    )
                    if trend:
                        self.detected_trends[patient_id].append(trend)
                        
        except Exception as e:
            logger.error(f"Failed to analyze diagnostic trends: {e}")
    
    async def _analyze_clinical_trends(self, patient_id: str, timeline: List[LongitudinalDataPoint]):
        """Analyze clinical metric trends"""
        try:
            # Collect all clinical metric names
            all_metrics = set()
            for point in timeline:
                all_metrics.update(point.clinical_metrics.keys())
            
            for metric_name in all_metrics:
                timestamps = []
                values = []
                
                for point in timeline:
                    if metric_name in point.clinical_metrics:
                        timestamps.append(point.timestamp)
                        values.append(point.clinical_metrics[metric_name])
                
                if len(values) >= self.trend_detection_config['min_data_points']:
                    trend = await self._detect_trend(
                        patient_id, f"clinical_{metric_name}", timestamps, values
                    )
                    if trend:
                        self.detected_trends[patient_id].append(trend)
                        
        except Exception as e:
            logger.error(f"Failed to analyze clinical trends: {e}")
    
    async def _detect_trend(self, patient_id: str, metric_name: str, 
                          timestamps: List[datetime], values: List[float]) -> Optional[ProgressionTrend]:
        """
        Detect trend in time series data using statistical analysis
        
        Args:
            patient_id: Patient identifier
            metric_name: Name of the metric being analyzed
            timestamps: List of timestamps
            values: List of corresponding values
            
        Returns:
            ProgressionTrend if significant trend detected, None otherwise
        """
        try:
            if len(values) < 2:
                return None
            
            # Convert timestamps to days since first measurement
            time_days = [(ts - timestamps[0]).days for ts in timestamps]
            
            # Apply smoothing if configured
            if self.trend_detection_config['smoothing_window'] > 1:
                values = self._apply_smoothing(values, self.trend_detection_config['smoothing_window'])
            
            # Remove outliers
            values, time_days = self._remove_outliers(values, time_days)
            
            if len(values) < 2:
                return None
            
            # Perform linear regression
            slope, intercept, r_value, p_value, std_err = self._linear_regression(time_days, values)
            
            # Determine trend type
            trend_type = self._classify_trend(slope, p_value)
            
            # Calculate confidence
            confidence = max(0, 1 - p_value) if p_value <= 1 else 0
            
            # Check if trend is significant
            if (p_value <= self.trend_detection_config['significance_threshold'] and 
                confidence >= self.trend_detection_config['confidence_threshold']):
                
                return ProgressionTrend(
                    patient_id=patient_id,
                    metric_name=metric_name,
                    trend_type=trend_type,
                    slope=slope,
                    confidence=confidence,
                    start_date=timestamps[0],
                    end_date=timestamps[-1],
                    data_points=len(values),
                    significance=p_value
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to detect trend for {metric_name}: {e}")
            return None
    
    def _apply_smoothing(self, values: List[float], window_size: int) -> List[float]:
        """Apply moving average smoothing to values"""
        if len(values) < window_size:
            return values
        
        smoothed = []
        for i in range(len(values)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(values), i + window_size // 2 + 1)
            smoothed.append(np.mean(values[start_idx:end_idx]))
        
        return smoothed
    
    def _remove_outliers(self, values: List[float], time_days: List[int]) -> Tuple[List[float], List[int]]:
        """Remove outliers using z-score method"""
        if len(values) < 3:
            return values, time_days
        
        values_array = np.array(values)
        z_scores = np.abs((values_array - np.mean(values_array)) / np.std(values_array))
        
        threshold = self.trend_detection_config['outlier_threshold']
        mask = z_scores < threshold
        
        filtered_values = [v for i, v in enumerate(values) if mask[i]]
        filtered_time = [t for i, t in enumerate(time_days) if mask[i]]
        
        return filtered_values, filtered_time
    
    def _linear_regression(self, x: List[float], y: List[float]) -> Tuple[float, float, float, float, float]:
        """Perform linear regression and return statistics"""
        from scipy import stats
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        return slope, intercept, r_value, p_value, std_err
    
    def _classify_trend(self, slope: float, p_value: float) -> str:
        """Classify trend based on slope and significance"""
        if p_value > self.trend_detection_config['significance_threshold']:
            return 'stable'
        
        if abs(slope) < 1e-6:
            return 'stable'
        elif slope > 0:
            return 'improving'
        else:
            return 'declining'
    
    async def _check_progression_alerts(self, patient_id: str, data_point: LongitudinalDataPoint):
        """Check if new data point triggers any progression alerts"""
        try:
            timeline = self.patient_timelines[patient_id]
            
            if len(timeline) < 2:
                return  # Need at least 2 points for comparison
            
            # Check for rapid changes
            await self._check_rapid_change_alerts(patient_id, timeline)
            
            # Check for anomalies
            await self._check_anomaly_alerts(patient_id, data_point, timeline)
            
        except Exception as e:
            logger.error(f"Failed to check progression alerts: {e}")
    
    async def _check_rapid_change_alerts(self, patient_id: str, timeline: List[LongitudinalDataPoint]):
        """Check for rapid changes that might indicate critical progression"""
        try:
            if len(timeline) < 2:
                return
            
            current = timeline[-1]
            previous = timeline[-2]
            
            # Calculate time difference in months
            time_diff_months = (current.timestamp - previous.timestamp).days / 30.0
            
            if time_diff_months <= 0:
                return
            
            # Check biomarkers for rapid changes
            for biomarker, current_value in current.biomarkers.items():
                if biomarker in previous.biomarkers:
                    previous_value = previous.biomarkers[biomarker]
                    rate_of_change = (current_value - previous_value) / time_diff_months
                    
                    threshold = self.alert_thresholds['biomarkers']['rapid_change_rate']
                    if abs(rate_of_change) > threshold:
                        alert = ProgressionAlert(
                            patient_id=patient_id,
                            alert_type='rapid_decline' if rate_of_change < 0 else 'unexpected_improvement',
                            severity=self._determine_severity(abs(rate_of_change), threshold),
                            message=f"Rapid change in {biomarker}: {rate_of_change:.3f} per month",
                            triggered_at=datetime.now(),
                            metrics_involved=[biomarker],
                            threshold_exceeded={biomarker: rate_of_change},
                            recommended_actions=self._get_recommended_actions(biomarker, rate_of_change)
                        )
                        self.active_alerts.append(alert)
            
            # Check diagnostic scores
            for score_name, current_value in current.diagnostic_scores.items():
                if score_name in previous.diagnostic_scores:
                    previous_value = previous.diagnostic_scores[score_name]
                    rate_of_change = (current_value - previous_value) / time_diff_months
                    
                    threshold = self.alert_thresholds['cognitive_decline']['rapid_decline_rate']
                    if rate_of_change < threshold:  # Declining scores are concerning
                        alert = ProgressionAlert(
                            patient_id=patient_id,
                            alert_type='rapid_decline',
                            severity=self._determine_severity(abs(rate_of_change), abs(threshold)),
                            message=f"Rapid decline in {score_name}: {rate_of_change:.3f} per month",
                            triggered_at=datetime.now(),
                            metrics_involved=[score_name],
                            threshold_exceeded={score_name: rate_of_change},
                            recommended_actions=self._get_recommended_actions(score_name, rate_of_change)
                        )
                        self.active_alerts.append(alert)
                        
        except Exception as e:
            logger.error(f"Failed to check rapid change alerts: {e}")
    
    async def _check_anomaly_alerts(self, patient_id: str, current_point: LongitudinalDataPoint, 
                                  timeline: List[LongitudinalDataPoint]):
        """Check for anomalous values compared to patient's historical data"""
        try:
            if len(timeline) < 3:
                return  # Need more history for anomaly detection
            
            # Analyze each biomarker for anomalies
            for biomarker, current_value in current_point.biomarkers.items():
                historical_values = [
                    point.biomarkers[biomarker] 
                    for point in timeline[:-1] 
                    if biomarker in point.biomarkers
                ]
                
                if len(historical_values) >= 2:
                    z_score = self._calculate_z_score(current_value, historical_values)
                    threshold = self.alert_thresholds['biomarkers']['anomaly_zscore']
                    
                    if abs(z_score) > threshold:
                        alert = ProgressionAlert(
                            patient_id=patient_id,
                            alert_type='anomaly',
                            severity=self._determine_severity(abs(z_score), threshold),
                            message=f"Anomalous {biomarker} value: {current_value:.3f} (z-score: {z_score:.2f})",
                            triggered_at=datetime.now(),
                            metrics_involved=[biomarker],
                            threshold_exceeded={biomarker: z_score},
                            recommended_actions=self._get_recommended_actions(biomarker, z_score)
                        )
                        self.active_alerts.append(alert)
                        
        except Exception as e:
            logger.error(f"Failed to check anomaly alerts: {e}")
    
    def _calculate_z_score(self, value: float, historical_values: List[float]) -> float:
        """Calculate z-score for anomaly detection"""
        if len(historical_values) < 2:
            return 0.0
        
        mean = np.mean(historical_values)
        std = np.std(historical_values)
        
        if std == 0:
            return 0.0
        
        return (value - mean) / std
    
    def _determine_severity(self, value: float, threshold: float) -> str:
        """Determine alert severity based on how much threshold is exceeded"""
        ratio = value / threshold
        
        if ratio >= 3.0:
            return 'critical'
        elif ratio >= 2.0:
            return 'high'
        elif ratio >= 1.5:
            return 'medium'
        else:
            return 'low'
    
    def _get_recommended_actions(self, metric_name: str, change_value: float) -> List[str]:
        """Get recommended actions based on metric and change"""
        actions = []
        
        if 'cognitive' in metric_name.lower() or 'memory' in metric_name.lower():
            actions.extend([
                "Schedule cognitive assessment",
                "Review medication adherence",
                "Consider neuropsychological evaluation"
            ])
        
        if 'motor' in metric_name.lower() or 'gait' in metric_name.lower():
            actions.extend([
                "Schedule physical therapy evaluation",
                "Assess fall risk",
                "Review mobility aids"
            ])
        
        if abs(change_value) > 2.0:  # Significant change
            actions.extend([
                "Schedule urgent clinical review",
                "Consider additional imaging",
                "Review treatment plan"
            ])
        
        return actions
    
    def get_patient_progression_summary(self, patient_id: str) -> Dict[str, Any]:
        """Get comprehensive progression summary for a patient"""
        try:
            timeline = self.patient_timelines.get(patient_id, [])
            trends = self.detected_trends.get(patient_id, [])
            alerts = [alert for alert in self.active_alerts if alert.patient_id == patient_id]
            
            if not timeline:
                return {'error': 'No longitudinal data available for patient'}
            
            # Calculate summary statistics
            total_timespan_days = (timeline[-1].timestamp - timeline[0].timestamp).days
            data_points = len(timeline)
            modalities = list(set(point.modality for point in timeline))
            nodes_contributing = list(set(point.node_id for point in timeline))
            
            # Categorize trends
            trend_summary = {
                'improving': len([t for t in trends if t.trend_type == 'improving']),
                'declining': len([t for t in trends if t.trend_type == 'declining']),
                'stable': len([t for t in trends if t.trend_type == 'stable']),
                'fluctuating': len([t for t in trends if t.trend_type == 'fluctuating'])
            }
            
            # Alert summary
            alert_summary = {
                'total_alerts': len(alerts),
                'critical': len([a for a in alerts if a.severity == 'critical']),
                'high': len([a for a in alerts if a.severity == 'high']),
                'medium': len([a for a in alerts if a.severity == 'medium']),
                'low': len([a for a in alerts if a.severity == 'low'])
            }
            
            return {
                'patient_id': patient_id,
                'timeline_summary': {
                    'total_timespan_days': total_timespan_days,
                    'data_points': data_points,
                    'modalities': modalities,
                    'contributing_nodes': nodes_contributing,
                    'first_measurement': timeline[0].timestamp.isoformat(),
                    'last_measurement': timeline[-1].timestamp.isoformat()
                },
                'trend_summary': trend_summary,
                'alert_summary': alert_summary,
                'recent_trends': [
                    {
                        'metric': trend.metric_name,
                        'type': trend.trend_type,
                        'confidence': trend.confidence,
                        'slope': trend.slope
                    }
                    for trend in sorted(trends, key=lambda x: x.end_date, reverse=True)[:5]
                ],
                'active_alerts': [
                    {
                        'type': alert.alert_type,
                        'severity': alert.severity,
                        'message': alert.message,
                        'metrics': alert.metrics_involved
                    }
                    for alert in sorted(alerts, key=lambda x: x.triggered_at, reverse=True)[:5]
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to get progression summary for {patient_id}: {e}")
            return {'error': str(e)}
    
    def get_federated_progression_insights(self) -> Dict[str, Any]:
        """Get insights across all patients in the federated network"""
        try:
            total_patients = len(self.patient_timelines)
            total_data_points = sum(len(timeline) for timeline in self.patient_timelines.values())
            
            # Node contribution analysis
            node_contributions = defaultdict(int)
            for timeline in self.patient_timelines.values():
                for point in timeline:
                    node_contributions[point.node_id] += 1
            
            # Trend analysis across all patients
            all_trends = []
            for trends in self.detected_trends.values():
                all_trends.extend(trends)
            
            trend_distribution = {
                'improving': len([t for t in all_trends if t.trend_type == 'improving']),
                'declining': len([t for t in all_trends if t.trend_type == 'declining']),
                'stable': len([t for t in all_trends if t.trend_type == 'stable']),
                'fluctuating': len([t for t in all_trends if t.trend_type == 'fluctuating'])
            }
            
            # Alert analysis
            alert_distribution = {
                'rapid_decline': len([a for a in self.active_alerts if a.alert_type == 'rapid_decline']),
                'unexpected_improvement': len([a for a in self.active_alerts if a.alert_type == 'unexpected_improvement']),
                'anomaly': len([a for a in self.active_alerts if a.alert_type == 'anomaly'])
            }
            
            return {
                'network_summary': {
                    'total_patients': total_patients,
                    'total_data_points': total_data_points,
                    'contributing_nodes': len(node_contributions),
                    'node_contributions': dict(node_contributions)
                },
                'trend_distribution': trend_distribution,
                'alert_distribution': alert_distribution,
                'total_active_alerts': len(self.active_alerts)
            }
            
        except Exception as e:
            logger.error(f"Failed to get federated progression insights: {e}")
            return {'error': str(e)}