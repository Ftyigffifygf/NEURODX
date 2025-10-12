"""
Comprehensive unit tests for sensor data processing.

This module tests synchronization with simulated sensor data streams and validates
feature extraction accuracy as specified in task 4.3.

Requirements tested:
- Test synchronization with simulated sensor data streams
- Validate feature extraction accuracy
- Requirements: 2.2
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging

from src.services.wearable_sensor.sensor_data_collector import SensorDataCollector
from src.services.wearable_sensor import (
    TemporalSynchronizer,
    SynchronizationConfig,
    AlignedSensorData,
    TemporalFeatures
)
from src.models.patient import WearableSession


class TestSensorDataStreamSimulation:
    """Test sensor data processing with simulated multi-modal data streams."""
    
    def setup_method(self):
        """Set up test fixtures with simulated sensor data streams."""
        self.collector = SensorDataCollector()
        self.synchronizer = TemporalSynchronizer()
        self.base_time = datetime(2024, 1, 1, 10, 0, 0)
        
        # Create realistic simulated sensor data streams
        self.simulated_streams = self._create_simulated_sensor_streams()
    
    def _create_simulated_sensor_streams(self) -> Dict[str, Dict[str, Any]]:
        """Create realistic simulated sensor data for testing."""
        duration_minutes = 10
        
        # EEG simulation: 8-channel EEG with realistic frequency components
        eeg_sampling_rate = 256.0
        eeg_duration_samples = int(duration_minutes * 60 * eeg_sampling_rate)
        eeg_channels = 8
        
        # Simulate EEG with alpha (8-12Hz), beta (13-30Hz), and noise
        t_eeg = np.linspace(0, duration_minutes * 60, eeg_duration_samples)
        eeg_data = np.zeros((eeg_duration_samples, eeg_channels))
        
        for ch in range(eeg_channels):
            # Alpha rhythm (8-12 Hz)
            alpha_freq = 10 + np.random.uniform(-2, 2)
            alpha_component = 20 * np.sin(2 * np.pi * alpha_freq * t_eeg)
            
            # Beta rhythm (13-30 Hz)
            beta_freq = 20 + np.random.uniform(-7, 10)
            beta_component = 10 * np.sin(2 * np.pi * beta_freq * t_eeg)
            
            # Noise and artifacts
            noise = np.random.normal(0, 5, len(t_eeg))
            
            eeg_data[:, ch] = alpha_component + beta_component + noise
        
        # Heart rate simulation: realistic HR variability
        hr_sampling_rate = 1.0
        hr_duration_samples = int(duration_minutes * 60 * hr_sampling_rate)
        
        # Simulate heart rate with circadian variation and HRV
        t_hr = np.linspace(0, duration_minutes * 60, hr_duration_samples)
        base_hr = 70
        circadian_variation = 5 * np.sin(2 * np.pi * t_hr / (24 * 3600))  # Daily rhythm
        hrv_noise = np.random.normal(0, 3, len(t_hr))  # Heart rate variability
        hr_data = base_hr + circadian_variation + hrv_noise
        hr_data = np.clip(hr_data, 50, 120)  # Physiological limits
        
        # Sleep stage simulation: realistic sleep architecture
        sleep_sampling_rate = 1/30  # 30-second epochs
        sleep_duration_samples = int(duration_minutes * 60 * sleep_sampling_rate)
        
        # Simulate sleep stages: 0=Wake, 1=Light, 2=Deep, 3=REM
        sleep_stages = np.zeros(sleep_duration_samples)
        
        # Simple sleep progression simulation
        for i in range(sleep_duration_samples):
            progress = i / sleep_duration_samples
            if progress < 0.1:  # Initial wake
                sleep_stages[i] = 0
            elif progress < 0.3:  # Light sleep
                sleep_stages[i] = 1
            elif progress < 0.6:  # Deep sleep
                sleep_stages[i] = 2 if np.random.random() > 0.3 else 1
            elif progress < 0.8:  # REM cycles
                sleep_stages[i] = 3 if np.random.random() > 0.5 else 1
            else:  # Wake up
                sleep_stages[i] = 0 if np.random.random() > 0.7 else 1
        
        # Gait simulation: 3-axis accelerometer during walking
        gait_sampling_rate = 100.0
        gait_duration_samples = int(duration_minutes * 60 * gait_sampling_rate)
        
        # Simulate walking pattern with step frequency ~1.5 Hz
        t_gait = np.linspace(0, duration_minutes * 60, gait_duration_samples)
        step_freq = 1.5  # Steps per second
        
        # Vertical axis (Z) - main walking pattern
        z_accel = 9.8 + 2 * np.sin(2 * np.pi * step_freq * t_gait) + np.random.normal(0, 0.5, len(t_gait))
        
        # Anterior-posterior (Y) and medial-lateral (X) components
        y_accel = 0.5 * np.sin(2 * np.pi * step_freq * t_gait + np.pi/4) + np.random.normal(0, 0.3, len(t_gait))
        x_accel = 0.3 * np.sin(2 * np.pi * step_freq * t_gait + np.pi/2) + np.random.normal(0, 0.2, len(t_gait))
        
        gait_data = np.column_stack([x_accel, y_accel, z_accel])
        
        return {
            "EEG": {
                "data": eeg_data,
                "sampling_rate": eeg_sampling_rate,
                "metadata": {"manufacturer": "NeuroSky", "model": "EEG-8000"}
            },
            "HeartRate": {
                "heart_rate": hr_data,
                "sampling_rate": hr_sampling_rate,
                "metadata": {"manufacturer": "Polar", "model": "H10"}
            },
            "Sleep": {
                "sleep_stages": sleep_stages,
                "sampling_rate": sleep_sampling_rate,
                "metadata": {"manufacturer": "Oura", "model": "Ring-3"}
            },
            "Gait": {
                "acceleration": gait_data,
                "sampling_rate": gait_sampling_rate,
                "metadata": {"manufacturer": "Apple", "model": "Watch-Series-8"}
            }
        }
    
    def test_multi_modal_data_collection(self):
        """Test collection of multiple sensor data streams."""
        sessions = []
        
        for device_type, stream_data in self.simulated_streams.items():
            session = self.collector.collect_sensor_data(
                device_type=device_type,
                data_source=stream_data,
                start_time=self.base_time,
                end_time=self.base_time + timedelta(minutes=10)
            )
            sessions.append(session)
        
        # Verify all sessions were created successfully
        assert len(sessions) == 4
        device_types = [s.device_type for s in sessions]
        assert "EEG" in device_types
        assert "HeartRate" in device_types
        assert "Sleep" in device_types
        assert "Gait" in device_types
        
        # Verify data shapes and sampling rates
        eeg_session = next(s for s in sessions if s.device_type == "EEG")
        assert eeg_session.raw_data.shape[1] == 8  # 8 channels
        assert eeg_session.sampling_rate == 256.0
        
        hr_session = next(s for s in sessions if s.device_type == "HeartRate")
        assert hr_session.raw_data.ndim == 1  # 1D heart rate data
        assert hr_session.sampling_rate == 1.0
        
        gait_session = next(s for s in sessions if s.device_type == "Gait")
        assert gait_session.raw_data.shape[1] == 3  # 3-axis accelerometer
        assert gait_session.sampling_rate == 100.0
    
    def test_synchronization_with_different_sampling_rates(self):
        """Test synchronization of sensor streams with different sampling rates."""
        # Collect all sensor sessions
        sessions = []
        for device_type, stream_data in self.simulated_streams.items():
            session = self.collector.collect_sensor_data(
                device_type=device_type,
                data_source=stream_data,
                start_time=self.base_time,
                end_time=self.base_time + timedelta(minutes=10)
            )
            sessions.append(session)
        
        # Synchronize sessions
        sync_result = self.synchronizer.synchronize_sessions(sessions)
        
        # Verify synchronization results
        assert len(sync_result.synchronized_sessions) == 4
        assert sync_result.synchronization_quality > 0.8
        
        # Verify common time range
        start_time, end_time = sync_result.common_time_range
        assert start_time >= self.base_time
        assert end_time <= self.base_time + timedelta(minutes=10)
        
        # Verify all sessions have the same time range after synchronization
        for session in sync_result.synchronized_sessions:
            assert session.start_time == start_time
            assert session.end_time == end_time
    
    def test_timestamp_alignment_accuracy(self):
        """Test accuracy of timestamp alignment across different sampling rates."""
        # Create sessions with precise timing
        sessions = []
        for device_type, stream_data in self.simulated_streams.items():
            session = self.collector.collect_sensor_data(
                device_type=device_type,
                data_source=stream_data,
                start_time=self.base_time,
                end_time=self.base_time + timedelta(minutes=5)  # Shorter for precision testing
            )
            sessions.append(session)
        
        # Align timestamps
        aligned_data = self.synchronizer.align_timestamps(sessions)
        
        # Verify alignment properties
        assert isinstance(aligned_data, AlignedSensorData)
        assert len(aligned_data.data_streams) == 4
        
        # All data streams should have the same number of samples
        sample_counts = [data.shape[0] for data in aligned_data.data_streams.values()]
        assert len(set(sample_counts)) == 1  # All should be the same
        
        # Verify timestamp array properties
        expected_duration = 5 * 60  # 5 minutes in seconds
        expected_samples = int(expected_duration * aligned_data.sampling_rate)
        assert len(aligned_data.timestamps) == expected_samples
        
        # Verify timestamp spacing
        time_diffs = np.diff(aligned_data.timestamps)
        expected_interval = 1.0 / aligned_data.sampling_rate
        np.testing.assert_allclose(time_diffs, expected_interval, rtol=1e-10)
    
    def test_interpolation_accuracy_with_missing_data(self):
        """Test interpolation accuracy when handling missing data points."""
        # Create heart rate data with intentional gaps
        hr_data = self.simulated_streams["HeartRate"]["heart_rate"].copy()
        
        # Introduce missing data (NaN) at specific intervals
        gap_indices = [100, 101, 102, 200, 201, 300]  # Multiple small gaps
        hr_data[gap_indices] = np.nan
        
        modified_stream = {
            "heart_rate": hr_data,
            "sampling_rate": 1.0,
            "metadata": {"manufacturer": "Test", "model": "Test"}
        }
        
        # Collect session with missing data
        session = self.collector.collect_sensor_data(
            device_type="HeartRate",
            data_source=modified_stream,
            start_time=self.base_time,
            end_time=self.base_time + timedelta(minutes=10)
        )
        
        # Test interpolation
        timestamps = np.arange(len(hr_data))
        interpolated_data, regions = self.synchronizer.handle_missing_data(
            session.raw_data, timestamps
        )
        
        # Verify interpolation results
        assert not np.any(np.isnan(interpolated_data))  # No NaN values should remain
        assert len(regions) > 0  # Should have detected interpolated regions
        
        # Verify interpolated values are reasonable
        for start_idx, end_idx in regions:
            if start_idx > 0 and end_idx < len(interpolated_data):
                # Check that interpolated values are between neighboring values
                before_val = interpolated_data[start_idx - 1]
                after_val = interpolated_data[end_idx]
                interp_vals = interpolated_data[start_idx:end_idx]
                
                assert np.all(interp_vals >= min(before_val, after_val))
                assert np.all(interp_vals <= max(before_val, after_val))
    
    def test_feature_extraction_accuracy(self):
        """Test accuracy of temporal feature extraction from synchronized data."""
        # Collect and synchronize all sensor sessions
        sessions = []
        for device_type, stream_data in self.simulated_streams.items():
            session = self.collector.collect_sensor_data(
                device_type=device_type,
                data_source=stream_data,
                start_time=self.base_time,
                end_time=self.base_time + timedelta(minutes=5)
            )
            sessions.append(session)
        
        # Align and extract features
        aligned_data = self.synchronizer.align_timestamps(sessions)
        features = self.synchronizer.extract_temporal_features(aligned_data)
        
        # Verify feature extraction results
        assert isinstance(features, TemporalFeatures)
        assert len(features.feature_vector) > 0
        assert len(features.feature_names) == len(features.feature_vector)
        
        # Verify statistical features for each device type
        for device_type in ["EEG", "HeartRate", "Sleep", "Gait"]:
            assert device_type in features.temporal_statistics
            stats = features.temporal_statistics[device_type]
            
            # Check that all expected statistical features are present
            expected_stats = ["mean", "std", "min", "max", "median", "skewness", "kurtosis"]
            for stat in expected_stats:
                assert stat in stats
                assert isinstance(stats[stat], (int, float))
                assert not np.isnan(stats[stat])
        
        # Verify cross-modal correlations
        assert len(features.cross_modal_correlations) > 0
        for corr_name, corr_value in features.cross_modal_correlations.items():
            assert -1.0 <= corr_value <= 1.0
            assert not np.isnan(corr_value)
        
        # Verify temporal windows
        assert len(features.temporal_windows) > 0
        for device_type, windows in features.temporal_windows.items():
            for window_name, window_data in windows.items():
                assert isinstance(window_data, np.ndarray)
                assert window_data.shape[1] == 3  # mean, std, range features
    
    def test_feature_extraction_known_signals(self):
        """Test feature extraction accuracy with known synthetic signals."""
        # Create known synthetic signals for validation
        duration = 60  # 1 minute
        sampling_rate = 10.0  # 10 Hz
        t = np.linspace(0, duration, int(duration * sampling_rate))
        
        # Known sine wave signal
        frequency = 1.0  # 1 Hz
        amplitude = 5.0
        sine_signal = amplitude * np.sin(2 * np.pi * frequency * t)
        
        # Create session with known signal
        known_session = WearableSession(
            session_id="WEAR_HeartRate_20240101_100000",
            device_type="HeartRate",
            start_time=self.base_time,
            end_time=self.base_time + timedelta(seconds=duration),
            sampling_rate=sampling_rate,
            raw_data=sine_signal
        )
        
        # Extract features
        aligned_data = self.synchronizer.align_timestamps([known_session])
        features = self.synchronizer.extract_temporal_features(aligned_data)
        
        # Verify known properties
        hr_stats = features.temporal_statistics["HeartRate"]
        
        # For a sine wave, mean should be close to 0
        assert abs(hr_stats["mean"]) < 0.1
        
        # Standard deviation should be related to amplitude (amplitude/sqrt(2) for sine)
        expected_std = amplitude / np.sqrt(2)
        assert abs(hr_stats["std"] - expected_std) < 0.5
        
        # Min and max should be close to -amplitude and +amplitude
        assert abs(hr_stats["min"] - (-amplitude)) < 0.5
        assert abs(hr_stats["max"] - amplitude) < 0.5
        
        # Zero crossings should be predictable for sine wave
        assert hr_stats["zero_crossings"] > 0
    
    def test_synchronization_quality_metrics(self):
        """Test synchronization quality assessment with various data conditions."""
        # Test with perfect data (no missing values)
        perfect_sessions = []
        for device_type, stream_data in self.simulated_streams.items():
            session = self.collector.collect_sensor_data(
                device_type=device_type,
                data_source=stream_data,
                start_time=self.base_time,
                end_time=self.base_time + timedelta(minutes=5)
            )
            perfect_sessions.append(session)
        
        perfect_result = self.synchronizer.synchronize_sessions(perfect_sessions)
        perfect_quality = perfect_result.synchronization_quality
        
        # Test with data containing gaps
        degraded_streams = self.simulated_streams.copy()
        hr_data_with_gaps = degraded_streams["HeartRate"]["heart_rate"].copy()
        hr_data_with_gaps[100:150] = np.nan  # Large gap
        degraded_streams["HeartRate"]["heart_rate"] = hr_data_with_gaps
        
        degraded_sessions = []
        for device_type, stream_data in degraded_streams.items():
            session = self.collector.collect_sensor_data(
                device_type=device_type,
                data_source=stream_data,
                start_time=self.base_time,
                end_time=self.base_time + timedelta(minutes=5)
            )
            degraded_sessions.append(session)
        
        degraded_result = self.synchronizer.synchronize_sessions(degraded_sessions)
        degraded_quality = degraded_result.synchronization_quality
        
        # Quality should be lower for data with gaps
        assert degraded_quality < perfect_quality
        assert 0.0 <= degraded_quality <= 1.0
        assert 0.0 <= perfect_quality <= 1.0
    
    def test_cross_modal_correlation_accuracy(self):
        """Test accuracy of cross-modal correlation calculations."""
        # Create correlated synthetic signals
        duration = 300  # 5 minutes
        base_signal = np.random.randn(duration)
        
        # Create sessions with known correlations
        correlated_sessions = []
        
        # Session 1: Base signal
        session1 = WearableSession(
            session_id="WEAR_HeartRate_20240101_100000",
            device_type="HeartRate",
            start_time=self.base_time,
            end_time=self.base_time + timedelta(seconds=duration),
            sampling_rate=1.0,
            raw_data=base_signal
        )
        correlated_sessions.append(session1)
        
        # Session 2: Highly correlated signal (r ≈ 0.9)
        noise_level = 0.1
        correlated_signal = base_signal + noise_level * np.random.randn(duration)
        session2 = WearableSession(
            session_id="WEAR_Sleep_20240101_100001",
            device_type="Sleep",
            start_time=self.base_time,
            end_time=self.base_time + timedelta(seconds=duration),
            sampling_rate=1.0,
            raw_data=correlated_signal
        )
        correlated_sessions.append(session2)
        
        # Extract features and correlations
        aligned_data = self.synchronizer.align_timestamps(correlated_sessions)
        features = self.synchronizer.extract_temporal_features(aligned_data)
        
        # Verify correlation is detected
        assert "HeartRate_Sleep" in features.cross_modal_correlations
        correlation = features.cross_modal_correlations["HeartRate_Sleep"]
        
        # Should be highly correlated (> 0.8)
        assert correlation > 0.8
        assert correlation <= 1.0
    
    def test_temporal_window_feature_consistency(self):
        """Test consistency of temporal window features across different window sizes."""
        # Create session with known periodic signal
        duration = 600  # 10 minutes
        sampling_rate = 2.0  # 2 Hz
        t = np.linspace(0, duration, int(duration * sampling_rate))
        
        # Create signal with multiple frequency components
        signal = (
            3 * np.sin(2 * np.pi * 0.1 * t) +  # 0.1 Hz component
            2 * np.sin(2 * np.pi * 0.5 * t) +  # 0.5 Hz component
            np.random.normal(0, 0.5, len(t))    # Noise
        )
        
        session = WearableSession(
            session_id="WEAR_HeartRate_20240101_100000",
            device_type="HeartRate",
            start_time=self.base_time,
            end_time=self.base_time + timedelta(seconds=duration),
            sampling_rate=sampling_rate,
            raw_data=signal
        )
        
        # Extract features
        aligned_data = self.synchronizer.align_timestamps([session])
        features = self.synchronizer.extract_temporal_features(aligned_data)
        
        # Verify temporal windows
        assert "HeartRate" in features.temporal_windows
        hr_windows = features.temporal_windows["HeartRate"]
        
        # Should have different window sizes
        window_types = [key for key in hr_windows.keys() if "window" in key]
        assert len(window_types) > 0
        
        # Verify window features are consistent
        for window_name, window_data in hr_windows.items():
            assert window_data.shape[1] == 3  # mean, std, range
            
            # All window features should be finite
            assert np.all(np.isfinite(window_data))
            
            # Standard deviation should be non-negative
            assert np.all(window_data[:, 1] >= 0)  # std column
            
            # Range should be non-negative
            assert np.all(window_data[:, 2] >= 0)  # range column


class TestSensorDataProcessingEdgeCases:
    """Test edge cases and error conditions in sensor data processing."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.collector = SensorDataCollector()
        self.synchronizer = TemporalSynchronizer()
        self.base_time = datetime(2024, 1, 1, 10, 0, 0)
    
    def test_synchronization_with_minimal_overlap(self):
        """Test synchronization behavior with minimal time overlap."""
        # Create sessions with minimal overlap
        session1 = WearableSession(
            session_id="WEAR_HeartRate_20240101_100000",
            device_type="HeartRate",
            start_time=self.base_time,
            end_time=self.base_time + timedelta(seconds=90),
            sampling_rate=1.0,
            raw_data=np.random.randn(90)
        )
        
        session2 = WearableSession(
            session_id="WEAR_EEG_20240101_100100",
            device_type="EEG",
            start_time=self.base_time + timedelta(seconds=60),  # 30s overlap
            end_time=self.base_time + timedelta(seconds=150),
            sampling_rate=1.0,
            raw_data=np.random.randn(90)
        )
        
        # Should succeed with 30s overlap (> min_overlap_seconds default of 60s would fail)
        config = SynchronizationConfig(min_overlap_seconds=25.0)
        synchronizer = TemporalSynchronizer(config)
        
        result = synchronizer.synchronize_sessions([session1, session2])
        assert len(result.synchronized_sessions) == 2
        
        # Verify overlap duration
        start, end = result.common_time_range
        overlap_duration = (end - start).total_seconds()
        assert overlap_duration >= 25.0
    
    def test_feature_extraction_with_constant_signal(self):
        """Test feature extraction with constant (no variation) signals."""
        # Create constant signal
        constant_value = 75.0
        constant_signal = np.full(600, constant_value)  # 10 minutes at 1Hz
        
        session = WearableSession(
            session_id="WEAR_HeartRate_20240101_100000",
            device_type="HeartRate",
            start_time=self.base_time,
            end_time=self.base_time + timedelta(minutes=10),
            sampling_rate=1.0,
            raw_data=constant_signal
        )
        
        # Extract features
        aligned_data = self.synchronizer.align_timestamps([session])
        features = self.synchronizer.extract_temporal_features(aligned_data)
        
        # Verify constant signal properties
        hr_stats = features.temporal_statistics["HeartRate"]
        assert hr_stats["mean"] == constant_value
        assert hr_stats["std"] == 0.0
        assert hr_stats["min"] == constant_value
        assert hr_stats["max"] == constant_value
        assert hr_stats["zero_crossings"] == 0
    
    def test_synchronization_with_large_time_gaps(self):
        """Test handling of sessions with large gaps that cannot be interpolated."""
        # Create data with large gap
        hr_data = np.random.randn(600)
        hr_data[200:400] = np.nan  # 200-second gap (> max_gap_seconds default)
        
        session = WearableSession(
            session_id="WEAR_HeartRate_20240101_100000",
            device_type="HeartRate",
            start_time=self.base_time,
            end_time=self.base_time + timedelta(minutes=10),
            sampling_rate=1.0,
            raw_data=hr_data
        )
        
        # Test interpolation handling
        timestamps = np.arange(len(hr_data))
        interpolated_data, regions = self.synchronizer.handle_missing_data(
            hr_data, timestamps
        )
        
        # Large gap should not be interpolated
        assert len(regions) == 0  # No regions should be interpolated
        assert np.any(np.isnan(interpolated_data[200:400]))  # Gap should remain
    
    def test_feature_extraction_with_extreme_values(self):
        """Test feature extraction robustness with extreme values."""
        # Create signal with extreme outliers
        normal_signal = np.random.normal(70, 5, 590)  # Normal heart rate
        extreme_values = np.array([1000, -500, 2000, -1000, 1500])  # Extreme outliers
        
        # Insert extreme values
        signal_with_outliers = np.concatenate([
            normal_signal[:295],
            extreme_values,
            normal_signal[295:]
        ])
        
        session = WearableSession(
            session_id="WEAR_HeartRate_20240101_100000",
            device_type="HeartRate",
            start_time=self.base_time,
            end_time=self.base_time + timedelta(minutes=10),
            sampling_rate=1.0,
            raw_data=signal_with_outliers
        )
        
        # Extract features
        aligned_data = self.synchronizer.align_timestamps([session])
        features = self.synchronizer.extract_temporal_features(aligned_data)
        
        # Verify features are computed despite extreme values
        hr_stats = features.temporal_statistics["HeartRate"]
        assert all(np.isfinite(val) for val in hr_stats.values())
        
        # Min and max should reflect extreme values
        assert hr_stats["min"] < 0
        assert hr_stats["max"] > 1000
    
    def test_empty_and_invalid_data_handling(self):
        """Test handling of empty or invalid sensor data."""
        # Test with empty data
        empty_session = WearableSession(
            session_id="WEAR_HeartRate_20240101_100000",
            device_type="HeartRate",
            start_time=self.base_time,
            end_time=self.base_time + timedelta(minutes=1),
            sampling_rate=1.0,
            raw_data=np.array([])
        )
        
        # Should handle empty data gracefully
        aligned_data = self.synchronizer.align_timestamps([empty_session])
        features = self.synchronizer.extract_temporal_features(aligned_data)
        
        # Should return empty or default features
        assert len(features.feature_vector) >= 0  # May be empty or have default values
        
        # Test with all-NaN data
        nan_data = np.full(60, np.nan)
        nan_session = WearableSession(
            session_id="WEAR_HeartRate_20240101_100001",
            device_type="HeartRate",
            start_time=self.base_time,
            end_time=self.base_time + timedelta(minutes=1),
            sampling_rate=1.0,
            raw_data=nan_data
        )
        
        # Should handle all-NaN data gracefully
        aligned_data = self.synchronizer.align_timestamps([nan_session])
        features = self.synchronizer.extract_temporal_features(aligned_data)
        
        # Features should be empty or default for all-NaN data
        if "HeartRate" in features.temporal_statistics:
            # If statistics are computed, they should be empty
            assert len(features.temporal_statistics["HeartRate"]) == 0


class TestSensorDataProcessingPerformance:
    """Test performance characteristics of sensor data processing."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.collector = SensorDataCollector()
        self.synchronizer = TemporalSynchronizer()
        self.base_time = datetime(2024, 1, 1, 10, 0, 0)
    
    def test_large_dataset_processing(self):
        """Test processing of large sensor datasets."""
        # Create large dataset (1 hour of high-frequency data)
        duration_hours = 1
        eeg_sampling_rate = 256.0
        n_samples = int(duration_hours * 3600 * eeg_sampling_rate)
        n_channels = 32  # High-density EEG
        
        large_eeg_data = np.random.randn(n_samples, n_channels)
        
        large_stream = {
            "data": large_eeg_data,
            "sampling_rate": eeg_sampling_rate,
            "metadata": {"manufacturer": "Test", "model": "Large"}
        }
        
        # Test collection and processing
        import time
        start_time = time.time()
        
        session = self.collector.collect_sensor_data(
            device_type="EEG",
            data_source=large_stream,
            start_time=self.base_time,
            end_time=self.base_time + timedelta(hours=duration_hours)
        )
        
        collection_time = time.time() - start_time
        
        # Verify successful processing
        assert session.raw_data.shape == (n_samples, n_channels)
        assert session.sampling_rate == eeg_sampling_rate
        
        # Performance should be reasonable (< 10 seconds for collection)
        assert collection_time < 10.0
    
    def test_concurrent_stream_processing(self):
        """Test processing multiple concurrent sensor streams."""
        # Create multiple streams with different characteristics
        streams = {}
        
        for i in range(5):  # 5 concurrent streams
            device_type = f"HeartRate"  # All same type for simplicity
            hr_data = np.random.uniform(60, 100, 3600)  # 1 hour at 1Hz
            
            streams[f"stream_{i}"] = {
                "heart_rate": hr_data,
                "sampling_rate": 1.0,
                "metadata": {"manufacturer": f"Device_{i}", "model": f"Model_{i}"}
            }
        
        # Process all streams
        sessions = []
        for stream_id, stream_data in streams.items():
            session = self.collector.collect_sensor_data(
                device_type="HeartRate",
                data_source=stream_data,
                start_time=self.base_time,
                end_time=self.base_time + timedelta(hours=1)
            )
            sessions.append(session)
        
        # Verify all sessions processed successfully
        assert len(sessions) == 5
        for session in sessions:
            assert session.device_type == "HeartRate"
            assert len(session.raw_data) == 3600


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])