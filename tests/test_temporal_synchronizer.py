"""
Unit tests for temporal synchronization service.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.services.wearable_sensor.temporal_synchronizer import (
    TemporalSynchronizer,
    SynchronizationConfig,
    SynchronizationResult,
    AlignedSensorData,
    TemporalFeatures
)
from src.models.patient import WearableSession


class TestTemporalSynchronizer:
    """Test cases for TemporalSynchronizer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = SynchronizationConfig(
            target_sampling_rate=1.0,
            interpolation_method="linear",
            max_gap_seconds=60.0,
            alignment_tolerance_seconds=1.0,
            min_overlap_seconds=30.0
        )
        self.synchronizer = TemporalSynchronizer(self.config)
        
        # Create test sessions
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        
        # EEG session (high sampling rate) - 2 minutes duration
        eeg_data = np.random.randn(15360, 8)  # 120 seconds at 128Hz, 8 channels
        self.eeg_session = WearableSession(
            session_id="WEAR_EEG_20240101_120000",
            device_type="EEG",
            start_time=base_time,
            end_time=base_time + timedelta(seconds=120),
            sampling_rate=128.0,
            raw_data=eeg_data
        )
        
        # Heart rate session (low sampling rate) - 2 minutes duration
        hr_data = np.random.randint(60, 100, 120)  # 120 seconds at 1Hz
        self.hr_session = WearableSession(
            session_id="WEAR_HeartRate_20240101_120000",
            device_type="HeartRate",
            start_time=base_time,
            end_time=base_time + timedelta(seconds=120),
            sampling_rate=1.0,
            raw_data=hr_data.astype(float)
        )
        
        # Gait session with slight time offset - 2 minutes duration
        gait_data = np.random.randn(6000, 3)  # 120 seconds at 50Hz, 3 axes
        self.gait_session = WearableSession(
            session_id="WEAR_Gait_20240101_120001",
            device_type="Gait",
            start_time=base_time + timedelta(seconds=10),
            end_time=base_time + timedelta(seconds=130),
            sampling_rate=50.0,
            raw_data=gait_data
        )
    
    def test_initialization_default_config(self):
        """Test synchronizer initialization with default config."""
        sync = TemporalSynchronizer()
        assert sync.config.target_sampling_rate == 1.0
        assert sync.config.interpolation_method == "linear"
        assert sync.config.max_gap_seconds == 300.0
    
    def test_initialization_custom_config(self):
        """Test synchronizer initialization with custom config."""
        custom_config = SynchronizationConfig(
            target_sampling_rate=2.0,
            interpolation_method="cubic",
            max_gap_seconds=120.0
        )
        sync = TemporalSynchronizer(custom_config)
        assert sync.config.target_sampling_rate == 2.0
        assert sync.config.interpolation_method == "cubic"
        assert sync.config.max_gap_seconds == 120.0
    
    def test_synchronize_sessions_empty_list(self):
        """Test synchronization with empty session list."""
        with pytest.raises(ValueError, match="No sessions provided"):
            self.synchronizer.synchronize_sessions([])
    
    def test_synchronize_sessions_single_session(self):
        """Test synchronization with single session."""
        result = self.synchronizer.synchronize_sessions([self.eeg_session])
        
        assert isinstance(result, SynchronizationResult)
        assert len(result.synchronized_sessions) == 1
        assert result.synchronization_quality == 1.0
        assert len(result.interpolated_gaps) == 0
    
    def test_synchronize_sessions_multiple_overlapping(self):
        """Test synchronization with multiple overlapping sessions."""
        sessions = [self.eeg_session, self.hr_session]
        result = self.synchronizer.synchronize_sessions(sessions)
        
        assert isinstance(result, SynchronizationResult)
        assert len(result.synchronized_sessions) == 2
        assert result.synchronization_quality > 0.0
        
        # Check common time range
        start, end = result.common_time_range
        assert start >= max(s.start_time for s in sessions)
        assert end <= min(s.end_time for s in sessions)
    
    def test_synchronize_sessions_no_overlap(self):
        """Test synchronization with non-overlapping sessions."""
        # Create non-overlapping session
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        future_session = WearableSession(
            session_id="WEAR_HeartRate_20240101_130000",
            device_type="HeartRate",
            start_time=base_time + timedelta(hours=1),
            end_time=base_time + timedelta(hours=1, seconds=120),
            sampling_rate=1.0,
            raw_data=np.random.randn(120)
        )
        
        with pytest.raises(ValueError, match="No overlapping time range"):
            self.synchronizer.synchronize_sessions([self.eeg_session, future_session])
    
    def test_synchronize_sessions_insufficient_overlap(self):
        """Test synchronization with insufficient overlap."""
        # Create session with minimal overlap
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        minimal_overlap_session = WearableSession(
            session_id="WEAR_HeartRate_20240101_120009",
            device_type="HeartRate",
            start_time=base_time + timedelta(seconds=119),  # Only 1s overlap
            end_time=base_time + timedelta(seconds=239),
            sampling_rate=1.0,
            raw_data=np.random.randn(120)
        )
        
        with pytest.raises(ValueError, match="Insufficient overlap"):
            self.synchronizer.synchronize_sessions([self.eeg_session, minimal_overlap_session])
    
    def test_align_timestamps_empty_sessions(self):
        """Test timestamp alignment with empty session list."""
        with pytest.raises(ValueError, match="No sessions provided"):
            self.synchronizer.align_timestamps([])
    
    def test_align_timestamps_valid_sessions(self):
        """Test timestamp alignment with valid sessions."""
        sessions = [self.eeg_session, self.hr_session]
        aligned_data = self.synchronizer.align_timestamps(sessions)
        
        assert isinstance(aligned_data, AlignedSensorData)
        assert len(aligned_data.data_streams) == 2
        assert "EEG" in aligned_data.data_streams
        assert "HeartRate" in aligned_data.data_streams
        assert aligned_data.sampling_rate == self.config.target_sampling_rate
        
        # Check data shapes - should be based on overlap between sessions
        # EEG: 0-120s, HeartRate: 0-120s, so overlap is 120s
        expected_samples = int(120 * self.config.target_sampling_rate)  # 120 seconds
        assert len(aligned_data.timestamps) == expected_samples
        assert aligned_data.data_streams["EEG"].shape[0] == expected_samples
        assert aligned_data.data_streams["HeartRate"].shape[0] == expected_samples
    
    def test_align_timestamps_missing_raw_data(self):
        """Test alignment with session missing raw data."""
        session_no_data = WearableSession(
            session_id="WEAR_HeartRate_20240101_120000",
            device_type="HeartRate",
            start_time=datetime(2024, 1, 1, 12, 0, 0),
            end_time=datetime(2024, 1, 1, 12, 2, 0),
            sampling_rate=1.0,
            raw_data=None
        )
        
        with pytest.raises(ValueError, match="No raw data available"):
            self.synchronizer.align_timestamps([session_no_data])
    
    def test_extract_temporal_features_valid_data(self):
        """Test temporal feature extraction with valid aligned data."""
        sessions = [self.eeg_session, self.hr_session]
        aligned_data = self.synchronizer.align_timestamps(sessions)
        features = self.synchronizer.extract_temporal_features(aligned_data)
        
        assert isinstance(features, TemporalFeatures)
        assert len(features.feature_vector) > 0
        assert len(features.feature_names) == len(features.feature_vector)
        assert len(features.temporal_statistics) == 2  # EEG and HeartRate
        
        # Check that we have statistical features for each device type
        assert "EEG" in features.temporal_statistics
        assert "HeartRate" in features.temporal_statistics
        
        # Check cross-modal correlations
        assert "EEG_HeartRate" in features.cross_modal_correlations
    
    def test_extract_temporal_features_empty_data(self):
        """Test feature extraction with empty data streams."""
        empty_aligned_data = AlignedSensorData(
            timestamps=np.array([]),
            data_streams={},
            sampling_rate=1.0,
            time_range=(datetime.now(), datetime.now()),
            quality_metrics={},
            interpolated_regions={}
        )
        
        features = self.synchronizer.extract_temporal_features(empty_aligned_data)
        assert len(features.feature_vector) == 0
        assert len(features.feature_names) == 0
    
    def test_handle_missing_data_no_missing(self):
        """Test missing data handling with complete data."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        timestamps = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        
        result_data, regions = self.synchronizer.handle_missing_data(data, timestamps)
        
        np.testing.assert_array_equal(result_data, data)
        assert len(regions) == 0
    
    def test_handle_missing_data_with_gaps(self):
        """Test missing data handling with NaN gaps."""
        data = np.array([1.0, np.nan, np.nan, 4.0, 5.0])
        timestamps = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        
        result_data, regions = self.synchronizer.handle_missing_data(data, timestamps)
        
        # Should interpolate the middle values
        assert not np.isnan(result_data[1])
        assert not np.isnan(result_data[2])
        assert len(regions) == 1
        assert regions[0] == (1, 3)
    
    def test_handle_missing_data_large_gap(self):
        """Test missing data handling with gap too large to interpolate."""
        # Create data with large gap
        data = np.array([1.0] + [np.nan] * 100 + [2.0])
        timestamps = np.linspace(0, 101, 102)  # Gap > max_gap_seconds
        
        result_data, regions = self.synchronizer.handle_missing_data(data, timestamps)
        
        # Large gap should not be interpolated
        assert len(regions) == 0
        assert np.isnan(result_data[1])  # Gap should remain
    
    def test_statistical_features_extraction(self):
        """Test statistical feature extraction."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        features = self.synchronizer._extract_statistical_features(data, "test")
        
        expected_keys = ["mean", "std", "min", "max", "median", "skewness", "kurtosis", "energy", "zero_crossings"]
        for key in expected_keys:
            assert key in features
        
        assert features["mean"] == 3.0
        assert features["min"] == 1.0
        assert features["max"] == 5.0
        assert features["median"] == 3.0
    
    def test_statistical_features_multidimensional(self):
        """Test statistical features with multi-dimensional data."""
        data = np.random.randn(100, 3)  # 3-channel data
        features = self.synchronizer._extract_statistical_features(data, "test")
        
        assert "mean" in features
        assert "std" in features
        assert isinstance(features["mean"], float)
    
    def test_statistical_features_empty_data(self):
        """Test statistical features with empty data."""
        data = np.array([])
        features = self.synchronizer._extract_statistical_features(data, "test")
        assert len(features) == 0
    
    def test_statistical_features_all_nan(self):
        """Test statistical features with all NaN data."""
        data = np.array([np.nan, np.nan, np.nan])
        features = self.synchronizer._extract_statistical_features(data, "test")
        assert len(features) == 0
    
    def test_window_features_extraction(self):
        """Test temporal window feature extraction."""
        data = np.random.randn(1000)  # Long enough for multiple windows
        windows = self.synchronizer._extract_window_features(data, "test")
        
        # Should have different window sizes
        window_keys = [key for key in windows.keys() if "window" in key]
        assert len(window_keys) > 0
        
        for key in window_keys:
            assert isinstance(windows[key], np.ndarray)
            assert windows[key].shape[1] == 3  # mean, std, range
    
    def test_cross_modal_correlations(self):
        """Test cross-modal correlation calculation."""
        # Create correlated data
        base_signal = np.random.randn(100)
        data_streams = {
            "EEG": base_signal + 0.1 * np.random.randn(100),
            "HeartRate": base_signal + 0.2 * np.random.randn(100)
        }
        
        correlations = self.synchronizer._calculate_cross_modal_correlations(data_streams)
        
        assert "EEG_HeartRate" in correlations
        assert -1.0 <= correlations["EEG_HeartRate"] <= 1.0
    
    def test_cross_modal_correlations_multidimensional(self):
        """Test correlations with multi-dimensional data."""
        data_streams = {
            "EEG": np.random.randn(100, 8),  # 8 channels
            "Gait": np.random.randn(100, 3)  # 3 axes
        }
        
        correlations = self.synchronizer._calculate_cross_modal_correlations(data_streams)
        
        assert "EEG_Gait" in correlations
        assert isinstance(correlations["EEG_Gait"], float)
    
    def test_cross_modal_correlations_insufficient_data(self):
        """Test correlations with insufficient valid data points."""
        data_streams = {
            "EEG": np.array([np.nan] * 5),  # All NaN
            "HeartRate": np.array([1, 2, 3, 4, 5])
        }
        
        correlations = self.synchronizer._calculate_cross_modal_correlations(data_streams)
        
        # Should not compute correlation due to insufficient valid data
        assert len(correlations) == 0
    
    def test_find_missing_regions(self):
        """Test finding continuous missing data regions."""
        missing_mask = np.array([False, True, True, False, False, True, False])
        regions = self.synchronizer._find_missing_regions(missing_mask)
        
        expected_regions = [(1, 3), (5, 6)]
        assert regions == expected_regions
    
    def test_find_missing_regions_at_boundaries(self):
        """Test finding missing regions at data boundaries."""
        missing_mask = np.array([True, True, False, False, True, True])
        regions = self.synchronizer._find_missing_regions(missing_mask)
        
        expected_regions = [(0, 2), (4, 6)]
        assert regions == expected_regions
    
    def test_interpolate_region_linear(self):
        """Test linear interpolation of missing region."""
        data = np.array([1.0, np.nan, np.nan, 4.0])
        result = self.synchronizer._interpolate_region(data, 1, 3, "linear")
        
        # Should interpolate linearly between 1.0 and 4.0
        expected = np.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_interpolate_region_at_start(self):
        """Test interpolation at data start (forward fill)."""
        data = np.array([np.nan, np.nan, 3.0, 4.0])
        result = self.synchronizer._interpolate_region(data, 0, 2, "linear")
        
        # Should forward fill with first valid value
        expected = np.array([3.0, 3.0, 3.0, 4.0])
        np.testing.assert_array_equal(result, expected)
    
    def test_interpolate_region_at_end(self):
        """Test interpolation at data end (backward fill)."""
        data = np.array([1.0, 2.0, np.nan, np.nan])
        result = self.synchronizer._interpolate_region(data, 2, 4, "linear")
        
        # Should backward fill with last valid value
        expected = np.array([1.0, 2.0, 2.0, 2.0])
        np.testing.assert_array_equal(result, expected)
    
    def test_calculate_skewness(self):
        """Test skewness calculation."""
        # Symmetric data should have near-zero skewness
        symmetric_data = np.array([-2, -1, 0, 1, 2])
        skewness = self.synchronizer._calculate_skewness(symmetric_data)
        assert abs(skewness) < 0.1
        
        # Right-skewed data should have positive skewness
        right_skewed = np.array([1, 1, 1, 2, 10])
        skewness = self.synchronizer._calculate_skewness(right_skewed)
        assert skewness > 0
    
    def test_calculate_kurtosis(self):
        """Test kurtosis calculation."""
        # Normal-like data should have near-zero excess kurtosis
        normal_data = np.random.randn(1000)
        kurtosis = self.synchronizer._calculate_kurtosis(normal_data)
        assert -1 < kurtosis < 1  # Should be close to 0 for normal distribution
    
    def test_count_zero_crossings(self):
        """Test zero crossing count."""
        # Sine wave should have predictable zero crossings
        t = np.linspace(0, 4*np.pi, 1000)
        sine_wave = np.sin(t)
        crossings = self.synchronizer._count_zero_crossings(sine_wave)
        
        # Should have some zero crossings (exact count depends on discretization)
        assert crossings >= 2  # At least some crossings should be detected
    
    def test_synchronization_quality_calculation(self):
        """Test synchronization quality score calculation."""
        sessions = [self.eeg_session, self.hr_session]
        interpolated_gaps = {"session1": [(0, 10)], "session2": []}
        
        quality = self.synchronizer._calculate_synchronization_quality(sessions, interpolated_gaps)
        
        assert 0.0 <= quality <= 1.0
        # Quality should be reduced due to interpolated gaps
        assert quality < 1.0
    
    def test_alignment_quality_calculation(self):
        """Test alignment quality calculation."""
        data = np.random.randn(100)
        interpolated_regions = [(10, 20), (50, 55)]  # 15 out of 100 samples interpolated
        
        quality = self.synchronizer._calculate_alignment_quality(data, interpolated_regions)
        
        assert 0.0 <= quality <= 1.0
        # Quality should be reduced due to interpolation
        assert quality < 1.0
    
    def test_alignment_quality_no_interpolation(self):
        """Test alignment quality with no interpolation."""
        data = np.random.randn(100)
        interpolated_regions = []
        
        quality = self.synchronizer._calculate_alignment_quality(data, interpolated_regions)
        
        assert quality == 1.0
    
    def test_edge_case_very_short_data(self):
        """Test handling of very short data sequences."""
        short_data = np.array([1.0, 2.0])
        features = self.synchronizer._extract_statistical_features(short_data, "test")
        
        # Should still compute basic statistics
        assert "mean" in features
        assert features["mean"] == 1.5
    
    def test_edge_case_single_value_data(self):
        """Test handling of single-value data."""
        single_data = np.array([5.0])
        features = self.synchronizer._extract_statistical_features(single_data, "test")
        
        assert features["mean"] == 5.0
        assert features["std"] == 0.0
        assert features["min"] == features["max"] == 5.0


class TestSynchronizationConfig:
    """Test cases for SynchronizationConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = SynchronizationConfig()
        
        assert config.target_sampling_rate == 1.0
        assert config.interpolation_method == "linear"
        assert config.max_gap_seconds == 300.0
        assert config.alignment_tolerance_seconds == 1.0
        assert config.min_overlap_seconds == 60.0
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = SynchronizationConfig(
            target_sampling_rate=2.0,
            interpolation_method="cubic",
            max_gap_seconds=120.0
        )
        
        assert config.target_sampling_rate == 2.0
        assert config.interpolation_method == "cubic"
        assert config.max_gap_seconds == 120.0


class TestSynchronizationResult:
    """Test cases for SynchronizationResult dataclass."""
    
    def test_initialization(self):
        """Test result initialization."""
        sessions = []
        time_range = (datetime.now(), datetime.now())
        gaps = {}
        
        result = SynchronizationResult(
            synchronized_sessions=sessions,
            common_time_range=time_range,
            interpolated_gaps=gaps,
            synchronization_quality=0.95
        )
        
        assert result.synchronized_sessions == sessions
        assert result.common_time_range == time_range
        assert result.interpolated_gaps == gaps
        assert result.synchronization_quality == 0.95
        assert result.warnings == []
        assert result.errors == []


class TestAlignedSensorData:
    """Test cases for AlignedSensorData dataclass."""
    
    def test_initialization(self):
        """Test aligned data initialization."""
        timestamps = np.array([0, 1, 2, 3, 4])
        data_streams = {"EEG": np.random.randn(5, 8)}
        time_range = (datetime.now(), datetime.now())
        
        aligned_data = AlignedSensorData(
            timestamps=timestamps,
            data_streams=data_streams,
            sampling_rate=1.0,
            time_range=time_range,
            quality_metrics={"EEG": 0.9},
            interpolated_regions={"EEG": [(1, 2)]}
        )
        
        assert len(aligned_data.timestamps) == 5
        assert "EEG" in aligned_data.data_streams
        assert aligned_data.sampling_rate == 1.0


class TestTemporalFeatures:
    """Test cases for TemporalFeatures dataclass."""
    
    def test_initialization(self):
        """Test temporal features initialization."""
        feature_vector = np.array([1.0, 2.0, 3.0])
        feature_names = ["feature1", "feature2", "feature3"]
        
        features = TemporalFeatures(
            feature_vector=feature_vector,
            feature_names=feature_names,
            temporal_windows={},
            cross_modal_correlations={},
            temporal_statistics={}
        )
        
        assert len(features.feature_vector) == 3
        assert len(features.feature_names) == 3
        np.testing.assert_array_equal(features.feature_vector, feature_vector)