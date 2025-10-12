"""
Unit tests for sensor data collection interface.

Tests the SensorDataCollector class and device-specific collectors for EEG,
heart rate, sleep, and gait data with various input scenarios and validation.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import json
import csv

from src.services.wearable_sensor import (
    SensorDataCollector,
    SensorDataQuality,
    SensorDataValidationResult
)
from src.models.patient import WearableSession


class TestSensorDataCollector:
    """Test cases for the main SensorDataCollector class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.collector = SensorDataCollector()
        self.start_time = datetime(2024, 1, 1, 10, 0, 0)
        self.end_time = datetime(2024, 1, 1, 11, 0, 0)
    
    def test_supported_device_types(self):
        """Test that all expected device types are supported."""
        expected_types = ["EEG", "HeartRate", "Sleep", "Gait"]
        for device_type in expected_types:
            assert device_type in self.collector.collectors
    
    def test_get_supported_formats(self):
        """Test getting supported file formats for each device type."""
        formats = self.collector.get_supported_formats("EEG")
        assert ".edf" in formats
        assert ".csv" in formats
        
        formats = self.collector.get_supported_formats("HeartRate")
        assert ".csv" in formats
        assert ".json" in formats
    
    def test_get_quality_thresholds(self):
        """Test getting quality thresholds for each device type."""
        thresholds = self.collector.get_quality_thresholds("EEG")
        assert "min_quality_score" in thresholds
        assert "max_missing_percentage" in thresholds
        assert "min_sampling_rate" in thresholds
        assert "max_sampling_rate" in thresholds
    
    def test_unsupported_device_type(self):
        """Test error handling for unsupported device types."""
        with pytest.raises(ValueError, match="Unsupported device type"):
            self.collector.collect_sensor_data(
                "InvalidDevice", {}, self.start_time, self.end_time
            )


class TestEEGCollector:
    """Test cases for EEG data collection."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.collector = EEGCollector()
        self.start_time = datetime(2024, 1, 1, 10, 0, 0)
        self.end_time = datetime(2024, 1, 1, 10, 10, 0)
    
    def test_collect_data_from_dict(self):
        """Test collecting EEG data from dictionary."""
        # Create synthetic EEG data (10 channels, 1000 samples)
        eeg_data = np.random.randn(10, 1000) * 50  # Typical EEG amplitude range
        
        data_dict = {
            "data": eeg_data,
            "sampling_rate": 256.0,
            "metadata": {
                "manufacturer": "TestDevice",
                "model": "EEG-1000"
            }
        }
        
        session = self.collector.collect_data(data_dict, self.start_time, self.end_time)
        
        assert isinstance(session, WearableSession)
        assert session.device_type == "EEG"
        assert session.sampling_rate == 256.0
        assert session.raw_data.shape == (10, 1000)
        assert session.device_manufacturer == "TestDevice"
        assert session.device_model == "EEG-1000"
    
    def test_validate_data_valid_eeg(self):
        """Test validation of valid EEG data."""
        # Create valid EEG data
        eeg_data = np.random.randn(10, 1000) * 50
        sampling_rate = 256.0
        
        result = self.collector.validate_data(eeg_data, sampling_rate)
        
        assert result.is_valid
        assert result.quality_metrics.quality_score > 0.8
        assert len(result.validation_errors) == 0
    
    def test_validate_data_invalid_sampling_rate(self):
        """Test validation with invalid sampling rate."""
        eeg_data = np.random.randn(10, 1000)
        invalid_sampling_rate = 50.0  # Below minimum
        
        result = self.collector.validate_data(eeg_data, invalid_sampling_rate)
        
        assert not result.is_valid
        assert len(result.validation_errors) > 0
        assert "Invalid sampling rate" in result.validation_errors[0]
    
    def test_validate_data_wrong_dimensions(self):
        """Test validation with wrong data dimensions."""
        eeg_data = np.random.randn(1000)  # 1D instead of 2D
        sampling_rate = 256.0
        
        result = self.collector.validate_data(eeg_data, sampling_rate)
        
        assert not result.is_valid
        assert "must be 2D" in result.validation_errors[0]
    
    def test_validate_data_high_missing_data(self):
        """Test validation with high percentage of missing data."""
        eeg_data = np.random.randn(10, 1000)
        eeg_data[:, :100] = np.nan  # 10% missing data
        sampling_rate = 256.0
        
        result = self.collector.validate_data(eeg_data, sampling_rate)
        
        assert result.is_valid  # Still valid but with warnings
        assert len(result.validation_warnings) > 0
        assert "high_missing_data" in result.quality_metrics.quality_flags


class TestHeartRateCollector:
    """Test cases for heart rate data collection."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.collector = HeartRateCollector()
        self.start_time = datetime(2024, 1, 1, 10, 0, 0)
        self.end_time = datetime(2024, 1, 1, 11, 0, 0)
    
    def test_collect_data_from_dict(self):
        """Test collecting heart rate data from dictionary."""
        # Create synthetic heart rate data (60-100 BPM)
        hr_data = np.random.uniform(60, 100, 3600)  # 1 hour of data at 1Hz
        
        data_dict = {
            "heart_rate": hr_data,
            "sampling_rate": 1.0,
            "metadata": {
                "manufacturer": "FitBit",
                "model": "Charge-5"
            }
        }
        
        session = self.collector.collect_data(data_dict, self.start_time, self.end_time)
        
        assert isinstance(session, WearableSession)
        assert session.device_type == "HeartRate"
        assert session.sampling_rate == 1.0
        assert len(session.raw_data) == 3600
        assert "mean_hr" in session.processed_features
        assert "std_hr" in session.processed_features
    
    def test_extract_features(self):
        """Test heart rate feature extraction."""
        hr_data = np.array([70, 72, 68, 75, 73, 71, 69, 74])
        features = self.collector._extract_features(hr_data)
        
        assert "mean_hr" in features
        assert "std_hr" in features
        assert "min_hr" in features
        assert "max_hr" in features
        assert "rmssd" in features
        
        assert features["mean_hr"] == pytest.approx(71.5, rel=1e-2)
        assert features["min_hr"] == 68
        assert features["max_hr"] == 75
    
    def test_validate_data_abnormal_values(self):
        """Test validation with abnormal heart rate values."""
        hr_data = np.array([300, 350, 400])  # Abnormally high values
        sampling_rate = 1.0
        
        result = self.collector.validate_data(hr_data, sampling_rate)
        
        assert result.is_valid  # Valid but with warnings
        assert len(result.validation_warnings) > 0
        assert "abnormal_values" in result.quality_metrics.quality_flags


class TestSleepCollector:
    """Test cases for sleep data collection."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.collector = SleepCollector()
        self.start_time = datetime(2024, 1, 1, 22, 0, 0)
        self.end_time = datetime(2024, 1, 2, 6, 0, 0)
    
    def test_collect_data_from_dict(self):
        """Test collecting sleep data from dictionary."""
        # Create synthetic sleep stage data (8 hours, 1 epoch per minute)
        sleep_stages = np.random.choice([0, 1, 2, 3], size=480)  # 8 hours * 60 minutes
        
        data_dict = {
            "sleep_stages": sleep_stages,
            "sampling_rate": 1/60,  # 1 epoch per minute
            "metadata": {
                "manufacturer": "Oura",
                "model": "Ring-3"
            }
        }
        
        session = self.collector.collect_data(data_dict, self.start_time, self.end_time)
        
        assert isinstance(session, WearableSession)
        assert session.device_type == "Sleep"
        assert session.sampling_rate == 1/60
        assert len(session.raw_data) == 480
        assert "sleep_efficiency" in session.processed_features
        assert "total_sleep_time" in session.processed_features
    
    def test_extract_features(self):
        """Test sleep feature extraction."""
        # Create sleep data: awake(0), light(1), deep(2), rem(3)
        sleep_data = np.array([0, 0, 1, 1, 1, 2, 2, 3, 3, 1, 1, 0])
        features = self.collector._extract_features(sleep_data)
        
        assert "sleep_efficiency" in features
        assert "awake_percentage" in features
        assert "light_percentage" in features
        assert "deep_percentage" in features
        assert "rem_percentage" in features
        
        # Check percentages add up to 100%
        total_percentage = (
            features["awake_percentage"] + 
            features["light_percentage"] + 
            features["deep_percentage"] + 
            features["rem_percentage"]
        )
        assert total_percentage == pytest.approx(100.0, rel=1e-2)


class TestGaitCollector:
    """Test cases for gait data collection."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.collector = GaitCollector()
        self.start_time = datetime(2024, 1, 1, 10, 0, 0)
        self.end_time = datetime(2024, 1, 1, 10, 30, 0)
    
    def test_collect_data_from_dict(self):
        """Test collecting gait data from dictionary."""
        # Create synthetic 3-axis accelerometer data
        duration_seconds = 1800  # 30 minutes
        sampling_rate = 100.0
        n_samples = int(duration_seconds * sampling_rate)
        
        accel_data = np.random.randn(n_samples, 3) * 2  # 3-axis accelerometer
        
        data_dict = {
            "acceleration": accel_data,
            "sampling_rate": sampling_rate,
            "metadata": {
                "manufacturer": "Apple",
                "model": "Watch-Series-8"
            }
        }
        
        session = self.collector.collect_data(data_dict, self.start_time, self.end_time)
        
        assert isinstance(session, WearableSession)
        assert session.device_type == "Gait"
        assert session.sampling_rate == sampling_rate
        assert session.raw_data.shape == (n_samples, 3)
        assert "mean_acceleration" in session.processed_features
        assert "std_acceleration" in session.processed_features
    
    def test_extract_features(self):
        """Test gait feature extraction."""
        # Create simple accelerometer data
        accel_data = np.array([
            [1.0, 0.0, 9.8],
            [1.5, 0.2, 9.9],
            [0.8, -0.1, 9.7],
            [1.2, 0.1, 9.8]
        ])
        
        features = self.collector._extract_features(accel_data, 100.0)
        
        assert "mean_acceleration" in features
        assert "std_acceleration" in features
        assert "max_acceleration" in features
        assert "min_acceleration" in features
        
        # Check that magnitude calculation is reasonable
        assert features["mean_acceleration"] > 9.0  # Should be close to gravity
        assert features["mean_acceleration"] < 11.0
    
    def test_validate_data_high_acceleration(self):
        """Test validation with abnormally high acceleration values."""
        accel_data = np.array([[500, 0, 0], [600, 0, 0]])  # Very high values
        sampling_rate = 100.0
        
        result = self.collector.validate_data(accel_data, sampling_rate)
        
        assert result.is_valid  # Valid but with warnings
        assert len(result.validation_warnings) > 0
        assert "high_acceleration" in result.quality_metrics.quality_flags


class TestFileIO:
    """Test cases for file input/output operations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.hr_collector = HeartRateCollector()
        self.start_time = datetime(2024, 1, 1, 10, 0, 0)
        self.end_time = datetime(2024, 1, 1, 11, 0, 0)
    
    def test_load_heart_rate_from_csv(self):
        """Test loading heart rate data from CSV file."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'heart_rate'])
            
            for i in range(100):
                timestamp = i * 1.0  # 1 second intervals
                hr = 70 + np.sin(i * 0.1) * 10  # Simulated heart rate
                writer.writerow([timestamp, hr])
            
            csv_path = f.name
        
        try:
            raw_data, sampling_rate, metadata = self.hr_collector._load_csv(Path(csv_path))
            
            assert len(raw_data) == 100
            assert sampling_rate == pytest.approx(1.0, rel=1e-1)
            assert metadata["format"] == "csv"
            
        finally:
            Path(csv_path).unlink()  # Clean up
    
    def test_load_heart_rate_from_json(self):
        """Test loading heart rate data from JSON file."""
        # Create temporary JSON file
        hr_data = [70 + np.sin(i * 0.1) * 10 for i in range(100)]
        
        data = {
            "heart_rate": hr_data,
            "sampling_rate": 1.0,
            "metadata": {
                "device": "test_device"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            json_path = f.name
        
        try:
            raw_data, sampling_rate, metadata = self.hr_collector._load_json(Path(json_path))
            
            assert len(raw_data) == 100
            assert sampling_rate == 1.0
            assert metadata["format"] == "json"
            assert metadata["device"] == "test_device"
            
        finally:
            Path(json_path).unlink()  # Clean up


class TestDataQuality:
    """Test cases for data quality assessment."""
    
    def test_sensor_data_quality_initialization(self):
        """Test SensorDataQuality initialization."""
        quality = SensorDataQuality()
        
        assert quality.signal_to_noise_ratio is None
        assert quality.missing_data_percentage == 0.0
        assert quality.artifact_percentage == 0.0
        assert quality.quality_score == 1.0
        assert quality.quality_flags == []
    
    def test_validation_result_initialization(self):
        """Test SensorDataValidationResult initialization."""
        quality = SensorDataQuality(quality_score=0.8)
        result = SensorDataValidationResult(
            is_valid=True,
            quality_metrics=quality
        )
        
        assert result.is_valid
        assert result.quality_metrics.quality_score == 0.8
        assert result.validation_errors == []
        assert result.validation_warnings == []