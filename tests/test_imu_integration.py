# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 Ahmed Essam <aessam.dahy@gmail.com>

"""Unit tests for IMU integration module."""

import numpy as np
import pytest
from scipy import signal

from blender_addon import imu_integration


@pytest.mark.unit
@pytest.mark.imu
class TestIMUSample:
    """Test IMUSample dataclass."""
    
    def test_imusample_creation(self):
        """Test creating an IMU sample."""
        sample = imu_integration.IMUSample(
            timestamp_ns=1000000000,
            accel=np.array([0.0, 0.0, -9.81]),
            gyro=np.array([0.01, 0.02, 0.03])
        )
        assert sample.timestamp_ns == 1000000000
        assert np.allclose(sample.accel, [0.0, 0.0, -9.81])
        assert np.allclose(sample.gyro, [0.01, 0.02, 0.03])


@pytest.mark.unit
@pytest.mark.imu
class TestIMUData:
    """Test IMUData container."""
    
    def test_imu_data_creation(self, sample_imu_data, sample_video_timestamps):
        """Test creating IMUData from samples."""
        samples = [
            imu_integration.IMUSample(
                timestamp_ns=int(ts),
                accel=accel,
                gyro=gyro
            )
            for ts, accel, gyro in zip(
                sample_imu_data['timestamps'],
                sample_imu_data['accel'],
                sample_imu_data['gyro']
            )
        ]
        
        imu_data = imu_integration.IMUData(
            samples=samples,
            video_timestamps=sample_video_timestamps
        )
        
        assert len(imu_data) == len(samples)
        assert len(imu_data.video_timestamps) == len(sample_video_timestamps)
    
    def test_get_sample_at_timestamp(self, sample_imu_data):
        """Test getting sample closest to timestamp."""
        samples = [
            imu_integration.IMUSample(
                timestamp_ns=int(ts),
                accel=accel,
                gyro=gyro
            )
            for ts, accel, gyro in zip(
                sample_imu_data['timestamps'],
                sample_imu_data['accel'],
                sample_imu_data['gyro']
            )
        ]
        
        imu_data = imu_integration.IMUData(
            samples=samples,
            video_timestamps=np.array([1000000000])
        )
        
        # Test exact match
        sample = imu_data.get_sample_at_timestamp(samples[500].timestamp_ns)
        assert sample is not None
        assert sample.timestamp_ns == samples[500].timestamp_ns
        
        # Test closest match
        mid_timestamp = (samples[100].timestamp_ns + samples[101].timestamp_ns) // 2
        sample = imu_data.get_sample_at_timestamp(mid_timestamp)
        assert sample is not None
    
    def test_interpolate_at_timestamp(self, sample_imu_data):
        """Test interpolating IMU data at timestamp."""
        samples = [
            imu_integration.IMUSample(
                timestamp_ns=int(ts),
                accel=accel,
                gyro=gyro
            )
            for ts, accel, gyro in zip(
                sample_imu_data['timestamps'][:10],  # Use fewer samples for speed
                sample_imu_data['accel'][:10],
                sample_imu_data['gyro'][:10]
            )
        ]
        
        imu_data = imu_integration.IMUData(
            samples=samples,
            video_timestamps=np.array([1000000000])
        )
        
        # Interpolate at middle timestamp
        mid_timestamp = (samples[4].timestamp_ns + samples[5].timestamp_ns) // 2
        sample = imu_data.interpolate_at_timestamp(mid_timestamp)
        
        assert sample is not None
        assert sample.timestamp_ns == mid_timestamp
        # Interpolated values should be between the two samples
        assert np.all(np.abs(sample.accel) < 20.0)  # Reasonable bounds


@pytest.mark.unit
@pytest.mark.imu
class TestIMUProcessor:
    """Test IMUProcessor class."""
    
    def test_processor_initialization(self, sample_imu_data, sample_video_timestamps):
        """Test initializing IMU processor."""
        samples = [
            imu_integration.IMUSample(
                timestamp_ns=int(ts),
                accel=accel,
                gyro=gyro
            )
            for ts, accel, gyro in zip(
                sample_imu_data['timestamps'],
                sample_imu_data['accel'],
                sample_imu_data['gyro']
            )
        ]
        
        imu_data = imu_integration.IMUData(
            samples=samples,
            video_timestamps=sample_video_timestamps
        )
        
        processor = imu_integration.IMUProcessor(imu_data)
        assert processor is not None
        assert processor._gravity_vector is not None
        assert processor._gravity_vector_normalized is not None
    
    def test_gravity_vector_extraction(self, sample_imu_data, sample_video_timestamps):
        """Test gravity vector extraction from accelerometer."""
        samples = [
            imu_integration.IMUSample(
                timestamp_ns=int(ts),
                accel=accel,
                gyro=gyro
            )
            for ts, accel, gyro in zip(
                sample_imu_data['timestamps'],
                sample_imu_data['accel'],
                sample_imu_data['gyro']
            )
        ]
        
        imu_data = imu_integration.IMUData(
            samples=samples,
            video_timestamps=sample_video_timestamps
        )
        
        processor = imu_integration.IMUProcessor(imu_data)
        gravity = processor.get_gravity_vector()
        
        # Gravity should point downward (negative Z in Blender)
        assert gravity.shape == (3,)
        assert np.linalg.norm(gravity) > 0.9  # Should be normalized
        assert gravity[2] < 0  # Should point down
    
    def test_gravity_consistency(self, sample_imu_data, sample_video_timestamps):
        """Test gravity consistency calculation."""
        samples = [
            imu_integration.IMUSample(
                timestamp_ns=int(ts),
                accel=accel,
                gyro=gyro
            )
            for ts, accel, gyro in zip(
                sample_imu_data['timestamps'],
                sample_imu_data['accel'],
                sample_imu_data['gyro']
            )
        ]
        
        imu_data = imu_integration.IMUData(
            samples=samples,
            video_timestamps=sample_video_timestamps
        )
        
        processor = imu_integration.IMUProcessor(imu_data)
        consistency = processor.get_gravity_consistency()
        
        # Consistency should be between 0 and 1
        assert 0.0 <= consistency <= 1.0
    
    def test_gyro_drift(self, sample_imu_data, sample_video_timestamps):
        """Test gyroscope drift calculation."""
        samples = [
            imu_integration.IMUSample(
                timestamp_ns=int(ts),
                accel=accel,
                gyro=gyro
            )
            for ts, accel, gyro in zip(
                sample_imu_data['timestamps'],
                sample_imu_data['accel'],
                sample_imu_data['gyro']
            )
        ]
        
        imu_data = imu_integration.IMUData(
            samples=samples,
            video_timestamps=sample_video_timestamps
        )
        
        processor = imu_integration.IMUProcessor(imu_data)
        drift = processor.get_gyro_drift()
        
        # Drift should be non-negative
        assert drift >= 0.0
    
    def test_orientation_from_gravity(self, sample_imu_data, sample_video_timestamps):
        """Test computing orientation from gravity vector."""
        from blender_addon import imu_integration
        mathutils = imu_integration.mathutils
        
        samples = [
            imu_integration.IMUSample(
                timestamp_ns=int(ts),
                accel=accel,
                gyro=gyro
            )
            for ts, accel, gyro in zip(
                sample_imu_data['timestamps'],
                sample_imu_data['accel'],
                sample_imu_data['gyro']
            )
        ]
        
        imu_data = imu_integration.IMUData(
            samples=samples,
            video_timestamps=sample_video_timestamps
        )
        
        processor = imu_integration.IMUProcessor(imu_data)
        orientation = processor.get_orientation_from_gravity()
        
        assert isinstance(orientation, mathutils.Quaternion)
        # Quaternion should be normalized
        assert abs(orientation.magnitude - 1.0) < 0.01
    
    def test_constrain_z_axis_to_gravity(self, sample_imu_data, sample_video_timestamps):
        """Test Z-axis constraint to gravity."""
        from blender_addon import imu_integration
        mathutils = imu_integration.mathutils
        
        samples = [
            imu_integration.IMUSample(
                timestamp_ns=int(ts),
                accel=accel,
                gyro=gyro
            )
            for ts, accel, gyro in zip(
                sample_imu_data['timestamps'],
                sample_imu_data['accel'],
                sample_imu_data['gyro']
            )
        ]
        
        imu_data = imu_integration.IMUData(
            samples=samples,
            video_timestamps=sample_video_timestamps
        )
        
        processor = imu_integration.IMUProcessor(imu_data)
        initial_orientation = mathutils.Quaternion((1, 0, 0, 0))
        
        # Test with weight 0 (should return original)
        constrained = processor.constrain_z_axis_to_gravity(initial_orientation, weight=0.0)
        assert constrained == initial_orientation
        
        # Test with weight 1.0 (should return gravity-aligned)
        constrained = processor.constrain_z_axis_to_gravity(initial_orientation, weight=1.0)
        assert isinstance(constrained, mathutils.Quaternion)
        assert abs(constrained.magnitude - 1.0) < 0.01


@pytest.mark.unit
@pytest.mark.imu
class TestIMULoading:
    """Test IMU data loading functions."""
    
    def test_load_opencamera_csv(self, opencamera_csv_files):
        """Test loading OpenCamera-Sensors CSV format."""
        imu_data = imu_integration.load_opencamera_csv(
            opencamera_csv_files['accel_path'],
            opencamera_csv_files['gyro_path'],
            opencamera_csv_files['timestamps_path']
        )
        
        assert imu_data is not None
        assert len(imu_data) > 0
        assert len(imu_data.video_timestamps) > 0
    
    def test_load_opencamera_csv_missing_file(self, temp_dir):
        """Test loading with missing files."""
        imu_data = imu_integration.load_opencamera_csv(
            str(temp_dir / "nonexistent_accel.csv"),
            str(temp_dir / "nonexistent_gyro.csv"),
            str(temp_dir / "nonexistent_timestamps.csv")
        )
        
        assert imu_data is None
    
    def test_load_opencamera_csv_invalid_format(self, temp_dir):
        """Test loading with invalid CSV format."""
        # Create invalid CSV file
        invalid_path = temp_dir / "invalid.csv"
        invalid_path.write_text("invalid,data\n1,2\n")
        
        imu_data = imu_integration.load_opencamera_csv(
            str(invalid_path),
            str(invalid_path),
            str(invalid_path)
        )
        
        assert imu_data is None
    
    def test_detect_camm_in_mp4_nonexistent_file(self, temp_dir):
        """Test CAMM detection with non-existent file."""
        video_path = str(temp_dir / "nonexistent.mp4")
        imu_data = imu_integration.detect_camm_in_mp4(video_path)
        assert imu_data is None
    
    def test_detect_camm_in_mp4_no_camm(self, temp_dir):
        """Test CAMM detection with file that has no CAMM data."""
        # Create empty file (no CAMM data)
        video_path = temp_dir / "test.mp4"
        video_path.write_bytes(b"fake mp4 data")
        
        imu_data = imu_integration.detect_camm_in_mp4(str(video_path))
        # Should return None when no CAMM is found
        assert imu_data is None


@pytest.mark.unit
@pytest.mark.imu
class TestIMUEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_imu_data(self):
        """Test handling of empty IMU data."""
        imu_data = imu_integration.IMUData(
            samples=[],
            video_timestamps=np.array([])
        )
        
        assert len(imu_data) == 0
        assert imu_data.get_sample_at_timestamp(1000000000) is None
        assert imu_data.interpolate_at_timestamp(1000000000) is None
    
    def test_single_sample(self):
        """Test handling of single IMU sample."""
        sample = imu_integration.IMUSample(
            timestamp_ns=1000000000,
            accel=np.array([0.0, 0.0, -9.81]),
            gyro=np.array([0.0, 0.0, 0.0])
        )
        
        imu_data = imu_integration.IMUData(
            samples=[sample],
            video_timestamps=np.array([1000000000])
        )
        
        retrieved = imu_data.get_sample_at_timestamp(1000000000)
        assert retrieved is not None
        assert retrieved.timestamp_ns == sample.timestamp_ns
    
    def test_timestamp_out_of_range(self, sample_imu_data):
        """Test handling of timestamps outside data range."""
        samples = [
            imu_integration.IMUSample(
                timestamp_ns=int(ts),
                accel=accel,
                gyro=gyro
            )
            for ts, accel, gyro in zip(
                sample_imu_data['timestamps'][:10],
                sample_imu_data['accel'][:10],
                sample_imu_data['gyro'][:10]
            )
        ]
        
        imu_data = imu_integration.IMUData(
            samples=samples,
            video_timestamps=np.array([1000000000])
        )
        
        # Test timestamp before range
        sample = imu_data.get_sample_at_timestamp(0)
        assert sample is not None
        assert sample.timestamp_ns == samples[0].timestamp_ns
        
        # Test timestamp after range
        sample = imu_data.get_sample_at_timestamp(999999999999)
        assert sample is not None
        assert sample.timestamp_ns == samples[-1].timestamp_ns

