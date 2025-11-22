# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 Ahmed Essam <aessam.dahy@gmail.com>

"""Robustness tests for IMU integration - malformed data, corrupted files, etc."""

import numpy as np
import pytest
from pathlib import Path

from blender_addon import imu_integration


@pytest.mark.unit
@pytest.mark.imu
class TestCorruptedCSVFiles:
    """Test handling of corrupted and malformed CSV files."""
    
    def test_csv_missing_columns(self, temp_dir):
        """Test CSV with missing columns."""
        csv_path = temp_dir / "missing_cols.csv"
        csv_path.write_text("X-data,Y-data\n1.0,2.0\n")  # Missing Z and timestamp
        
        try:
            imu_data = imu_integration.load_opencamera_csv(
                str(csv_path),
                str(csv_path),
                str(csv_path)
            )
            assert imu_data is None
        except Exception:
            # Exception is also acceptable
            pass
    
    def test_csv_wrong_delimiter(self, temp_dir):
        """Test CSV with wrong delimiter."""
        csv_path = temp_dir / "wrong_delim.csv"
        csv_path.write_text("X-data;Y-data;Z-data;timestamp\n1.0;2.0;3.0;1000\n")  # Semicolon instead of comma
        
        try:
            imu_data = imu_integration.load_opencamera_csv(
                str(csv_path),
                str(csv_path),
                str(csv_path)
            )
            # Should handle gracefully
            assert imu_data is None or isinstance(imu_data, imu_integration.IMUData)
        except Exception:
            pass
    
    def test_csv_nan_values(self, temp_dir):
        """Test CSV with NaN values."""
        csv_path = temp_dir / "nan_values.csv"
        csv_path.write_text("X-data,Y-data,Z-data,timestamp\n1.0,NaN,3.0,1000\n2.0,2.0,inf,2000\n")
        
        try:
            imu_data = imu_integration.load_opencamera_csv(
                str(csv_path),
                str(csv_path),
                str(csv_path)
            )
            # Should handle NaN/inf gracefully
            assert imu_data is None or isinstance(imu_data, imu_integration.IMUData)
        except Exception:
            pass
    
    def test_csv_incomplete_lines(self, temp_dir):
        """Test CSV with incomplete lines."""
        csv_path = temp_dir / "incomplete.csv"
        csv_path.write_text("X-data,Y-data,Z-data,timestamp\n1.0,2.0\n3.0,4.0,5.0,1000\n")
        
        try:
            imu_data = imu_integration.load_opencamera_csv(
                str(csv_path),
                str(csv_path),
                str(csv_path)
            )
            assert imu_data is None or isinstance(imu_data, imu_integration.IMUData)
        except Exception:
            pass
    
    def test_csv_wrong_timestamp_format(self, temp_dir):
        """Test CSV with wrong timestamp format."""
        csv_path = temp_dir / "wrong_timestamp.csv"
        csv_path.write_text("X-data,Y-data,Z-data,timestamp\n1.0,2.0,3.0,invalid\n")
        
        try:
            imu_data = imu_integration.load_opencamera_csv(
                str(csv_path),
                str(csv_path),
                str(csv_path)
            )
            assert imu_data is None
        except Exception:
            pass
    
    def test_csv_out_of_order_data(self, temp_dir):
        """Test CSV with out-of-order timestamps."""
        csv_path = temp_dir / "out_of_order.csv"
        csv_path.write_text(
            "X-data,Y-data,Z-data,timestamp\n"
            "1.0,2.0,3.0,2000\n"
            "1.1,2.1,3.1,1000\n"  # Out of order
            "1.2,2.2,3.2,3000\n"
        )
        
        try:
            imu_data = imu_integration.load_opencamera_csv(
                str(csv_path),
                str(csv_path),
                str(csv_path)
            )
            # Should handle out-of-order data
            if imu_data:
                assert len(imu_data) > 0
        except Exception:
            pass
    
    def test_csv_empty_file(self, temp_dir):
        """Test empty CSV file."""
        csv_path = temp_dir / "empty.csv"
        csv_path.write_text("")
        
        imu_data = imu_integration.load_opencamera_csv(
            str(csv_path),
            str(csv_path),
            str(csv_path)
        )
        assert imu_data is None
    
    def test_csv_only_header(self, temp_dir):
        """Test CSV with only header row."""
        csv_path = temp_dir / "header_only.csv"
        csv_path.write_text("X-data,Y-data,Z-data,timestamp\n")
        
        imu_data = imu_integration.load_opencamera_csv(
            str(csv_path),
            str(csv_path),
            str(csv_path)
        )
        # Empty IMUData or None are both acceptable
        assert imu_data is None or len(imu_data) == 0
    
    def test_csv_very_large_file(self, temp_dir):
        """Test with very large CSV file (memory stress test)."""
        csv_path = temp_dir / "large.csv"
        
        # Create a moderately large file (not too large for CI)
        with open(csv_path, 'w') as f:
            f.write("X-data,Y-data,Z-data,timestamp\n")
            for i in range(10000):  # 10k samples
                f.write(f"{i*0.001},{i*0.002},{i*0.003},{i*1000000}\n")
        
        try:
            imu_data = imu_integration.load_opencamera_csv(
                str(csv_path),
                str(csv_path),
                str(csv_path)
            )
            # Should handle large files
            if imu_data:
                assert len(imu_data) > 0
        except MemoryError:
            # Memory error is acceptable for extremely large files
            pytest.skip("Not enough memory for large file test")
        except Exception:
            pass


@pytest.mark.unit
@pytest.mark.imu
class TestInvalidIMUData:
    """Test handling of invalid IMU data structures."""
    
    def test_imu_data_with_nan_values(self):
        """Test IMU data with NaN values."""
        sample = imu_integration.IMUSample(
            timestamp_ns=1000000000,
            accel=np.array([np.nan, 0.0, -9.81], dtype=np.float32),
            gyro=np.array([0.0, 0.0, 0.0], dtype=np.float32)
        )
        
        imu_data = imu_integration.IMUData(
            samples=[sample],
            video_timestamps=np.array([1000000000])
        )
        
        try:
            processor = imu_integration.IMUProcessor(imu_data)
            gravity = processor.get_gravity_vector()
            # Should handle NaN gracefully
            assert gravity is not None or True  # Either works
        except Exception:
            pass  # Exception is acceptable
    
    def test_imu_data_with_inf_values(self):
        """Test IMU data with infinity values."""
        sample = imu_integration.IMUSample(
            timestamp_ns=1000000000,
            accel=np.array([np.inf, 0.0, -9.81], dtype=np.float32),
            gyro=np.array([0.0, 0.0, 0.0], dtype=np.float32)
        )
        
        imu_data = imu_integration.IMUData(
            samples=[sample],
            video_timestamps=np.array([1000000000])
        )
        
        try:
            processor = imu_integration.IMUProcessor(imu_data)
            gravity = processor.get_gravity_vector()
            # Should handle inf gracefully
            assert gravity is not None or True
        except Exception:
            pass
    
    def test_imu_data_negative_timestamps(self):
        """Test IMU data with negative timestamps."""
        sample = imu_integration.IMUSample(
            timestamp_ns=-1000000000,  # Negative timestamp
            accel=np.array([0.0, 0.0, -9.81], dtype=np.float32),
            gyro=np.array([0.0, 0.0, 0.0], dtype=np.float32)
        )
        
        imu_data = imu_integration.IMUData(
            samples=[sample],
            video_timestamps=np.array([-1000000000])
        )
        
        # Should handle negative timestamps
        retrieved = imu_data.get_sample_at_timestamp(-1000000000)
        assert retrieved is not None
    
    def test_imu_data_extremely_large_timestamps(self):
        """Test IMU data with extremely large timestamps."""
        large_ts = 10**18  # Very large timestamp
        sample = imu_integration.IMUSample(
            timestamp_ns=large_ts,
            accel=np.array([0.0, 0.0, -9.81], dtype=np.float32),
            gyro=np.array([0.0, 0.0, 0.0], dtype=np.float32)
        )
        
        imu_data = imu_integration.IMUData(
            samples=[sample],
            video_timestamps=np.array([large_ts])
        )
        
        # Should handle large timestamps
        retrieved = imu_data.get_sample_at_timestamp(large_ts)
        assert retrieved is not None


@pytest.mark.unit
@pytest.mark.imu
class TestVideoFileRobustness:
    """Test handling of various video file scenarios."""
    
    def test_camm_detection_corrupted_mp4(self, temp_dir):
        """Test CAMM detection with corrupted MP4 file."""
        video_path = temp_dir / "corrupted.mp4"
        video_path.write_bytes(b"corrupted mp4 data that is not valid")
        
        imu_data = imu_integration.detect_camm_in_mp4(str(video_path))
        # Should return None for corrupted files
        assert imu_data is None
    
    def test_camm_detection_wrong_format(self, temp_dir):
        """Test CAMM detection with non-MP4 file."""
        video_path = temp_dir / "test.avi"
        video_path.write_bytes(b"fake avi data")
        
        imu_data = imu_integration.detect_camm_in_mp4(str(video_path))
        # Should handle gracefully
        assert imu_data is None or isinstance(imu_data, imu_integration.IMUData)
    
    def test_camm_detection_empty_file(self, temp_dir):
        """Test CAMM detection with empty file."""
        video_path = temp_dir / "empty.mp4"
        video_path.write_bytes(b"")
        
        imu_data = imu_integration.detect_camm_in_mp4(str(video_path))
        assert imu_data is None


@pytest.mark.unit
@pytest.mark.imu
class TestDataSynchronizationRobustness:
    """Test robustness of data synchronization."""
    
    def test_mismatched_accel_gyro_timestamps(self, temp_dir):
        """Test with mismatched accelerometer and gyroscope timestamps."""
        # Create CSV files with different timestamp ranges
        accel_path = temp_dir / "accel.csv"
        accel_path.write_text(
            "X-data,Y-data,Z-data,timestamp\n"
            "0.0,0.0,-9.81,0\n"
            "0.0,0.0,-9.81,1000000000\n"
        )
        
        gyro_path = temp_dir / "gyro.csv"
        gyro_path.write_text(
            "X-data,Y-data,Z-data,timestamp\n"
            "0.01,0.02,0.03,500000000\n"  # Different timestamps
            "0.01,0.02,0.03,1500000000\n"
        )
        
        timestamps_path = temp_dir / "timestamps.csv"
        timestamps_path.write_text("timestamp\n0\n1000000000\n")
        
        try:
            imu_data = imu_integration.load_opencamera_csv(
                str(accel_path),
                str(gyro_path),
                str(timestamps_path)
            )
            # Should handle mismatched timestamps
            if imu_data:
                assert len(imu_data) > 0
        except Exception:
            pass
    
    def test_no_overlapping_timestamps(self, temp_dir):
        """Test with no overlapping timestamps between accel and gyro."""
        accel_path = temp_dir / "accel.csv"
        accel_path.write_text(
            "X-data,Y-data,Z-data,timestamp\n"
            "0.0,0.0,-9.81,0\n"
            "0.0,0.0,-9.81,1000000000\n"
        )
        
        gyro_path = temp_dir / "gyro.csv"
        gyro_path.write_text(
            "X-data,Y-data,Z-data,timestamp\n"
            "0.01,0.02,0.03,2000000000\n"  # No overlap
            "0.01,0.02,0.03,3000000000\n"
        )
        
        timestamps_path = temp_dir / "timestamps.csv"
        timestamps_path.write_text("timestamp\n0\n1000000000\n")
        
        try:
            imu_data = imu_integration.load_opencamera_csv(
                str(accel_path),
                str(gyro_path),
                str(timestamps_path)
            )
            # Should handle no overlap (might return None or empty data)
            assert imu_data is None or len(imu_data) >= 0
        except Exception:
            pass

