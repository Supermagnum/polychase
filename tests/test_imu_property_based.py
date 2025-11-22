# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 Ahmed Essam <aessam.dahy@gmail.com>

"""Property-based testing for IMU integration using Hypothesis."""

import numpy as np
import pytest
from hypothesis import given, strategies as st, assume, settings

from blender_addon import imu_integration
from tests.data_generators import generate_synthetic_imu_data, generate_video_timestamps


@pytest.mark.unit
@pytest.mark.imu
class TestIMUPropertyBased:
    """Property-based tests for IMU processing."""
    
    @given(
        accel_x=st.floats(min_value=-20.0, max_value=20.0, allow_nan=False, allow_infinity=False),
        accel_y=st.floats(min_value=-20.0, max_value=20.0, allow_nan=False, allow_infinity=False),
        accel_z=st.floats(min_value=-20.0, max_value=20.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50, deadline=None)
    def test_gravity_vector_normalization(self, accel_x, accel_y, accel_z):
        """Test that gravity vector is always normalized."""
        # Create IMU data with multiple samples (filter needs at least 15)
        num_samples = 20
        samples = [
            imu_integration.IMUSample(
                timestamp_ns=int(i * 1e9 / 200),  # 200 Hz
                accel=np.array([accel_x, accel_y, accel_z], dtype=np.float32),
                gyro=np.array([0.0, 0.0, 0.0], dtype=np.float32)
            )
            for i in range(num_samples)
        ]
        
        imu_data = imu_integration.IMUData(
            samples=samples,
            video_timestamps=np.linspace(0, num_samples * 1e9 / 200, 10, dtype=np.int64)
        )
        
        processor = imu_integration.IMUProcessor(imu_data)
        gravity = processor.get_gravity_vector()
        
        # Gravity vector should be normalized (length ~1.0)
        gravity_length = np.linalg.norm(gravity)
        assert 0.0 <= gravity_length <= 1.1, f"Gravity vector not normalized: {gravity_length}"
    
    @given(
        gyro_x=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        gyro_y=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        gyro_z=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50, deadline=None)
    def test_gyro_integration_quaternion_properties(self, gyro_x, gyro_y, gyro_z):
        """Test that gyro integration produces valid quaternions."""
        from blender_addon import imu_integration
        mathutils = imu_integration.mathutils
        
        # Create IMU data with enough samples for filter
        num_samples = 20
        timestamps = np.linspace(0, 1.0, num_samples) * 1e9
        samples = [
            imu_integration.IMUSample(
                timestamp_ns=int(ts),
                accel=np.array([0.0, 0.0, -9.81], dtype=np.float32),
                gyro=np.array([gyro_x, gyro_y, gyro_z], dtype=np.float32)
            )
            for ts in timestamps
        ]
        
        imu_data = imu_integration.IMUData(
            samples=samples,
            video_timestamps=timestamps.astype(np.int64)
        )
        
        processor = imu_integration.IMUProcessor(imu_data)
        initial_orientation = mathutils.Quaternion((1, 0, 0, 0))
        
        # Integrate gyro
        result = processor.integrate_gyro(1, initial_orientation)
        
        # Quaternion should be normalized
        assert abs(result.magnitude - 1.0) < 0.1, f"Quaternion not normalized: {result.magnitude}"
    
    @given(
        num_samples=st.integers(min_value=20, max_value=1000),  # Need at least 20 for filter
        sample_rate=st.floats(min_value=10.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=20, deadline=None)
    def test_imu_data_scalability(self, num_samples, sample_rate):
        """Test IMU processing with various data sizes."""
        assume(num_samples >= 20)  # Filter needs minimum samples
        assume(sample_rate > 0)
        
        imu_data_dict = generate_synthetic_imu_data(
            num_samples=num_samples,
            sample_rate_hz=sample_rate
        )
        
        samples = [
            imu_integration.IMUSample(
                timestamp_ns=int(ts),
                accel=accel,
                gyro=gyro
            )
            for ts, accel, gyro in zip(
                imu_data_dict['timestamps'],
                imu_data_dict['accel'],
                imu_data_dict['gyro']
            )
        ]
        
        video_timestamps = generate_video_timestamps(
            num_frames=max(10, num_samples // 10)
        )
        
        imu_data = imu_integration.IMUData(
            samples=samples,
            video_timestamps=video_timestamps
        )
        
        processor = imu_integration.IMUProcessor(imu_data)
        
        # Should not crash and should produce valid results
        gravity = processor.get_gravity_vector()
        assert gravity is not None
        assert len(gravity) == 3
        
        consistency = processor.get_gravity_consistency()
        assert 0.0 <= consistency <= 1.0
    
    @given(
        timestamp1=st.integers(min_value=0, max_value=10**12),
        timestamp2=st.integers(min_value=0, max_value=10**12),
    )
    @settings(max_examples=30, deadline=None)
    def test_timestamp_interpolation(self, timestamp1, timestamp2):
        """Test timestamp interpolation with various timestamp values."""
        assume(timestamp1 != timestamp2)
        
        # Create samples with given timestamps
        ts_min = min(timestamp1, timestamp2)
        ts_max = max(timestamp1, timestamp2)
        
        samples = [
            imu_integration.IMUSample(
                timestamp_ns=ts_min,
                accel=np.array([0.0, 0.0, -9.81], dtype=np.float32),
                gyro=np.array([0.0, 0.0, 0.0], dtype=np.float32)
            ),
            imu_integration.IMUSample(
                timestamp_ns=ts_max,
                accel=np.array([0.0, 0.0, -9.81], dtype=np.float32),
                gyro=np.array([0.0, 0.0, 0.0], dtype=np.float32)
            ),
        ]
        
        imu_data = imu_integration.IMUData(
            samples=samples,
            video_timestamps=np.array([ts_min, ts_max])
        )
        
        # Interpolate at middle timestamp
        mid_ts = (ts_min + ts_max) // 2
        sample = imu_data.interpolate_at_timestamp(mid_ts)
        
        # Should return a valid sample
        assert sample is not None
        assert sample.timestamp_ns == mid_ts
    
    @given(
        weight=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=20, deadline=None)
    def test_orientation_blending_weights(self, weight):
        """Test orientation blending with various weights."""
        from blender_addon import imu_integration
        mathutils = imu_integration.mathutils
        
        imu_data_dict = generate_synthetic_imu_data(num_samples=100)
        samples = [
            imu_integration.IMUSample(
                timestamp_ns=int(ts),
                accel=accel,
                gyro=gyro
            )
            for ts, accel, gyro in zip(
                imu_data_dict['timestamps'],
                imu_data_dict['accel'],
                imu_data_dict['gyro']
            )
        ]
        
        imu_data = imu_integration.IMUData(
            samples=samples,
            video_timestamps=generate_video_timestamps()
        )
        
        processor = imu_integration.IMUProcessor(imu_data)
        initial_orientation = mathutils.Quaternion((1, 0, 0, 0))
        
        # Test constraint with various weights
        constrained = processor.constrain_z_axis_to_gravity(initial_orientation, weight=weight)
        
        # Result should be a valid quaternion
        assert abs(constrained.magnitude - 1.0) < 0.1


@pytest.mark.unit
@pytest.mark.imu
class TestIMUBoundaryConditions:
    """Test boundary conditions and edge cases."""
    
    def test_zero_acceleration(self):
        """Test with zero acceleration values."""
        # Need multiple samples for filter
        samples = [
            imu_integration.IMUSample(
                timestamp_ns=int(i * 1e9 / 200),
                accel=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                gyro=np.array([0.0, 0.0, 0.0], dtype=np.float32)
            )
            for i in range(20)
        ]
        
        imu_data = imu_integration.IMUData(
            samples=samples,
            video_timestamps=np.linspace(0, 20 * 1e9 / 200, 10, dtype=np.int64)
        )
        
        processor = imu_integration.IMUProcessor(imu_data)
        gravity = processor.get_gravity_vector()
        
        # Should handle zero acceleration gracefully
        assert gravity is not None
        assert len(gravity) == 3
    
    def test_maximum_sensor_values(self):
        """Test with maximum sensor values."""
        # Typical IMU max values: accel ~16g, gyro ~2000 deg/s
        max_accel = 16.0 * 9.81  # 16g in m/s²
        max_gyro = 2000.0 * np.pi / 180.0  # 2000 deg/s in rad/s
        
        # Need multiple samples for filter
        samples = [
            imu_integration.IMUSample(
                timestamp_ns=int(i * 1e9 / 200),
                accel=np.array([max_accel, max_accel, max_accel], dtype=np.float32),
                gyro=np.array([max_gyro, max_gyro, max_gyro], dtype=np.float32)
            )
            for i in range(20)
        ]
        
        imu_data = imu_integration.IMUData(
            samples=samples,
            video_timestamps=np.linspace(0, 20 * 1e9 / 200, 10, dtype=np.int64)
        )
        
        processor = imu_integration.IMUProcessor(imu_data)
        
        # Should handle maximum values without crashing
        gravity = processor.get_gravity_vector()
        assert gravity is not None
    
    def test_timestamp_duplicates(self):
        """Test with duplicate timestamps."""
        samples = [
            imu_integration.IMUSample(
                timestamp_ns=1000000000,
                accel=np.array([0.0, 0.0, -9.81], dtype=np.float32),
                gyro=np.array([0.0, 0.0, 0.0], dtype=np.float32)
            ),
            imu_integration.IMUSample(
                timestamp_ns=1000000000,  # Duplicate
                accel=np.array([0.1, 0.1, -9.81], dtype=np.float32),
                gyro=np.array([0.01, 0.01, 0.01], dtype=np.float32)
            ),
        ]
        
        imu_data = imu_integration.IMUData(
            samples=samples,
            video_timestamps=np.array([1000000000, 1000000000])
        )
        
        # Should handle duplicates
        sample = imu_data.get_sample_at_timestamp(1000000000)
        assert sample is not None
    
    def test_timestamp_gaps(self):
        """Test with large timestamp gaps."""
        samples = [
            imu_integration.IMUSample(
                timestamp_ns=0,
                accel=np.array([0.0, 0.0, -9.81], dtype=np.float32),
                gyro=np.array([0.0, 0.0, 0.0], dtype=np.float32)
            ),
            imu_integration.IMUSample(
                timestamp_ns=10**12,  # 1 second gap
                accel=np.array([0.0, 0.0, -9.81], dtype=np.float32),
                gyro=np.array([0.0, 0.0, 0.0], dtype=np.float32)
            ),
        ]
        
        imu_data = imu_integration.IMUData(
            samples=samples,
            video_timestamps=np.array([0, 10**12])
        )
        
        # Should handle gaps
        mid_ts = 5 * 10**11
        sample = imu_data.interpolate_at_timestamp(mid_ts)
        assert sample is not None
    
    def test_video_imu_desynchronization(self):
        """Test with desynchronized video and IMU timestamps."""
        # IMU timestamps: 0 to 1 second
        imu_samples = [
            imu_integration.IMUSample(
                timestamp_ns=int(i * 1e9 / 200),  # 200 Hz
                accel=np.array([0.0, 0.0, -9.81], dtype=np.float32),
                gyro=np.array([0.0, 0.0, 0.0], dtype=np.float32)
            )
            for i in range(200)
        ]
        
        # Video timestamps: offset by 0.5 seconds
        video_timestamps = np.array([
            int((i * 1e9 / 30) + 0.5e9)  # 30 fps, offset by 0.5s
            for i in range(30)
        ], dtype=np.int64)
        
        imu_data = imu_integration.IMUData(
            samples=imu_samples,
            video_timestamps=video_timestamps
        )
        
        # Should handle desynchronization
        processor = imu_integration.IMUProcessor(imu_data)
        gravity = processor.get_gravity_vector()
        assert gravity is not None

