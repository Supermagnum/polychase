# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 Ahmed Essam <aessam.dahy@gmail.com>

"""Performance benchmarks for IMU integration."""

import numpy as np
import pytest

try:
    import pytest_benchmark
    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False
    pytest_benchmark = None

from blender_addon import imu_integration
from tests.data_generators import generate_synthetic_imu_data, generate_video_timestamps

pytestmark = pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not installed")


@pytest.mark.performance
@pytest.mark.imu
class TestIMUPerformance:
    """Performance benchmarks for IMU processing."""
    
    def test_imu_data_loading_performance(self, benchmark, opencamera_csv_files):
        """Benchmark IMU data loading from CSV files."""
        def load_data():
            return imu_integration.load_opencamera_csv(
                opencamera_csv_files['accel_path'],
                opencamera_csv_files['gyro_path'],
                opencamera_csv_files['timestamps_path']
            )
        
        result = benchmark(load_data)
        assert result is not None
    
    def test_gravity_vector_computation_performance(self, benchmark):
        """Benchmark gravity vector computation."""
        imu_data_dict = generate_synthetic_imu_data(num_samples=10000)
        
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
        
        def compute_gravity():
            processor = imu_integration.IMUProcessor(imu_data)
            return processor.get_gravity_vector()
        
        result = benchmark(compute_gravity)
        assert result is not None
    
    def test_gyro_integration_performance(self, benchmark):
        """Benchmark gyroscope integration."""
        from blender_addon import imu_integration
        mathutils = imu_integration.mathutils
        
        imu_data_dict = generate_synthetic_imu_data(num_samples=1000)
        
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
        
        video_timestamps = generate_video_timestamps(num_frames=100)
        imu_data = imu_integration.IMUData(
            samples=samples,
            video_timestamps=video_timestamps
        )
        
        processor = imu_integration.IMUProcessor(imu_data)
        initial_orientation = mathutils.Quaternion((1, 0, 0, 0))
        
        def integrate_gyro():
            result = initial_orientation
            for frame_idx in range(len(video_timestamps)):
                result = processor.integrate_gyro(frame_idx, result)
            return result
        
        result = benchmark(integrate_gyro)
        assert result is not None
    
    def test_timestamp_interpolation_performance(self, benchmark):
        """Benchmark timestamp interpolation."""
        imu_data_dict = generate_synthetic_imu_data(num_samples=5000)
        
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
        
        video_timestamps = generate_video_timestamps(num_frames=300)
        imu_data = imu_integration.IMUData(
            samples=samples,
            video_timestamps=video_timestamps
        )
        
        def interpolate_all():
            results = []
            for ts in video_timestamps:
                sample = imu_data.interpolate_at_timestamp(int(ts))
                results.append(sample)
            return results
        
        result = benchmark(interpolate_all)
        assert len(result) == len(video_timestamps)


@pytest.mark.performance
@pytest.mark.slow
class TestIMUScalability:
    """Test IMU processing with large datasets."""
    
    @pytest.mark.parametrize("num_samples", [1000, 10000, 100000])
    def test_large_dataset_loading(self, num_samples, benchmark):
        """Test loading performance with varying dataset sizes."""
        imu_data_dict = generate_synthetic_imu_data(num_samples=num_samples)
        
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
        
        def create_imu_data():
            return imu_integration.IMUData(
                samples=samples,
                video_timestamps=generate_video_timestamps()
            )
        
        result = benchmark(create_imu_data)
        assert result is not None

