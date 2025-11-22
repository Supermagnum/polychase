# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 Ahmed Essam <aessam.dahy@gmail.com>

"""Memory profiling tests for IMU integration."""

import numpy as np
import pytest

from blender_addon import imu_integration
from tests.data_generators import generate_synthetic_imu_data, generate_video_timestamps


@pytest.mark.memory
@pytest.mark.imu
class TestIMUMemoryUsage:
    """Memory usage tests for IMU processing."""
    
    def test_imu_data_memory_footprint(self):
        """Test memory footprint of IMU data structures."""
        import tracemalloc
        
        tracemalloc.start()
        
        # Generate large dataset
        imu_data_dict = generate_synthetic_imu_data(num_samples=100000)
        
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
        
        snapshot1 = tracemalloc.take_snapshot()
        
        imu_data = imu_integration.IMUData(
            samples=samples,
            video_timestamps=generate_video_timestamps(num_frames=1000)
        )
        
        snapshot2 = tracemalloc.take_snapshot()
        
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        
        # Check that memory increase is reasonable (< 100 MB for 100k samples)
        total_mb = sum(stat.size_diff for stat in top_stats) / (1024 * 1024)
        assert total_mb < 100, f"Memory usage too high: {total_mb:.2f} MB"
        
        tracemalloc.stop()
    
    def test_processor_memory_usage(self):
        """Test memory usage of IMU processor."""
        import tracemalloc
        
        tracemalloc.start()
        
        imu_data_dict = generate_synthetic_imu_data(num_samples=50000)
        
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
        
        snapshot1 = tracemalloc.take_snapshot()
        
        processor = imu_integration.IMUProcessor(imu_data)
        
        snapshot2 = tracemalloc.take_snapshot()
        
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        
        # Processor should not use excessive memory
        total_mb = sum(stat.size_diff for stat in top_stats) / (1024 * 1024)
        assert total_mb < 50, f"Processor memory usage too high: {total_mb:.2f} MB"
        
        tracemalloc.stop()
    
    def test_no_memory_leaks_in_processing(self):
        """Test for memory leaks during repeated processing."""
        import tracemalloc
        import gc
        
        tracemalloc.start()
        
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
        
        # Process multiple times
        for _ in range(10):
            processor = imu_integration.IMUProcessor(imu_data)
            _ = processor.get_gravity_vector()
            _ = processor.get_gravity_consistency()
            _ = processor.get_gyro_drift()
            del processor
            gc.collect()
        
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        # Check that memory is not growing unbounded
        total_mb = sum(stat.size for stat in top_stats[:10]) / (1024 * 1024)
        assert total_mb < 200, f"Potential memory leak detected: {total_mb:.2f} MB"
        
        tracemalloc.stop()

