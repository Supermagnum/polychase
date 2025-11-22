# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 Ahmed Essam <aessam.dahy@gmail.com>

"""Pytest configuration and shared fixtures."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_imu_data():
    """Generate sample IMU data for testing."""
    np.random.seed(42)  # For reproducibility
    
    # Generate 1000 samples at 200 Hz
    num_samples = 1000
    sample_rate = 200.0
    duration = num_samples / sample_rate
    timestamps = np.linspace(0, duration * 1e9, num_samples, dtype=np.int64)
    
    # Generate accelerometer data with gravity + noise
    gravity = np.array([0.0, 0.0, -9.81])
    accel_data = np.tile(gravity, (num_samples, 1))
    accel_data += np.random.normal(0, 0.1, (num_samples, 3))  # Add noise
    
    # Generate gyroscope data (mostly stationary with small movements)
    gyro_data = np.random.normal(0, 0.01, (num_samples, 3))
    
    return {
        'timestamps': timestamps,
        'accel': accel_data,
        'gyro': gyro_data,
        'sample_rate': sample_rate,
    }


@pytest.fixture
def sample_video_timestamps():
    """Generate sample video frame timestamps (30 fps)."""
    num_frames = 100
    fps = 30.0
    duration = num_frames / fps
    timestamps = np.linspace(0, duration * 1e9, num_frames, dtype=np.int64)
    return timestamps


@pytest.fixture
def opencamera_csv_files(temp_dir, sample_imu_data, sample_video_timestamps):
    """Create OpenCamera-Sensors format CSV files."""
    base_name = "test_video"
    
    # Accelerometer CSV
    accel_path = temp_dir / f"{base_name}_accel.csv"
    accel_data = np.column_stack([
        sample_imu_data['accel'][:, 0],
        sample_imu_data['accel'][:, 1],
        sample_imu_data['accel'][:, 2],
        sample_imu_data['timestamps']
    ])
    np.savetxt(accel_path, accel_data, delimiter=',', 
               header='X-data,Y-data,Z-data,timestamp (ns)',
               comments='', fmt='%.6f,%.6f,%.6f,%d')
    
    # Gyroscope CSV
    gyro_path = temp_dir / f"{base_name}_gyro.csv"
    gyro_data = np.column_stack([
        sample_imu_data['gyro'][:, 0],
        sample_imu_data['gyro'][:, 1],
        sample_imu_data['gyro'][:, 2],
        sample_imu_data['timestamps']
    ])
    np.savetxt(gyro_path, gyro_data, delimiter=',',
               header='X-data,Y-data,Z-data,timestamp (ns)',
               comments='', fmt='%.6f,%.6f,%.6f,%d')
    
    # Timestamps CSV
    timestamps_path = temp_dir / f"{base_name}_timestamps.csv"
    np.savetxt(timestamps_path, sample_video_timestamps, delimiter=',',
               header='timestamp (ns)', comments='', fmt='%d')
    
    return {
        'accel_path': str(accel_path),
        'gyro_path': str(gyro_path),
        'timestamps_path': str(timestamps_path),
        'base_name': base_name,
    }


@pytest.fixture
def mock_blender_context(monkeypatch):
    """Mock Blender context for testing without Blender."""
    class MockContext:
        scene = None
        window_manager = None
        area = None
        region = None
        
    context = MockContext()
    monkeypatch.setattr('bpy.context', context)
    return context

