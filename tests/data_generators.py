# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 Ahmed Essam <aessam.dahy@gmail.com>

"""Mock data generators for testing IMU integration."""

import numpy as np
from pathlib import Path


def generate_synthetic_imu_data(
    num_samples: int = 1000,
    sample_rate_hz: float = 200.0,
    duration: float = None,
    noise_level: float = 0.1,
    include_motion: bool = True,
    gyro_drift: float = 0.001,
) -> dict:
    """
    Generate synthetic IMU data with known ground truth.
    
    Args:
        num_samples: Number of IMU samples to generate
        sample_rate_hz: Sample rate in Hz
        duration: Duration in seconds (overrides num_samples if provided)
        noise_level: Standard deviation of noise (m/s² for accel, rad/s for gyro)
        include_motion: Whether to include simulated camera motion
        gyro_drift: Gyroscope drift rate (rad/s)
    
    Returns:
        Dictionary with timestamps, accel, gyro, and ground truth data
    """
    np.random.seed(42)
    
    if duration is not None:
        num_samples = int(duration * sample_rate_hz)
    
    dt = 1.0 / sample_rate_hz
    timestamps = np.arange(num_samples) * dt * 1e9  # Convert to nanoseconds
    timestamps = timestamps.astype(np.int64)
    
    # Generate ground truth gravity vector (pointing down in Blender)
    gravity_truth = np.array([0.0, 0.0, -9.81])
    
    # Generate accelerometer data
    accel_data = np.tile(gravity_truth, (num_samples, 1))
    
    if include_motion:
        # Add sinusoidal motion in X and Y
        t = np.arange(num_samples) * dt
        motion_x = 0.5 * np.sin(2 * np.pi * 0.5 * t)  # 0.5 Hz oscillation
        motion_y = 0.3 * np.cos(2 * np.pi * 0.3 * t)  # 0.3 Hz oscillation
        accel_data[:, 0] += motion_x
        accel_data[:, 1] += motion_y
    
    # Add noise
    accel_data += np.random.normal(0, noise_level, (num_samples, 3))
    
    # Generate gyroscope data
    if include_motion:
        # Simulate rotation around Z-axis
        t = np.arange(num_samples) * dt
        angular_velocity_z = 0.1 * np.sin(2 * np.pi * 0.2 * t)
        gyro_data = np.zeros((num_samples, 3))
        gyro_data[:, 2] = angular_velocity_z
    else:
        gyro_data = np.zeros((num_samples, 3))
    
    # Add gyroscope bias and drift
    gyro_bias = np.array([0.001, -0.001, 0.002])
    gyro_data += gyro_bias
    gyro_data += np.random.normal(0, noise_level * 0.1, (num_samples, 3))
    
    # Add drift over time
    drift = np.linspace(0, gyro_drift, num_samples)
    gyro_data[:, 0] += drift
    
    return {
        'timestamps': timestamps,
        'accel': accel_data.astype(np.float32),
        'gyro': gyro_data.astype(np.float32),
        'sample_rate': sample_rate_hz,
        'gravity_truth': gravity_truth,
        'gyro_bias_truth': gyro_bias,
    }


def generate_video_timestamps(
    num_frames: int = 100,
    fps: float = 30.0,
    start_time_ns: int = 0,
) -> np.ndarray:
    """
    Generate video frame timestamps.
    
    Args:
        num_frames: Number of video frames
        fps: Frames per second
        start_time_ns: Start timestamp in nanoseconds
    
    Returns:
        Array of timestamps in nanoseconds
    """
    frame_interval = 1.0 / fps
    timestamps = start_time_ns + np.arange(num_frames) * frame_interval * 1e9
    return timestamps.astype(np.int64)


def create_opencamera_csv_files(
    output_dir: Path,
    base_name: str = "test_video",
    imu_data: dict = None,
    video_timestamps: np.ndarray = None,
) -> dict:
    """
    Create OpenCamera-Sensors format CSV files.
    
    Args:
        output_dir: Directory to write CSV files
        base_name: Base name for files
        imu_data: IMU data dictionary (from generate_synthetic_imu_data)
        video_timestamps: Video frame timestamps
    
    Returns:
        Dictionary with paths to created files
    """
    if imu_data is None:
        imu_data = generate_synthetic_imu_data()
    
    if video_timestamps is None:
        video_timestamps = generate_video_timestamps()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Accelerometer CSV
    accel_path = output_dir / f"{base_name}_accel.csv"
    accel_data = np.column_stack([
        imu_data['accel'][:, 0],
        imu_data['accel'][:, 1],
        imu_data['accel'][:, 2],
        imu_data['timestamps']
    ])
    np.savetxt(
        accel_path,
        accel_data,
        delimiter=',',
        header='X-data,Y-data,Z-data,timestamp (ns)',
        comments='',
        fmt='%.6f,%.6f,%.6f,%d'
    )
    
    # Gyroscope CSV
    gyro_path = output_dir / f"{base_name}_gyro.csv"
    gyro_data = np.column_stack([
        imu_data['gyro'][:, 0],
        imu_data['gyro'][:, 1],
        imu_data['gyro'][:, 2],
        imu_data['timestamps']
    ])
    np.savetxt(
        gyro_path,
        gyro_data,
        delimiter=',',
        header='X-data,Y-data,Z-data,timestamp (ns)',
        comments='',
        fmt='%.6f,%.6f,%.6f,%d'
    )
    
    # Timestamps CSV
    timestamps_path = output_dir / f"{base_name}_timestamps.csv"
    np.savetxt(
        timestamps_path,
        video_timestamps,
        delimiter=',',
        header='timestamp (ns)',
        comments='',
        fmt='%d'
    )
    
    return {
        'accel_path': str(accel_path),
        'gyro_path': str(gyro_path),
        'timestamps_path': str(timestamps_path),
        'imu_data': imu_data,
        'video_timestamps': video_timestamps,
    }


def generate_corrupted_csv(output_dir: Path, base_name: str = "corrupted") -> dict:
    """Generate corrupted CSV files for error handling tests."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Missing columns
    missing_cols_path = output_dir / f"{base_name}_missing_cols.csv"
    missing_cols_path.write_text("X-data,Y-data\n1.0,2.0\n")
    
    # Invalid data
    invalid_data_path = output_dir / f"{base_name}_invalid.csv"
    invalid_data_path.write_text("X-data,Y-data,Z-data,timestamp\ninvalid,data,here,123\n")
    
    # Empty file
    empty_path = output_dir / f"{base_name}_empty.csv"
    empty_path.write_text("")
    
    return {
        'missing_cols': str(missing_cols_path),
        'invalid_data': str(invalid_data_path),
        'empty': str(empty_path),
    }


def generate_noisy_imu_data(
    base_data: dict = None,
    noise_multiplier: float = 5.0,
) -> dict:
    """Generate IMU data with high noise levels."""
    if base_data is None:
        base_data = generate_synthetic_imu_data()
    
    noisy_data = base_data.copy()
    noise_level = base_data.get('noise_level', 0.1) * noise_multiplier
    
    # Add high noise to accelerometer
    noisy_data['accel'] += np.random.normal(0, noise_level, noisy_data['accel'].shape)
    
    # Add high noise to gyroscope
    noisy_data['gyro'] += np.random.normal(0, noise_level * 0.1, noisy_data['gyro'].shape)
    
    return noisy_data


def generate_drift_imu_data(
    base_data: dict = None,
    drift_rate: float = 0.1,
) -> dict:
    """Generate IMU data with significant gyroscope drift."""
    if base_data is None:
        base_data = generate_synthetic_imu_data()
    
    drift_data = base_data.copy()
    num_samples = len(drift_data['timestamps'])
    
    # Add linear drift to all gyro axes
    t = np.arange(num_samples) / drift_data['sample_rate']
    drift = np.column_stack([
        drift_rate * t,
        -drift_rate * 0.5 * t,
        drift_rate * 0.3 * t,
    ])
    
    drift_data['gyro'] = drift_data['gyro'].astype(np.float64) + drift
    drift_data['gyro'] = drift_data['gyro'].astype(np.float32)
    
    return drift_data

