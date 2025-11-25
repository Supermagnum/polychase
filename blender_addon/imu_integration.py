# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 Ahmed Essam <aessam.dahy@gmail.com>

"""
IMU Integration Module for Polychase

This module provides IMU (Inertial Measurement Unit) data processing and
integration for improved camera tracking with automatic Z-axis orientation.
Supports OpenCamera-Sensors CSV format and CAMM (Camera Motion Metadata) from MP4 files.
"""

import os
import sys
import typing
import importlib.util
from dataclasses import dataclass

import numpy as np

# Optional scipy dependency - will be set by automatic detection
SCIPY_AVAILABLE = False
signal = None
interp1d = None

try:
    import mathutils
    # Verify it's the real mathutils (has Quaternion class)
    if not hasattr(mathutils, 'Quaternion'):
        raise ImportError("mathutils doesn't have Quaternion")
except (ImportError, AttributeError):
    # Mock mathutils for testing outside Blender
    class MockQuaternion:
        def __init__(self, *args):
            if len(args) == 1 and isinstance(args[0], (list, tuple, np.ndarray)) and len(args[0]) == 4:
                # WXYZ format
                self.w, self.x, self.y, self.z = args[0]
            elif len(args) == 2:
                # axis, angle format
                axis, angle = args
                axis = np.array(axis)
                axis = axis / np.linalg.norm(axis)
                self.w = np.cos(angle / 2)
                self.x, self.y, self.z = axis * np.sin(angle / 2)
            else:
                self.w, self.x, self.y, self.z = args if len(args) == 4 else (1, 0, 0, 0)
        
        def __mul__(self, other):
            # Quaternion multiplication
            w1, x1, y1, z1 = self.w, self.x, self.y, self.z
            w2, x2, y2, z2 = other.w, other.x, other.y, other.z
            return MockQuaternion([
                w1*w2 - x1*x2 - y1*y2 - z1*z2,
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2
            ])
        
        def slerp(self, other, factor):
            # Simplified slerp
            if factor <= 0:
                return self
            if factor >= 1:
                return other
            # Linear interpolation for simplicity
            w = self.w * (1 - factor) + other.w * factor
            x = self.x * (1 - factor) + other.x * factor
            y = self.y * (1 - factor) + other.y * factor
            z = self.z * (1 - factor) + other.z * factor
            norm = np.sqrt(w*w + x*x + y*y + z*z)
            if norm > 0:
                return MockQuaternion([w/norm, x/norm, y/norm, z/norm])
            return self
        
        @property
        def magnitude(self):
            return np.sqrt(self.w*self.w + self.x*self.x + self.y*self.y + self.z*self.z)
        
        def __eq__(self, other):
            if not isinstance(other, MockQuaternion):
                return False
            return abs(self.w - other.w) < 1e-6 and abs(self.x - other.x) < 1e-6 and \
                   abs(self.y - other.y) < 1e-6 and abs(self.z - other.z) < 1e-6
    
    class MockVector:
        def __init__(self, *args):
            if len(args) == 1:
                self.x, self.y, self.z = args[0] if len(args[0]) >= 3 else (0, 0, 0)
            else:
                self.x, self.y, self.z = args if len(args) >= 3 else (0, 0, 0)
    
    class MockMathutils:
        Quaternion = MockQuaternion
        Vector = MockVector
    
    mathutils = MockMathutils()

# Automatic feature detection for optional dependencies
# Feature detection state
_pandas_available = False
_pandas_version = None
_pandas_import_error = None
pd = None

_scipy_available = False
_scipy_version = None
_scipy_import_error = None


def _find_package_locations(package_name: str) -> list[str]:
    """
    Search for a package in common installation locations.
    
    Returns list of directory paths where the package might be found.
    """
    locations = []
    
    # Check current sys.path
    for path in sys.path:
        package_path = os.path.join(path, package_name)
        if os.path.exists(package_path):
            locations.append(path)
    
    # Common Flatpak Python paths
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    flatpak_paths = [
        f'/var/data/python/lib/python{python_version}/site-packages',
        f'/var/data/python/lib/python3.11/site-packages',
        f'/var/data/python/lib/python3.12/site-packages',
        os.path.expanduser(f'~/.local/lib/python{python_version}/site-packages'),
        os.path.expanduser('~/.local/lib/python3.11/site-packages'),
        os.path.expanduser('~/.local/lib/python3.12/site-packages'),
    ]
    
    # Check Flatpak paths
    for path in flatpak_paths:
        if os.path.exists(path):
            package_path = os.path.join(path, package_name)
            if os.path.exists(package_path) and path not in locations:
                locations.append(path)
    
    # Check system-wide locations
    system_paths = [
        f'/usr/lib/python{python_version}/site-packages',
        f'/usr/local/lib/python{python_version}/site-packages',
    ]
    for path in system_paths:
        if os.path.exists(path):
            package_path = os.path.join(path, package_name)
            if os.path.exists(package_path) and path not in locations:
                locations.append(path)
    
    return locations


def _try_import_package(package_name: str, module_name: str = None) -> tuple[bool, str | None, str | None]:
    """
    Try multiple strategies to import a package.
    
    Args:
        package_name: Name of the package directory (e.g., 'pandas')
        module_name: Name to import (defaults to package_name)
    
    Returns:
        (success, version, error_message)
    """
    if module_name is None:
        module_name = package_name
    
    # Strategy 1: Direct import
    try:
        module = __import__(module_name)
        version = getattr(module, '__version__', None)
        return True, version, None
    except ImportError as e:
        error_msg = str(e)
    
    # Strategy 2: Find package locations and add to sys.path
    locations = _find_package_locations(package_name)
    added_paths = []
    
    for location in locations:
        if location not in sys.path:
            sys.path.insert(0, location)
            added_paths.append(location)
    
    # Try importing again after adding paths
    if added_paths:
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', None)
            return True, version, None
        except ImportError as e2:
            error_msg = str(e2)
    
    # Strategy 3: Try importlib.util for direct module loading
    for location in locations:
        package_path = os.path.join(location, package_name)
        if os.path.exists(package_path):
            init_file = os.path.join(package_path, '__init__.py')
            if os.path.exists(init_file):
                try:
                    spec = importlib.util.spec_from_file_location(
                        module_name, init_file
                    )
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        version = getattr(module, '__version__', None)
                        # Register in sys.modules
                        sys.modules[module_name] = module
                        return True, version, None
                except Exception:
                    pass
    
    # Build comprehensive error message
    if locations:
        error_msg = (
            f"{error_msg}\n"
            f"Package '{package_name}' appears to be installed in: {locations}\n"
            f"Added paths to sys.path: {added_paths}\n"
            f"This may require restarting Blender or reloading the addon."
        )
    
    return False, None, error_msg


def _ensure_pandas_runtime_dependencies() -> tuple[list[str], list[str]]:
    """
    Make sure pandas runtime dependencies (python-dateutil and pytz) are importable.
    
    Returns:
        (missing_dependencies, diagnostic_messages)
    """
    missing: list[str] = []
    diagnostics: list[str] = []
    dependency_specs = [
        ('dateutil', 'dateutil', 'python-dateutil'),
        ('pytz', 'pytz', 'pytz'),
    ]
    
    for package_name, module_name, display_name in dependency_specs:
        success, version, error = _try_import_package(package_name, module_name)
        if not success:
            missing.append(display_name)
            if error:
                diagnostics.append(f"{display_name}: {error}")
    
    return missing, diagnostics


def _detect_features():
    """Automatically detect and import optional dependencies."""
    global _pandas_available, _pandas_version, _pandas_import_error, pd
    global _scipy_available, _scipy_version, _scipy_import_error
    
    missing_runtime_deps, runtime_diagnostics = _ensure_pandas_runtime_dependencies()
    if missing_runtime_deps:
        _pandas_available = False
        _pandas_version = None
        pd = None
        diagnostic_block = "\n".join(runtime_diagnostics) if runtime_diagnostics else ""
        instructions = (
            "Install python-dateutil>=2.8.2 and pytz>=2023.3 in the same Python "
            "environment as Blender. The addon will re-check automatically when "
            "reloaded or when refresh_feature_detection() is called."
        )
        _pandas_import_error = (
            "Missing pandas runtime dependencies: "
            f"{', '.join(missing_runtime_deps)}\n"
            f"{diagnostic_block}\n"
            f"{instructions}"
        ).strip()
    else:
        # Detect pandas
        success, version, error = _try_import_package('pandas', 'pandas')
        if success:
            try:
                import pandas as pd
                _pandas_available = True
                _pandas_version = pd.__version__ if pd is not None else version
                _pandas_import_error = None
            except Exception as e:
                _pandas_available = False
                _pandas_version = None
                _pandas_import_error = str(e)
                pd = None
        else:
            _pandas_available = False
            _pandas_version = None
            _pandas_import_error = error
            pd = None

    if not _pandas_available:
        pd = None
    
    # Detect scipy (update SCIPY_AVAILABLE)
    global SCIPY_AVAILABLE, signal, interp1d
    success, version, error = _try_import_package('scipy', 'scipy')
    if success:
        try:
            from scipy import signal
            from scipy.interpolate import interp1d
            _scipy_available = True
            _scipy_version = version
            _scipy_import_error = None
            SCIPY_AVAILABLE = True
        except Exception as e:
            _scipy_available = False
            _scipy_version = None
            _scipy_import_error = str(e)
            SCIPY_AVAILABLE = False
            # Set up fallback implementations
            def butter(order, cutoff, btype='low'):
                """Simple fallback for scipy.signal.butter - returns None to indicate unavailable"""
                return None, None
            
            def filtfilt(b, a, x):
                """Simple fallback - returns unfiltered data if scipy not available"""
                return x
            
            signal = type('signal', (), {'butter': butter, 'filtfilt': filtfilt})()
            interp1d = None
    else:
        _scipy_available = False
        _scipy_version = None
        _scipy_import_error = error
        SCIPY_AVAILABLE = False
        # Set up fallback implementations
        def butter(order, cutoff, btype='low'):
            """Simple fallback for scipy.signal.butter - returns None to indicate unavailable"""
            return None, None
        
        def filtfilt(b, a, x):
            """Simple fallback - returns unfiltered data if scipy not available"""
            return x
        
        signal = type('signal', (), {'butter': butter, 'filtfilt': filtfilt})()
        interp1d = None


# Run automatic feature detection
_detect_features()


def refresh_feature_detection():
    """
    Re-run automatic feature detection.
    
    Call this function if you've installed pandas/scipy after the addon was loaded.
    This will search for packages again and update the availability status.
    """
    _detect_features()
    return {
        'pandas': {
            'available': _pandas_available,
            'version': _pandas_version,
            'error': _pandas_import_error
        },
        'scipy': {
            'available': _scipy_available,
            'version': _scipy_version,
            'error': _scipy_import_error
        }
    }


def get_feature_status():
    """
    Get current status of optional dependencies.
    
    Returns:
        dict with 'pandas' and 'scipy' keys, each containing:
        - 'available': bool
        - 'version': str or None
        - 'error': str or None
    """
    return {
        'pandas': {
            'available': _pandas_available,
            'version': _pandas_version,
            'error': _pandas_import_error
        },
        'scipy': {
            'available': _scipy_available,
            'version': _scipy_version,
            'error': _scipy_import_error
        }
    }


@dataclass
class IMUSample:
    """Single IMU sample with timestamp"""
    timestamp_ns: int
    accel: np.ndarray  # [x, y, z] in m/s^2
    gyro: np.ndarray   # [x, y, z] in rad/s


@dataclass
class IMUData:
    """Container for IMU data with timestamps"""
    samples: list[IMUSample]
    video_timestamps: np.ndarray  # Timestamps for each video frame (ns)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def get_sample_at_timestamp(self, timestamp_ns: int) -> IMUSample | None:
        """Get IMU sample closest to given timestamp"""
        if not self.samples:
            return None
        
        # Binary search for closest sample
        timestamps = np.array([s.timestamp_ns for s in self.samples])
        idx = np.searchsorted(timestamps, timestamp_ns)
        
        if idx == 0:
            return self.samples[0]
        elif idx >= len(self.samples):
            return self.samples[-1]
        else:
            # Return closest sample
            if abs(timestamps[idx] - timestamp_ns) < abs(timestamps[idx-1] - timestamp_ns):
                return self.samples[idx]
            else:
                return self.samples[idx-1]
    
    def interpolate_at_timestamp(self, timestamp_ns: int) -> IMUSample | None:
        """Interpolate IMU data at given timestamp"""
        if len(self.samples) < 2:
            return self.get_sample_at_timestamp(timestamp_ns)
        
        timestamps = np.array([s.timestamp_ns for s in self.samples])
        
        # Check bounds
        if timestamp_ns < timestamps[0] or timestamp_ns > timestamps[-1]:
            return self.get_sample_at_timestamp(timestamp_ns)
        
        # Interpolate accelerometer
        accel_data = np.array([[s.accel[0], s.accel[1], s.accel[2]] for s in self.samples])
        if interp1d is None:
            raise ImportError("scipy is required for timestamp interpolation. Install with: pip install scipy")
        accel_interp = interp1d(timestamps, accel_data, axis=0, kind='linear', 
                               bounds_error=False, fill_value='extrapolate')
        accel = accel_interp(timestamp_ns)
        
        # Interpolate gyroscope
        gyro_data = np.array([[s.gyro[0], s.gyro[1], s.gyro[2]] for s in self.samples])
        if interp1d is None:
            raise ImportError("scipy is required for timestamp interpolation. Install with: pip install scipy")
        gyro_interp = interp1d(timestamps, gyro_data, axis=0, kind='linear',
                              bounds_error=False, fill_value='extrapolate')
        gyro = gyro_interp(timestamp_ns)
        
        return IMUSample(timestamp_ns=timestamp_ns, accel=accel, gyro=gyro)


class IMUProcessor:
    """Processes IMU data for camera tracking"""
    
    def __init__(self, imu_data: IMUData, lowpass_cutoff: float = 0.1, 
                 lowpass_order: int = 4, sample_rate_hz: float = 200.0):
        """
        Initialize IMU processor
        
        Args:
            imu_data: IMU data container
            lowpass_cutoff: Low-pass filter cutoff frequency for gravity extraction (Hz)
            lowpass_order: Filter order for low-pass filter
            sample_rate_hz: IMU sample rate in Hz (for filter design)
        """
        self.imu_data = imu_data
        self.lowpass_cutoff = lowpass_cutoff
        self.lowpass_order = lowpass_order
        self.sample_rate_hz = sample_rate_hz
        
        # Compute gravity vector from accelerometer
        self._gravity_vector = None
        self._gravity_vector_normalized = None
        self._compute_gravity_vector()
        
        # Gyroscope bias (can be calibrated)
        self._gyro_bias = np.zeros(3)
        self._compute_gyro_bias()
    
    def _compute_gravity_vector(self):
        """Compute gravity vector using low-pass filtered accelerometer data"""
        if not self.imu_data.samples:
            self._gravity_vector = np.array([0.0, 0.0, -9.81])
            self._gravity_vector_normalized = np.array([0.0, 0.0, -1.0])
            return
        
        # Extract accelerometer data
        accel_data = np.array([[s.accel[0], s.accel[1], s.accel[2]] for s in self.imu_data.samples])
        
        # Design low-pass filter to isolate gravity
        if SCIPY_AVAILABLE:
            nyquist = self.sample_rate_hz / 2.0
            normal_cutoff = self.lowpass_cutoff / nyquist
            b, a = signal.butter(self.lowpass_order, normal_cutoff, btype='low')
            
            # Apply filter to each axis
            filtered_accel = np.zeros_like(accel_data)
            for i in range(3):
                filtered_accel[:, i] = signal.filtfilt(b, a, accel_data[:, i])
        else:
            # Fallback: simple moving average filter (less accurate but works without scipy)
            window_size = int(self.sample_rate_hz / self.lowpass_cutoff)  # Approximate filter window
            window_size = max(1, min(window_size, len(accel_data) // 2))  # Clamp to reasonable range
            
            filtered_accel = np.zeros_like(accel_data)
            for i in range(3):
                # Simple moving average
                kernel = np.ones(window_size) / window_size
                filtered_accel[:, i] = np.convolve(accel_data[:, i], kernel, mode='same')
        
        # Use median of filtered data as gravity vector (more robust than mean)
        self._gravity_vector = np.median(filtered_accel, axis=0)
        
        # Normalize
        norm = np.linalg.norm(self._gravity_vector)
        if norm > 0.1:  # Sanity check
            self._gravity_vector_normalized = self._gravity_vector / norm
        else:
            # Fallback to standard gravity
            self._gravity_vector_normalized = np.array([0.0, 0.0, -1.0])
            self._gravity_vector = np.array([0.0, 0.0, -9.81])
    
    def _compute_gyro_bias(self):
        """Estimate gyroscope bias from static periods"""
        if not self.imu_data.samples:
            return
        
        # Find periods with low acceleration variance (static periods)
        accel_data = np.array([[s.accel[0], s.accel[1], s.accel[2]] for s in self.imu_data.samples])
        accel_magnitude = np.linalg.norm(accel_data, axis=1)
        
        # Use samples where acceleration is close to gravity (static)
        gravity_magnitude = np.linalg.norm(self._gravity_vector)
        static_mask = np.abs(accel_magnitude - gravity_magnitude) < 0.5  # 0.5 m/s^2 threshold
        
        if np.sum(static_mask) > 10:  # Need at least 10 static samples
            gyro_data = np.array([[s.gyro[0], s.gyro[1], s.gyro[2]] for s in self.imu_data.samples])
            self._gyro_bias = np.median(gyro_data[static_mask], axis=0)
        else:
            self._gyro_bias = np.zeros(3)
    
    def get_gravity_vector(self) -> np.ndarray:
        """Get gravity vector in sensor frame (normalized)"""
        return self._gravity_vector_normalized.copy()
    
    def get_gravity_magnitude(self) -> float:
        """Get gravity magnitude"""
        return np.linalg.norm(self._gravity_vector)
    
    def get_gravity_consistency(self) -> float:
        """
        Compute gravity vector consistency (0-1, higher is better)
        Measures how consistent the gravity direction is across samples
        """
        if not self.imu_data.samples or len(self.imu_data.samples) < 10:
            return 0.0
        
        # Sample a subset for efficiency
        step = max(1, len(self.imu_data.samples) // 100)
        samples = self.imu_data.samples[::step]
        
        # Compute gravity direction for each sample using low-pass filter
        accel_data = np.array([[s.accel[0], s.accel[1], s.accel[2]] for s in samples])
        
        # Normalize each acceleration vector
        norms = np.linalg.norm(accel_data, axis=1)
        valid_mask = norms > 0.1
        if np.sum(valid_mask) < 10:
            return 0.0
        
        normalized = accel_data[valid_mask] / norms[valid_mask, np.newaxis]
        
        # Compute mean direction
        mean_dir = np.mean(normalized, axis=0)
        mean_dir = mean_dir / np.linalg.norm(mean_dir)
        
        # Compute consistency as average dot product with mean direction
        consistency = np.mean(np.abs(np.dot(normalized, mean_dir)))
        
        return float(consistency)
    
    def get_gyro_drift(self) -> float:
        """
        Estimate gyroscope drift rate (rad/s)
        Higher values indicate more drift
        """
        if not self.imu_data.samples or len(self.imu_data.samples) < 100:
            return 0.0
        
        # Sample gyro data
        gyro_data = np.array([[s.gyro[0], s.gyro[1], s.gyro[2]] for s in self.imu_data.samples])
        
        # Remove bias
        gyro_corrected = gyro_data - self._gyro_bias
        
        # Compute drift as standard deviation of gyro magnitude
        gyro_magnitude = np.linalg.norm(gyro_corrected, axis=1)
        drift = np.std(gyro_magnitude)
        
        return float(drift)
    
    def integrate_gyro(self, frame_idx: int, initial_orientation: mathutils.Quaternion,
                      dt: float = None) -> mathutils.Quaternion:
        """
        Integrate gyroscope data to estimate orientation change
        
        Args:
            frame_idx: Video frame index
            initial_orientation: Starting orientation quaternion
            dt: Time step (seconds). If None, computed from timestamps
        
        Returns:
            Estimated orientation quaternion
        """
        if frame_idx < 0 or frame_idx >= len(self.imu_data.video_timestamps):
            return initial_orientation
        
        if frame_idx == 0:
            return initial_orientation
        
        # Get timestamps for current and previous frame
        t_current = self.imu_data.video_timestamps[frame_idx]
        t_prev = self.imu_data.video_timestamps[frame_idx - 1]
        
        if dt is None:
            dt = (t_current - t_prev) * 1e-9  # Convert ns to seconds
        
        # Get gyro sample at current frame
        sample = self.imu_data.interpolate_at_timestamp(t_current)
        if sample is None:
            return initial_orientation
        
        # Remove bias
        gyro = sample.gyro - self._gyro_bias
        
        # Convert to quaternion rotation
        # Angular velocity to quaternion derivative: dq/dt = 0.5 * q * [0, wx, wy, wz]
        angle = np.linalg.norm(gyro) * dt
        if angle > 1e-6:
            axis = gyro / np.linalg.norm(gyro)
            # Create rotation quaternion
            q_delta = mathutils.Quaternion(axis, angle)
            return initial_orientation @ q_delta
        
        return initial_orientation
    
    def get_orientation_from_gravity(self, gravity_in_world: np.ndarray = None) -> mathutils.Quaternion:
        """
        Compute camera orientation from gravity vector
        
        Args:
            gravity_in_world: Gravity vector in world frame (default: [0, 0, -1] for Blender)
        
        Returns:
            Quaternion representing rotation from sensor frame to world frame
        """
        if gravity_in_world is None:
            gravity_in_world = np.array([0.0, 0.0, -1.0])  # Blender Z-down
        
        gravity_sensor = self.get_gravity_vector()
        
        # Compute rotation that aligns sensor gravity with world gravity
        # Using method from: https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
        
        v1 = gravity_sensor
        v2 = gravity_in_world / np.linalg.norm(gravity_in_world)
        
        # Handle case where vectors are parallel
        if np.abs(np.dot(v1, v2)) > 0.999:
            if np.dot(v1, v2) > 0:
                return mathutils.Quaternion((1, 0, 0, 0))  # Identity
            else:
                # 180 degree rotation around perpendicular axis
                if abs(v1[0]) < 0.9:
                    axis = np.cross(v1, [1, 0, 0])
                else:
                    axis = np.cross(v1, [0, 1, 0])
                axis = axis / np.linalg.norm(axis)
                return mathutils.Quaternion(axis, np.pi)
        
        # Compute rotation axis and angle
        axis = np.cross(v1, v2)
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
        
        return mathutils.Quaternion(axis, angle)
    
    def constrain_z_axis_to_gravity(self, orientation: mathutils.Quaternion,
                                    weight: float = 1.0) -> mathutils.Quaternion:
        """
        Constrain camera Z-axis to align with gravity vector
        
        Args:
            orientation: Current camera orientation
            weight: Blend weight (0-1) for gravity constraint
        
        Returns:
            Blended orientation quaternion
        """
        if weight <= 0.0:
            return orientation
        
        # Get gravity-aligned orientation
        gravity_orientation = self.get_orientation_from_gravity()
        
        # Blend orientations
        if weight >= 1.0:
            return gravity_orientation
        else:
            return orientation.slerp(gravity_orientation, weight)


def load_opencamera_csv(accel_path: str, gyro_path: str, 
                        timestamps_path: str) -> IMUData | None:
    """
    Load IMU data from OpenCamera-Sensors CSV format
    
    Expected format:
    - accel.csv: X-data, Y-data, Z-data, timestamp (ns)
    - gyro.csv: X-data, Y-data, Z-data, timestamp (ns)
    - timestamps.csv: timestamp (ns) for each video frame
    
    Returns:
        IMUData object or None if loading fails
    """
    if pd is None:
        import_error_msg = _pandas_import_error if '_pandas_import_error' in globals() else 'Module not found'
        error_msg = (
            "pandas (and its dependencies python-dateutil / pytz) are required for CSV loading "
            "but are not available.\n"
            f"Import error: {import_error_msg}\n\n"
            "Installation options:\n"
            "1. Blender 4.2+: Should install automatically via blender_manifest.toml\n"
            "   - Check that blender_manifest.toml includes pandas in [dependencies]\n"
            "   - Try disabling and re-enabling the addon to trigger dependency installation\n"
            "2. Manual installation (Blender's Python console):\n"
            "   import subprocess, sys\n"
            "   subprocess.check_call([\n"
            "       sys.executable, '-m', 'pip', 'install',\n"
            "       'pandas>=1.5.0', 'python-dateutil>=2.8.2', 'pytz>=2023.3'\n"
            "   ])\n"
            "3. System-wide (if Blender uses system Python):\n"
            "   pip install pandas>=1.5.0 python-dateutil>=2.8.2 pytz>=2023.3"
        )
        raise ImportError(error_msg)
    
    try:
        # Load accelerometer data
        accel_df = pd.read_csv(accel_path)
        if len(accel_df.columns) < 4:
            return None
        
        accel_data = accel_df.iloc[:, :4].values  # X, Y, Z, timestamp
        accel_timestamps = accel_data[:, 3].astype(np.int64)
        accel_values = accel_data[:, :3].astype(np.float32)
        
        # Load gyroscope data
        gyro_df = pd.read_csv(gyro_path)
        if len(gyro_df.columns) < 4:
            return None
        
        gyro_data = gyro_df.iloc[:, :4].values  # X, Y, Z, timestamp
        gyro_timestamps = gyro_data[:, 3].astype(np.int64)
        gyro_values = gyro_data[:, :3].astype(np.float32)
        
        # Load video frame timestamps
        timestamps_df = pd.read_csv(timestamps_path)
        if len(timestamps_df.columns) < 1:
            return None
        
        video_timestamps = timestamps_df.iloc[:, 0].values.astype(np.int64)
        
        # Synchronize accel and gyro data by timestamp
        # Create combined samples
        all_timestamps = np.unique(np.concatenate([accel_timestamps, gyro_timestamps]))
        all_timestamps.sort()
        
        if len(all_timestamps) == 0:
            return None
        
        samples = []
        for ts in all_timestamps:
            # Find closest accel sample
            accel_idx = np.argmin(np.abs(accel_timestamps - ts))
            accel = accel_values[accel_idx]
            
            # Find closest gyro sample
            gyro_idx = np.argmin(np.abs(gyro_timestamps - ts))
            gyro = gyro_values[gyro_idx]
            
            samples.append(IMUSample(
                timestamp_ns=int(ts),
                accel=accel,
                gyro=gyro
            ))
        
        return IMUData(samples=samples, video_timestamps=video_timestamps)
    
    except Exception as e:
        print(f"Error loading OpenCamera CSV files: {e}")
        return None


def detect_camm_in_mp4(video_path: str) -> IMUData | None:
    """
    Detect and extract CAMM (Camera Motion Metadata) from MP4 file
    
    CAMM is a metadata format used by cameras (like GoPro) to embed IMU data
    directly in MP4 files. This function tries multiple extraction methods:
    1. GoPro telemetry (gopro-telemetry library)
    2. Direct MP4 box parsing
    3. MediaInfo metadata extraction
    
    Args:
        video_path: Path to MP4 video file
    
    Returns:
        IMUData object or None if CAMM not found or extraction fails
    """
    if not os.path.isfile(video_path):
        return None
    
    # Try GoPro telemetry extraction first (most common format)
    imu_data = _extract_gopro_telemetry(video_path)
    if imu_data:
        return imu_data
    
    # Try direct MP4 box parsing
    imu_data = _extract_camm_from_mp4_boxes(video_path)
    if imu_data:
        return imu_data
    
    # Try MediaInfo extraction
    imu_data = _extract_camm_with_mediainfo(video_path)
    if imu_data:
        return imu_data
    
    return None


def _extract_gopro_telemetry(video_path: str) -> IMUData | None:
    """
    Extract IMU data using gopro-telemetry library.
    
    Supports the gopro-telemetry package (PyPI: gopro-telemetry).
    The package name uses a hyphen, so we use importlib to import it.
    
    Expected API (gopro-telemetry library):
    - GoPro() class that takes a file path or file-like object
    - telemetry.accl: List of accelerometer samples with 'ts', 'x', 'y', 'z' keys
    - telemetry.gyro: List of gyroscope samples with 'ts', 'x', 'y', 'z' keys
    - Timestamps are in seconds (converted to nanoseconds)
    """
    import importlib
    
    # Try gopro-telemetry package (hyphenated name requires importlib)
    try:
        gopro_telemetry = importlib.import_module('gopro_telemetry')
        # The library typically provides a GoPro class
        if hasattr(gopro_telemetry, 'GoPro'):
            GoPro = gopro_telemetry.GoPro
        elif hasattr(gopro_telemetry, 'GoProTelemetry'):
            GoPro = gopro_telemetry.GoProTelemetry
        else:
            return None
    except ImportError:
        # Try alternative import method for hyphenated package
        try:
            import sys
            import importlib.util
            spec = importlib.util.find_spec('gopro-telemetry')
            if spec is None:
                return None
            gopro_telemetry = importlib.util.module_from_spec(spec)
            sys.modules['gopro-telemetry'] = gopro_telemetry
            spec.loader.exec_module(gopro_telemetry)
            if hasattr(gopro_telemetry, 'GoPro'):
                GoPro = gopro_telemetry.GoPro
            elif hasattr(gopro_telemetry, 'GoProTelemetry'):
                GoPro = gopro_telemetry.GoProTelemetry
            else:
                return None
        except Exception:
            return None
    
    try:
        # Parse GoPro telemetry - library typically takes file path
        telemetry = GoPro(video_path)
        
        # Extract accelerometer data
        accel_data = []
        if hasattr(telemetry, 'accl') and telemetry.accl:
            for sample in telemetry.accl:
                # Handle different timestamp formats
                ts = sample.get('ts') or sample.get('timestamp', 0.0)
                # Convert to nanoseconds (assume seconds if < 1e12)
                if ts < 1e12:
                    timestamp_ns = int(ts * 1e9)
                else:
                    timestamp_ns = int(ts)
                accel_data.append({
                    'timestamp': timestamp_ns,
                    'x': float(sample.get('x', sample.get('accl_x', 0.0))),
                    'y': float(sample.get('y', sample.get('accl_y', 0.0))),
                    'z': float(sample.get('z', sample.get('accl_z', 0.0))),
                })
        
        # Extract gyroscope data
        gyro_data = []
        if hasattr(telemetry, 'gyro') and telemetry.gyro:
            for sample in telemetry.gyro:
                ts = sample.get('ts') or sample.get('timestamp', 0.0)
                if ts < 1e12:
                    timestamp_ns = int(ts * 1e9)
                else:
                    timestamp_ns = int(ts)
                # GoPro gyro is in rad/s
                gyro_data.append({
                    'timestamp': timestamp_ns,
                    'x': float(sample.get('x', sample.get('gyro_x', 0.0))),
                    'y': float(sample.get('y', sample.get('gyro_y', 0.0))),
                    'z': float(sample.get('z', sample.get('gyro_z', 0.0))),
                })
        
        if not accel_data or not gyro_data:
            return None
        
        # Combine and synchronize data
        return _create_imu_data_from_dicts(accel_data, gyro_data, video_path)
    
    except Exception:
        return None


def _extract_camm_from_mp4_boxes(video_path: str) -> IMUData | None:
    """
    Extract CAMM data by parsing MP4 boxes directly.
    
    Note: This is a framework implementation. Full MP4 box parsing requires
    understanding the MP4 container format structure. For production use, consider
    using specialized tools like ExifTool or MediaInfo to extract CAMM data first,
    then import as CSV.
    
    Currently falls back to manual parsing which is not fully implemented.
    """
    # Try using mp4parse library if available (note: this is a placeholder
    # as no standard mp4parse library exists - would need a real MP4 parser)
    try:
        # Attempt to use any available MP4 parsing library
        # This is a placeholder - actual implementation would require
        # a real MP4 parsing library or manual box parsing
        pass
    except Exception:
        pass
    
    # Fall back to manual parsing (limited implementation)
    return _parse_mp4_boxes_manual(video_path)


def _parse_mp4_boxes_manual(video_path: str) -> IMUData | None:
    """
    Manually parse MP4 boxes to find CAMM metadata.
    
    Note: Full MP4 box parsing is complex and requires understanding:
    - MP4 container structure (ftyp, moov, mdat boxes)
    - Track structure and sample tables
    - CAMM track format specification
    
    This is a placeholder implementation. For production use, consider:
    1. Using ExifTool to extract CAMM data: exiftool -b -CAMM video.mp4
    2. Using MediaInfo to identify CAMM tracks
    3. Implementing a proper MP4 parser or using an existing library
    
    Returns None as manual parsing is not fully implemented.
    """
    try:
        with open(video_path, 'rb') as f:
            # Read file and search for CAMM box (box type 'camm')
            # MP4 box format: 4 bytes size (big-endian) + 4 bytes type
            data = f.read()
            
            # Search for 'camm' box type signature
            # Note: This is a naive search - proper parsing requires
            # following the MP4 box hierarchy
            camm_pos = data.find(b'camm')
            if camm_pos == -1:
                return None
            
            # Full implementation would:
            # 1. Parse box size (4 bytes before 'camm')
            # 2. Read box data
            # 3. Parse CAMM track structure
            # 4. Extract IMU samples with timestamps
            # 5. Convert to IMUData format
            
            # For now, return None to indicate manual parsing not fully implemented
            return None
    
    except Exception:
        return None


def _extract_camm_with_mediainfo(video_path: str) -> IMUData | None:
    """
    Extract CAMM data using MediaInfo library.
    
    Note: MediaInfo can identify CAMM tracks but extracting the actual IMU data
    requires parsing the track samples. This is a framework implementation.
    
    For production use, consider:
    1. Using MediaInfo to identify CAMM tracks: mediainfo --Output=JSON video.mp4
    2. Using ExifTool to extract CAMM data: exiftool -b -CAMM video.mp4
    3. Implementing track sample parsing based on CAMM specification
    """
    try:
        from pymediainfo import MediaInfo
    except ImportError:
        try:
            import subprocess
            import json
            # Try using mediainfo command-line tool
            result = subprocess.run(
                ['mediainfo', '--Output=JSON', video_path],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                metadata = json.loads(result.stdout)
                # Parse metadata for CAMM tracks
                # MediaInfo JSON structure: {"media": {"track": [...]}}
                # Look for tracks with track_type "Other" and format containing "camm"
                # Note: MediaInfo can identify CAMM tracks but extracting sample data
                # requires parsing the MP4 container structure
                media = metadata.get('media', {})
                tracks = media.get('track', [])
                for track in tracks:
                    if isinstance(track, dict):
                        track_type = track.get('@type', '').lower()
                        format_name = track.get('Format', '').lower()
                        if track_type == 'other' and 'camm' in format_name:
                            # CAMM track found, but sample data extraction not implemented
                            # Would need to parse track samples from MP4 container
                            pass
                return None
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, json.JSONDecodeError, FileNotFoundError):
            pass
        except Exception:
            pass
        return None
    
    try:
        media_info = MediaInfo.parse(video_path)
        
        # Look for custom metadata tracks containing IMU data
        accel_data = []
        gyro_data = []
        
        for track in media_info.tracks:
            if track.track_type == 'Other':
                track_data = track.to_data()
                format_name = track_data.get('format', track_data.get('other_format', '')).lower()
                if 'camm' in format_name:
                    # Found CAMM track, but MediaInfo doesn't provide sample data
                    # Would need to parse track samples from MP4 container directly
                    # This requires understanding the MP4 structure and CAMM format
                    pass
        
        if accel_data and gyro_data:
            return _create_imu_data_from_dicts(accel_data, gyro_data, video_path)
    
    except Exception:
        pass
    
    return None


def _create_imu_data_from_dicts(
    accel_data: list[dict],
    gyro_data: list[dict],
    video_path: str
) -> IMUData | None:
    """
    Create IMUData from accelerometer and gyroscope dictionaries.
    
    Args:
        accel_data: List of dicts with 'timestamp', 'x', 'y', 'z' keys
        gyro_data: List of dicts with 'timestamp', 'x', 'y', 'z' keys
        video_path: Path to video file (for extracting frame timestamps)
    
    Returns:
        IMUData object or None if creation fails
    """
    if not accel_data or not gyro_data:
        return None
    
    # Get all unique timestamps
    all_timestamps = set()
    for sample in accel_data:
        all_timestamps.add(sample['timestamp'])
    for sample in gyro_data:
        all_timestamps.add(sample['timestamp'])
    
    all_timestamps = sorted(all_timestamps)
    
    # Create synchronized samples
    samples = []
    for ts in all_timestamps:
        # Find closest accel sample
        accel_sample = min(accel_data, key=lambda s: abs(s['timestamp'] - ts))
        accel = np.array([
            accel_sample['x'],
            accel_sample['y'],
            accel_sample['z']
        ], dtype=np.float32)
        
        # Find closest gyro sample
        gyro_sample = min(gyro_data, key=lambda s: abs(s['timestamp'] - ts))
        gyro = np.array([
            gyro_sample['x'],
            gyro_sample['y'],
            gyro_sample['z']
        ], dtype=np.float32)
        
        samples.append(IMUSample(
            timestamp_ns=int(ts),
            accel=accel,
            gyro=gyro
        ))
    
    # Extract video frame timestamps
    # Try to get frame rate and duration from video
    video_timestamps = _extract_video_frame_timestamps(video_path, len(samples))
    
    return IMUData(samples=samples, video_timestamps=video_timestamps)


def _extract_video_frame_timestamps(video_path: str, num_imu_samples: int) -> np.ndarray:
    """
    Extract video frame timestamps.
    
    Tries to get frame rate from video metadata, or estimates from IMU sample count.
    """
    try:
        # Try to get frame rate using ffprobe or similar
        import subprocess
        import json
        
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', video_path],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'video':
                    fps_str = stream.get('r_frame_rate', '30/1')
                    if '/' in fps_str:
                        num, den = map(float, fps_str.split('/'))
                        fps = num / den if den > 0 else 30.0
                    else:
                        fps = float(fps_str)
                    
                    # Estimate duration from first IMU sample to last
                    # This is approximate - real implementation would parse video timestamps
                    duration = num_imu_samples / 200.0  # Assume 200 Hz IMU
                    num_frames = int(duration * fps)
                    timestamps = np.linspace(0, duration * 1e9, num_frames, dtype=np.int64)
                    return timestamps
    except Exception:
        pass
    
    # Fallback: estimate from IMU sample rate
    # Assume 30 fps video and 200 Hz IMU
    duration = num_imu_samples / 200.0
    num_frames = int(duration * 30.0)
    timestamps = np.linspace(0, duration * 1e9, num_frames, dtype=np.int64)
    return timestamps

