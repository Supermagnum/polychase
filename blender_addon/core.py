# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 Ahmed Essam <aessam.dahy@gmail.com>

import typing

import bpy
import mathutils
import numpy as np

from . import properties, utils

def _install_wheel_if_needed():
    """Extract and install wheel for Blender versions < 4.2 that don't support blender_manifest.toml"""
    import sys
    import os
    import zipfile
    import platform
    
    addon_path = os.path.dirname(os.path.abspath(__file__))
    wheels_dir = os.path.join(addon_path, 'wheels')
    lib_dir = os.path.join(addon_path, 'lib')
    
    # Check if .so file already exists (already installed)
    so_file_found = False
    if os.path.exists(lib_dir):
        for f in os.listdir(lib_dir):
            if f.startswith('polychase_core') and (f.endswith('.so') or f.endswith('.pyd')):
                so_file_found = True
                break
    
    if so_file_found:
        return
    
    if not os.path.exists(wheels_dir):
        return
    
    # Detect platform and find matching wheel
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    wheel_pattern = None
    if system == 'linux' and ('x86_64' in machine or 'amd64' in machine):
        wheel_pattern = 'manylinux'
    elif system == 'windows' and ('x86_64' in machine or 'amd64' in machine):
        wheel_pattern = 'win_amd64'
    
    if not wheel_pattern:
        return
    
    # Find matching wheel
    wheel_file = None
    if os.path.exists(wheels_dir):
        for filename in os.listdir(wheels_dir):
            if filename.endswith('.whl') and wheel_pattern in filename:
                wheel_file = os.path.join(wheels_dir, filename)
                break
    
    if not wheel_file:
        return
    
    # Extract wheel to lib directory
    try:
        os.makedirs(lib_dir, exist_ok=True)
        with zipfile.ZipFile(wheel_file, 'r') as wheel:
            # Extract all polychase_core files (both .so files and dist-info)
            for member in wheel.namelist():
                # Extract .so files and dist-info directories
                if (member.startswith('polychase_core') or 
                    member.startswith('polychase_core-') or
                    (os.sep in member and 'polychase_core' in member.split(os.sep)[0])):
                    # Preserve directory structure
                    target_path = os.path.join(lib_dir, member)
                    # Create parent directories if needed
                    parent_dir = os.path.dirname(target_path)
                    if parent_dir and parent_dir != lib_dir:
                        os.makedirs(parent_dir, exist_ok=True)
                    wheel.extract(member, lib_dir)
        
        # Verify extraction - check for .so file
        so_extracted = False
        if os.path.exists(lib_dir):
            for root, dirs, files in os.walk(lib_dir):
                for f in files:
                    if f.startswith('polychase_core') and (f.endswith('.so') or f.endswith('.pyd')):
                        so_extracted = True
                        # If .so is in a subdirectory, we need to add that to path
                        # We'll load the module explicitly later via importlib
                        break
                if so_extracted:
                    break
        
        # When the .so exists we rely on explicit importlib loading later
    except Exception as e:
        # Log the error for debugging but continue to try other import methods
        import traceback
        print(f"Warning: Failed to extract wheel: {e}")
        traceback.print_exc()
        pass


if typing.TYPE_CHECKING:
    # Import generated C++ stubs to make development easier
    from .lib.polychase_core import *
else:
    # Try to install wheel for Blender < 4.2
    _install_wheel_if_needed()
    
    # Try multiple import strategies
    polychase_core_imported = False
    import_error = None
    
    # Strategy 1: Direct import (works when wheel is installed by Blender 4.2+)
    try:
        import sys
        from polychase_core import *
        # Verify that key symbols were actually imported
        polychase_core_module = sys.modules.get('polychase_core')
        if 'Pose' in globals() or (polychase_core_module and hasattr(polychase_core_module, 'Pose')):
            polychase_core_imported = True
        else:
            # Import appeared to succeed but symbols aren't available
            # This can happen if the module loads but doesn't export properly
            # Try to get symbols from the module directly
            if polychase_core_module:
                try:
                    for name in dir(polychase_core_module):
                        if not name.startswith('_'):
                            value = getattr(polychase_core_module, name)
                            if not isinstance(value, type(sys)):
                                globals()[name] = value
                    # Check again if Pose is now available
                    if 'Pose' in globals():
                        polychase_core_imported = True
                    else:
                        import_error = ImportError("polychase_core imported but Pose symbol not found")
                except Exception:
                    import_error = ImportError("polychase_core imported but symbols not accessible")
            else:
                import_error = ImportError("polychase_core imported but module not in sys.modules")
    except ImportError as e:
        import_error = e
    
    # Strategy 2: Import from lib directory (for manually extracted wheels)
    if not polychase_core_imported:
        import sys
        import os
        import importlib.util
        addon_path = os.path.dirname(os.path.abspath(__file__))
        lib_dir = os.path.join(addon_path, 'lib')
        
        if os.path.exists(lib_dir):
            # Find the .so file and try to load it directly
            so_file = None
            for root, dirs, files in os.walk(lib_dir):
                for f in files:
                    if f.startswith('polychase_core') and (f.endswith('.so') or f.endswith('.pyd')):
                        so_file = os.path.join(root, f)
                        break
                if so_file:
                    break
            
            if so_file:
                # Try importing using importlib (more reliable for .so files)
                try:
                    spec = importlib.util.spec_from_file_location("polychase_core", so_file)
                    if spec and spec.loader:
                        # Register the module in sys.modules first
                        import sys
                        module = importlib.util.module_from_spec(spec)
                        sys.modules["polychase_core"] = module
                        spec.loader.exec_module(module)
                        # Import all symbols into current namespace
                        # Use dir() to get all attributes, including those from pybind11
                        # Then get the actual values from getattr()
                        current_module = sys.modules[__name__]
                        for name in dir(module):
                            if not name.startswith('_'):
                                try:
                                    value = getattr(module, name)
                                    # Only import non-module attributes (classes, functions, etc.)
                                    if not isinstance(value, type(sys)):
                                        globals()[name] = value
                                        setattr(current_module, name, value)
                                except (AttributeError, TypeError):
                                    pass
                        polychase_core_imported = True
                except Exception as e:
                    import_error = e
                    # Try to diagnose the issue - check if it's a dependency problem
                    try:
                        import ctypes
                        # Try loading with ctypes to see if it's a dependency issue
                        ctypes.CDLL(so_file, mode=ctypes.RTLD_GLOBAL)
                        # If ctypes can load it, the issue is with Python import, not dependencies
                    except Exception as ctypes_error:
                        # If ctypes also fails, it's likely a dependency issue
                        import_error = Exception(
                            f"Failed to load .so file. Import error: {e}. "
                            f"ctypes error: {ctypes_error}. "
                            f"This may indicate missing system dependencies or Python version mismatch."
                        )
            
            # If direct loading failed, try normal import
            if not polychase_core_imported:
                try:
                    from polychase_core import *
                    polychase_core_imported = True
                except ImportError as e:
                    import_error = e
    
    # Strategy 3: Relative import from .lib (for development)
    if not polychase_core_imported:
        try:
            from .lib.polychase_core import *
            polychase_core_imported = True
        except ImportError as e:
            import_error = e
    
    # Final verification: Ensure Pose and other key symbols are accessible
    # This handles cases where the module loads but symbols aren't properly exposed
    if polychase_core_imported:
        import sys
        polychase_core_module = sys.modules.get('polychase_core')
        if polychase_core_module:
            # Get the current module (core module) to set attributes on
            current_module = sys.modules.get(__name__)
            if current_module is None:
                # If module not in sys.modules yet, create a reference
                import types
                current_module = types.ModuleType(__name__)
                sys.modules[__name__] = current_module
            
            # Copy all public symbols from polychase_core to both globals() and module attributes
            key_symbols = ['Pose', 'CameraState', 'CameraIntrinsics', 'TrackerThread']
            for symbol_name in key_symbols:
                if hasattr(polychase_core_module, symbol_name):
                    value = getattr(polychase_core_module, symbol_name)
                    # Set in globals() for direct access
                    globals()[symbol_name] = value
                    # Set as module attribute for core.Pose access
                    setattr(current_module, symbol_name, value)
            
            # Also copy all other public symbols (not just key ones)
            for name in dir(polychase_core_module):
                if not name.startswith('_') and name not in key_symbols:
                    try:
                        value = getattr(polychase_core_module, name)
                        # Only import non-module attributes
                        if not isinstance(value, type(sys)):
                            globals()[name] = value
                            setattr(current_module, name, value)
                    except (AttributeError, TypeError):
                        pass
            
            # Final check: verify Pose is accessible
            if 'Pose' not in globals() and not hasattr(current_module, 'Pose'):
                # Some builds expose CameraPose instead of Pose; alias it if available
                if hasattr(polychase_core_module, 'CameraPose'):
                    value = getattr(polychase_core_module, 'CameraPose')
                    globals()['Pose'] = value
                    setattr(current_module, 'Pose', value)
                else:
                    polychase_core_imported = False
                    import_error = ImportError("polychase_core imported but Pose symbol not found after symbol copy")
    
    # If all imports failed, raise a helpful error
    if not polychase_core_imported:
        import sys
        import os
        addon_path = os.path.dirname(os.path.abspath(__file__))
        wheels_dir = os.path.join(addon_path, 'wheels')
        lib_dir = os.path.join(addon_path, 'lib')
        
        # Check what files are actually in lib_dir
        lib_contents = []
        if os.path.exists(lib_dir):
            for root, dirs, files in os.walk(lib_dir):
                for f in files:
                    lib_contents.append(os.path.relpath(os.path.join(root, f), lib_dir))
        
        # Check sys.path
        relevant_paths = [p for p in sys.path if 'polychase' in p.lower() or 'addon' in p.lower()]
        
        error_msg = (
            f"Failed to import polychase_core.\n"
            f"Addon path: {addon_path}\n"
            f"Wheels directory: {wheels_dir} ({'exists' if os.path.exists(wheels_dir) else 'missing'})\n"
            f"Lib directory: {lib_dir} ({'exists' if os.path.exists(lib_dir) else 'missing'})\n"
        )
        
        if lib_contents:
            error_msg += f"Files in lib directory: {', '.join(lib_contents[:10])}\n"
        else:
            error_msg += "Lib directory is empty or doesn't exist\n"
        
        if relevant_paths:
            error_msg += f"Relevant sys.path entries: {', '.join(relevant_paths[:5])}\n"
        
        if import_error:
            error_msg += f"Original import error: {import_error}\n"
        else:
            error_msg += "No import error captured\n"
        
        # Check Python version compatibility
        import sys
        import re
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        python_version_short = f"{sys.version_info.major}.{sys.version_info.minor}"
        error_msg += f"\nPython Version Information:\n"
        error_msg += f"  Blender version: {bpy.app.version_string}\n"
        error_msg += f"  Blender Python version: {python_version}\n"
        
        # Check which wheels are available
        available_wheels = []
        if os.path.exists(wheels_dir):
            for filename in os.listdir(wheels_dir):
                if filename.endswith('.whl'):
                    # Extract Python version from wheel name (e.g., cp312, cp311)
                    match = re.search(r'cp(\d)(\d+)', filename)
                    if match:
                        wheel_py_major = match.group(1)
                        wheel_py_minor = match.group(2)
                        available_wheels.append(f"Python {wheel_py_major}.{wheel_py_minor} (cp{wheel_py_major}{wheel_py_minor})")
        
        if available_wheels:
            error_msg += f"  Available wheels: {', '.join(available_wheels)}\n"
        else:
            error_msg += f"  No wheels found in wheels directory\n"
        
        # Check if we have a matching wheel
        required_wheel_tag = f"cp{sys.version_info.major}{sys.version_info.minor:02d}"
        has_matching_wheel = any(required_wheel_tag in w for w in available_wheels) if available_wheels else False
        
        if not has_matching_wheel:
            error_msg += f"\nSOLUTION: Python version mismatch detected!\n"
            error_msg += f"  Blender is using Python {python_version}, but no matching wheel found.\n"
            error_msg += f"  Required wheel tag: {required_wheel_tag}\n"
            error_msg += f"  Python extension modules (.so files) are version-specific and cannot be loaded across versions.\n\n"
            error_msg += f"Options to fix this:\n"
            error_msg += f"  1. Upgrade to Blender 4.2+ which supports automatic wheel installation via blender_manifest.toml\n"
            error_msg += f"  2. Build a wheel for Python {python_version_short} (see build_wheel.sh in the repository)\n"
            error_msg += f"  3. Contact the maintainer to request a wheel for Python {python_version_short}\n"
        
        raise ImportError(error_msg) from None


class _Trackers:

    def __init__(self):
        self.trackers = {}

    def get_tracker(self, id: int, geom: bpy.types.Object):
        geom_id = geom.id_data.name_full

        if id not in self.trackers or self.trackers[id].geom_id != geom_id:
            self.trackers[id] = Tracker(id, geom)

        assert id in self.trackers
        tracker = self.trackers[id]
        tracker.geom = geom    # Update geom as well
        return tracker

    def delete_tracker(self, id: int):
        if id in self.trackers:
            del self.trackers[id]


Trackers = _Trackers()


class PinModeData:

    _tracker_id: int
    _points: np.ndarray
    _is_selected: np.ndarray
    _distances: np.ndarray  # Optional distance constraints per pin (NaN = no constraint)
    _points_version_number: int
    _selected_pin_idx: int

    def __init__(self, tracker_id: int):
        self._tracker_id = tracker_id
        self._points = np.empty((0, 3), dtype=np.float32)
        self._is_selected = np.empty((0,), dtype=np.uint32)
        self._distances = np.empty((0,), dtype=np.float32)
        self._points_version_number = 0
        self._selected_pin_idx = -1

    def reset_points_if_necessary(self, tracker: properties.PolychaseTracker):
        if tracker.points_version_number != self._points_version_number:
            if tracker.points_version_number == 0:
                assert tracker.selected_pin_idx == -1
                self._points = np.empty((0, 3), dtype=np.float32)
                self._is_selected = np.empty((0,), dtype=np.uint32)
                self._distances = np.empty((0,), dtype=np.float32)
                self._selected_pin_idx = -1
            else:
                self._points = np.frombuffer(
                    tracker.points, dtype=np.float32).reshape((-1, 3))
                self._is_selected = np.zeros(
                    (self._points.shape[0],), dtype=np.uint32)
                # Load distances if available
                if tracker.pin_distances:
                    distances_raw = np.frombuffer(
                        tracker.pin_distances, dtype=np.float32)
                    if len(distances_raw) == len(self._points):
                        self._distances = distances_raw.copy()
                    else:
                        # Mismatch - initialize with NaN (no constraints)
                        self._distances = np.full(
                            (len(self._points),), np.nan, dtype=np.float32)
                else:
                    # No distances stored - initialize with NaN
                    self._distances = np.full(
                        (len(self._points),), np.nan, dtype=np.float32)
                self._selected_pin_idx = tracker.selected_pin_idx
                if self._selected_pin_idx > 0:
                    self._is_selected[self._selected_pin_idx] = 1

            self._points_version_number = tracker.points_version_number

        if tracker.selected_pin_idx != self._selected_pin_idx:
            self._is_selected[self._selected_pin_idx] = 0
            self._is_selected[tracker.selected_pin_idx] = 1
            self._selected_pin_idx = tracker.selected_pin_idx

    def _update_points(self, tracker: properties.PolychaseTracker):
        assert self._points_version_number == tracker.points_version_number

        self._points_version_number += 1
        tracker.points_version_number += 1

        tracker.points = self._points.tobytes()
        # Update distances to match points length
        if len(self._distances) != len(self._points):
            # Resize distances array to match points
            if len(self._distances) < len(self._points):
                # Add NaN for new pins
                new_distances = np.full(
                    (len(self._points) - len(self._distances),), 
                    np.nan, dtype=np.float32)
                self._distances = np.append(self._distances, new_distances)
            else:
                # Truncate if points were deleted
                self._distances = self._distances[:len(self._points)]
        tracker.pin_distances = self._distances.tobytes()

    def _update_selected_pin_idx(
            self, idx, tracker: properties.PolychaseTracker):
        assert self._selected_pin_idx == tracker.selected_pin_idx

        self._selected_pin_idx = idx
        tracker.selected_pin_idx = idx

    @property
    def points(self) -> np.ndarray:
        tracker = properties.PolychaseState.get_tracker_by_id(self._tracker_id)
        assert tracker

        self.reset_points_if_necessary(tracker)
        return self._points

    @property
    def is_selected(self) -> np.ndarray:
        tracker = properties.PolychaseState.get_tracker_by_id(self._tracker_id)
        assert tracker

        self.reset_points_if_necessary(tracker)
        return self._is_selected
    
    @property
    def distances(self) -> np.ndarray:
        """Get distance constraints for pins. NaN means no constraint."""
        tracker = properties.PolychaseState.get_tracker_by_id(self._tracker_id)
        assert tracker

        self.reset_points_if_necessary(tracker)
        return self._distances
    
    def get_pin_distance(self, pin_idx: int) -> float | None:
        """Get distance constraint for a specific pin. Returns None if no constraint."""
        distances = self.distances
        if pin_idx < 0 or pin_idx >= len(distances):
            return None
        dist = distances[pin_idx]
        if np.isnan(dist) or dist <= 0:
            return None
        return float(dist)
    
    def set_pin_distance(self, pin_idx: int, distance: float | None):
        """Set distance constraint for a specific pin. None removes the constraint."""
        tracker = properties.PolychaseState.get_tracker_by_id(self._tracker_id)
        assert tracker

        self.reset_points_if_necessary(tracker)
        
        if pin_idx < 0 or pin_idx >= len(self._distances):
            return
        
        if distance is None or distance <= 0:
            self._distances[pin_idx] = np.nan
        else:
            self._distances[pin_idx] = float(distance)
        
        self._update_points(tracker)

    def is_out_of_date(self) -> bool:
        tracker = properties.PolychaseState.get_tracker_by_id(self._tracker_id)
        assert tracker

        return self._points_version_number != tracker.points_version_number

    def create_pin(self, point: np.ndarray, select: bool = False, distance: float | None = None):
        tracker = properties.PolychaseState.get_tracker_by_id(self._tracker_id)
        assert tracker

        self.reset_points_if_necessary(tracker)

        self._points = np.append(
            self._points, np.array([point], dtype=np.float32), axis=0)
        self._is_selected = np.append(
            self._is_selected, np.array([0], dtype=np.uint32), axis=0)
        # Add distance constraint (NaN if not provided)
        dist_value = np.nan if distance is None or distance <= 0 else float(distance)
        self._distances = np.append(
            self._distances, np.array([dist_value], dtype=np.float32), axis=0)
        self._update_points(tracker)

        if select:
            self.select_pin(len(self._points) - 1)

    def delete_pin(self, idx: int):
        tracker = properties.PolychaseState.get_tracker_by_id(self._tracker_id)
        assert tracker

        self.reset_points_if_necessary(tracker)

        if idx < 0 or idx >= len(self._points):
            return

        if self._selected_pin_idx == idx:
            self._update_selected_pin_idx(-1, tracker)
        elif self._selected_pin_idx > idx:
            self._update_selected_pin_idx(self._selected_pin_idx - 1, tracker)

        self._points = np.delete(self._points, idx, axis=0)
        self._is_selected = np.delete(self._is_selected, idx, axis=0)
        self._distances = np.delete(self._distances, idx, axis=0)

        self._update_points(tracker)

    def select_pin(self, pin_idx: int):
        tracker = properties.PolychaseState.get_tracker_by_id(self._tracker_id)
        assert tracker

        self.reset_points_if_necessary(tracker)

        self.unselect_pin()
        self._update_selected_pin_idx(pin_idx, tracker)
        self._is_selected[self._selected_pin_idx] = 1

    def unselect_pin(self):
        tracker = properties.PolychaseState.get_tracker_by_id(self._tracker_id)
        assert tracker

        self.reset_points_if_necessary(tracker)

        if self._selected_pin_idx != -1:
            self._is_selected[self._selected_pin_idx] = 0
        self._update_selected_pin_idx(-1, tracker)


class Tracker:

    def __init__(self, tracker_id: int, geom: bpy.types.Object):
        geom_id = geom.id_data.name_full

        self.tracker_id = tracker_id
        self.geom_id = geom_id
        self.geom = geom
        self.pin_mode = PinModeData(tracker_id=self.tracker_id)

        self.init_accel_mesh()

    def init_accel_mesh(self):
        tracker = properties.PolychaseState.get_tracker_by_id(self.tracker_id)
        assert tracker

        geom = tracker.geometry
        assert geom

        depsgraph = bpy.context.evaluated_depsgraph_get()
        evaluated_geom = geom.evaluated_get(depsgraph)
        mesh = evaluated_geom.to_mesh()

        mesh.calc_loop_triangles()

        num_vertices = len(mesh.vertices)
        num_triangles = len(mesh.loop_triangles)

        vertices: np.ndarray = np.empty((num_vertices, 3), dtype=np.float32)
        triangles: np.ndarray = np.empty((num_triangles, 3), dtype=np.uint32)
        triangle_polygons: np.ndarray = np.empty(
            (num_triangles,), dtype=np.uint32)

        assert len(mesh.loop_triangles) == len(mesh.loop_triangle_polygons)

        mesh.vertices.foreach_get("co", vertices.ravel())
        mesh.loop_triangles.foreach_get("vertices", triangles.ravel())
        mesh.loop_triangle_polygons.foreach_get(
            "value", triangle_polygons.ravel())

        # Sort triangles and triangle_polygons
        sort_indices = np.argsort(triangle_polygons, axis=0)
        triangles = triangles[sort_indices]
        triangle_polygons = triangle_polygons[sort_indices]

        masked_triangles: np.ndarray
        if hasattr(self, "accel_mesh"):
            masked_triangles = self.accel_mesh.inner().masked_triangles
        else:
            # Check if masked_triangles buffer is valid
            if tracker.masked_triangles and len(tracker.masked_triangles) > 0:
                # Blender v5.0.0 introduced a bug where BYTE_STRING properties insert
                # an extra null terminator.
                # See: https://projects.blender.org/blender/blender/issues/150431
                length = len(
                    tracker.masked_triangles) - len(tracker.masked_triangles) % 4
                if length > 0:
                    masked_triangles = np.frombuffer(
                        tracker.masked_triangles[:length], dtype=np.uint32)
                else:
                    # Invalid buffer size, create empty array
                    masked_triangles = np.empty((0,), dtype=np.uint32)
            else:
                # Empty buffer, create empty array
                masked_triangles = np.empty((0,), dtype=np.uint32)

        try:
            self.accel_mesh = AcceleratedMesh(
                vertices, triangles, masked_triangles)
        except:
            self.accel_mesh = AcceleratedMesh(vertices, triangles)
            tracker.masked_triangles = self.accel_mesh.inner(
            ).masked_triangles.tobytes()

        # Are we sure we want to store edges here?
        self.edges_indices = np.empty((len(mesh.edges), 2), dtype=np.uint32)
        mesh.edges.foreach_get("vertices", self.edges_indices.ravel())

        # Are we also sure that we want to handle polygon/triangle mapping here,
        # and not in C++ land?
        self.triangle_polygons = triangle_polygons

    def ray_cast(
            self,
            region: bpy.types.Region,
            rv3d: bpy.types.RegionView3D,
            region_x: int,
            region_y: int,
            check_mask: bool):
        return ray_cast(
            accel_mesh=self.accel_mesh,
            scene_transform=SceneTransformations(
                model_matrix=typing.cast(np.ndarray, self.geom.matrix_world),
                view_matrix=typing.cast(np.ndarray, rv3d.view_matrix),
                intrinsics=camera_intrinsics_from_proj(rv3d.window_matrix),
            ),
            pos=typing.cast(np.ndarray, utils.ndc(region, region_x, region_y)),
            check_mask=check_mask,
        )

    def set_polygon_mask_using_triangle_idx(self, tri_idx: int):
        polygon = self.triangle_polygons[tri_idx]
        idx = tri_idx
        while idx < len(self.triangle_polygons
                       ) and self.triangle_polygons[idx] == polygon:
            self.accel_mesh.inner_mut().mask_triangle(idx)
            idx += 1

        idx = tri_idx - 1
        while idx >= 0 and self.triangle_polygons[idx] == polygon:
            self.accel_mesh.inner_mut().mask_triangle(idx)
            idx -= 1

    def clear_polygon_mask_using_triangle_idx(self, tri_idx: int):
        polygon = self.triangle_polygons[tri_idx]
        idx = tri_idx
        while idx < len(self.triangle_polygons
                       ) and self.triangle_polygons[idx] == polygon:
            self.accel_mesh.inner_mut().unmask_triangle(idx)
            idx += 1

        idx = tri_idx - 1
        while idx >= 0 and self.triangle_polygons[idx] == polygon:
            self.accel_mesh.inner_mut().unmask_triangle(idx)
            idx -= 1

    @classmethod
    def get(
        cls,
        tracker: properties.PolychaseTracker,
    ) -> typing.Self | None:
        return Trackers.get_tracker(
            tracker.id, tracker.geometry) if tracker.geometry else None


# TODO: Remove these from here?
def camera_intrinsics(
    camera: bpy.types.Object,
    width: float = 1.0,
    height: float = 1.0,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> CameraIntrinsics:
    assert isinstance(camera.data, bpy.types.Camera)

    return camera_intrinsics_expanded(
        lens=camera.data.lens,
        shift_x=camera.data.shift_x,
        shift_y=camera.data.shift_y,
        sensor_width=camera.data.sensor_width,
        sensor_height=camera.data.sensor_height,
        sensor_fit=camera.data.sensor_fit,
        width=width,
        height=height,
        scale_x=scale_x,
        scale_y=scale_y,
    )


def camera_intrinsics_expanded(
    lens: float,
    shift_x: float,
    shift_y: float,
    sensor_width: float,
    sensor_height: float,
    sensor_fit: str,
    width: float = 1.0,
    height: float = 1.0,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
):
    fx, fy, cx, cy = utils.calc_camera_params_expanded(
        lens=lens,
        shift_x=shift_x,
        shift_y=shift_y,
        sensor_width=sensor_width,
        sensor_height=sensor_height,
        sensor_fit=sensor_fit,
        width=width,
        height=height,
        scale_x=scale_x,
        scale_y=scale_y,
    )
    return CameraIntrinsics(
        fx=-fx,
        fy=-fy,
        cx=-cx,
        cy=-cy,
        aspect_ratio=fx / fy,
        width=width,
        height=height,
        convention=CameraConvention.OpenGL,
    )


def set_camera_intrinsics(
        camera: bpy.types.Object, intrinsics: CameraIntrinsics):
    utils.set_camera_params(
        camera,
        intrinsics.width,
        intrinsics.height,
        -intrinsics.fx,
        -intrinsics.fy,
        -intrinsics.cx,
        -intrinsics.cy,
    )


def camera_intrinsics_from_proj(
        proj: mathutils.Matrix,
        width: float = 2.0,
        height: float = 2.0) -> CameraIntrinsics:
    fx, fy, cx, cy = utils.calc_camera_params_from_proj(proj)
    return CameraIntrinsics(
        fx=-fx,
        fy=-fy,
        cx=-cx,
        cy=-cy,
        aspect_ratio=fx / fy,
        width=width,
        height=height,
        convention=CameraConvention.OpenGL,
    )
