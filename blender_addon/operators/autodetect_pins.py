# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 Ahmed Essam <aessam.dahy@gmail.com>

import typing
import math

import bpy
import bpy.types
import mathutils
import numpy as np

from .. import core, utils
from ..properties import PolychaseTracker, PolychaseState

# Try to import OpenCV for feature detection
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def detect_features_opencv(clip: bpy.types.MovieClip, frame_number: int,
                           max_features: int = 100,
                           quality_level: float = 0.01,
                           min_distance: float = 10.0) -> list[tuple[float, float]]:
    """
    Detect features in a movie clip frame using OpenCV.
    
    Args:
        clip: Blender movie clip
        frame_number: Frame number to detect features in
        max_features: Maximum number of features to detect
        quality_level: Quality threshold (0-1, lower = more features)
        min_distance: Minimum distance between features (pixels)
    
    Returns:
        List of (x, y) feature coordinates in normalized clip coordinates [0, 1]
    """
    if not CV2_AVAILABLE:
        return []
    
    # Get clip dimensions
    width, height = clip.size
    
    # Load frame as image
    clip.frame_set(frame_number)
    frame_path = clip.frame_path
    
    # Read image using OpenCV
    # frame_path is relative, need to make it absolute
    import os
    abs_frame_path = bpy.path.abspath(frame_path)
    
    if not os.path.exists(abs_frame_path):
        return []
    
    img = cv2.imread(abs_frame_path)
    if img is None:
        return []
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect features using Good Features to Track (GFTT)
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_features,
        qualityLevel=quality_level,
        minDistance=min_distance,
        blockSize=3,
        useHarrisDetector=False,
        k=0.04
    )
    
    if corners is None:
        return []
    
    # Convert to normalized coordinates [0, 1]
    features = []
    for corner in corners:
        x, y = corner[0]
        # Normalize to [0, 1] range
        features.append((x / width, y / height))
    
    return features


def detect_features_blender_tracking(clip: bpy.types.MovieClip, 
                                     frame_number: int) -> list[tuple[float, float]]:
    """
    Detect features using Blender's motion tracking markers.
    
    Args:
        clip: Blender movie clip
        frame_number: Frame number to detect features in
    
    Returns:
        List of (x, y) feature coordinates in normalized clip coordinates [0, 1]
    """
    if not clip.tracking:
        return []
    
    width, height = clip.size
    features = []
    
    # Get all tracking markers that are enabled and have data at this frame
    for track in clip.tracking.tracks:
        if not track.mute:
            # Find marker at or near current frame
            marker = track.markers.find_frame(frame_number)
            if marker and marker.mute == 0:
                # Get normalized coordinates
                co = marker.co
                features.append((co[0], co[1]))
    
    return features


def filter_features_by_distance(features: list[tuple[float, float]],
                                min_distance: float) -> list[tuple[float, float]]:
    """
    Filter features to ensure minimum spacing between them.
    
    Uses a simple greedy algorithm: keep features that are far enough from
    previously kept features.
    
    Args:
        features: List of (x, y) normalized coordinates
        min_distance: Minimum distance in normalized coordinates [0, 1]
    
    Returns:
        Filtered list of features
    """
    if not features:
        return []
    
    filtered = [features[0]]
    
    for feat in features[1:]:
        # Check distance to all previously kept features
        too_close = False
        for kept in filtered:
            dx = feat[0] - kept[0]
            dy = feat[1] - kept[1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < min_distance:
                too_close = True
                break
        
        if not too_close:
            filtered.append(feat)
    
    return filtered


def clip_coords_to_ndc(clip: bpy.types.MovieClip,
                       clip_x: float, clip_y: float,
                       camera: bpy.types.Object) -> np.ndarray:
    """
    Convert normalized clip coordinates [0, 1] to NDC coordinates for ray-casting.
    
    Blender's clip coordinates are normalized [0, 1] where (0, 0) is bottom-left.
    NDC coordinates are [-1, 1] where (0, 0) is center.
    
    The CameraIntrinsics.Unproject function expects NDC coordinates where:
    - X: [-1, 1], -1 is left, 1 is right
    - Y: [-1, 1], -1 is bottom, 1 is top
    
    Args:
        clip: Movie clip
        clip_x, clip_y: Normalized clip coordinates [0, 1] (Blender format: bottom-left origin)
        camera: Camera object
    
    Returns:
        NDC coordinates as numpy array [x, y]
    """
    # Convert normalized clip coords [0, 1] to NDC [-1, 1]
    # Clip: (0, 0) = bottom-left, (1, 1) = top-right
    # NDC: (-1, -1) = bottom-left, (1, 1) = top-right
    # So we just scale and offset: ndc = clip * 2.0 - 1.0
    
    # For Y, clip has bottom-left origin, NDC also has bottom-left origin
    # So no flipping needed
    ndc_x = clip_x * 2.0 - 1.0
    ndc_y = clip_y * 2.0 - 1.0
    
    return np.array([ndc_x, ndc_y], dtype=np.float32)


class PC_OT_AutodetectPins(bpy.types.Operator):
    bl_idname = "polychase.autodetect_pins"
    bl_options = {"REGISTER", "UNDO"}
    bl_label = "Auto-detect Pins"
    bl_description = "Automatically detect tracking features and create pins on geometry"
    
    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        state = PolychaseState.from_context(context)
        if not state:
            return False
        
        tracker = state.active_tracker
        if not tracker:
            return False
        
        return (
            tracker.clip is not None and
            tracker.camera is not None and
            tracker.geometry is not None
        )
    
    def execute(self, context: bpy.types.Context) -> set:
        state = PolychaseState.from_context(context)
        if not state:
            self.report({"ERROR"}, "Polychase state not found")
            return {"CANCELLED"}
        
        tracker = state.active_tracker
        if not tracker:
            self.report({"ERROR"}, "No active tracker")
            return {"CANCELLED"}
        
        clip = tracker.clip
        camera = tracker.camera
        geometry = tracker.geometry
        
        if not clip or not camera or not geometry:
            self.report({"ERROR"}, "Clip, camera, or geometry not set")
            return {"CANCELLED"}
        
        # Get current frame
        current_frame = context.scene.frame_current
        
        # Check if frame is within clip range
        clip_start = clip.frame_start
        clip_end = clip.frame_start + clip.frame_duration - 1
        if current_frame < clip_start or current_frame > clip_end:
            self.report(
                {"ERROR"},
                f"Current frame {current_frame} is outside clip range [{clip_start}, {clip_end}]"
            )
            return {"CANCELLED"}
        
        # Get tracker core for ray-casting
        tracker_core = core.Tracker.get(tracker)
        if not tracker_core:
            self.report({"ERROR"}, "Failed to get tracker core")
            return {"CANCELLED"}
        
        # Get viewport region and 3D view data for scene transformation
        # We need to find the 3D viewport
        area = None
        region = None
        rv3d = None
        
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                for region in area.regions:
                    if region.type == 'WINDOW':
                        space = area.spaces.active
                        if isinstance(space, bpy.types.SpaceView3D):
                            rv3d = space.region_3d
                            break
                if rv3d:
                    break
        
        if not rv3d:
            self.report(
                {"ERROR"},
                "Could not find 3D viewport. Please ensure a 3D viewport is open."
            )
            return {"CANCELLED"}
        
        # Detect features
        features = []
        
        # Try Blender tracking markers first (if available)
        if tracker.autodetect_use_blender_tracking:
            features = detect_features_blender_tracking(clip, current_frame)
        
        # Fall back to OpenCV if no Blender tracking or if explicitly requested
        if not features or tracker.autodetect_use_opencv:
            if not CV2_AVAILABLE:
                self.report(
                    {"WARNING"},
                    "OpenCV not available. Install with: pip install opencv-python"
                )
            else:
                opencv_features = detect_features_opencv(
                    clip,
                    current_frame,
                    max_features=tracker.autodetect_max_pins,
                    quality_level=tracker.autodetect_quality_threshold,
                    min_distance=tracker.autodetect_min_distance
                )
                features.extend(opencv_features)
        
        if not features:
            self.report({"WARNING"}, "No features detected in current frame")
            return {"CANCELLED"}
        
        # Filter features for good distribution
        if tracker.autodetect_min_spacing > 0:
            features = filter_features_by_distance(
                features,
                tracker.autodetect_min_spacing
            )
        
        # Limit to maximum number of pins
        if len(features) > tracker.autodetect_max_pins:
            features = features[:tracker.autodetect_max_pins]
        
        # Get pin mode data
        pin_mode_data = tracker_core.pin_mode
        
        # Ray-cast each feature onto geometry and create pins
        successful_pins = 0
        failed_pins = 0
        
        # Create scene transformation for ray-casting
        # Use camera's actual view matrix and intrinsics based on clip size
        clip_width, clip_height = clip.size
        scene_transform = core.SceneTransformations(
            model_matrix=typing.cast(np.ndarray, geometry.matrix_world),
            view_matrix=typing.cast(np.ndarray, camera.matrix_world.inverted()),
            intrinsics=core.camera_intrinsics(camera, width=clip_width, height=clip_height),
        )
        
        for feat_x, feat_y in features:
            # Convert clip coordinates to NDC
            ndc = clip_coords_to_ndc(clip, feat_x, feat_y, camera)
            
            # Ray-cast onto geometry
            rayhit = core.ray_cast(
                accel_mesh=tracker_core.accel_mesh,
                scene_transform=scene_transform,
                pos=ndc,
                check_mask=True  # Respect masked triangles
            )
            
            if rayhit:
                # Create pin at hit location
                pin_mode_data.create_pin(rayhit.pos, select=False)
                successful_pins += 1
            else:
                failed_pins += 1
        
        # Report results
        if successful_pins > 0:
            self.report(
                {"INFO"},
                f"Created {successful_pins} pins from {len(features)} detected features"
            )
            if failed_pins > 0:
                self.report(
                    {"WARNING"},
                    f"{failed_pins} features did not hit geometry"
                )
        else:
            self.report(
                {"WARNING"},
                f"No pins created. {failed_pins} features did not hit geometry"
            )
        
        return {"FINISHED"}

