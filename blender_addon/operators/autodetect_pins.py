# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 Ahmed Essam <aessam.dahy@gmail.com>

import typing
import math
import os

import bpy
import bpy.types
import numpy as np

from .. import core
from ..properties import PolychaseState

# Try to import OpenCV for feature detection
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def _scene_frame_to_clip_frame(clip: bpy.types.MovieClip, scene_frame: int) -> int:
    """Convert a scene frame index to the clip-local frame index (1-based)."""
    clip_start = int(getattr(clip, "frame_start", 1))
    frame_offset = int(getattr(clip, "frame_offset", 0))
    return frame_offset + (scene_frame - clip_start) + 1


def _clip_frame_bounds(clip: bpy.types.MovieClip) -> tuple[int, int]:
    """Return inclusive clip frame bounds in clip-local indices (1-based)."""
    frame_offset = int(getattr(clip, "frame_offset", 0))
    duration = max(int(getattr(clip, "frame_duration", 0)), 0)
    first = frame_offset + 1
    last = frame_offset + duration if duration > 0 else frame_offset + 1
    return first, last


def _resolve_clip_frame_path(clip: bpy.types.MovieClip, frame_number: int) -> str | None:
    """Return absolute path to the cached frame image if Blender generated one."""
    frame_path_attr = getattr(clip, "frame_path", None)
    path: str | None = None

    if callable(frame_path_attr):
        try:
            path = frame_path_attr(frame_number)
        except TypeError:
            try:
                path = frame_path_attr()
            except TypeError:
                path = None
    elif isinstance(frame_path_attr, str):
        path = frame_path_attr

    if not path:
        return None

    abs_path = bpy.path.abspath(path)
    return abs_path


def _decode_movie_frame_with_opencv(clip: bpy.types.MovieClip,
                                    frame_number: int) -> np.ndarray | None:
    """Decode a frame directly from the movie file using OpenCV."""
    clip_filepath = getattr(clip, "filepath", "")
    if not clip_filepath:
        return None

    abs_clip_path = bpy.path.abspath(clip_filepath)
    if not abs_clip_path or not os.path.exists(abs_clip_path):
        return None

    try:
        capture = cv2.VideoCapture(abs_clip_path)
    except Exception:
        return None

    if not capture or not capture.isOpened():
        if capture:
            capture.release()
        return None

    try:
        target_index = max(frame_number - 1, 0)
        capture.set(cv2.CAP_PROP_POS_FRAMES, float(target_index))
        success, frame = capture.read()
    finally:
        capture.release()

    if not success or frame is None:
        return None

    return frame


def detect_features_opencv(clip: bpy.types.MovieClip, frame_number: int,
                           max_features: int = 100,
                           quality_level: float = 0.01,
                           min_distance: float = 10.0) -> list[tuple[float, float]]:
    """
    Detect features in a movie clip frame using OpenCV.

    Args:
        clip: Blender movie clip
        frame_number: Clip-local frame number (1-based)
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

    # Validate frame range (defensive)
    clip_first, clip_last = _clip_frame_bounds(clip)
    if frame_number < clip_first or frame_number > clip_last:
        return []

    # Load frame as image (prefer cached frame, fallback to decoding source video)
    abs_frame_path = _resolve_clip_frame_path(clip, frame_number)

    img = None
    if abs_frame_path and os.path.exists(abs_frame_path):
        img = cv2.imread(abs_frame_path)

    if img is None and getattr(clip, "source", "MOVIE") == "MOVIE":
        img = _decode_movie_frame_with_opencv(clip, frame_number)

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
        frame_number: Clip-local frame number (1-based)

    Returns:
        List of (x, y) feature coordinates in normalized clip coordinates [0, 1]
    """
    clip_first, clip_last = _clip_frame_bounds(clip)
    if frame_number < clip_first or frame_number > clip_last:
        return []

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
    Convert normalized clip coordinates [0, 1] to NDC coordinates [-1, 1].

    Args:
        clip: Movie clip
        clip_x, clip_y: Normalized clip coordinates [0, 1]
        camera: Camera object (unused, kept for API compatibility)

    Returns:
        NDC coordinates as numpy array [x, y]
    """
    ndc_scale = 2.0
    ndc_offset = 1.0
    ndc_x = clip_x * ndc_scale - ndc_offset
    ndc_y = clip_y * ndc_scale - ndc_offset

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

        # Check if frame is within clip range (scene timeline)
        clip_scene_start = int(getattr(clip, "frame_start", 1))
        clip_scene_end = clip_scene_start + max(int(getattr(clip, "frame_duration", 0)) - 1, 0)
        if current_frame < clip_scene_start or current_frame > clip_scene_end:
            self.report(
                {"ERROR"},
                (
                    f"Current frame {current_frame} is outside clip range "
                    f"[{clip_scene_start}, {clip_scene_end}]"
                )
            )
            return {"CANCELLED"}

        # Convert to clip-local frame index and validate against actual video frames
        clip_frame_number = _scene_frame_to_clip_frame(clip, current_frame)
        clip_frame_start, clip_frame_end = _clip_frame_bounds(clip)
        if clip_frame_number < clip_frame_start or clip_frame_number > clip_frame_end:
            self.report(
                {"ERROR"},
                (
                    f"Current frame {current_frame} maps to clip frame {clip_frame_number}, "
                    f"but valid clip frames are [{clip_frame_start}, {clip_frame_end}]"
                )
            )
            return {"CANCELLED"}

        # Get tracker core for ray-casting
        tracker_core = core.Tracker.get(tracker)
        if not tracker_core:
            self.report({"ERROR"}, "Failed to get tracker core")
            return {"CANCELLED"}

        rv3d = None
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                space = area.spaces.active
                if isinstance(space, bpy.types.SpaceView3D):
                    rv3d = space.region_3d
                    break

        if not rv3d:
            self.report(
                {"ERROR"},
                "Could not find 3D viewport. Please ensure a 3D viewport is open."
            )
            return {"CANCELLED"}

        features = []
        if tracker.autodetect_use_blender_tracking:
            features = detect_features_blender_tracking(clip, clip_frame_number)

        if not features or tracker.autodetect_use_opencv:
            if not CV2_AVAILABLE:
                self.report(
                    {"WARNING"},
                    "OpenCV not available. Install with: pip install opencv-python"
                )
            else:
                opencv_features = detect_features_opencv(
                    clip,
                    clip_frame_number,
                    max_features=tracker.autodetect_max_pins,
                    quality_level=tracker.autodetect_quality_threshold,
                    min_distance=tracker.autodetect_min_distance
                )
                features.extend(opencv_features)

        if not features:
            self.report({"WARNING"}, "No features detected in current frame")
            return {"CANCELLED"}

        if tracker.autodetect_min_spacing > 0:
            features = filter_features_by_distance(features, tracker.autodetect_min_spacing)

        if len(features) > tracker.autodetect_max_pins:
            features = features[:tracker.autodetect_max_pins]

        pin_mode_data = tracker_core.pin_mode
        successful_pins = 0
        failed_pins = 0
        clip_width, clip_height = clip.size
        scene_transform = core.SceneTransformations(
            model_matrix=typing.cast(np.ndarray, geometry.matrix_world),
            view_matrix=typing.cast(np.ndarray, camera.matrix_world.inverted()),
            intrinsics=core.camera_intrinsics(camera, width=clip_width, height=clip_height),
        )

        for feat_x, feat_y in features:
            ndc = clip_coords_to_ndc(clip, feat_x, feat_y, camera)
            rayhit = core.ray_cast(
                accel_mesh=tracker_core.accel_mesh,
                scene_transform=scene_transform,
                pos=ndc,
                check_mask=True
            )

            if rayhit:
                pin_mode_data.create_pin(rayhit.pos, select=False)
                successful_pins += 1
            else:
                failed_pins += 1
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

