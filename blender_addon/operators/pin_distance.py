# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 Ahmed Essam <aessam.dahy@gmail.com>

import bpy
import bpy.types
import numpy as np

from .. import core
from ..properties import PolychaseState, PolychaseTracker


class PC_OT_SetPinDistance(bpy.types.Operator):
    bl_idname = "polychase.set_pin_distance"
    bl_options = {"REGISTER", "UNDO"}
    bl_label = "Set Pin Distance"
    bl_description = "Set distance constraint for selected pin"
    
    distance: bpy.props.FloatProperty(
        name="Distance",
        description="Distance from camera to pin (in Blender units). Set to 0 to remove constraint.",
        default=0.0,
        min=0.0,
        soft_max=1000.0,
        unit="LENGTH"
    )
    
    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        state = PolychaseState.from_context(context)
        if not state:
            return False
        
        tracker = state.active_tracker
        if not tracker:
            return False
        
        return tracker.selected_pin_idx >= 0
    
    def invoke(self, context: bpy.types.Context, event: bpy.types.Event) -> set:
        state = PolychaseState.from_context(context)
        if not state:
            return {"CANCELLED"}
        
        tracker = state.active_tracker
        if not tracker or tracker.selected_pin_idx < 0:
            return {"CANCELLED"}
        
        # Get current distance if set
        tracker_core = core.Tracker.get(tracker)
        if tracker_core:
            pin_mode = tracker_core.pin_mode
            current_distance = pin_mode.get_pin_distance(tracker.selected_pin_idx)
            if current_distance is not None:
                self.distance = current_distance
        
        return context.window_manager.invoke_props_dialog(self)
    
    def execute(self, context: bpy.types.Context) -> set:
        state = PolychaseState.from_context(context)
        if not state:
            self.report({"ERROR"}, "Polychase state not found")
            return {"CANCELLED"}
        
        tracker = state.active_tracker
        if not tracker or tracker.selected_pin_idx < 0:
            self.report({"ERROR"}, "No pin selected")
            return {"CANCELLED"}
        
        tracker_core = core.Tracker.get(tracker)
        if not tracker_core:
            self.report({"ERROR"}, "Failed to get tracker core")
            return {"CANCELLED"}
        
        pin_mode = tracker_core.pin_mode
        
        # Set distance (None if 0 or negative)
        if self.distance > 0:
            pin_mode.set_pin_distance(tracker.selected_pin_idx, self.distance)
            self.report(
                {"INFO"},
                f"Set distance constraint: {self.distance:.3f} Blender units"
            )
        else:
            pin_mode.set_pin_distance(tracker.selected_pin_idx, None)
            self.report({"INFO"}, "Removed distance constraint")
        
        return {"FINISHED"}


class PC_OT_SnapPinToDistance(bpy.types.Operator):
    bl_idname = "polychase.snap_pin_to_distance"
    bl_options = {"REGISTER", "UNDO"}
    bl_label = "Snap Pin to Known Distance"
    bl_description = "Move pin to match a known distance from camera"
    
    distance: bpy.props.FloatProperty(
        name="Known Distance",
        description="Known distance from camera to pin (in Blender units)",
        default=1.0,
        min=0.001,
        soft_max=1000.0,
        unit="LENGTH"
    )
    
    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        state = PolychaseState.from_context(context)
        if not state:
            return False
        
        tracker = state.active_tracker
        if not tracker:
            return False
        
        return (
            tracker.selected_pin_idx >= 0 and
            tracker.camera is not None and
            tracker.geometry is not None
        )
    
    def invoke(self, context: bpy.types.Context, event: bpy.types.Event) -> set:
        state = PolychaseState.from_context(context)
        if not state:
            return {"CANCELLED"}
        
        tracker = state.active_tracker
        if not tracker or tracker.selected_pin_idx < 0:
            return {"CANCELLED"}
        
        # Calculate current distance
        tracker_core = core.Tracker.get(tracker)
        if tracker_core:
            pin_mode = tracker_core.pin_mode
            points = pin_mode.points
            if tracker.selected_pin_idx < len(points):
                pin_pos_world = np.array(tracker.geometry.matrix_world) @ np.append(
                    points[tracker.selected_pin_idx], 1.0)
                camera_pos = np.array(tracker.camera.matrix_world.translation)
                current_distance = np.linalg.norm(pin_pos_world[:3] - camera_pos)
                self.distance = current_distance
        
        return context.window_manager.invoke_props_dialog(self)
    
    def execute(self, context: bpy.types.Context) -> set:
        state = PolychaseState.from_context(context)
        if not state:
            self.report({"ERROR"}, "Polychase state not found")
            return {"CANCELLED"}
        
        tracker = state.active_tracker
        if not tracker or tracker.selected_pin_idx < 0:
            self.report({"ERROR"}, "No pin selected")
            return {"CANCELLED"}
        
        if not tracker.camera or not tracker.geometry:
            self.report({"ERROR"}, "Camera or geometry not set")
            return {"CANCELLED"}
        
        tracker_core = core.Tracker.get(tracker)
        if not tracker_core:
            self.report({"ERROR"}, "Failed to get tracker core")
            return {"CANCELLED"}
        
        pin_mode = tracker_core.pin_mode
        points = pin_mode.points
        
        if tracker.selected_pin_idx >= len(points):
            self.report({"ERROR"}, "Invalid pin index")
            return {"CANCELLED"}
        
        # Get pin position in world space
        pin_pos_local = points[tracker.selected_pin_idx]
        pin_pos_world = np.array(tracker.geometry.matrix_world) @ np.append(
            pin_pos_local, 1.0)
        pin_pos_world = pin_pos_world[:3]
        
        # Get camera position
        camera_pos = np.array(tracker.camera.matrix_world.translation)
        
        # Calculate direction from camera to pin
        direction = pin_pos_world - camera_pos
        current_distance = np.linalg.norm(direction)
        
        if current_distance < 1e-6:
            self.report({"ERROR"}, "Pin is too close to camera")
            return {"CANCELLED"}
        
        # Normalize direction and scale to desired distance
        direction_normalized = direction / current_distance
        new_pin_pos_world = camera_pos + direction_normalized * self.distance
        
        # Convert back to local space
        geometry_matrix_inv = np.linalg.inv(np.array(tracker.geometry.matrix_world))
        new_pin_pos_local = (geometry_matrix_inv @ np.append(new_pin_pos_world, 1.0))[:3]
        
        # Update pin position
        points[tracker.selected_pin_idx] = new_pin_pos_local
        pin_mode._update_points(tracker)
        
        # Set distance constraint
        pin_mode.set_pin_distance(tracker.selected_pin_idx, self.distance)
        
        self.report(
            {"INFO"},
            f"Snapped pin to distance: {self.distance:.3f} Blender units"
        )
        
        return {"FINISHED"}

