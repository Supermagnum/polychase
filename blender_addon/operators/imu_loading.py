# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 Ahmed Essam <aessam.dahy@gmail.com>

import os
import typing

import bpy
import bpy.types

from .. import imu_integration
from ..properties import PolychaseState, PolychaseTracker


class PC_OT_LoadIMUCSV(bpy.types.Operator):
    """Load IMU data from OpenCamera-Sensors CSV files"""
    bl_idname = "polychase.load_imu_csv"
    bl_label = "Load IMU CSV Files"
    bl_options = {"REGISTER", "UNDO"}

    filepath: bpy.props.StringProperty(
        name="File Path",
        description="Path to accelerometer CSV file",
        subtype="FILE_PATH")

    filter_glob: bpy.props.StringProperty(
        default="*.csv",
        options={"HIDDEN"})

    @classmethod
    def poll(cls, context):
        state = PolychaseState.from_context(context)
        return state is not None and state.is_tracking_active()

    def invoke(self, context, event):
        tracker = PolychaseState.from_context(context).active_tracker
        if not tracker:
            return {"CANCELLED"}

        # Set default path based on clip name if available
        if tracker.clip:
            clip_name = tracker.clip.filepath
            if clip_name:
                base_name = os.path.splitext(os.path.basename(clip_name))[0]
                default_dir = os.path.dirname(clip_name)
                self.filepath = os.path.join(default_dir, f"{base_name}_accel.csv")

        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        state = PolychaseState.from_context(context)
        if not state:
            return {"CANCELLED"}

        tracker = state.active_tracker
        if not tracker:
            return {"CANCELLED"}

        # Get base path from selected file
        filepath = bpy.path.abspath(self.filepath)
        base_dir = os.path.dirname(filepath)
        base_name = os.path.basename(filepath)

        # Try to infer other file names
        if "_accel.csv" in base_name:
            video_name = base_name.replace("_accel.csv", "")
        elif "_gyro.csv" in base_name:
            video_name = base_name.replace("_gyro.csv", "")
        elif "_timestamps.csv" in base_name:
            video_name = base_name.replace("_timestamps.csv", "")
        else:
            # Try to extract from clip name
            if tracker.clip:
                clip_name = os.path.splitext(os.path.basename(tracker.clip.filepath))[0]
                video_name = clip_name
            else:
                video_name = os.path.splitext(base_name)[0]

        accel_path = os.path.join(base_dir, f"{video_name}_accel.csv")
        gyro_path = os.path.join(base_dir, f"{video_name}_gyro.csv")
        timestamps_path = os.path.join(base_dir, f"{video_name}_timestamps.csv")

        # Check which files exist
        if "_accel.csv" in base_name:
            accel_path = filepath
        elif "_gyro.csv" in base_name:
            gyro_path = filepath
        elif "_timestamps.csv" in base_name:
            timestamps_path = filepath

        # Set paths in tracker
        tracker.imu_accel_csv_path = accel_path
        tracker.imu_gyro_csv_path = gyro_path
        tracker.imu_timestamps_csv_path = timestamps_path

        # Try to load and validate
        accel_abs = bpy.path.abspath(accel_path)
        gyro_abs = bpy.path.abspath(gyro_path)
        timestamps_abs = bpy.path.abspath(timestamps_path)

        if not os.path.isfile(accel_abs):
            self.report({"WARNING"}, f"Accelerometer file not found: {accel_path}")
        if not os.path.isfile(gyro_abs):
            self.report({"WARNING"}, f"Gyroscope file not found: {gyro_path}")
        if not os.path.isfile(timestamps_abs):
            self.report({"WARNING"}, f"Timestamps file not found: {timestamps_path}")

        if os.path.isfile(accel_abs) and os.path.isfile(gyro_abs) and os.path.isfile(timestamps_abs):
            # Try to load and validate IMU data
            try:
                imu_data = imu_integration.load_opencamera_csv(
                    accel_abs, gyro_abs, timestamps_abs)
                if imu_data:
                    self.report({"INFO"}, f"IMU data loaded: {len(imu_data)} samples")
                    tracker.imu_enabled = True
                else:
                    self.report({"ERROR"}, "Failed to load IMU data. Check file format.")
                    tracker.imu_enabled = False
            except Exception as e:
                self.report({"ERROR"}, f"Error loading IMU data: {str(e)}")
                tracker.imu_enabled = False
        else:
            self.report({"INFO"}, "IMU file paths set. Enable IMU after all files are available.")

        return {"FINISHED"}


class PC_OT_DetectCAMM(bpy.types.Operator):
    """Detect and load CAMM (Camera Motion Metadata) from MP4 file"""
    bl_idname = "polychase.detect_camm"
    bl_label = "Detect CAMM in Video"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        state = PolychaseState.from_context(context)
        if not state or not state.is_tracking_active():
            return False
        tracker = state.active_tracker
        return tracker is not None and tracker.clip is not None

    def execute(self, context):
        state = PolychaseState.from_context(context)
        if not state:
            return {"CANCELLED"}

        tracker = state.active_tracker
        if not tracker or not tracker.clip:
            return {"CANCELLED"}

        video_path = bpy.path.abspath(tracker.clip.filepath)
        if not os.path.isfile(video_path):
            self.report({"ERROR"}, "Video file not found")
            return {"CANCELLED"}

        # Try to detect CAMM
        imu_data = imu_integration.detect_camm_in_mp4(video_path)
        if imu_data:
            self.report({"INFO"}, f"CAMM data detected: {len(imu_data)} samples")
            tracker.imu_enabled = True
        else:
            self.report({"INFO"}, "No CAMM metadata found in video file. Use CSV import instead.")

        return {"FINISHED"}

