# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 Ahmed Essam <aessam.dahy@gmail.com>

import dataclasses
import typing

import bpy
import bpy.props
import bpy.types

from . import background_images, utils

# TOOD: Move this from here
T = typing.TypeVar("T")


class BCollectionProperty(typing.Generic[T]):

    def __getitem__(self, index: int) -> T:
        ...

    def __iter__(self) -> typing.Iterator[T]:
        ...

    def __len__(self) -> int:
        ...

    def add(self) -> T:
        ...

    def remove(self, index: int) -> None:
        ...


def on_tracking_mesh_changed(
        self: bpy.types.bpy_struct, context: bpy.types.Context):
    tracker = typing.cast(PolychaseTracker, self)
    tracker.points = b""
    tracker.points_version_number = 0
    tracker.masked_triangles = b""
    tracker.selected_pin_idx = -1
    store_geom_cam_transform(tracker)


def on_clip_changed(self: bpy.types.bpy_struct, context: bpy.types.Context):
    tracker = typing.cast(PolychaseTracker, self)

    if tracker.clip and context.scene:
        context.scene.frame_start = tracker.clip.frame_start
        context.scene.frame_end = tracker.clip.frame_start + tracker.clip.frame_duration - 1

    if tracker.camera and tracker.clip:
        assert isinstance(tracker.camera.data, bpy.types.Camera)
        camera_data = tracker.camera.data
        camera_data.background_images.clear()
        background_images.create_background_image_for_clip(
            camera_data=camera_data,
            clip=tracker.clip,
            alpha=1.0,
        )


def on_camera_changed(self: bpy.types.bpy_struct, context: bpy.types.Context):
    tracker = typing.cast(PolychaseTracker, self)
    if tracker.camera and tracker.clip:
        assert isinstance(tracker.camera.data, bpy.types.Camera)
        camera_data = tracker.camera.data
        camera_data.background_images.clear()
        background_images.create_background_image_for_clip(
            camera_data=camera_data,
            clip=tracker.clip,
            alpha=1.0,
        )


class PolychaseTracker(bpy.types.PropertyGroup):
    if typing.TYPE_CHECKING:
        id: int
        name: str
        clip: bpy.types.MovieClip | None
        geometry: bpy.types.Object | None
        camera: bpy.types.Object | None
        tracking_target: typing.Literal["CAMERA", "GEOMETRY"]
        database_path: str

        # State for pinmode
        points: bytes
        points_version_number: int
        selected_pin_idx: int
        pinmode_optimize_focal_length: bool
        pinmode_optimize_principal_point: bool
        pin_distances: bytes  # Optional distance constraints per pin (float32 array)

        # State for drawing 3D masks
        mask_selection_radius: float
        masked_triangles: bytes

        # Camera options
        variable_focal_length: bool
        variable_principal_point: bool

        # Appearance
        default_pin_color: tuple[float, float, float, float]
        selected_pin_color: tuple[float, float, float, float]
        pin_distance_color: tuple[float, float, float, float]
        pin_radius: float
        wireframe_color: tuple[float, float, float, float]
        wireframe_width: float
        mask_color: tuple[float, float, float, float]

        # Scene
        geometry_loc: tuple[float, float, float]
        geometry_rot: tuple[float, float, float, float]
        geometry_scale: tuple[float, float, float]
        camera_loc: tuple[float, float, float]
        camera_rot: tuple[float, float, float, float]

        # IMU settings
        imu_enabled: bool
        imu_accel_csv_path: str
        imu_gyro_csv_path: str
        imu_timestamps_csv_path: str
        imu_influence_weight: float
        imu_lock_z_axis: bool
        imu_visualize_gravity: bool
        
        # Auto-detect pins settings
        autodetect_max_pins: int
        autodetect_min_distance: float
        autodetect_min_spacing: float
        autodetect_quality_threshold: float
        autodetect_use_blender_tracking: bool
        autodetect_use_opencv: bool

    else:
        id: bpy.props.IntProperty(default=0)
        name: bpy.props.StringProperty(name="Name")
        clip: bpy.props.PointerProperty(
            name="Clip",
            type=bpy.types.MovieClip,
            update=on_clip_changed,
        )
        geometry: bpy.props.PointerProperty(
            name="Geometry",
            type=bpy.types.Object,
            poll=utils.bpy_poll_is_mesh,
            update=on_tracking_mesh_changed)
        camera: bpy.props.PointerProperty(
            name="Camera",
            type=bpy.types.Object,
            poll=utils.bpy_poll_is_camera,
            update=on_camera_changed,
        )
        tracking_target: bpy.props.EnumProperty(
            name="Tracking Target",
            items=(
                ("CAMERA", "Camera", "Track Camera"),
                ("GEOMETRY", "Geometry", "Track Geometry")))
        database_path: bpy.props.StringProperty(
            name="Database",
            description="Optical flow database path",
            subtype="FILE_PATH")

        # State for pinmode
        points: bpy.props.StringProperty(subtype="BYTE_STRING")
        points_version_number: bpy.props.IntProperty(default=0)
        selected_pin_idx: bpy.props.IntProperty(default=-1)
        pinmode_optimize_focal_length: bpy.props.BoolProperty(default=False)
        pinmode_optimize_principal_point: bpy.props.BoolProperty(default=False)
        pin_distances: bpy.props.StringProperty(subtype="BYTE_STRING")  # Optional distance constraints per pin

        # State for drawing 3D masks
        mask_selection_radius: bpy.props.FloatProperty(
            default=25.0, min=1.0, max=100.0)
        masked_triangles: bpy.props.StringProperty(subtype="BYTE_STRING")

        # Camera options
        variable_focal_length: bpy.props.BoolProperty(
            name="Variable Focal Length",
            default=False,
            description=
            "Whether or not to estimate focal length for each tracked frame",
        )
        variable_principal_point: bpy.props.BoolProperty(
            name="Variable Principal Length",
            default=False,
            description=
            "Whether or not to estimate principal point for each tracked frame",
        )

        # Appearance
        default_pin_color: bpy.props.FloatVectorProperty(
            name="Pin Color",
            subtype="COLOR",
            size=4,
            min=0.0,
            max=1.0,
            default=[0.0, 0.0, 1.0, 1.0])
        selected_pin_color: bpy.props.FloatVectorProperty(
            name="Pin Color",
            subtype="COLOR",
            size=4,
            min=0.0,
            max=1.0,
            default=[1.0, 0.0, 0.0, 1.0])
        pin_distance_color: bpy.props.FloatVectorProperty(
            name="Pin Distance Color",
            subtype="COLOR",
            size=4,
            min=0.0,
            max=1.0,
            default=[0.0, 1.0, 0.0, 1.0],
            description="Color for pins with distance constraints")
        pin_radius: bpy.props.FloatProperty(
            name="Pin Radius", min=0.0, max=100.0, default=10.0)
        wireframe_color: bpy.props.FloatVectorProperty(
            name="Wireframe Color",
            subtype="COLOR",
            size=4,
            min=0.0,
            max=1.0,
            default=[0.0, 1.0, 0.0, 1.0])
        wireframe_width: bpy.props.IntProperty(
            name="Wireframe width", default=1, min=1, max=10)
        mask_color: bpy.props.FloatVectorProperty(
            name="3D Mask Color",
            subtype="COLOR",
            size=4,
            min=0.0,
            max=1.0,
            default=[0.5, 0.1, 0.0, 0.5])

        # Scene
        geometry_loc: bpy.props.FloatVectorProperty(
            size=3, subtype="TRANSLATION")
        geometry_rot: bpy.props.FloatVectorProperty(
            size=4, subtype="QUATERNION")
        geometry_scale: bpy.props.FloatVectorProperty(
            size=3, default=[1.0, 1.0, 1.0])
        camera_loc: bpy.props.FloatVectorProperty(size=3, subtype="TRANSLATION")
        camera_rot: bpy.props.FloatVectorProperty(size=4, subtype="QUATERNION")

        # IMU settings
        imu_enabled: bpy.props.BoolProperty(
            name="Enable IMU",
            default=False,
            description="Enable IMU data integration for camera tracking")
        imu_accel_csv_path: bpy.props.StringProperty(
            name="Accelerometer CSV",
            description="Path to accelerometer CSV file (OpenCamera-Sensors format)",
            subtype="FILE_PATH")
        imu_gyro_csv_path: bpy.props.StringProperty(
            name="Gyroscope CSV",
            description="Path to gyroscope CSV file (OpenCamera-Sensors format)",
            subtype="FILE_PATH")
        imu_timestamps_csv_path: bpy.props.StringProperty(
            name="Timestamps CSV",
            description="Path to video frame timestamps CSV file",
            subtype="FILE_PATH")
        imu_influence_weight: bpy.props.FloatProperty(
            name="IMU Influence",
            description="Weight of IMU data in tracking (0.0 = disabled, 1.0 = full IMU)",
            default=0.5,
            min=0.0,
            max=1.0)
        imu_lock_z_axis: bpy.props.BoolProperty(
            name="Lock Z-Axis to Gravity",
            default=False,
            description="Constrain camera Z-axis to align with gravity vector")
        imu_visualize_gravity: bpy.props.BoolProperty(
            name="Visualize Gravity Vector",
            default=False,
            description="Display gravity vector in 3D viewport")
        
        # Auto-detect pins settings
        autodetect_max_pins: bpy.props.IntProperty(
            name="Max Pins",
            default=50,
            min=1,
            max=500,
            description="Maximum number of pins to create")
        autodetect_min_distance: bpy.props.FloatProperty(
            name="Min Distance",
            default=10.0,
            min=1.0,
            max=100.0,
            description="Minimum distance between features in pixels (OpenCV)")
        autodetect_min_spacing: bpy.props.FloatProperty(
            name="Min Spacing",
            default=0.05,
            min=0.0,
            max=1.0,
            description="Minimum spacing between pins in normalized coordinates [0, 1]")
        autodetect_quality_threshold: bpy.props.FloatProperty(
            name="Quality Threshold",
            default=0.01,
            min=0.001,
            max=1.0,
            description="Feature quality threshold (lower = more features, OpenCV)")
        autodetect_use_blender_tracking: bpy.props.BoolProperty(
            name="Use Blender Tracking",
            default=True,
            description="Use Blender's motion tracking markers if available")
        autodetect_use_opencv: bpy.props.BoolProperty(
            name="Use OpenCV",
            default=True,
            description="Use OpenCV feature detection (requires opencv-python)")

    def get_target_object(self) -> bpy.types.Object | None:
        if self.tracking_target == "CAMERA":
            return self.camera
        else:
            return self.geometry

    def store_geom_cam_transform(self):
        store_geom_cam_transform(self)


def store_geom_cam_transform(tracker: PolychaseTracker):
    if tracker.geometry:
        loc, rot, scale = tracker.geometry.matrix_world.decompose()
        tracker.geometry_loc = typing.cast(tuple, loc)
        tracker.geometry_rot = typing.cast(tuple, rot)
        tracker.geometry_scale = typing.cast(tuple, scale)

    if tracker.camera:
        loc, rot, _ = tracker.camera.matrix_world.decompose()
        tracker.camera_loc = typing.cast(tuple, loc)
        tracker.camera_rot = typing.cast(tuple, rot)


@dataclasses.dataclass
class _PolychaseTransientState:
    in_pinmode: bool = False
    should_stop_pin_mode: bool = False

    is_preprocessing: bool = False
    should_stop_preprocessing: bool = False
    preprocessing_progress: float = 0.0
    preprocessing_message: str = ""

    is_tracking: bool = False
    should_stop_tracking: bool = False
    tracking_progress: float = 0.0
    tracking_message: str = ""

    is_refining: bool = False
    should_stop_refining: bool = False
    refining_progress: float = 0.0
    refining_message: str = ""


_transient_state = _PolychaseTransientState()


class PolychaseState(bpy.types.PropertyGroup):
    if typing.TYPE_CHECKING:
        trackers: BCollectionProperty[PolychaseTracker]
        active_tracker_idx: int
        num_created_trackers: int

    else:
        trackers: bpy.props.CollectionProperty(
            type=PolychaseTracker, name="Trackers")
        active_tracker_idx: bpy.props.IntProperty(default=-1)
        num_created_trackers: bpy.props.IntProperty(default=0)

    @classmethod
    def register(cls):
        setattr(
            bpy.types.Scene,
            "polychase_data",
            bpy.props.PointerProperty(type=cls))

    @classmethod
    def unregister(cls):
        delattr(bpy.types.Scene, "polychase_data")

    @classmethod
    def from_context(
            cls,
            context: bpy.types.Context | None = None) -> typing.Self | None:
        if context is None:
            context = bpy.context
        return getattr(context.scene, "polychase_data", None)

    @classmethod
    def get_transient_state(cls) -> _PolychaseTransientState:
        return _transient_state

    @property
    def active_tracker(self) -> PolychaseTracker | None:
        if self.active_tracker_idx < 0 or self.active_tracker_idx >= len(
                self.trackers):
            return None
        return self.trackers[self.active_tracker_idx]

    @staticmethod
    def get_tracker_by_id(
            id: int,
            context: bpy.types.Context | None = None
    ) -> PolychaseTracker | None:
        state = PolychaseState.from_context(context)
        if not state:
            return None

        tracker: PolychaseTracker
        for tracker in state.trackers:
            if tracker.id == id:
                return tracker

        return None

    def is_tracking_active(self) -> bool:
        return self.active_tracker is not None
