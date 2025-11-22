# Polychase

A free and open-source motion tracking addon for Blender, inspired by KeenTools GeoTracker.

## Overview

Polychase is a 3D motion tracking solution that allows you to track camera movement or object motion in video footage within Blender. It uses optical flow analysis and PnP, aided by user input to provide accurate tracking results.

## Features

### Core Tracking Capabilities
- **3D Pin Mode**: Place and manage tracking pins on 3D geometry
- **Camera/Geometry Tracking**: Track camera/geometry movement through 3D space
- **Trajectory Refinement**: Refine tracking results using bundle adjustment

### Advanced Features
- **Variable Camera Parameters**: Support for estimating focal length and principal point
- **Keyframe Management**: Complete keyframe control for tracked animation
- **Scene Transformation**: Transform entire tracked scenes
- **Animation Conversion**: Convert between camera and object tracking
- **Real-time Preview**: Live tracking progress and results
- **Mask Support**: 3D masking for selective tracking
- **IMU Integration**: Inertial Measurement Unit data support for improved camera tracking with automatic Z-axis orientation

### User Interface
- **Integrated Blender UI**: Native Blender panels and operators
- **Visual Feedback**: Color-coded pins, wireframes, and progress indicators
- **Customizable Appearance**: Adjustable pin colors, sizes, and wireframe styles

## Usage

### Basic Workflow

1. **Setup Scene:**
   - Import your video footage as a movie clip
   - Add or import the 3D geometry you want to track
   - Set up a camera object

2. **Create Tracker:**
   - Open the Polychase panel in Blender's 3D viewport
   - Create a new tracker
   - Assign your clip, geometry, and camera

3. **Analyze Video:**
   - Set the database path for optical flow storage
   - Run "Analyze Video" to generate optical flow data

4. **Pin Mode:**
   - Enter pin mode to place tracking points on your 3D geometry
   - Add pins by clicking on the geometry surface
   - Drag the pins to adjust the pose of the geometry/camera

5. **Track Sequence:**
   - Choose tracking direction (forward/backward)
   - Select tracking target (camera or geometry)
   - Run tracking to generate keyframes

6. **Refine Results:**
   - Use the refine sequence tool to improve tracking accuracy

7. **IMU Integration (Optional):**
   - Load IMU data from OpenCamera-Sensors CSV files or detect CAMM metadata in MP4
   - Enable IMU integration in the IMU Settings panel
   - Adjust IMU influence weight to balance between optical flow and IMU data
   - Optionally lock Z-axis to gravity vector for stable vertical orientation

### Pin Mode Controls

- **Left Click**: Add new pin
- **Right Click**: Delete pin
- **M**: Go to mask drawing mode
- **ESC**: Exit pin mode

## Technical Details

### Architecture

- **C++ Core**: High-performance tracking algorithms written in C++
- **Python Bindings**: pybind11 integration for Blender compatibility  
- **Blender Integration**: Native Blender addon with custom operators and panels

### Algorithms

- **Optical Flow**: Off-the-shelf OpenCV solution
- **3D Tracking**: PnP (Perspective-n-Point) solving for camera pose estimation
- **Bundle Adjustment**: Global non-linear optimization for trajectory refinement
- **Ray Casting**: Accelerated mesh intersection using Embree
- **IMU Processing**: Gravity vector extraction via low-pass filtering, gyroscope integration for orientation estimation, and sensor fusion with optical flow tracking

## IMU Integration

Polychase supports IMU (Inertial Measurement Unit) data integration for improved camera tracking, especially useful for handheld or mobile device footage.

### Supported Formats

Polychase supports two main IMU data formats:

1. **OpenCamera-Sensors CSV Format** (Recommended):
   - `{VIDEO_NAME}_accel.csv`: X-data, Y-data, Z-data, timestamp (ns)
   - `{VIDEO_NAME}_gyro.csv`: X-data, Y-data, Z-data, timestamp (ns)
   - `{VIDEO_NAME}_timestamps.csv`: timestamp (ns) for each video frame

2. **CAMM (Camera Motion Metadata)** embedded in MP4:
   - Gyroscope: radians/second around XYZ axes
   - Accelerometer: m/s² along XYZ axes
   - Timestamps synchronized with video frames

### CAMM Format Details

CAMM (Camera Motion Metadata) is a specification that allows MP4 files to embed metadata about camera motion including:

- **Gyroscope**: Angular velocity in radians/second around XYZ axes
- **Accelerometer**: Linear acceleration in m/s² along XYZ axes
- **Magnetometer**: Magnetic field data (if available)
- **GPS**: Location data (if available)
- **3DoF/6DoF Pose**: Camera orientation and position data

The metadata is stored as a metadata track within the MP4 file, making it easy to extract IMU data directly from video files without separate CSV files.

**Currently Supported CAMM Formats:**

- **GoPro GPMF (General Purpose Metadata Format)**: ✅ **Supported**
  - HERO5 and later store accelerometer, gyroscope and high frequency GPS as a track within MP4 files
  - Extraction code implemented - requires compatible GoPro telemetry extraction library
  - Automatically detected and extracted from GoPro MP4 files when library is available
  - **Note:** The implementation uses a generic GoPro telemetry interface. You may need to install a compatible library such as `gopro-telemetry-extractor` or similar, depending on what's available for your Python version.

**CAMM-Compatible Devices (Framework Ready, May Require Additional Libraries):**

The codebase includes a framework for extracting CAMM data from MP4 files using multiple methods:
1. GoPro telemetry extraction (fully implemented)
2. Direct MP4 box parsing (requires `mp4parse` library or similar)
3. MediaInfo extraction (requires `pymediainfo` library or `mediainfo` command-line tool)

**Note:** While the framework supports generic CAMM extraction, specific implementations for the following formats may require additional libraries or format-specific parsers:

- **Sony XAVC RTMD**: Sony cameras like DSC-RX100M7 and DSC-RX0M2 record raw per-frame gyro/accelerometer data in XAVC video files. May work with generic CAMM parsing if the format matches standard CAMM structure.
- **Insta360**: Insta360 Pro2 uses CAMM standard with AngularVelocity (gyroscope in radians/second) and Acceleration (accelerometer in m/s²). May be extractable via ExifTool or MediaInfo if the metadata structure matches.

For best results with CAMM extraction, ensure you have the appropriate libraries installed:
- GoPro videos: `pip install gopro-telemetry`
- Generic CAMM: `pip install pymediainfo` or install `mediainfo` command-line tool

### OpenCamera-Sensors Setup

To record IMU data with OpenCamera Sensors on Android devices (e.g., Pixel 9a):

1. **Open OpenCamera Sensors** app
2. **Go to Settings** (gear icon)
3. **Enable Camera2 API**
4. **Go to "IMU settings..."**
   - Enable "Enable sync video IMU recording"
   - Check "Timestamp source" shows "realtime" (REQUIRED)
5. **Optionally disable video stabilization** in video preferences (for better IMU accuracy)
6. **Test recording** a short clip
7. **Verify CSV files** are created in `DCIM/OpenCamera/{VIDEO_DATE}/`

The CSV files will be automatically named based on your video filename:
- `{VIDEO_NAME}_accel.csv`
- `{VIDEO_NAME}_gyro.csv`
- `{VIDEO_NAME}_timestamps.csv`

**Important:** Ensure the timestamp source is set to "realtime" for proper synchronization between video frames and IMU samples.

### IMU Workflow

1. **Prepare IMU Data**:
   - Record video with OpenCamera app or similar that exports sensor data
   - Ensure CSV files are in the same directory as your video file
   - Files should follow the naming convention: `{VIDEO_NAME}_accel.csv`, `{VIDEO_NAME}_gyro.csv`, `{VIDEO_NAME}_timestamps.csv`

2. **Load IMU Data**:
   - Open the IMU Settings panel in Polychase
   - Click "Load IMU CSV Files" and select one of the CSV files
   - The addon will automatically detect the other required files based on naming convention
   - Alternatively, use "Detect CAMM in Video" if your video contains embedded metadata

3. **Configure IMU Settings**:
   - Enable "Enable IMU Integration" checkbox
   - Adjust "IMU Influence Weight" (0.0 = optical flow only, 1.0 = IMU only, 0.5 = balanced)
   - Enable "Lock Z-Axis to Gravity" to constrain vertical orientation
   - Check IMU data quality indicators (gravity consistency, gyro drift)

4. **Track with IMU**:
   - IMU constraints are automatically applied during tracking
   - Gravity vector helps maintain stable vertical orientation
   - Gyroscope data provides smooth frame-to-frame rotation estimates
   - Optical flow tracking is enhanced with IMU orientation priors

### IMU Data Quality

The IMU Settings panel displays quality indicators:

- **Gravity Consistency**: Measures how consistent the gravity direction is across samples (0-1, higher is better). Values below 0.7 indicate potential issues.
- **Gyro Drift**: Estimates gyroscope drift rate in rad/s. Higher values indicate more drift. Values above 0.1 rad/s may need calibration.
- **Sample Count**: Total number of IMU samples loaded

### Technical Details

- **Gravity Extraction**: Low-pass Butterworth filter (default cutoff: 0.1 Hz) isolates gravity from motion acceleration
- **Gyroscope Integration**: Quaternion-based orientation integration with automatic bias estimation
- **Sensor Fusion**: Weighted blending between optical flow tracking and IMU orientation estimates
- **Coordinate Systems**: Handles transformations between sensor frame and camera frame

### Dependencies

**Required:**
- `numpy`: For numerical operations
- `scipy`: For signal processing and filtering

**Optional (but recommended):**
- `pandas`: For CSV file parsing (required for OpenCamera-Sensors CSV format)
- `gopro-telemetry`: For GoPro GPMF/CAMM extraction from GoPro videos
- `pymediainfo`: For generic CAMM extraction via MediaInfo (alternative: install `mediainfo` command-line tool)

Install dependencies with:
```bash
# Minimum required
pip install numpy scipy

# For CSV support (OpenCamera-Sensors)
pip install pandas

# For GoPro CAMM support
pip install gopro-telemetry

# For generic CAMM support (optional)
pip install pymediainfo
# OR install mediainfo command-line tool via system package manager
```

## Development and Testing

### Setting Up Development Environment

Install development dependencies:

```bash
pip install -e .[dev]
# Or use the setup script:
bash setup_testing.sh
```

### Running Tests

Run all tests:

```bash
pytest
# Or use Make:
make test
```

Run tests with coverage:

```bash
pytest --cov=blender_addon --cov-report=html
# Or:
make test-cov
```

Run specific test categories:

```bash
pytest -m imu          # IMU-related tests
pytest -m performance # Performance benchmarks
pytest -m unit        # Unit tests only
```

### Code Quality

Run all quality checks:

```bash
make quality
```

Individual checks:

```bash
make format-check  # Check code formatting
make lint          # Run linters
make type-check    # Type checking
make security      # Security scan
```

Format code:

```bash
make format
```

### Pre-commit Hooks

Install pre-commit hooks:

```bash
pre-commit install
```

Run hooks manually:

```bash
make pre-commit
```

### Test Coverage

Coverage goals:
- Overall: 80% minimum
- IMU integration: 95% minimum
- Critical paths: 95% minimum

View coverage reports in `htmlcov/index.html` after running tests with coverage.

For detailed testing documentation, see [tests/README.md](tests/README.md) or [TESTING.md](TESTING.md).

**Test Results:** See [test-results.md](test-results.md) for the latest test execution results and coverage information.

**IMU Format Support:** See [IMU_FORMAT_SUPPORT.md](IMU_FORMAT_SUPPORT.md) for detailed implementation status of each supported format.

### Demo & Technical Walkthrough
[![Watch the technical walkthrough on YouTube](https://img.youtube.com/vi/W4HNmcjFuLw/hqdefault.jpg)](https://youtu.be/W4HNmcjFuLw)
