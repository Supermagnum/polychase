# IMU Format Support Status

This document details the actual implementation status of IMU data format support in Polychase.

## Fully Implemented and Tested

### OpenCamera-Sensors CSV Format

**Status:** Fully implemented and tested

**File Format:**
- `{VIDEO_NAME}_accel.csv`: 4 columns - X-data, Y-data, Z-data, timestamp (ns)
- `{VIDEO_NAME}_gyro.csv`: 4 columns - X-data, Y-data, Z-data, timestamp (ns)
- `{VIDEO_NAME}_timestamps.csv`: 1 column - timestamp (ns) for each video frame

**Implementation:**
- Function: `load_opencamera_csv()` in `blender_addon/imu_integration.py`
- Uses pandas for CSV parsing
- Handles timestamp synchronization between accel and gyro data
- Validates file format and handles errors gracefully
- Tested with 17+ unit tests

**Requirements:**
- `pandas` library (required)

## Partially Implemented (Framework Ready)

### CAMM (Camera Motion Metadata) Detection

**Status:** Framework implemented, requires compatible libraries

**Implementation Approach:**
The code tries multiple extraction methods in order:
1. GoPro telemetry extraction (via gopro-telemetry libraries)
2. Direct MP4 box parsing (via mp4parse or similar)
3. MediaInfo extraction (via pymediainfo or mediainfo CLI)

**GoPro GPMF Support:**
- **Code Status:** Implemented
- **Function:** `_extract_gopro_telemetry()` in `blender_addon/imu_integration.py`
- **Library Support:** Tries multiple library names:
  - `gopro_telemetry`
  - `gopro_telemetry_extractor`
  - `gpmf_parser`
- **Expected API:** Library should provide `GoProTelemetry` class with:
  - `telemetry.accl`: List of accelerometer samples with 'ts', 'x', 'y', 'z' keys
  - `telemetry.gyro`: List of gyroscope samples with 'ts', 'x', 'y', 'z' keys
- **Test Status:** Framework tested, requires actual GoPro video files for full validation

**Generic CAMM Support:**
- **Code Status:** Framework exists, not fully functional
- **Function:** `_extract_camm_from_mp4_boxes()` in `blender_addon/imu_integration.py`
- **Library Support:** Requires `mp4parse` library (not a standard library)
- **Manual Parsing:** `_parse_mp4_boxes_manual()` exists but returns None (not fully implemented)
- **Limitations:** Full MP4 box parsing requires proper MP4 container structure understanding

**MediaInfo Support:**
- **Code Status:** Framework exists, placeholder implementation
- **Function:** `_extract_camm_with_mediainfo()` in `blender_addon/imu_integration.py`
- **Library Support:** Tries `pymediainfo` or `mediainfo` command-line tool
- **Limitations:** Actual CAMM track parsing is not implemented (placeholder code)

## Not Specifically Implemented

### Sony XAVC RTMD
- **Status:** Not specifically implemented
- **Note:** May work through generic CAMM parsing if format matches standard CAMM structure
- **Action Required:** Would need format-specific parser for XAVC RTMD structure

### Insta360 CAMM
- **Status:** Not specifically implemented
- **Note:** May be extractable via ExifTool or MediaInfo if metadata structure matches
- **Action Required:** Would need format-specific parser or ExifTool integration

## Summary

| Format | Implementation Status | Testing Status | Library Requirements |
|--------|----------------------|----------------|---------------------|
| OpenCamera-Sensors CSV | Fully Implemented | 17+ tests passing | pandas (required) |
| GoPro GPMF | Code Implemented | Framework tested | gopro-telemetry* (optional) |
| Generic CAMM | Framework Only | Basic tests | mp4parse/pymediainfo (optional) |
| Sony XAVC RTMD | Not Implemented | Not tested | N/A |
| Insta360 CAMM | Not Implemented | Not tested | N/A |

*Library name may vary - code tries multiple common names

## Recommendations

1. **For Production Use:** Use OpenCamera-Sensors CSV format (fully tested and reliable)
2. **For GoPro Videos:** Install a compatible GoPro telemetry library and test with actual GoPro video files
3. **For Other CAMM Formats:** Consider implementing format-specific parsers or using external tools (ExifTool, MediaInfo) to extract data first, then import as CSV

## Testing

All implemented formats are covered by the test suite:
- OpenCamera CSV: Comprehensive unit tests, property-based tests, robustness tests
- CAMM detection: Framework tests with mocked libraries
- Error handling: Extensive tests for malformed data

See [test-results.md](test-results.md) for detailed test coverage.

