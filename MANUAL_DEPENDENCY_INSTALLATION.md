# Manual Dependency Installation for Polychase

This guide explains how to manually install Python dependencies for the Polychase addon if automatic installation via `blender_manifest.toml` is not working.

## Required Dependencies

The following packages are required for Polychase to function:

- `numpy >= 1.21.0` - Numerical operations
- `scipy >= 1.9.0` - Signal processing and filtering
- `pandas >= 1.5.0` - CSV file parsing (for IMU data)
- `python-dateutil >= 2.8.2` - Pandas runtime dependency
- `pytz >= 2023.3` - Pandas runtime dependency
- `opencv-contrib-python >= 4.9.0.80` - Feature detection (optional but recommended)

## Finding Blender's Python Executable

### Method 1: From Blender's Python Console

1. Open Blender
2. Switch to the "Scripting" workspace
3. Open the Python console
4. Run the following command:

```python
import sys
print(sys.executable)
```

This will print the path to Blender's Python executable.

### Method 2: Platform-Specific Locations

**Linux:**
- Flatpak: `/app/bin/python3` or check with `which python3` in Blender's console
- AppImage: Usually in the AppImage bundle's Python directory
- System install: May use system Python, check with `sys.executable` in console

**Windows:**
- Typically: `C:\Program Files\Blender Foundation\Blender <version>\<version>\python\bin\python.exe`
- Or: `C:\Users\<username>\AppData\Local\Programs\Blender Foundation\Blender <version>\<version>\python\bin\python.exe`

**macOS:**
- Typically: `/Applications/Blender.app/Contents/Resources/<version>/python/bin/python3.10` (version may vary)

## Installation Methods

### Method 1: Using Blender's Python Console (Recommended)

1. Open Blender
2. Switch to the "Scripting" workspace
3. Open the Python console
4. Copy and paste the following code:

```python
import subprocess
import sys

dependencies = [
    "numpy>=1.21.0",
    "scipy>=1.9.0",
    "pandas>=1.5.0",
    "python-dateutil>=2.8.2",
    "pytz>=2023.3",
    "opencv-contrib-python>=4.9.0.80"
]

print("Installing dependencies...")
for dep in dependencies:
    print(f"Installing {dep}...")
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '--user', dep
        ])
        print(f"Successfully installed {dep}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {dep}: {e}")
        # Try without --user flag
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', dep
            ])
            print(f"Successfully installed {dep} (without --user flag)")
        except subprocess.CalledProcessError as e2:
            print(f"Failed to install {dep} (both methods failed): {e2}")

print("\nInstallation complete!")
print("Please restart Blender for changes to take effect.")
```

5. Press Enter to execute
6. Wait for all packages to install
7. Restart Blender

### Method 2: Using Command Line (Linux/macOS)

1. Find Blender's Python executable (see "Finding Blender's Python Executable" above)
2. Open a terminal
3. Run the following commands (replace `<blender_python>` with the actual path):

```bash
<blender_python> -m pip install --user numpy>=1.21.0
<blender_python> -m pip install --user scipy>=1.9.0
<blender_python> -m pip install --user pandas>=1.5.0
<blender_python> -m pip install --user python-dateutil>=2.8.2
<blender_python> -m pip install --user pytz>=2023.3
<blender_python> -m pip install --user opencv-contrib-python>=4.9.0.80
```

Or install all at once:

```bash
<blender_python> -m pip install --user \
    numpy>=1.21.0 \
    scipy>=1.9.0 \
    pandas>=1.5.0 \
    python-dateutil>=2.8.2 \
    pytz>=2023.3 \
    opencv-contrib-python>=4.9.0.80
```

### Method 3: Using Command Line (Windows)

1. Find Blender's Python executable (see "Finding Blender's Python Executable" above)
2. Open Command Prompt or PowerShell
3. Run the following commands (replace `<blender_python>` with the actual path):

```cmd
<blender_python> -m pip install --user numpy>=1.21.0
<blender_python> -m pip install --user scipy>=1.9.0
<blender_python> -m pip install --user pandas>=1.5.0
<blender_python> -m pip install --user python-dateutil>=2.8.2
<blender_python> -m pip install --user pytz>=2023.3
<blender_python> -m pip install --user opencv-contrib-python>=4.9.0.80
```

Or install all at once:

```cmd
<blender_python> -m pip install --user numpy>=1.21.0 scipy>=1.9.0 pandas>=1.5.0 python-dateutil>=2.8.2 pytz>=2023.3 opencv-contrib-python>=4.9.0.80
```

## Verification

After installation, verify that all packages are installed correctly:

1. Open Blender
2. Switch to the "Scripting" workspace
3. Open the Python console
4. Run the following verification script:

```python
import sys

packages = {
    'numpy': '1.21.0',
    'scipy': '1.9.0',
    'pandas': '1.5.0',
    'dateutil': '2.8.2',
    'pytz': '2023.3',
    'cv2': '4.9.0.80'
}

print("Verifying installed packages...")
print("=" * 60)

all_ok = True
for module_name, min_version in packages.items():
    try:
        if module_name == 'cv2':
            import cv2
            version = cv2.__version__
            module_display = 'opencv-contrib-python'
        elif module_name == 'dateutil':
            import dateutil
            version = dateutil.__version__
            module_display = 'python-dateutil'
        else:
            mod = __import__(module_name)
            version = mod.__version__
            module_display = module_name
        
        print(f"OK: {module_display} version {version}")
    except ImportError:
        print(f"FAIL: {module_display} not found")
        all_ok = False

print("=" * 60)
if all_ok:
    print("All packages installed successfully!")
else:
    print("Some packages are missing. Please install them using the methods above.")
```

## Troubleshooting

### Permission Errors

If you encounter permission errors, try:

1. **Without --user flag**: Remove `--user` from the pip install command
2. **Run as administrator** (Windows) or **use sudo** (Linux/macOS) - not recommended as it may install to system Python instead of Blender's Python

### Package Not Found After Installation

If packages are installed but not found by Blender:

1. **Check Python path**: Verify that Blender is using the correct Python executable
2. **Restart Blender**: Always restart Blender after installing packages
3. **Check installation location**: Packages installed with `--user` go to user site-packages, which should be in Blender's Python path

### Flatpak-Specific Issues

If using Blender from Flatpak:

1. Packages may need to be installed in Flatpak's Python environment
2. The Python path may be `/app/bin/python3` or similar
3. User site-packages may be in `/var/data/python/lib/python3.X/site-packages`
4. You may need to add paths manually (though the addon tries to do this automatically)

### OpenCV Installation Issues

If `opencv-contrib-python` fails to install:

1. Try `opencv-python` instead (has fewer features but may install more easily)
2. Check that you have sufficient disk space
3. On some systems, you may need to install system dependencies first (e.g., `libopencv-dev` on Linux)

### Import Errors After Installation

If you see import errors for pandas dependencies (pytz, dateutil):

1. These are runtime dependencies of pandas and must be installed separately
2. Install them explicitly using the commands above
3. Make sure to install them in the same Python environment as pandas

## Alternative: Using the Installation Script

A helper script `install_pandas_blender.py` is included in the addon. You can use it from Blender's Python console:

```python
exec(open('/path/to/polychase/install_pandas_blender.py').read())
```

Replace `/path/to/polychase/` with the actual path to the addon directory.

## Notes

- Blender 4.2+ should automatically install dependencies listed in `blender_manifest.toml`
- If automatic installation fails, use the manual methods above
- Always restart Blender after installing packages
- Some features (like automatic feature detection) require OpenCV but the addon will work without it (with reduced functionality)

