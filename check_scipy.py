#!/usr/bin/env python3
"""Check if scipy is installed in Blender's Python environment"""
import sys

print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Python path: {sys.path[:3]}...")

try:
    import scipy
    print(f"\n✓ scipy is installed")
    print(f"  Version: {scipy.__version__}")
    print(f"  Location: {scipy.__file__}")
    
    # Check specific modules
    try:
        from scipy import signal
        print(f"  ✓ scipy.signal available")
    except ImportError as e:
        print(f"  ✗ scipy.signal not available: {e}")
    
    try:
        from scipy.interpolate import interp1d
        print(f"  ✓ scipy.interpolate.interp1d available")
    except ImportError as e:
        print(f"  ✗ scipy.interpolate.interp1d not available: {e}")
        
except ImportError as e:
    print(f"\n✗ scipy is NOT installed")
    print(f"  Error: {e}")
    print(f"\nTo install scipy in Blender:")
    print(f"  1. Blender 4.2+: Should install automatically via blender_manifest.toml")
    print(f"  2. Manual: Run Blender's Python and use pip:")
    print(f"     import subprocess; subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'scipy'])")

# Also check other dependencies
print("\n" + "="*50)
print("Other dependencies:")

for module_name in ['numpy', 'pandas']:
    try:
        mod = __import__(module_name)
        print(f"✓ {module_name}: {mod.__version__}")
    except ImportError:
        print(f"✗ {module_name}: NOT installed")

