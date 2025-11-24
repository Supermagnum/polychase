"""Check scipy in Blender - Run this in Blender's Python console"""
import sys
import bpy

print("="*60)
print("Blender Python Environment Check")
print("="*60)
print(f"Blender version: {bpy.app.version_string}")
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"\nPython path (first 5):")
for i, path in enumerate(sys.path[:5]):
    print(f"  {i+1}. {path}")

print("\n" + "="*60)
print("Checking scipy:")
print("="*60)

try:
    import scipy
    print(f"✓ scipy is installed")
    print(f"  Version: {scipy.__version__}")
    print(f"  Location: {scipy.__file__}")
    
    # Check specific modules
    try:
        from scipy import signal
        print(f"  ✓ scipy.signal available")
        print(f"     Location: {signal.__file__}")
    except ImportError as e:
        print(f"  ✗ scipy.signal not available: {e}")
    
    try:
        from scipy.interpolate import interp1d
        print(f"  ✓ scipy.interpolate.interp1d available")
    except ImportError as e:
        print(f"  ✗ scipy.interpolate.interp1d not available: {e}")
        
except ImportError as e:
    print(f"✗ scipy is NOT installed")
    print(f"  Error: {e}")
    print(f"\nTo install scipy in Blender:")
    print(f"  Run in Blender's Python console:")
    print(f"  import subprocess, sys")
    print(f"  subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'scipy'])")

print("\n" + "="*60)
print("Other dependencies:")
print("="*60)

for module_name in ['numpy', 'pandas']:
    try:
        mod = __import__(module_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✓ {module_name}: {version}")
    except ImportError:
        print(f"✗ {module_name}: NOT installed")

print("\n" + "="*60)
print("Checking polychase_core:")
print("="*60)

try:
    import polychase_core
    print(f"✓ polychase_core is installed")
    print(f"  Location: {polychase_core.__file__}")
    
    # Check for Pose
    if hasattr(polychase_core, 'Pose'):
        print(f"  ✓ Pose class available")
    else:
        print(f"  ✗ Pose class NOT available")
        print(f"  Available attributes: {[x for x in dir(polychase_core) if not x.startswith('_')][:10]}")
except ImportError as e:
    print(f"✗ polychase_core is NOT installed")
    print(f"  Error: {e}")

