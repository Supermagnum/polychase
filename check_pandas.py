#!/usr/bin/env python3
"""Check if pandas is installed and can be imported correctly"""
import sys

print("="*60)
print("Pandas Installation Check")
print("="*60)
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Python path (first 3 entries):")
for p in sys.path[:3]:
    print(f"  {p}")

print("\n" + "-"*60)
print("Testing pandas import...")
print("-"*60)

try:
    import pandas as pd
    print(f"✓ pandas is installed")
    print(f"  Version: {pd.__version__}")
    print(f"  Location: {pd.__file__}")
    
    # Test basic functionality
    try:
        import io
        test_csv = "x,y,z,timestamp\n1.0,2.0,3.0,1000\n4.0,5.0,6.0,2000\n"
        df = pd.read_csv(io.StringIO(test_csv))
        print(f"  ✓ pd.read_csv() works correctly")
        print(f"    Test data shape: {df.shape}")
    except Exception as e:
        print(f"  ✗ pd.read_csv() failed: {e}")
        
except ImportError as e:
    print(f"✗ pandas is NOT installed")
    print(f"  Error: {e}")
    print(f"\nInstallation instructions:")
    print(f"  1. Blender 4.2+: Should install automatically via blender_manifest.toml")
    print(f"  2. Manual: Run in Blender's Python console:")
    print(f"     import subprocess, sys")
    print(f"     subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pandas>=1.5.0'])")
    print(f"  3. System-wide: pip install pandas>=1.5.0")

print("\n" + "-"*60)
print("Checking blender_manifest.toml...")
print("-"*60)

try:
    import tomllib
    with open("blender_addon/blender_manifest.toml", "rb") as f:
        manifest = tomllib.load(f)
    if "dependencies" in manifest:
        deps = manifest["dependencies"]
        if "pandas" in deps:
            print(f"✓ pandas found in blender_manifest.toml")
            print(f"  Requirement: {deps['pandas']}")
        else:
            print(f"✗ pandas NOT found in blender_manifest.toml dependencies")
    else:
        print(f"✗ No [dependencies] section in blender_manifest.toml")
except Exception as e:
    print(f"Could not check blender_manifest.toml: {e}")

print("\n" + "="*60)

