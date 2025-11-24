"""Verify scipy and pandas installation in Blender"""
import sys

print("Verifying installation...")
print("="*50)

try:
    import scipy
    print(f"✓ scipy: {scipy.__version__}")
    
    from scipy import signal
    from scipy.interpolate import interp1d
    print("✓ scipy.signal available")
    print("✓ scipy.interpolate.interp1d available")
except ImportError as e:
    print(f"✗ scipy error: {e}")

try:
    import pandas as pd
    print(f"✓ pandas: {pd.__version__}")
except ImportError as e:
    print(f"✗ pandas error: {e}")

try:
    import numpy as np
    print(f"✓ numpy: {np.__version__}")
except ImportError as e:
    print(f"✗ numpy error: {e}")

print("="*50)
print("All dependencies should now be available!")
print("\nNext steps:")
print("1. Restart Blender or reload the addon")
print("2. Try enabling the Polychase addon again")
print("3. The scipy import error should be resolved")

