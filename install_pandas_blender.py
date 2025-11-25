#!/usr/bin/env python3
"""
Script to install pandas (and other dependencies) in Blender's Python environment.
Run this from Blender's Python console or as a script.

Usage in Blender Python console:
    exec(open('/path/to/install_pandas_blender.py').read())
"""

import sys
import subprocess
import os

def install_dependencies():
    """Install required dependencies in Blender's Python environment"""
    print("="*60)
    print("Installing Polychase dependencies in Blender's Python")
    print("="*60)
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    
    dependencies = [
        "pandas>=1.5.0",
        "python-dateutil>=2.8.2",
        "pytz>=2023.3",
        "scipy>=1.9.0",
        "numpy>=1.21.0"
    ]
    
    print("\nInstalling dependencies...")
    for dep in dependencies:
        print(f"\nInstalling {dep}...")
        try:
            subprocess.check_call(
                [sys.executable, '-m', 'pip', 'install', '--user', dep],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print(f"✓ {dep} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {dep}")
            print(f"  Error: {e}")
            # Try without --user flag
            try:
                subprocess.check_call(
                    [sys.executable, '-m', 'pip', 'install', dep],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                print(f"✓ {dep} installed successfully (without --user flag)")
            except subprocess.CalledProcessError as e2:
                print(f"✗ Failed to install {dep} (both methods failed)")
                print(f"  Error: {e2}")
    
    print("\n" + "="*60)
    print("Verifying installation...")
    print("="*60)
    
    for module_name in ['pandas', 'dateutil', 'pytz', 'scipy', 'numpy']:
        try:
            mod = __import__(module_name)
            print(f"✓ {module_name}: {mod.__version__}")
        except ImportError:
            print(f"✗ {module_name}: NOT installed")
    
    print("\n" + "="*60)
    print("Installation complete!")
    print("="*60)
    print("\nNote: You may need to restart Blender for changes to take effect.")

if __name__ == "__main__":
    install_dependencies()

