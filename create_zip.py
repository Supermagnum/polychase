#!/usr/bin/env python3
"""Create zip archive with polychase directory structure"""
import os
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

def create_zip():
    project_root = Path(__file__).parent
    date_str = datetime.now().strftime("%Y%m%d")
    zip_name = f"polychase-imu-full-{date_str}.zip"
    
    # Remove old zip files
    for old_zip in project_root.glob("polychase-imu-full-*.zip"):
        old_zip.unlink()
    
    # Create temporary directory
    tmpdir = project_root / "tmp_zip_build"
    if tmpdir.exists():
        shutil.rmtree(tmpdir)
    tmpdir.mkdir()
    
    try:
        # Copy blender_addon as polychase
        shutil.copytree(project_root / "blender_addon", tmpdir / "polychase")
        
        # Copy other files
        for file in ["README.md", "test-results.md", "IMU_FORMAT_SUPPORT.md", "TESTING.md", "setup.py"]:
            src = project_root / file
            if src.exists():
                shutil.copy2(src, tmpdir / file)
        
        # Copy tests directory
        if (project_root / "tests").exists():
            shutil.copytree(project_root / "tests", tmpdir / "tests")
        
        # Create zip file
        zip_path = project_root / zip_name
        exclude_patterns = {".pyc", "__pycache__", ".git", ".pytest_cache", ".DS_Store"}
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(tmpdir):
                # Filter out excluded directories
                dirs[:] = [d for d in dirs if not any(pattern in d for pattern in exclude_patterns)]
                
                for file in files:
                    # Skip excluded files
                    if any(pattern in file for pattern in exclude_patterns):
                        continue
                    
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(tmpdir)
                    zf.write(file_path, arcname)
        
        print(f"Zip created: {zip_name}")
        print(f"Size: {zip_path.stat().st_size / 1024 / 1024:.1f} MB")
        
        # Verify key files
        with zipfile.ZipFile(zip_path, 'r') as zf:
            files = zf.namelist()
            if "polychase/core.py" in files:
                print("✓ polychase/core.py included")
            if any("cp312" in f and "wheels" in f for f in files):
                print("✓ Python 3.12 wheel included")
            else:
                print("⚠ Python 3.12 wheel not found in zip")
        
    finally:
        # Cleanup
        if tmpdir.exists():
            shutil.rmtree(tmpdir)

if __name__ == "__main__":
    create_zip()

