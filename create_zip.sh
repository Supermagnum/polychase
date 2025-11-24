#!/bin/bash
# Create zip archive with polychase directory structure

cd "$(dirname "$0")"

# Remove old zip files
rm -f polychase-imu-full-*.zip

# Create temporary directory
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

# Copy files with polychase directory name
cp -r blender_addon "$TMPDIR/polychase"
cp README.md test-results.md IMU_FORMAT_SUPPORT.md TESTING.md setup.py "$TMPDIR/" 2>/dev/null
cp -r tests "$TMPDIR/" 2>/dev/null

# Create zip file
cd "$TMPDIR"
zip -r polychase-imu-full-$(date +%Y%m%d).zip \
  polychase/ \
  README.md \
  test-results.md \
  IMU_FORMAT_SUPPORT.md \
  TESTING.md \
  tests/ \
  setup.py \
  -x "*.pyc" \
  -x "*__pycache__*" \
  -x "*.git*" \
  -x "*.pytest_cache*" \
  -x "*/.DS_Store" > /dev/null 2>&1

# Move zip to project directory
mv polychase-imu-full-*.zip "$OLDPWD/"

echo "Zip created: polychase-imu-full-$(date +%Y%m%d).zip"

