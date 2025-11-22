#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 Ahmed Essam <aessam.dahy@gmail.com>
#
# Setup script for testing and quality assurance infrastructure

set -e

echo "Setting up Polychase testing infrastructure..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Upgrade pip
echo "Upgrading pip..."
python3 -m pip install --upgrade pip

# Install development dependencies
echo "Installing development dependencies..."
python3 -m pip install -e .[dev] || {
    echo "Installing dependencies manually..."
    python3 -m pip install \
        pytest pytest-cov pytest-benchmark \
        black isort \
        flake8 flake8-docstrings flake8-bugbear flake8-comprehensions \
        pylint mypy bandit \
        memory-profiler line-profiler \
        pre-commit \
        numpy scipy pandas
}

# Install pre-commit hooks
if command -v pre-commit &> /dev/null; then
    echo "Installing pre-commit hooks..."
    pre-commit install
    echo "Pre-commit hooks installed successfully."
else
    echo "Warning: pre-commit not found. Skipping hook installation."
fi

# Create test data directory
echo "Creating test data directories..."
mkdir -p tests/data/video_samples
mkdir -p tests/data/imu_data
mkdir -p tests/data/expected_results

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Run tests: pytest"
echo "  2. Check formatting: black --check ."
echo "  3. Run linting: flake8 blender_addon/ tests/"
echo "  4. Run type checking: mypy blender_addon/"
echo ""
echo "For more information, see tests/README.md"

