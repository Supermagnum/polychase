#!/bin/bash

SCRIPT="$(readlink -f "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT")"

cd "${SCRIPT_DIR}"

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if user has Docker permissions
if ! docker ps &> /dev/null; then
    echo "Error: Docker permission denied. You may need to:"
    echo "  1. Add your user to the docker group: sudo usermod -aG docker $USER"
    echo "  2. Log out and log back in, or run: newgrp docker"
    echo "  3. Or run this script with sudo (not recommended)"
    exit 1
fi

echo "Building Docker image for Python 3.12 wheel..."
docker build -f ./docker/Dockerfile.linux . -t polychase:py312

echo "Running container to build wheel..."
docker run --name polychase-py312 -d -i -t polychase:py312 /bin/sh

echo "Copying wheel from container..."
docker cp polychase-py312:/work/polychase-src/wheelhouse/. ./blender_addon/wheels

echo "Cleaning up container..."
docker container stop polychase-py312
docker container rm polychase-py312

echo "Build complete! Check blender_addon/wheels/ for the new wheel file."
