# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 Ahmed Essam <aessam.dahy@gmail.com>
#
# Makefile for Polychase development tasks

.PHONY: help test test-cov lint format type-check security clean install-dev

help:
	@echo "Polychase Development Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  install-dev    Install development dependencies"
	@echo "  test           Run all tests"
	@echo "  test-cov       Run tests with coverage report"
	@echo "  test-fast      Run fast tests only (skip slow/performance)"
	@echo "  test-imu       Run IMU-related tests only"
	@echo "  lint           Run all linters (flake8, pylint)"
	@echo "  format         Format code with Black and isort"
	@echo "  format-check   Check code formatting without modifying"
	@echo "  type-check     Run type checking with mypy"
	@echo "  security       Run security scan with bandit"
	@echo "  quality        Run all quality checks (format, lint, type, security)"
	@echo "  clean          Clean generated files and caches"
	@echo "  pre-commit     Run pre-commit hooks on all files"

install-dev:
	pip install -e .[dev]

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=blender_addon --cov-report=html --cov-report=term-missing

test-fast:
	pytest tests/ -v -m "not slow and not performance"

test-imu:
	pytest tests/ -v -m imu

lint:
	flake8 blender_addon/ tests/
	pylint blender_addon/ --exit-zero

format:
	black .
	isort .

format-check:
	black --check .
	isort --check-only .

type-check:
	mypy blender_addon/ --ignore-missing-imports

security:
	bandit -r blender_addon/ -ll

quality: format-check lint type-check security

clean:
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} + 2>/dev/null || true
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf dist
	rm -rf build

pre-commit:
	pre-commit run --all-files

