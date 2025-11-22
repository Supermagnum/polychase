# Testing and Quality Assurance Guide

Quick reference for Polychase testing and code quality.

## Quick Start

```bash
# Setup
bash setup_testing.sh

# Run all tests
pytest

# Run quality checks
make quality
```

## Test Commands

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=blender_addon --cov-report=html

# Specific markers
pytest -m imu          # IMU tests only
pytest -m unit         # Unit tests only
pytest -m performance  # Benchmarks
pytest -m "not slow"   # Skip slow tests

# Verbose output
pytest -v -s
```

### Code Quality

```bash
# Format code
black .
isort .

# Check formatting
black --check .
isort --check-only .

# Lint
flake8 blender_addon/ tests/
pylint blender_addon/

# Type check
mypy blender_addon/

# Security scan
bandit -r blender_addon/
```

### Using Make

```bash
make test          # Run tests
make test-cov      # Tests with coverage
make lint          # Run linters
make format        # Format code
make type-check    # Type checking
make security      # Security scan
make quality       # All quality checks
make clean         # Clean generated files
```

## Pre-commit Hooks

Hooks run automatically on `git commit`. They check:

- Code formatting (Black, isort)
- Linting (Flake8)
- Type checking (mypy)
- Security (Bandit)

Install: `pre-commit install`

Run manually: `pre-commit run --all-files`

## Local Testing

All tests and quality checks should be run locally before committing:

```bash
# Run all checks
make quality
make test-cov
```

## Coverage Requirements

- Overall: 80% minimum
- IMU module: 95% minimum
- Critical paths: 95% minimum

View reports: `htmlcov/index.html`

## Writing Tests

### Test Structure

```python
@pytest.mark.unit
@pytest.mark.imu
class TestMyFeature:
    """Test description."""
    
    def test_something(self, sample_imu_data):
        """Test case."""
        # Arrange
        data = sample_imu_data
        
        # Act
        result = process(data)
        
        # Assert
        assert result is not None
```

### Using Fixtures

Available fixtures (in `conftest.py`):

- `temp_dir`: Temporary directory
- `sample_imu_data`: Synthetic IMU data
- `sample_video_timestamps`: Video frame timestamps
- `opencamera_csv_files`: CSV files in OpenCamera format
- `mock_blender_context`: Mock Blender context

### Mock Data Generators

Use `tests/data_generators.py`:

```python
from tests.data_generators import (
    generate_synthetic_imu_data,
    create_opencamera_csv_files,
    generate_noisy_imu_data,
    generate_drift_imu_data,
)
```

## Performance Testing

### Benchmarks

```bash
# Run benchmarks
pytest tests/test_performance.py --benchmark-only

# Compare with previous run
pytest tests/test_performance.py --benchmark-only --benchmark-compare
```

### Memory Profiling

```bash
# Run memory tests
pytest tests/test_memory.py -v

# Profile specific function
python -m memory_profiler tests/test_imu_integration.py
```

## Troubleshooting

### Import Errors

Run from project root:

```bash
cd /path/to/polychase
pytest
```

### Missing Dependencies

```bash
pip install -e .[dev]
```

### Blender API

Tests requiring Blender API use mocks. If issues occur, check mock implementations in `conftest.py`.

## File Structure

```
polychase/
├── tests/
│   ├── conftest.py              # Fixtures
│   ├── data_generators.py      # Mock data
│   ├── test_imu_integration.py # IMU tests
│   ├── test_performance.py     # Benchmarks
│   └── test_memory.py          # Memory tests
├── .pre-commit-config.yaml     # Pre-commit hooks
├── .github/workflows/test.yml  # CI/CD
├── pyproject.toml              # Tool configs
├── .flake8                     # Flake8 config
├── .pylintrc                    # Pylint config
├── mypy.ini                     # Mypy config
├── pytest.ini                   # Pytest config
├── Makefile                     # Development tasks
└── setup_testing.sh            # Setup script
```

## Best Practices

1. **Write tests first** (TDD)
2. **Maintain coverage** above thresholds
3. **Run quality checks** before committing
4. **Use fixtures** for test data
5. **Mark tests** appropriately (unit, integration, slow, etc.)
6. **Document** complex test cases
7. **Keep tests fast** - use markers for slow tests

## Property-Based Testing

Polychase uses Hypothesis for property-based testing, which automatically generates test cases:

```bash
pytest tests/test_imu_property_based.py -v
```

Property-based tests verify:
- Gravity vector normalization invariants
- Quaternion properties (normalization, valid rotations)
- Scalability with various data sizes
- Timestamp interpolation correctness
- Orientation blending weights

## Robustness Testing

Comprehensive tests for malformed data and edge cases:

```bash
pytest tests/test_imu_robustness.py -v
```

Tests cover:
- **Corrupted CSV files**: Missing columns, wrong delimiters, NaN values, incomplete lines
- **Invalid IMU data**: NaN/inf values, negative timestamps, extremely large values
- **Video file issues**: Corrupted MP4, wrong formats, empty files
- **Synchronization problems**: Mismatched timestamps, no overlapping data
- **Boundary conditions**: Zero values, maximum sensor values, timestamp gaps

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [Hypothesis property-based testing](https://hypothesis.readthedocs.io/)
- [Black formatting](https://black.readthedocs.io/)
- [mypy type checking](https://mypy.readthedocs.io/)
- [Pre-commit hooks](https://pre-commit.com/)

