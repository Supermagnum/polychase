# Polychase Test Suite

This directory contains the comprehensive test suite for the Polychase Blender addon with IMU integration.

## Test Structure

```
tests/
├── __init__.py
├── conftest.py              # Pytest fixtures and configuration
├── data_generators.py       # Mock data generators for testing
├── test_imu_integration.py  # Unit tests for IMU module
├── test_performance.py      # Performance benchmarks
├── test_memory.py           # Memory profiling tests
└── README.md                # This file
```

## Running Tests

### Prerequisites

Install test dependencies:

```bash
pip install -e .[dev]
# Or manually:
pip install pytest pytest-cov pytest-benchmark
pip install numpy scipy pandas
```

### Basic Test Execution

Run all tests:

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=blender_addon --cov-report=html
```

View coverage report:

```bash
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

### Running Specific Test Categories

Run only unit tests:

```bash
pytest -m unit
```

Run only IMU-related tests:

```bash
pytest -m imu
```

Run performance benchmarks:

```bash
pytest -m performance
```

Run memory profiling tests:

```bash
pytest -m memory
```

Skip slow tests:

```bash
pytest -m "not slow"
```

### Verbose Output

Get detailed test output:

```bash
pytest -v
```

Show print statements:

```bash
pytest -s
```

### Test Markers

Available test markers:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.imu` - IMU-related tests
- `@pytest.mark.performance` - Performance benchmarks
- `@pytest.mark.memory` - Memory profiling tests
- `@pytest.mark.slow` - Slow-running tests

### Property-Based Testing

Run property-based tests using Hypothesis:

```bash
pytest tests/test_imu_property_based.py -v
```

These tests automatically generate random but valid test cases to find edge cases and verify invariants.

### Robustness Testing

Test handling of malformed data and edge cases:

```bash
pytest tests/test_imu_robustness.py -v
```

Tests include:
- Corrupted CSV files
- Malformed data formats
- Boundary conditions
- Data synchronization issues

## Code Quality Checks

### Formatting

Check code formatting:

```bash
black --check .
```

Format code:

```bash
black .
```

### Import Sorting

Check import order:

```bash
isort --check-only .
```

Sort imports:

```bash
isort .
```

### Linting

Run flake8:

```bash
flake8 blender_addon/ tests/
```

Run pylint:

```bash
pylint blender_addon/
```

### Type Checking

Run mypy:

```bash
mypy blender_addon/
```

### Security Scanning

Run bandit:

```bash
bandit -r blender_addon/
```

## Pre-commit Hooks

Install pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

Run hooks manually:

```bash
pre-commit run --all-files
```

## Performance Profiling

### Using pytest-benchmark

Run benchmarks:

```bash
pytest tests/test_performance.py --benchmark-only
```

Compare benchmarks:

```bash
pytest tests/test_performance.py --benchmark-only --benchmark-compare
```

### Memory Profiling

Run memory tests:

```bash
pytest tests/test_memory.py -v
```

Profile specific function:

```bash
python -m memory_profiler tests/test_imu_integration.py
```

## Running Tests Locally

All tests should be run locally before committing changes. Use the commands described above to ensure code quality.

## Test Data

Test data is generated programmatically using `data_generators.py`. This includes:

- Synthetic IMU data with known ground truth
- OpenCamera-Sensors format CSV files
- Corrupted data for error handling tests
- Noisy and drift-affected data for robustness testing

## Writing New Tests

### Unit Test Template

```python
@pytest.mark.unit
@pytest.mark.imu
class TestMyFeature:
    """Test description."""
    
    def test_something(self):
        """Test case description."""
        # Arrange
        data = generate_synthetic_imu_data()
        
        # Act
        result = process_data(data)
        
        # Assert
        assert result is not None
```

### Using Fixtures

```python
def test_with_fixture(sample_imu_data, temp_dir):
    """Test using fixtures."""
    # Use sample_imu_data and temp_dir
    pass
```

### Parametrized Tests

```python
@pytest.mark.parametrize("num_samples", [100, 1000, 10000])
def test_scalability(num_samples):
    """Test with different parameters."""
    data = generate_synthetic_imu_data(num_samples=num_samples)
    # Test...
```

## Coverage Goals

- Overall coverage: 80% minimum
- IMU integration module: 95% minimum
- Critical paths: 95% minimum

View coverage reports in `htmlcov/` after running tests with coverage.

## Troubleshooting

### Import Errors

If you get import errors, ensure you're running from the project root:

```bash
cd /path/to/polychase
pytest
```

### Missing Dependencies

Install all dependencies:

```bash
pip install -e .[dev]
```

### Blender API Issues

Tests that require Blender API (bpy) are marked and may be skipped if Blender is not available. Mock implementations are used where possible.

## Contributing

When adding new features:

1. Write tests first (TDD approach)
2. Ensure all tests pass
3. Maintain or improve coverage
4. Run all quality checks
5. Update this README if needed

