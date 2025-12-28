# Tests for Neutron Event Analyzer

This directory contains the test suite for the neutron_event_analyzer package.

## Test Structure

```
tests/
├── __init__.py                  # Package initialization
├── conftest.py                  # Pytest fixtures and configuration
├── test_csv_loading.py          # Tests for CSV loading functionality
├── data/                        # Test data directory
│   └── neutrons/               # Simulation-generated test data
│       ├── eventFiles/         # Binary event files (.empirevent)
│       ├── photonFiles/        # Binary photon files (.empirphot)
│       ├── ExportedEvents/     # Pre-exported event CSV files
│       └── ExportedPhotons/    # Pre-exported photon CSV files
└── README.md                    # This file
```

## Running Tests

### Install Test Dependencies

```bash
pip install pytest pytest-cov
```

### Run All Tests

```bash
# From the project root
pytest tests/

# With verbose output
pytest -v tests/

# With coverage report
pytest --cov=neutron_event_analyzer tests/
```

### Run Specific Test Files

```bash
# Test CSV loading functionality
pytest tests/test_csv_loading.py

# Run specific test class
pytest tests/test_csv_loading.py::TestCSVLoading

# Run specific test method
pytest tests/test_csv_loading.py::TestCSVLoading::test_load_with_existing_csv_files
```

## Test Coverage

The current test suite covers:

### CSV Loading Functionality (`test_csv_loading.py`)

1. **TestCSVLoading**: Core CSV loading tests
   - Loading from existing CSV files without empir binaries
   - Data type validation
   - Association functionality with CSV data
   - Query filtering
   - Row limiting
   - File pairing by basename
   - Ellipticity computation
   - Different association methods (kdtree, window, simple)

2. **TestCSVFileFormat**: CSV format validation
   - Event CSV format validation
   - Photon CSV format validation
   - Column name verification

3. **TestErrorHandling**: Error condition tests
   - Missing CSV and missing empir binaries
   - Premature dataframe access

4. **TestDataIntegrity**: Data integrity checks
   - No data loss during loading
   - CSV filename matching with binary files

## Test Data

The test data in `tests/data/neutrons/` is simulation-generated data that includes:
- Binary files (.empirevent, .empirphot)
- Pre-exported CSV files matching the binary files

The test fixtures automatically:
- Create temporary copies of test data
- Fix CSV file naming to match binary basenames
- Fix CSV column names to match expected format
- Clean up after tests complete

## Writing New Tests

When adding new tests:

1. Use the provided fixtures (`temp_data_dir`, `temp_data_dir_no_csv`)
2. Follow the existing test structure and naming conventions
3. Add docstrings explaining what each test does
4. Run tests locally before committing
5. Ensure tests are isolated and don't depend on execution order

## Continuous Integration

These tests can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pip install -e .
    pip install pytest pytest-cov
    pytest --cov=neutron_event_analyzer tests/
```

## Test Fixtures

### `test_data_dir`
Returns the path to the test data directory.

### `temp_data_dir`
Creates a temporary copy of test data with properly formatted CSV files:
- Fixes photon CSV filename to match binary basename
- Fixes photon CSV column names (toa → t)
- Ensures event CSV has correct format with ' PSD value' column

### `temp_data_dir_no_csv`
Creates a temporary copy of test data WITHOUT CSV files to test empir binary fallback.

### `mock_export_dir`
Creates a temporary directory for mock empir binaries.
