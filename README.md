# Neutron Event Analyzer

A Python package for analyzing event-by-event data from neutron event camera measurements. This package processes .empirphot and .empirevent files, associates photons with events, computes ellipticity, and provides plotting utilities.

## Installation
Clone the repo using git:
```bash
git clone https://github.com/TsvikiHirsh/neutron_event_analyzer.git
```

Then Install the package using pip:
```python
pip install .
```

### Optional: empir Binaries

The empir binaries (`empir_export_events` and `empir_export_photons`) are only required if you need to convert binary .empirevent and .empirphot files to CSV format.

If you already have pre-exported CSV files in `ExportedEvents` and `ExportedPhotons` folders, the empir binaries are **not required**. The package will automatically use the CSV files when available.

## Requirements

- Python >= 3.8
- pandas >= 1.5.0
- numpy >= 1.21.0
- tqdm >= 4.62.0
- matplotlib >= 3.5.0
- scipy >= 1.7.0

## Directory Structure

The package can work with two different data folder structures:

### Option 1: Using Pre-Exported CSV Files (Recommended)

This option allows you to work with already exported CSV files without needing empir binaries:

```
data_folder/
├── photonFiles/
│   └── *.empirphot          (original binary files, used for file matching)
├── eventFiles/
│   └── *.empirevent         (original binary files, used for file matching)
├── ExportedPhotons/
│   └── *.csv                (pre-exported photon CSV files)
└── ExportedEvents/
    └── *.csv                (pre-exported event CSV files)
```

**Note:** The CSV filenames must match the base names of the original binary files (e.g., `data_00001.empirphot` → `data_00001.csv`).

### Option 2: Using empir Binaries

If you don't have pre-exported CSV files, the package can convert the binary files using empir:

```
data_folder/
├── photonFiles/
│   └── *.empirphot
└── eventFiles/
    └── *.empirevent
```

In this case, you must have the `empir_export_events` and `empir_export_photons` binaries available in the `export_dir`.


## Usage

### Basic Usage with Pre-Exported CSV Files (Recommended)

```python
import neutron_event_analyzer as nea

# Initialize the analyzer with pre-exported CSV files
# The export_dir parameter is optional when CSV files exist
analyser = nea.Analyse(data_folder='/path/to/data_folder')

# Load data - automatically uses CSV files from ExportedEvents/ExportedPhotons
analyser.load()

# Associate photons to events
analyser.associate(time_norm_ns=1.0, spatial_norm_px=1.0, dSpace_px=50, verbosity=1)

# Compute ellipticity
analyser.compute_ellipticity()

# Get combined DataFrame
combined_df = analyser.get_combined_dataframe()
```

### Usage with empir Binaries

If you need to convert binary files using empir:

```python
import neutron_event_analyzer as nea

# Initialize the analyzer with path to empir binaries
analyser = nea.Analyse(
    data_folder='/path/to/data_folder',
    export_dir='/path/to/export'  # Directory containing empir binaries
)

# Load data - will convert binary files using empir if CSV files don't exist
analyser.load()

# Associate photons to events
analyser.associate(time_norm_ns=1.0, spatial_norm_px=1.0, dSpace_px=50, verbosity=1)

# Get combined DataFrame
combined_df = analyser.get_combined_dataframe()
```

### Advanced Usage with Query and Filtering

```python
import neutron_event_analyzer as nea

# Initialize analyzer
analyser = nea.Analyse(data_folder='/path/to/data_folder')

# Load with filtering and limits
analyser.load(
    event_glob="[Ee]ventFiles/*.empirevent",
    photon_glob="[Pp]hotonFiles/*.empirphot",
    query="n > 2",  # Only load events with more than 2 photons
    limit=1000      # Limit to first 1000 rows
)

# Associate using different methods
analyser.associate(
    method='simple',     # Options: 'auto', 'kdtree', 'window', 'simple'
    dSpace_px=50,
    max_time_ns=500,
    verbosity=1
)

# Compute ellipticity
analyser.compute_ellipticity()
```

## API Reference

### Main Methods

- `load(event_glob, photon_glob, limit, query)`: Load paired event and photon files. Automatically uses pre-exported CSV files if available, otherwise converts using empir binaries.
- `associate(time_norm_ns, spatial_norm_px, dSpace_px, method, verbosity)`: Associate photons to events using various methods ('auto', 'kdtree', 'window', 'simple', 'lumacam').
- `compute_ellipticity(x_col, y_col, event_col, verbosity)`: Compute ellipticity and shape metrics for associated events.
- `get_combined_dataframe()`: Return the associated DataFrame with all computed metrics.

### Association Methods

- **`auto`**: Automatically selects the best method (lumacam if available, otherwise kdtree)
- **`kdtree`**: Full KDTree-based association on normalized space-time coordinates
- **`window`**: Time-window KDTree for nearly time-sorted data (more efficient)
- **`simple`**: Forward time-window with spatial selection (fastest for small time windows)
- **`lumacam`**: Uses lumacamTesting library (requires optional installation)

## Notes

- **CSV Files**: If you have pre-exported CSV files in `ExportedEvents` and `ExportedPhotons` folders, the empir binaries are not required.
- **empir Binaries**: Only required if converting .empirevent and .empirphot files. The `export_dir` must contain `empir_export_events` and `empir_export_photons` binaries.
- **Multithreading**: The package uses parallel processing for efficient file loading (default: 10 threads).
- **Temporary Files**: When using empir binaries, temporary CSV files are created in `/tmp` and automatically cleaned up after processing.
- **File Matching**: CSV filenames must match the base names of binary files (without extensions).

## Testing

The package includes a comprehensive test suite to verify functionality:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/

# Run with coverage report
pytest --cov=neutron_event_analyzer tests/

# Run specific test file
pytest tests/test_csv_loading.py -v
```

See [tests/README.md](tests/README.md) for detailed testing documentation.

## License

MIT License