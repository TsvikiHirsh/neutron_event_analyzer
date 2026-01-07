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

The empir binaries (`empir_export_events`, `empir_export_photons`, and `empir_pixel2photon`) are only required if you need to convert binary files (.empirevent, .empirphot, .tpx3) to CSV format.

If you already have pre-exported CSV files in `ExportedEvents`, `ExportedPhotons`, and `ExportedPixels` folders, the empir binaries are **not required**. The package will automatically use the CSV files when available.

## Requirements

- Python >= 3.8
- pandas >= 1.5.0
- numpy >= 1.21.0
- tqdm >= 4.62.0
- matplotlib >= 3.5.0
- scipy >= 1.7.0

## Directory Structure

The package can work with two different data folder structures, now including support for pixel-level data:

### Option 1: Using Pre-Exported CSV Files (Recommended)

This option allows you to work with already exported CSV files without needing empir binaries:

```
data_folder/
├── photonFiles/
│   └── *.empirphot          (original binary files, used for file matching)
├── eventFiles/
│   └── *.empirevent         (original binary files, used for file matching)
├── tpx3Files/               (NEW: pixel data)
│   └── *.tpx3               (TPX3 pixel files)
├── ExportedPhotons/
│   └── *.csv                (pre-exported photon CSV files)
├── ExportedEvents/
│   └── *.csv                (pre-exported event CSV files)
└── ExportedPixels/          (NEW: pixel data)
    └── *.csv                (pre-exported pixel CSV files)
```

**Note:** The CSV filenames must match the base names of the original binary files (e.g., `data_00001.empirphot` → `data_00001.csv`, `data_00001.tpx3` → `data_00001.csv`).

### Option 2: Using empir Binaries

If you don't have pre-exported CSV files, the package can convert the binary files using empir:

```
data_folder/
├── photonFiles/
│   └── *.empirphot
├── eventFiles/
│   └── *.empirevent
└── tpx3Files/               (NEW: pixel data)
    └── *.tpx3
```

In this case, you must have the `empir_export_events`, `empir_export_photons`, and `empir_pixel2photon` binaries available in the `export_dir`.


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

### Pixel-Level Analysis (NEW)

The package now supports pixel-level data analysis with multi-tier association capabilities.

#### Loading Pixel Data Only

```python
import neutron_event_analyzer as nea

# Initialize analyzer
analyser = nea.Analyse(data_folder='/path/to/data_folder')

# Load only pixel data
analyser.load(
    load_events=False,
    load_photons=False,
    load_pixels=True,
    verbosity=1
)

# Access pixel DataFrame
print(f"Loaded {len(analyser.pixels_df)} pixels")
print(analyser.pixels_df.head())
```

#### Two-Tier Association: Pixels → Photons

```python
import neutron_event_analyzer as nea

# Initialize analyzer
analyser = nea.Analyse(data_folder='/path/to/data_folder')

# Load pixels and photons
analyser.load(
    load_events=False,
    load_photons=True,
    load_pixels=True,
    verbosity=1
)

# Associate pixels to photons
result_df = analyser.associate_full(
    pixel_max_dist_px=10.0,      # Maximum spatial distance for pixel-photon matching
    pixel_max_time_ns=1000,      # Maximum time difference for pixel-photon matching
    verbosity=1
)

# Result contains pixels with associated photon information
print(f"Associated {result_df['assoc_photon_id'].notna().sum()} pixels to photons")

# Save results
analyser.save_associations(
    output_dir='/path/to/output',
    filename='pixel_photon_associations.csv',
    verbosity=1
)
```

#### Three-Tier Association: Pixels → Photons → Events

```python
import neutron_event_analyzer as nea

# Initialize analyzer
analyser = nea.Analyse(data_folder='/path/to/data_folder')

# Load all three data types
analyser.load(
    load_events=True,
    load_photons=True,
    load_pixels=True,
    verbosity=1
)

# Perform full three-tier association
result_df = analyser.associate_full(
    # Pixel → Photon parameters
    pixel_max_dist_px=10.0,
    pixel_max_time_ns=1000,

    # Photon → Event parameters
    photon_time_norm_ns=1.0,
    photon_spatial_norm_px=1.0,
    photon_dSpace_px=50.0,
    max_time_ns=500,

    method='simple',  # Association method for photon-event
    verbosity=1
)

# Result is pixel-centric with photon and event information
print(f"Total pixels: {len(result_df)}")
print(f"Pixels → Photons: {result_df['assoc_photon_id'].notna().sum()}")
print(f"Pixels → Events: {result_df['assoc_event_id'].notna().sum()}")

# Save complete association results
output_path = analyser.save_associations(verbosity=1)
print(f"Results saved to: {output_path}")
```

#### Flexible Loading Options

```python
import neutron_event_analyzer as nea

# Initialize analyzer
analyser = nea.Analyse(data_folder='/path/to/data_folder')

# Choose what to load with boolean flags
analyser.load(
    load_events=True,      # Load event data (default: True)
    load_photons=True,     # Load photon data (default: True)
    load_pixels=True,      # Load pixel data (default: False)

    # Optional glob patterns
    event_glob="[Ee]ventFiles/*.empirevent",
    photon_glob="[Pp]hotonFiles/*.empirphot",
    pixel_glob="[Tt]px3Files/*.tpx3",

    # Optional filtering
    query="n > 2",
    limit=1000,

    verbosity=1
)
```

### Using Settings Files for Association Parameters

You can provide empir configuration parameters via a settings JSON file or dictionary. These parameters will be used as defaults for association methods, making it easier to maintain consistency with your empir analysis pipeline.

#### Settings File Structure

```json
{
    "pixel2photon": {
        "dSpace": 2,
        "dTime": 100e-09,
        "nPxMin": 8,
        "nPxMax": 100,
        "TDC1": true
    },
    "photon2event": {
        "dSpace_px": 50.0,
        "dTime_s": 5e-08,
        "durationMax_s": 5e-07,
        "dTime_ext": 5
    }
}
```

#### Usage with Settings File

```python
import neutron_event_analyzer as nea

# Option 1: Load settings from JSON file
analyser = nea.Analyse(
    data_folder='/path/to/data',
    settings='path/to/settings.json'
)

# Option 2: Use settings dictionary
settings = {
    "pixel2photon": {
        "dSpace": 10.0,      # pixels
        "dTime": 1000e-09,   # seconds → converted to ns
    },
    "photon2event": {
        "dSpace_px": 50.0,   # pixels
        "dTime_s": 500e-09,  # seconds → converted to ns
    }
}

analyser = nea.Analyse(
    data_folder='/path/to/data',
    settings=settings
)

# Load and associate - parameters from settings are used as defaults
analyser.load(load_pixels=True, load_photons=True, load_events=True)
analyser.associate_full(verbosity=1)  # Uses settings defaults

# You can still override specific parameters
analyser.associate_full(
    pixel_max_dist_px=15.0,  # Override setting
    verbosity=1
)
```

**Parameter Mapping:**
- `pixel2photon.dSpace` → `pixel_max_dist_px` (pixels)
- `pixel2photon.dTime` → `pixel_max_time_ns` (seconds → nanoseconds)
- `photon2event.dSpace_px` → `photon_dSpace_px` (pixels)
- `photon2event.dTime_s` → `max_time_ns` (seconds → nanoseconds)

## Command-Line Tools (NEW)

The package now includes powerful CLI tools for running associations and optimizing parameters.

### nea-assoc: Run Associations

Process data and perform pixel-photon-event associations from the command line.

```bash
# Basic usage with auto-detected settings
nea-assoc /path/to/data

# Use specific settings preset
nea-assoc /path/to/data --settings in_focus

# Custom association parameters
nea-assoc /path/to/data --photon-dspace 60 --max-time 500

# Disable specific data types
nea-assoc /path/to/data --no-pixels -v 2
```

See [docs/CLI_USAGE.md](docs/CLI_USAGE.md) for complete CLI documentation.

### nea-optimize: Parameter Optimization (NEW)

Automatically find optimal association parameters by analyzing your real data.

#### Iterative Optimization

Run multiple iterations to progressively refine parameters:

```bash
# Basic usage
nea-optimize optimize /path/to/data --output results/

# Custom starting point and iterations
nea-optimize optimize /path/to/data \
  --spatial 25 \
  --temporal 150 \
  --iterations 10 \
  --output results/
```

#### Quick Parameter Suggestion

Get instant parameter recommendations:

```bash
# Analyze data and suggest improvements
nea-optimize suggest /path/to/data \
  --spatial 20 \
  --temporal 100 \
  --output suggested_params.json
```

#### Quality Analysis

Analyze association quality without making suggestions:

```bash
# Get detailed quality metrics
nea-optimize analyze /path/to/data \
  --spatial 20 \
  --temporal 100 \
  --output metrics.json
```

**Python API for Optimization:**

```python
import neutron_event_analyzer as nea

# Iterative optimization
best_params = nea.optimize_parameters_iteratively(
    data_folder='/path/to/data',
    initial_spatial_px=20.0,
    initial_temporal_ns=100.0,
    max_iterations=5,
    output_dir='results/'
)

print(f"Optimized spatial: {best_params['spatial_px']:.2f} px")
print(f"Optimized temporal: {best_params['temporal_ns']:.2f} ns")

# Quick suggestion
suggestion = nea.suggest_parameters_from_data(
    data_folder='/path/to/data',
    current_spatial_px=20.0,
    current_temporal_ns=100.0,
    output_path='suggested_params.json'
)
```

**Output Files:**

- `best_parameters.json`: Optimized parameters in empir format
- `optimization_history.json`: Full iteration history
- `summary.json`: Optimization summary with improvements

See [docs/ITERATIVE_OPTIMIZATION.md](docs/ITERATIVE_OPTIMIZATION.md) for complete optimization documentation.

## Parameter Optimization with Ground Truth (NEW)

For synthetic data with known ground truth, use the optimizer to find the best parameters systematically.

```python
import neutron_event_analyzer as nea

# Grid search optimization
optimizer = nea.AssociationOptimizer(
    synthetic_data_dir='path/to/synthetic_data',
    ground_truth_photons=photon_df,  # With event_id column
    ground_truth_events=event_df,    # With event_id column
    verbosity=1
)

best = optimizer.grid_search(
    methods=['simple', 'kdtree', 'window'],
    spatial_thresholds_px=[10.0, 20.0, 50.0],
    temporal_thresholds_ns=[50.0, 100.0, 500.0],
    metric='f1_score'
)

print(f"Best method: {best.method}")
print(f"Best spatial: {best.spatial_threshold_px} px")
print(f"Best temporal: {best.temporal_threshold_ns} ns")
print(f"F1 Score: {best.f1_score:.4f}")

# Save optimized parameters
optimizer.save_best_parameters('optimized_params.json')
```

See [docs/OPTIMIZER.md](docs/OPTIMIZER.md) for complete ground-truth optimization documentation.

## API Reference

### Analyse Class

#### `__init__()`
Initialize the Analyse object.

**Parameters:**
- `data_folder` (str): Path to the data folder containing input files
- `export_dir` (str): Path to directory with empir binaries (default: `"./export"`)
- `n_threads` (int): Number of threads for parallel processing (default: `10`)
- `use_lumacam` (bool): Prefer lumacam for association when `method='auto'` (default: `False`)
- `settings` (str or dict): Path to settings JSON file or settings dictionary with empir parameters (default: `None`)

### Main Methods

#### `load()`
Load event, photon, and/or pixel data files.

**Parameters:**
- `event_glob` (str): Glob pattern for event files (default: `"[Ee]ventFiles/*.empirevent"`)
- `photon_glob` (str): Glob pattern for photon files (default: `"[Pp]hotonFiles/*.empirphot"`)
- `pixel_glob` (str): Glob pattern for pixel files (default: `"[Tt]px3Files/*.tpx3"`)
- `load_events` (bool): Whether to load event data (default: `True`)
- `load_photons` (bool): Whether to load photon data (default: `True`)
- `load_pixels` (bool): Whether to load pixel data (default: `False`)
- `limit` (int): Maximum number of rows to load per file (default: `None`)
- `query` (str): Pandas query string for filtering (default: `None`)
- `verbosity` (int): Verbosity level (0=quiet, 1=normal, 2=debug) (default: `0`)

**Returns:** None (populates `events_df`, `photons_df`, and/or `pixels_df` attributes)

Automatically uses pre-exported CSV files if available, otherwise converts using empir binaries.

#### `associate()`
Associate photons to events using various methods.

**Parameters:**
- `time_norm_ns` (float): Time normalization factor in nanoseconds (default: `1.0`)
- `spatial_norm_px` (float): Spatial normalization factor in pixels (default: `1.0`)
- `dSpace_px` (float): Maximum spatial distance for association in pixels (default: `50.0`)
- `max_time_ns` (float): Maximum time window for association in nanoseconds (default: `500`)
- `method` (str): Association method - 'auto', 'kdtree', 'window', 'simple', 'lumacam' (default: `'auto'`)
- `verbosity` (int): Verbosity level (default: `0`)

**Returns:** pandas.DataFrame with associated photon-event data

#### `associate_full()`
Perform multi-tier association: pixels → photons → events.

**Parameters:**
- `pixel_max_dist_px` (float, optional): Maximum spatial distance for pixel-photon matching in pixels. If None, uses value from settings or defaults to `5.0`
- `pixel_max_time_ns` (float, optional): Maximum time difference for pixel-photon matching in nanoseconds. If None, uses value from settings or defaults to `500`
- `photon_time_norm_ns` (float): Time normalization for photon-event association (default: `1.0`)
- `photon_spatial_norm_px` (float): Spatial normalization for photon-event association (default: `1.0`)
- `photon_dSpace_px` (float, optional): Maximum spatial distance for photon-event association. If None, uses value from settings or defaults to `50.0`
- `max_time_ns` (float, optional): Maximum time window for photon-event association in nanoseconds. If None, uses value from settings or defaults to `500`
- `method` (str): Association method for photon-event - 'auto', 'kdtree', 'window', 'simple' (default: `'simple'`)
- `verbosity` (int): Verbosity level (default: `1`)

**Returns:** pandas.DataFrame with multi-tier association results

**Behavior:**
- If pixels, photons, and events are loaded: performs 3-tier association (pixels → photons → events)
- If only pixels and photons are loaded: performs 2-tier association (pixels → photons)
- If only photons and events are loaded: performs standard photon-event association
- Parameters marked as optional can be provided via settings file/dict or will use hard-coded defaults

#### `save_associations()`
Save association results to disk.

**Parameters:**
- `output_dir` (str): Output directory (default: `{data_folder}/AssociatedResults`)
- `filename` (str): Output filename (default: `"associated_data.csv"`)
- `format` (str): Output format - 'csv' or 'parquet' (default: `'csv'`)
- `verbosity` (int): Verbosity level (default: `1`)

**Returns:** str (path to saved file)

**Raises:** ValueError if no association data exists

#### `compute_ellipticity()`
Compute ellipticity and shape metrics for associated events.

**Parameters:**
- `x_col` (str): Column name for x coordinates (default: `'x'`)
- `y_col` (str): Column name for y coordinates (default: `'y'`)
- `event_col` (str): Column name for event ID (default: `'assoc_cluster_id'`)
- `verbosity` (int): Verbosity level (default: `0`)

**Returns:** None (adds ellipticity columns to `associated_df`)

#### `get_combined_dataframe()`
Return the associated DataFrame with all computed metrics.

**Returns:** pandas.DataFrame

### Association Methods

- **`auto`**: Automatically selects the best method (lumacam if available, otherwise kdtree)
- **`kdtree`**: Full KDTree-based association on normalized space-time coordinates
- **`window`**: Time-window KDTree for nearly time-sorted data (more efficient)
- **`simple`**: Forward time-window with spatial selection (fastest for small time windows)
- **`lumacam`**: Uses lumacamTesting library (requires optional installation)

## Notes

- **CSV Files**: If you have pre-exported CSV files in `ExportedEvents`, `ExportedPhotons`, and `ExportedPixels` folders, the empir binaries are not required.
- **empir Binaries**: Only required if converting binary files (.empirevent, .empirphot, .tpx3). The `export_dir` must contain:
  - `empir_export_events` - for .empirevent files
  - `empir_export_photons` - for .empirphot files
  - `empir_pixel2photon` - for .tpx3 pixel files
- **Multithreading**: The package uses parallel processing for efficient file loading (default: 10 threads).
- **Temporary Files**: When using empir binaries, temporary CSV files are created in `/tmp` and automatically cleaned up after processing.
- **File Matching**: CSV filenames must match the base names of binary files (without extensions).
- **Pixel Data Format**: Pixel CSV files must contain columns: `x [px]`, `y [px]`, `t [s]`, `tot [a.u.]`, `t_relToExtTrigger [s]` (or equivalent).
- **Association Results**: Multi-tier association results are automatically saved to `{data_folder}/AssociatedResults/` folder when using `save_associations()`.

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