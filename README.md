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

Ensure the `export` directory containing the binaries `empir_export_events` and `empir_export_photons` is accessible, or specify its path during initialization.

## Requirements

- Python >= 3.8
- pandas >= 1.5.0
- numpy >= 1.21.0
- tqdm >= 4.62.0
- matplotlib >= 3.5.0

## Directory Structure

The package expects a data folder with the following structure:

```
data_folder/
├── photonFiles/
│   └── *.empirphot
└── eventFiles/
    └── *.empirevent
```


## Usage
```python
import neutron_event_analyzer as nea

# Initialize the analyzer
analyser = nea.Analyse(data_folder='/path/to/data_folder', export_dir='/path/to/export')

# Load events and photons
analyser.load_events()
analyser.load_photons()

# Associate photons to events
analyser.associate(time_norm_ns=1.0, spatial_norm_px=1.0, dSpace_px=np.inf, verbosity=1)

# Compute ellipticity
analyser.compute_ellipticity()

# Get combined DataFrame
combined_df = analyser.get_combined_dataframe()

# Plot diagnostic plots
analyser.plot_four(name='my_run', min_n=1, max_n=1000, min_psd=1e-10, max_psd=1)
```

- `load_events(glob_string=None)`: Load `.empirevent` files into a DataFrame.
- `load_photons(glob_string=None)`: Load `.empirphot` files into a DataFrame.
- `associate(time_norm_ns, spatial_norm_px, dSpace_px, verbosity)`: Associate photons to events.
- `compute_ellipticity(x_col, y_col, event_col, verbosity)`: Compute ellipticity and related metrics.
- `plot_four(name, min_n, max_n, min_psd, max_psd, df)`: Generate four diagnostic plots.
- `get_combined_dataframe()`: Return the associated DataFrame.

## Notes

- The `export_dir` must contain the `empir_export_events` and `empir_export_photons` binaries.
- The package uses multithreading for efficient file processing (default: 10 threads).
- Temporary CSV files are created and cleaned up automatically during processing.

## License

MIT License