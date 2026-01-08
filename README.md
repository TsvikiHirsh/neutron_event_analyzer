# Neutron Event Analyzer

Analyze event-by-event data from neutron event cameras and optimize EMPIR reconstruction parameters.

## Quick Start

```bash
# Install
pip install git+https://github.com/TsvikiHirsh/neutron_event_analyzer.git

# Set path to EMPIR binaries (optional, used for reconstruction)
export EMPIR_PATH=/path/to/empir/binaries

# Get parameter suggestions for EMPIR reconstruction
nea-suggest /path/to/data

# Run association analysis
nea-assoc /path/to/data --settings in_focus
```

## What It Does

This tool analyzes EMPIR reconstruction data and suggests optimal parameters for:

- **pixel2photon**: Clustering pixels into photons (`dSpace`, `dTime`, `nPxMin`, `nPxMax`)
- **photon2event**: Clustering photons into events (`dSpace_px`, `dTime_s`, `durationMax_s`)

The suggestions are based on statistical analysis of reconstruction outputs - **no ground truth required**.

## Installation

### Requirements
- Python ≥ 3.8
- EMPIR binaries (for reconstruction)

### Install
```bash
git clone https://github.com/TsvikiHirsh/neutron_event_analyzer.git
cd neutron_event_analyzer
pip install -e .
```

## Data Structure

Your data folder should contain:
```
data_folder/
├── tpx3Files/          # Raw pixel data
│   └── *.tpx3
├── photonFiles/        # Reconstructed photons
│   └── *.empirphot
└── eventFiles/         # Reconstructed events
    └── *.empirevent
```

The tool will automatically:
- Export files to CSV when needed
- Run reconstruction if files are missing
- Load and analyze the data

## CLI Usage

### Parameter Optimization: `nea-suggest`

Suggest optimal EMPIR reconstruction parameters based on data analysis.

```bash
# Basic usage
nea-suggest <data> [options]

# Examples
nea-suggest ./my_data                              # Optimize both stages
nea-suggest ./my_data --stage photon2event         # Specific stage only
nea-suggest ./my_data --params current.json        # Use baseline
nea-suggest ./my_data -q                           # Quiet mode
```

**Options:**
- `--stage, -s`: Stage to optimize (`pixel2photon`, `photon2event`, `both`) [default: `both`]
- `--params`: Current parameters JSON for comparison
- `--output, -o`: Output file [default: `<data>/.suggestedSettingsParameters.json`]
- `--binaries`: EMPIR binaries directory [default: `$EMPIR_PATH`]
- `--verbose, -v`: Increase verbosity (`-vv` for debug)
- `--quiet, -q`: Minimal output

### Association Analysis: `nea-assoc`

Associate pixels, photons, and events with configurable parameters.

```bash
# Basic usage
nea-assoc <data> [options]

# Examples
nea-assoc ./my_data                                # Auto-detect settings
nea-assoc ./my_data --settings in_focus            # Use preset
nea-assoc ./my_data --no-pixels                    # Skip pixel data
nea-assoc ./my_data --photon-dspace 60             # Custom parameters
```

**Options:**
- `--settings, -s`: Settings preset or JSON file
- `--binaries`: EMPIR binaries directory [default: `$EMPIR_PATH`]
- `--no-events/--no-photons/--no-pixels`: Skip data types
- `--photon-dspace`: Spatial clustering threshold
- `--max-time`: Temporal window
- `--output-dir, -o`: Output directory
- `--verbose, -v`: Increase verbosity
- `--quiet, -q`: Minimal output

## Python API

```python
import neutron_event_analyzer as nea

# Load and analyze data
analyser = nea.Analyse(data_folder='./data')
analyser.load()

# Associate data (pixels → photons → events)
analyser.associate(method='simple', verbosity=1)

# Get combined dataframe
df = analyser.get_combined_dataframe()

# Compute event shapes
analyser.compute_ellipticity()

# Plot
plotter = nea.Plotter(analyser)
plotter.plot_event(event_id=1)
```

## How It Works

The tool analyzes intrinsic distributions in reconstruction outputs:

1. **Temporal Clustering**: Examines time differences within clusters to assess clustering quality
2. **Spatial Clustering**: Analyzes spatial spread to detect over/under-clustering
3. **Cluster Sizes**: Studies size distributions to suggest appropriate min/max thresholds
4. **Event Quality**: Evaluates multiplicity and duration patterns

Each parameter is optimized independently based on its specific impact on these distributions.

## Output

Results are saved as JSON with structure:
```json
{
  "pixel2photon": {
    "dSpace": 3.5,
    "dTime": 1e-7,
    "nPxMin": 3,
    "nPxMax": 20
  },
  "photon2event": {
    "dSpace_px": 75.0,
    "dTime_s": 5e-8,
    "durationMax_s": 5e-7
  }
}
```

## Documentation

Full documentation: https://neutron-event-analyzer.readthedocs.io

- [Parameter Optimization Guide](docs/parameter_optimization.md)
- [API Reference](docs/api.md)
- [Examples](docs/examples.md)

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{neutron_event_analyzer,
  author = {Hirsh, Tsviki Y.},
  title = {Neutron Event Analyzer},
  url = {https://github.com/TsvikiHirsh/neutron_event_analyzer},
  year = {2024}
}
```

## Contact

- **Author**: Tsviki Y. Hirsh
- **Email**: tsviki.hirsh@gmail.com
- **Issues**: https://github.com/TsvikiHirsh/neutron_event_analyzer/issues
