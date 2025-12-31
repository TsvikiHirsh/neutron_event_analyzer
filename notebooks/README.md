# Examples

This directory contains example notebooks and scripts demonstrating the neutron event analyzer package.

## Files

### Jupyter Notebooks

- **[pixel_association_demo.ipynb](pixel_association_demo.ipynb)** - Comprehensive demo of pixel-photon-event association
  - Loading pixel, photon, and event data
  - Three-tier association workflow
  - Visualization of association quality
  - Cluster size analysis
  - Spatial and temporal distribution plots

### Python Scripts

- **[settings_usage_example.py](settings_usage_example.py)** - Demonstrates using settings files for association parameters
  - Loading settings from JSON file
  - Using settings dictionary
  - Overriding settings with explicit parameters

### Configuration Files

- **[settings_example.json](settings_example.json)** - Example settings file with empir parameters
  - `pixel2photon` configuration
  - `photon2event` configuration
  - `event2image` configuration

## Running the Examples

### Jupyter Notebook

```bash
# Install Jupyter if needed
pip install jupyter

# Start Jupyter
jupyter notebook

# Navigate to examples/pixel_association_demo.ipynb
```

### Python Scripts

```bash
cd examples
python3 settings_usage_example.py
```

## Settings File Usage

The settings file allows you to specify empir analysis parameters that will be used as defaults for association methods:

```python
import neutron_event_analyzer as nea

# Load with settings
analyser = nea.Analyse(
    data_folder='../tests/data/neutrons',
    settings='settings_example.json'
)

# Association will use parameters from settings
analyser.load(load_pixels=True, load_photons=True, load_events=True)
analyser.associate_full(verbosity=1)
```

See [settings_usage_example.py](settings_usage_example.py) for detailed examples.
