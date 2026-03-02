# Neutron Event Analyzer (NEA)

Associate pixels → photons → events from neutron event camera data.

> **Note:** NEA assumes `ExportedPixels/`, `ExportedPhotons/`, and `ExportedEvents/`
> directories already exist (produced by [empirun](https://github.com/TsvikiHirsh/empirun)).
> Its sole job is to build the associated table.

## TLDR

```bash
pip install git+https://github.com/TsvikiHirsh/neutron_event_analyzer.git

# Associate data in a folder
nea-assoc ./my_data

# Use a settings preset and tag the output
nea-assoc ./my_data --settings in_focus --suffix run1

# Use ML-based association
nea-assoc ./my_data --method ml
```

Output: `my_data/AssociatedResults/associated_data[_suffix].csv`

---

## Install

```bash
git clone https://github.com/TsvikiHirsh/neutron_event_analyzer.git
cd neutron_event_analyzer
pip install -e .

# For ML methods (sklearn + torch)
pip install -e ".[ml]"
```

## Data Layout

```
my_data/
├── ExportedPixels/      # CSV pixel files (from empirun)
├── ExportedPhotons/     # CSV photon files
└── ExportedEvents/      # CSV event files
```

NEA reads all CSVs from each directory, associates across tiers, and writes results to:

```
my_data/
└── AssociatedResults/
    ├── associated_data.csv        # Associated table
    └── association_stats.json     # Match rates and distributions
```

## CLI Usage

```
nea-assoc <data> [--settings PRESET|FILE] [--method METHOD] [--suffix TEXT] [-v] [-q]
```

| Option | Description |
|---|---|
| `data` | Path to data folder |
| `--settings, -s` | Preset or JSON file (`in_focus`, `out_of_focus`, `fast_neutrons`, `hitmap`) |
| `--method, -m` | Association method: `simple` (default), `kdtree`, `window`, `mystic`, `ml` |
| `--suffix` | Tag output files (e.g. `run1` → `associated_data_run1.csv`) |
| `-v` / `-vv` | Verbose / debug output |
| `-q` | Quiet mode |
| `--advanced` | Reveal all advanced options |

Show full option list:

```bash
nea-assoc --advanced --help
```

### Settings presets

| Preset | Use case |
|---|---|
| `in_focus` | Standard in-focus neutron imaging |
| `out_of_focus` | Defocused / divergent beam |
| `fast_neutrons` | Fast neutron experiments |
| `hitmap` | High-rate hitmap mode |

A `parameterSettings.json` file in the data folder is auto-detected and used as settings.

### Association methods

| Method | Description |
|---|---|
| `simple` | Fast forward time-window with center-of-mass check (default) |
| `kdtree` | KDTree on normalized space-time with iterative CoM refinement |
| `window` | Symmetric sliding time-window KDTree |
| `mystic` | Constrained optimization (requires `mystic` package) |
| `ml` | Trained ML model (requires `scikit-learn`; trains automatically) |

## Python API

```python
import neutron_event_analyzer as nea

# Load and associate
analyser = nea.Analyse('./my_data', settings='in_focus')
analyser.load()
analyser.associate(method='simple', suffix='run1')

# Access results
df = analyser.associated_df
stats = analyser.get_association_stats()

# Save explicitly (also done automatically by associate())
analyser.save_associations(output_dir='./results', suffix='run1')
```

### ML training

```python
analyser.load()
analyser.associate(method='simple')          # Bootstrap labels
analyser.train_association_model()           # Train on bootstrapped data
analyser.associate(method='ml', suffix='ml') # Re-associate with ML model
```

## License

MIT — see [LICENSE](LICENSE).

## Citation

```bibtex
@software{neutron_event_analyzer,
  author = {Hirsh, Tsviki Y.},
  title  = {Neutron Event Analyzer},
  url    = {https://github.com/TsvikiHirsh/neutron_event_analyzer},
  year   = {2024}
}
```
