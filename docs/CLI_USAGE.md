# nea-assoc: Neutron Event Analyzer CLI Tool

Command-line interface for performing pixel-photon-event association on neutron detector data.

## Quick Start

The simplest usage requires only a data folder:

```bash
nea-assoc /path/to/run112_ZnS
```

This will:
1. Auto-detect the folder structure
2. Look for `.parameterSettings.json` in the folder
3. Load all available data (events, photons, pixels)
4. Perform association
5. Save results to `<data_folder>/AssociatedResults/associated_data.csv`

## Expected Folder Structure

The tool expects your data folder to contain:

```
run112_ZnS/
├── eventFiles/              # Event data (.empirevent files)
├── photonFiles/             # Photon data (.empirphot files)
├── tpx3Files/              # Pixel data (.tpx3 files)
├── ExportedEvents/         # CSV exports (optional, auto-used if present)
├── ExportedPhotons/        # CSV exports (optional, auto-used if present)
├── ExportedPixels/         # CSV exports (optional, auto-used if present)
└── .parameterSettings.json # Settings file (optional, auto-detected)
```

The tool is flexible with folder names - it will look for variations like `EventFiles`, `photonFiles`, `TPX3Files`, etc.

## Basic Usage Examples

### 1. Default Analysis
```bash
# Uses all available data with auto-detected settings
nea-assoc /path/to/data
```

### 2. Use a Settings Preset
```bash
# Use one of the built-in presets
nea-assoc /path/to/data --settings in_focus

# Available presets: in_focus, out_of_focus, fast_neutrons, hitmap
```

### 3. Custom Settings File
```bash
# Use your own settings JSON file
nea-assoc /path/to/data --settings my_custom_settings.json
```

### 4. Skip Pixel Data (Photon-Event Association Only)
```bash
# Faster if you don't need pixel-level association
nea-assoc /path/to/data --no-pixels
```

### 5. Test Run with Limited Data
```bash
# Process only first 100 rows per file (good for testing)
nea-assoc /path/to/data --limit 100
```

## Advanced Usage

### Control Verbosity

```bash
# Quiet mode - only show errors
nea-assoc /path/to/data --quiet

# Normal mode (default)
nea-assoc /path/to/data

# Verbose mode - show more details
nea-assoc /path/to/data -v

# Very verbose - show debug information
nea-assoc /path/to/data -vv
```

### Customize Association Parameters

```bash
# Override pixel-photon association parameters
nea-assoc /path/to/data \
  --pixel-max-dist 10 \
  --pixel-max-time 1000

# Override photon-event association parameters
nea-assoc /path/to/data \
  --photon-dspace 60 \
  --max-time 500 \
  --method kdtree
```

### Custom Output Location

```bash
# Specify output directory and filename
nea-assoc /path/to/data \
  --output-dir /path/to/results \
  --output-file my_results.csv

# Save as Parquet instead of CSV
nea-assoc /path/to/data --format parquet
```

### Parallel Processing

```bash
# Use specific number of threads
nea-assoc /path/to/data --threads 8

# Let it auto-detect (default)
nea-assoc /path/to/data
```

## Command-Line Options

### Data Loading Options

| Option | Description |
|--------|-------------|
| `--no-events` | Skip loading event data |
| `--no-photons` | Skip loading photon data |
| `--no-pixels` | Skip loading pixel data |
| `--limit N` | Load only first N rows per file (for testing) |
| `--query EXPR` | Pandas query to filter data (e.g., `"PSD > 0.5"`) |

### Settings and Configuration

| Option | Description |
|--------|-------------|
| `--settings PRESET\|FILE`, `-s` | Settings preset or JSON file path |
| `--threads N`, `-j` | Number of parallel threads |

### Association Parameters

| Option | Description |
|--------|-------------|
| `--pixel-max-dist PIXELS` | Max spatial distance for pixel-photon (pixels) |
| `--pixel-max-time NS` | Max time difference for pixel-photon (nanoseconds) |
| `--photon-dspace PIXELS` | Max COM distance for photon-event (pixels) |
| `--max-time NS` | Max time window for associations (nanoseconds) |
| `--method METHOD` | Association method: simple, kdtree, window, auto |

### Output Options

| Option | Description |
|--------|-------------|
| `--output-dir DIR`, `-o` | Output directory path |
| `--output-file FILE`, `-f` | Output filename |
| `--format FORMAT` | Output format: csv or parquet |

### Display Options

| Option | Description |
|--------|-------------|
| `--verbose`, `-v` | Increase verbosity (repeat for more: -v, -vv) |
| `--quiet`, `-q` | Suppress all output except errors |
| `--version` | Show version and exit |
| `--help`, `-h` | Show help message and exit |

## Settings File Format

The `.parameterSettings.json` file should follow this structure:

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
    "dSpace_px": 0.001,
    "dTime_s": 5e-08,
    "durationMax_s": 5e-07,
    "dTime_ext": 5
  },
  "event2image": {
    "size_x": 512,
    "size_y": 512,
    "nPhotons_min": 1,
    "nPhotons_max": 1,
    "psd_min": 0
  }
}
```

## Output Files

After running, you'll find in the output directory:

1. **associated_data.csv** - The main results file with clean column names:
   - Pixel columns: `px\x`, `px\y`, `px\toa`, `px\tot`, `px\tof`, `px\dt`, `px\dr`
   - Photon columns: `ph\x`, `ph\y`, `ph\toa`, `ph\tof`, `ph\id`
   - Event columns: `ev\x`, `ev\y`, `ev\toa`, `ev\n`, `ev\psd`, `ev\id`

2. **README.md** - Comprehensive documentation about the data structure, column definitions, and units

## Workflow Examples

### Complete Analysis Pipeline

```bash
#!/bin/bash
# Full analysis with all data types

DATA_DIR="/path/to/run112_ZnS"
OUTPUT_DIR="/path/to/results"

nea-assoc "$DATA_DIR" \
  --settings in_focus \
  --output-dir "$OUTPUT_DIR" \
  --output-file run112_full_association.csv \
  --threads 8 \
  --verbose
```

### Quick Test Run

```bash
# Test with limited data before full run
nea-assoc /path/to/data \
  --limit 1000 \
  --verbose \
  --output-dir /tmp/test_run
```

### Photon-Event Only (Faster)

```bash
# Skip pixels for faster processing
nea-assoc /path/to/data \
  --no-pixels \
  --settings fast_neutrons \
  --output-file photon_event_only.csv
```

### Batch Processing

```bash
#!/bin/bash
# Process multiple runs

for run in run112_ZnS run113_CsI run114_LiF; do
  echo "Processing $run..."
  nea-assoc "/data/$run" \
    --output-dir "/results/$run" \
    --quiet
done
```

## Performance Tips

1. **Use `--no-pixels` if you don't need pixel-level data** - significantly faster
2. **Test with `--limit 1000` first** - verify parameters before full run
3. **Adjust `--threads` based on your CPU** - more threads = faster processing
4. **Use `--format parquet`** for large datasets - smaller file size, faster I/O

## Troubleshooting

### No data found
- Check folder structure matches expected layout
- Ensure files have correct extensions (.empirevent, .empirphot, .tpx3)
- Try with `--verbose` to see what the tool is detecting

### Association produces no results
- Check your association parameters - they might be too restrictive
- Try with a preset: `--settings in_focus`
- Increase `--pixel-max-dist` and `--photon-dspace` values

### Out of memory
- Use `--limit` to process less data
- Use `--no-pixels` if you don't need pixel data
- Process in smaller batches

### Slow performance
- Increase `--threads` value
- Use `--no-pixels` if not needed
- Try different `--method` (kdtree is often faster for large datasets)

## Getting Help

```bash
# Show all available options
nea-assoc --help

# Show version
nea-assoc --version
```

For more information, visit: https://github.com/nuclear/neutron_event_analyzer
