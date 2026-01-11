# Auto-Save and Enhanced CLI Guide

## What's New

### 1. Automatic Saving ✅
Association results are now **automatically saved** when you run `associate()` - no need to call `save_associated_data()` manually!

### 2. Enhanced CLI with Auto-Detection ✅
The `nea-assoc` command now:
- Auto-detects grouped folder structures
- Automatically processes all groups
- Automatically generates plots
- Supports the new `--relax` parameter

---

## Automatic Saving in Python API

### Single Folder - Auto-Saves

```python
import neutron_event_analyzer as nea

assoc = nea.Analyse("archive/pencilbeam/detector_model/intensifier_gain_50/")
result = assoc.associate(relax=1.5, verbosity=1)

# Output includes:
# ======================================================================
# Full Association Complete
# ======================================================================
# Final combined dataframe has 4459 rows
# 💾 Auto-saved results to: archive/.../intensifier_gain_50/AssociatedPixels/associated_data.csv
```

**What gets saved:**
- File: `{data_folder}/AssociatedPixels/associated_data.csv`
- Format: CSV with all association columns
- Automatic: Happens every time you run `associate()`

### Grouped Folders - Auto-Saves All Groups

```python
import neutron_event_analyzer as nea

assoc = nea.Analyse("archive/pencilbeam/detector_model")
results = assoc.associate(relax=1.5, verbosity=1)

# Output includes:
# ======================================================================
# Groupby Association Complete
# ======================================================================
# Processed 5/5 groups successfully
#
# 💾 Auto-saving results for all groups...
#    ✅ intensifier: .../intensifier/AssociatedPixels/associated_data.csv
#    ✅ intensifier_gain_50: .../intensifier_gain_50/AssociatedPixels/associated_data.csv
#    ...
# 💾 Saved results for 5/5 groups
```

**What gets saved:**
- Each group gets its own CSV file in its subfolder
- Format: `{data_folder}/{group_name}/AssociatedPixels/associated_data.csv`
- Automatic: All groups saved automatically

### Disabling Auto-Save (If Needed)

Auto-save happens by default, but you can manually control saving:

```python
# Auto-save still happens, but you can also call manually with custom settings
assoc = nea.Analyse("data/")
result = assoc.associate(relax=1.5)  # Auto-saves

# Manually save to custom location
assoc.save_associated_data(output_dir="/custom/path", format='parquet')
```

---

## Enhanced CLI Tool (`nea-assoc`)

### New Features

1. **Auto-detects grouped folders** - No manual looping needed
2. **Automatic plot generation** - Plots created automatically after association
3. **`--relax` parameter** - Easy parameter scaling
4. **Simplified workflow** - One command does it all

### Basic Usage

#### Single Folder

```bash
# Basic usage - auto-saves and auto-plots
nea-assoc /path/to/data

# With relaxed parameters
nea-assoc /path/to/data --relax 1.5

# Custom settings
nea-assoc /path/to/data --settings in_focus --relax 1.5
```

#### Grouped Folders - Automatic!

```bash
# Point to parent folder - automatically processes all groups
nea-assoc archive/pencilbeam/detector_model --relax 1.5

# Output:
# ======================================================================
# Neutron Event Analyzer - Association Tool
# ======================================================================
#
# 📁 Data folder: archive/pencilbeam/detector_model
# ⚙️  Auto-detected settings: .parameterSettings.json
#
# 🔧 Initializing analyzer...
# 📁 Detected groupby folder structure with 5 groups:
#    - intensifier
#    - intensifier_gain_50
#    - intensifier_gain_500
#    - intensifier_gain_5000
#    - intensifier_gain_50000
#
# ✨ Will process all groups automatically
#
# 🔗 Performing association...
# Processing groups: 100%|██████████| 5/5 [02:30<00:00]
#
# 💾 Auto-saving results for all groups...
# 💾 Saved results for 5/5 groups
#
# 📊 Generating plots...
# Plotting groups: 100%|██████████| 5/5 [00:45<00:00]
# ✅ Generated plots for 5 groups
```

### CLI Arguments

#### Core Arguments

```bash
nea-assoc <data_folder> [options]
```

**Positional:**
- `data_folder` - Path to data folder (single or grouped)

#### Association Parameters

```bash
--relax FACTOR          # Scale all parameters (e.g., 1.5 = 50% more relaxed)
--method {simple,kdtree,window,auto}  # Association method
--pixel-max-dist PIXELS # Max spatial distance for pixel-photon
--pixel-max-time NS     # Max time difference for pixel-photon
--photon-dspace PIXELS  # Max CoM distance for photon-event
--max-time NS           # Max time window
```

#### Settings

```bash
--settings PRESET|FILE  # Use preset or custom settings file
--binaries DIR          # EMPIR binaries directory (or use $EMPIR_PATH)
--threads N             # Number of threads
```

#### Data Loading

```bash
--no-events   # Don't load event data
--no-photons  # Don't load photon data
--no-pixels   # Don't load pixel data
--limit N     # Limit rows per file (testing)
--query EXPR  # Filter data (e.g., "PSD > 0.5")
```

#### Output Control

```bash
-v, --verbose {0,1,2}  # Verbosity level
-q, --quiet            # Suppress output
```

### Examples

#### Example 1: Basic Analysis with Relaxed Parameters

```bash
# Single folder
nea-assoc archive/pencilbeam/detector_model/intensifier_gain_50 --relax 1.5

# Grouped folders - all processed automatically
nea-assoc archive/pencilbeam/detector_model --relax 1.5
```

#### Example 2: Custom Settings and Parameters

```bash
nea-assoc /path/to/data \
  --settings in_focus \
  --relax 2.0 \
  --method kdtree \
  --threads 8 \
  --verbose 2
```

#### Example 3: Using Environment Variable

```bash
# Set EMPIR_PATH once
export EMPIR_PATH=/work/nuclear/lumacam/lumacam_measurementcontrol

# Then use nea-assoc without --binaries
nea-assoc archive/pencilbeam/detector_model --relax 1.5
```

#### Example 4: Processing Only Specific Data Types

```bash
# Only photon-event association (no pixels)
nea-assoc /path/to/data --no-pixels --relax 1.5
```

### Output Structure

#### Single Folder Output

```
data_folder/
├── AssociatedPixels/
│   └── associated_data.csv          # Auto-saved results
└── AssociatedResults/
    ├── association_rate.png         # Auto-generated plots
    ├── time_differences.png
    ├── spatial_differences.png
    └── ...
```

#### Grouped Folders Output

```
detector_model/
├── intensifier/
│   ├── AssociatedPixels/
│   │   └── associated_data.csv      # Auto-saved
│   └── AssociatedResults/
│       └── *.png                     # Auto-generated plots
├── intensifier_gain_50/
│   ├── AssociatedPixels/
│   │   └── associated_data.csv
│   └── AssociatedResults/
│       └── *.png
└── ...
```

---

## Comparison: Before vs After

### Before (Manual Process)

#### Python API

```python
import neutron_event_analyzer as nea

# Had to manually loop through groups
for group in ['intensifier', 'intensifier_gain_50', ...]:
    assoc = nea.Analyse(f"data/{group}", export_dir="/path/to/empir")
    assoc.associate()
    assoc.save_associated_data()  # Manual save
    assoc.plot_stats()             # Manual plotting
```

#### CLI

```bash
# Had to run command for each group
nea-assoc data/intensifier --binaries /path/to/empir
nea-assoc data/intensifier_gain_50 --binaries /path/to/empir
# ... repeat for each group
```

### After (Automatic)

#### Python API

```python
import os
os.environ['EMPIR_PATH'] = '/path/to/empir'

import neutron_event_analyzer as nea

# One call does everything
assoc = nea.Analyse("data")
results = assoc.associate(relax=1.5)  # Auto-saves all groups
plots = assoc.plot_stats()            # Auto-plots all groups
```

#### CLI

```bash
export EMPIR_PATH=/path/to/empir

# One command processes all groups
nea-assoc data --relax 1.5
# Auto-detects groups
# Auto-processes all
# Auto-saves all
# Auto-plots all
```

---

## Benefits

### 1. No More Forgetting to Save
Results are automatically saved after association. You can still access them via:
- `assoc.associated_df` (single folder)
- `assoc.groupby_results` (grouped folders)
- Saved CSV files in `AssociatedPixels/` folders

### 2. Simplified Workflow
```python
# Before: 4 separate calls
assoc = nea.Analyse("data/", export_dir="/path")
assoc.associate()
assoc.save_associated_data()
assoc.plot_stats()

# After: 2 calls (or 1 with CLI)
assoc = nea.Analyse("data/")
assoc.associate(relax=1.5)  # Auto-saves, then use plot_stats() if needed
```

### 3. CLI Matches Python API
The CLI now has the same unified, automatic behavior as the Python API:
- Auto-detects grouped folders
- Processes all groups automatically
- Saves and plots automatically

### 4. Easy Parameter Tuning
```bash
# Try different relax values quickly
nea-assoc data --relax 1.0  # Baseline
nea-assoc data --relax 1.5  # More relaxed
nea-assoc data --relax 2.0  # Very relaxed
```

---

## Advanced: Controlling Auto-Save Behavior

### Check What Was Saved

```python
assoc = nea.Analyse("data/")
results = assoc.associate(relax=1.5)

# Check saved files
import os
if assoc.is_groupby:
    for group in assoc.groupby_subdirs:
        path = os.path.join(assoc.data_folder, group, "AssociatedPixels", "associated_data.csv")
        print(f"{group}: {os.path.exists(path)}")
else:
    path = os.path.join(assoc.data_folder, "AssociatedPixels", "associated_data.csv")
    print(f"Saved: {os.path.exists(path)}")
```

### Re-save with Different Format

```python
assoc = nea.Analyse("data/")
results = assoc.associate(relax=1.5)  # Auto-saves as CSV

# Also save as Parquet
assoc.save_associated_data(format='parquet', output_dir="/custom/path")
```

---

## Troubleshooting

### Issue: Don't want auto-save

**Solution:** Auto-save is automatic and lightweight. If you really don't want the files:
```python
import os
# Delete after association if needed
os.remove("path/to/AssociatedPixels/associated_data.csv")
```

### Issue: Want to save to custom location

**Solution:** Use manual save in addition to auto-save:
```python
assoc.associate(relax=1.5)  # Auto-saves to default location
assoc.save_associated_data(output_dir="/custom/path")  # Additional save
```

### Issue: CLI taking too long with plots

**Solution:** Plots are generated after association. You can skip manual plotting by checking output directories:
```bash
# Plots are in AssociatedResults/ folders
ls data/*/AssociatedResults/
```

---

## Summary

**Key Changes:**

1. ✅ `associate()` now auto-saves results (both Python and CLI)
2. ✅ `nea-assoc` auto-detects grouped folders
3. ✅ `nea-assoc` auto-generates plots
4. ✅ New `--relax` parameter in CLI
5. ✅ Unified behavior between Python API and CLI

**Benefits:**

- Never forget to save results
- One command/call does everything
- Consistent workflow for single and grouped data
- Easy parameter tuning with `--relax`

**Usage:**

```bash
# CLI: Everything automatic
export EMPIR_PATH=/path/to/empir
nea-assoc data --relax 1.5

# Python: Everything automatic
import os
os.environ['EMPIR_PATH'] = '/path/to/empir'
import neutron_event_analyzer as nea

assoc = nea.Analyse("data")
results = assoc.associate(relax=1.5)  # Auto-saves!
plots = assoc.plot_stats()            # Auto-plots!
```

🎉 Fully automatic, unified, and simple!
