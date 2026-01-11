# New Features in neutron_event_analyzer

## Summary of Changes

This document describes three new features added to the `neutron_event_analyzer` package:

1. **EMPIR_PATH environment variable support** - Automatically use `EMPIR_PATH` for binary locations
2. **`relax` parameter** - Easily scale association parameters for better matching
3. **Groupby folder detection and parallel processing** - Analyze multiple experimental conditions in parallel

---

## 1. EMPIR_PATH Environment Variable Support

### Problem Solved
Previously, the `export_dir` parameter was required to specify the location of EMPIR binaries. Now, the `Analyse` class automatically checks the `EMPIR_PATH` environment variable if `export_dir` is not explicitly provided.

### Usage

**In Jupyter notebooks:**
```python
%env EMPIR_PATH /work/nuclear/lumacam/lumacam_measurementcontrol

import neutron_event_analyzer as nea

# No need to specify export_dir anymore!
assoc = nea.Analyse("archive/pencilbeam/detector_model/intensifier_gain_50/")
```

**In shell/scripts:**
```bash
export EMPIR_PATH=/path/to/empir/binaries
python your_script.py
```

### Implementation Details
- Checks for `EMPIR_PATH` when `export_dir` is not specified (defaults to `"./export"`)
- If `EMPIR_PATH` is set, uses it automatically
- Explicit `export_dir` parameter still takes precedence over environment variable
- Modified in [analyser.py:104-108](src/neutron_event_analyzer/analyser.py#L104-L108)

---

## 2. Relax Parameter for Association

### Problem Solved
Association parameters from `.parameterSettings.json` files can be too restrictive, leading to poor matching rates (e.g., 43.2% pixel-to-photon matching). The `relax` parameter allows you to easily scale all association parameters without manually specifying each one.

### Usage

```python
import neutron_event_analyzer as nea

# Load data
assoc = nea.Analyse("archive/pencilbeam/detector_model/intensifier_gain_50/")

# Run association with relaxed parameters (50% more permissive)
assoc.associate(relax=1.5, verbosity=2)

# Or more restrictive (20% tighter)
assoc.associate(relax=0.8, verbosity=2)

# Default is relax=1.0 (no scaling)
assoc.associate(verbosity=2)
```

### How It Works

The `relax` parameter is a **multiplicative scaling factor** applied to all association parameters:

- **Pixel-to-photon association:**
  - `pixel_max_dist_px` (spatial distance in pixels)
  - `pixel_max_time_ns` (time difference in nanoseconds)

- **Photon-to-event association:**
  - `photon_dSpace_px` (center-of-mass distance in pixels)
  - `max_time_ns` (time window in nanoseconds)

### Examples

If your settings file has:
```json
{
  "pixel2photon": {
    "dSpace": 2.0,
    "dTime": 100e-9
  },
  "photon2event": {
    "dSpace_px": 50.0,
    "dTime_s": 500e-9
  }
}
```

With `relax=1.5`:
- `pixel_max_dist_px`: 2.0 → 3.0 pixels
- `pixel_max_time_ns`: 100 → 150 ns
- `photon_dSpace_px`: 50.0 → 75.0 pixels
- `max_time_ns`: 500 → 750 ns

### When to Use

- **`relax > 1.0`**: Use when getting poor matching rates (e.g., < 60% for pixels-to-photons)
  - Example: `relax=1.5` for 50% more permissive matching
  - Common for out-of-focus data or high-gain intensifiers with larger blobs

- **`relax < 1.0`**: Use when getting too many false matches
  - Example: `relax=0.8` for 20% more restrictive matching
  - Common for high-precision, in-focus measurements

- **`relax = 1.0`**: Default, uses parameters from settings file as-is

### Output

With `verbosity=2`, you'll see the applied parameters:
```
======================================================================
Starting Full Multi-Tier Association
======================================================================
Using parameters from settings file
Relaxation factor applied: 1.5x
  Pixel-photon: max_dist=3.00px, max_time=150.0ns
  Photon-event: dSpace=75.00px, max_time=750.0ns
```

### Implementation Details
- Added as optional parameter to `associate()` method
- Applied in [analyser.py:790-794](src/neutron_event_analyzer/analyser.py#L790-L794)
- Works with all association methods (simple, kdtree, window, lumacam)

---

## 3. Groupby Folder Detection and Parallel Processing

### Problem Solved
When analyzing multiple experimental conditions (e.g., different intensifier gains), you previously had to manually loop through folders and create separate `Analyse` instances. The new groupby feature automatically detects folder structures with multiple conditions and can process them all in parallel.

### Folder Structure

The feature automatically detects "groupby" folder structures:

```
archive/pencilbeam/detector_model/
├── intensifier/
│   ├── ExportedEvents/
│   ├── ExportedPhotons/
│   ├── ExportedPixels/
│   └── ...
├── intensifier_gain_50/
│   ├── ExportedEvents/
│   ├── ExportedPhotons/
│   ├── ExportedPixels/
│   └── ...
├── intensifier_gain_500/
│   └── ...
├── intensifier_gain_5000/
│   └── ...
└── intensifier_gain_50000/
    └── ...
```

### Usage

**Automatic Detection:**
```python
import neutron_event_analyzer as nea

# Point to the parent folder containing multiple groups
assoc = nea.Analyse("archive/pencilbeam/detector_model")

# Output:
# 📁 Detected groupby folder structure with 5 groups:
#    - intensifier
#    - intensifier_gain_50
#    - intensifier_gain_500
#    - intensifier_gain_5000
#    - intensifier_gain_50000
#
# ℹ️  Use .associate_groupby() to run association on all groups in parallel
#     or access individual groups using: Analyse(f'{data_folder}/group_name')
```

**Run Association on All Groups:**
```python
# Process all groups in parallel with relaxed parameters
results = assoc.associate_groupby(relax=1.5, method='simple', verbosity=1)

# Output shows progress bar:
# ======================================================================
# Running Association on 5 Groups
# ======================================================================
# Processing groups: 100%|██████████| 5/5 [00:45<00:00,  9.2s/it]
#
# ======================================================================
# Groupby Association Complete
# ======================================================================
# Processed 5/5 groups successfully
```

**Access Results:**
```python
# Results is a dictionary mapping group names to associated dataframes
for group_name, df in results.items():
    print(f"{group_name}: {len(df)} pixels associated")
    print(f"  Match rate: {df['assoc_photon_id'].notna().sum() / len(df) * 100:.1f}%")

# Access specific group
intensifier_50 = results['intensifier_gain_50']
```

**Individual Group Analysis:**
```python
# You can still analyze individual groups if needed
assoc_50 = nea.Analyse("archive/pencilbeam/detector_model/intensifier_gain_50")
assoc_50.associate(relax=1.5)
```

### Detection Criteria

A folder is considered a groupby structure if:
1. It contains a `.groupby_metadata.json` file (created by lumacam's `lens.groupby()` method), OR
2. It contains ≥2 subdirectories that have data folders (`ExportedEvents`, `ExportedPhotons`, `ExportedPixels`, etc.)

### Parallel Processing

- Uses `ProcessPoolExecutor` to process groups in parallel
- Number of parallel workers controlled by `n_threads` parameter in `__init__`
- Each group runs with `n_threads=1` (parallelism is across groups, not within groups)
- Progress shown with `tqdm` progress bar

### API Reference

#### `Analyse._is_groupby_folder(folder_path)`
Static method to check if a folder is a groupby structure.

**Returns:**
- `tuple`: `(is_groupby, subdirs)` where `is_groupby` is bool and `subdirs` is list of group names

#### `Analyse.associate_groupby(**kwargs)`
Run association on all groups in parallel.

**Arguments:**
- `**kwargs`: Arguments to pass to `associate()` for each group
  - Common: `relax`, `method`, `verbosity`, `pixel_max_dist_px`, etc.

**Returns:**
- `dict`: Dictionary mapping group names to their associated dataframes

**Raises:**
- `ValueError`: If called on a non-groupby folder structure

**Example:**
```python
results = assoc.associate_groupby(
    relax=1.5,
    method='simple',
    verbosity=1,
    pixel_max_dist_px=10.0  # Override specific parameters
)
```

### Attributes

When a groupby folder is detected:
- `assoc.is_groupby` → `True`
- `assoc.groupby_subdirs` → List of group names (e.g., `['intensifier', 'intensifier_gain_50', ...]`)
- `assoc.groupby_results` → Dictionary of results after calling `associate_groupby()`
- Data loading is **skipped** (no `load()` called automatically)

### Implementation Details
- Detection: [analyser.py:27-67](src/neutron_event_analyzer/analyser.py#L27-L67)
- Initialization handling: [analyser.py:124-138](src/neutron_event_analyzer/analyser.py#L124-L138)
- Parallel processing: [analyser.py:990-1069](src/neutron_event_analyzer/analyser.py#L990-L1069)

---

## Complete Example Workflow

```python
import neutron_event_analyzer as nea
import os

# Set environment variable for EMPIR binaries
os.environ['EMPIR_PATH'] = '/work/nuclear/lumacam/lumacam_measurementcontrol'

# Analyze a single dataset with relaxed parameters
assoc = nea.Analyse("archive/pencilbeam/detector_model/intensifier_gain_50/")
assoc.associate(relax=1.5, verbosity=2)

# Or analyze multiple datasets in parallel
assoc_group = nea.Analyse("archive/pencilbeam/detector_model")
results = assoc_group.associate_groupby(relax=1.5, method='simple', verbosity=1)

# Compare results across groups
for group, df in results.items():
    pixel_match_rate = df['assoc_photon_id'].notna().sum() / len(df) * 100
    photon_match_rate = df['assoc_event_id'].notna().sum() / len(df) * 100
    print(f"{group}:")
    print(f"  Pixel→Photon: {pixel_match_rate:.1f}%")
    print(f"  Photon→Event: {photon_match_rate:.1f}%")

# Generate statistics for each group
for group, df in results.items():
    assoc_single = nea.Analyse(f"archive/pencilbeam/detector_model/{group}")
    assoc_single.associated_df = df
    stats = assoc_single.plot_stats()
    print(f"\n{group} statistics:\n{stats}")
```

---

## Backward Compatibility

All changes are **backward compatible**:

- Existing code without `relax` parameter continues to work (default `relax=1.0`)
- Existing code without `EMPIR_PATH` continues to work (explicit `export_dir` still supported)
- Non-groupby folders work exactly as before
- All existing methods and APIs unchanged

---

## Testing

Basic syntax validation completed:
```bash
python3 -m py_compile src/neutron_event_analyzer/analyser.py
# ✅ No syntax errors
```

Manual testing recommended:
1. Test EMPIR_PATH detection with your environment
2. Test relax parameter with different values (0.5, 1.0, 1.5, 2.0)
3. Test groupby detection with your folder structure
4. Compare association quality with and without relaxation

---

## Questions or Issues?

If you encounter any problems with these new features, please check:

1. **EMPIR_PATH not working:** Verify the environment variable is set (`echo $EMPIR_PATH` or `os.environ.get('EMPIR_PATH')`)
2. **Poor matching with relax:** Try increasing the `relax` value (start with 1.5, try up to 2.0 if needed)
3. **Groupby not detected:** Ensure subdirectories contain data folders (ExportedEvents, ExportedPhotons, etc.)
4. **Parallel processing slow:** Reduce `n_threads` parameter or process groups individually

For more details, see the source code comments in [analyser.py](src/neutron_event_analyzer/analyser.py).
