# Unified API Guide - Simple & Consistent

The `neutron_event_analyzer` now has a **unified API** that works seamlessly for both single folders and grouped folder structures. The same methods work for both cases!

## Core Principle

**Same methods, automatic detection, clean API** ✨

```python
# Single or grouped - same code works for both!
assoc = nea.Analyse("your/data/path")
results = assoc.associate(relax=1.5)
plots = assoc.plot_stats()
```

---

## Complete Workflow Example

### Single Folder Workflow

```python
import neutron_event_analyzer as nea
import os

# Set EMPIR path once
os.environ['EMPIR_PATH'] = '/work/nuclear/lumacam/lumacam_measurementcontrol'

# Load data
assoc = nea.Analyse("archive/pencilbeam/detector_model/intensifier_gain_50/")

# Associate with relaxed parameters
result = assoc.associate(relax=1.5, verbosity=1)
# result is a DataFrame

# Generate plots
plots = assoc.plot_stats(verbosity=1)
# plots is a list of file paths

# Save results
assoc.save_associated_data()
```

### Grouped Folder Workflow - Same Methods!

```python
import neutron_event_analyzer as nea
import os

# Set EMPIR path once
os.environ['EMPIR_PATH'] = '/work/nuclear/lumacam/lumacam_measurementcontrol'

# Load grouped data - automatically detected
assoc = nea.Analyse("archive/pencilbeam/detector_model")
# Output:
# 📁 Detected groupby folder structure with 5 groups:
#    - intensifier
#    - intensifier_gain_50
#    - intensifier_gain_500
#    - intensifier_gain_5000
#    - intensifier_gain_50000

# Associate - automatically processes all groups
results = assoc.associate(relax=1.5, verbosity=1)
# Processing groups: 100%|██████████| 5/5 [02:30<00:00, 30.2s/it]
# results is a dict: {group_name: DataFrame}

# Generate plots - automatically plots all groups
all_plots = assoc.plot_stats(verbosity=1)
# 📊 Generating plots for 5 groups...
# Plotting groups: 100%|██████████| 5/5 [00:45<00:00]
# all_plots is a dict: {group_name: [plot_files]}

# Or plot specific group
specific_plots = assoc.plot_stats(group='intensifier_gain_50')
# specific_plots is a list of file paths
```

---

## API Methods - Work for Both Single and Grouped

### 1. `Analyse(data_folder, ...)`

**Automatically detects** single vs grouped folder structures.

```python
# Single folder
assoc = nea.Analyse("data/single_folder/")

# Grouped folder - auto-detected
assoc = nea.Analyse("data/grouped_folder/")
# 📁 Detected groupby folder structure with N groups
```

**Return behavior:**
- Single: Loads data automatically
- Grouped: Detects groups, waits for `associate()` call

---

### 2. `associate(relax=1.0, method='simple', verbosity=1, ...)`

**Works for both!** Returns different types based on structure.

```python
# Single folder
result = assoc.associate(relax=1.5)
# Returns: DataFrame

# Grouped folder
results = assoc.associate(relax=1.5)
# Returns: dict {group_name: DataFrame}
```

**Parameters:**
- `relax` (float): Scale all association parameters (default: 1.0)
  - 1.0 = use settings as-is
  - 1.5 = 50% more permissive (recommended for poor matching)
  - 2.0 = 100% more permissive
- `method` (str): Association method ('simple', 'kdtree', 'window', 'lumacam')
- `verbosity` (int): 0=silent, 1=progress bars, 2=detailed

---

### 3. `plot_stats(output_dir=None, verbosity=1, group=None)`

**Generates plots for both!** Returns different types based on structure.

```python
# Single folder
plots = assoc.plot_stats()
# Returns: list of plot file paths

# Grouped folder - all groups
all_plots = assoc.plot_stats()
# 📊 Generating plots for 5 groups...
# Returns: dict {group_name: [plot_files]}

# Grouped folder - specific group
plots = assoc.plot_stats(group='intensifier_gain_50')
# Returns: list of plot file paths
```

**Parameters:**
- `output_dir` (str): Where to save plots (default: auto-determined)
- `verbosity` (int): Output level
- `group` (str): For grouped data, specify which group to plot (None = all groups)

---

### 4. `save_associated_data(output_dir=None, format='csv', verbosity=1)`

**Saves results for both!**

```python
# Single folder
path = assoc.save_associated_data()
# Saves single CSV file

# Grouped folder - saves each group
paths = assoc.save_associated_data()
# Saves CSV for each group
# Returns: dict {group_name: file_path}
```

---

## Quick Reference: Return Types

| Method | Single Folder | Grouped Folder |
|--------|--------------|----------------|
| `associate()` | `DataFrame` | `dict[str, DataFrame]` |
| `plot_stats()` | `list[str]` | `dict[str, list[str]]` |
| `plot_stats(group='name')` | N/A | `list[str]` |
| `save_associated_data()` | `str` (file path) | `dict[str, str]` |

---

## Accessing Results

### Single Folder Results

```python
assoc = nea.Analyse("data/single/")
result = assoc.associate(relax=1.5)

# result is a DataFrame - use directly
print(f"Total pixels: {len(result)}")
print(f"Match rate: {result['assoc_photon_id'].notna().sum() / len(result) * 100:.1f}%")

# Access via attribute too
df = assoc.associated_df
```

### Grouped Folder Results

```python
assoc = nea.Analyse("data/grouped/")
results = assoc.associate(relax=1.5)

# results is a dict - iterate or access by name
for group_name, df in results.items():
    print(f"{group_name}: {len(df)} pixels")

# Access specific group
df_50 = results['intensifier_gain_50']

# Or via attribute
all_results = assoc.groupby_results  # Same as results dict
```

---

## Practical Examples

### Example 1: Compare Association Quality Across Groups

```python
import neutron_event_analyzer as nea
import pandas as pd

assoc = nea.Analyse("archive/pencilbeam/detector_model")
results = assoc.associate(relax=1.5, verbosity=1)

# Create comparison DataFrame
comparison = []
for group_name, df in results.items():
    pix_phot = df['assoc_photon_id'].notna().sum() / len(df) * 100
    phot_evt = df['assoc_event_id'].notna().sum() / len(df) * 100
    comparison.append({
        'Group': group_name,
        'Pixels': len(df),
        'Pix→Phot %': pix_phot,
        'Phot→Evt %': phot_evt
    })

comparison_df = pd.DataFrame(comparison)
print(comparison_df.to_string(index=False))
```

### Example 2: Generate All Plots and Save All Data

```python
import neutron_event_analyzer as nea

assoc = nea.Analyse("archive/pencilbeam/detector_model")

# Associate all groups
results = assoc.associate(relax=1.5, verbosity=1)

# Generate plots for all groups
plots = assoc.plot_stats(verbosity=1)
print(f"Generated plots for {len(plots)} groups")

# Save CSV data for all groups
saved_paths = assoc.save_associated_data(verbosity=1)
print(f"Saved data to {len(saved_paths)} files")
```

### Example 3: Process Only One Group from Grouped Structure

```python
import neutron_event_analyzer as nea

# Option 1: Process all, analyze one
assoc = nea.Analyse("archive/pencilbeam/detector_model")
results = assoc.associate(relax=1.5)
plots = assoc.plot_stats(group='intensifier_gain_50')  # Only this group

# Option 2: Load specific group directly
assoc_50 = nea.Analyse("archive/pencilbeam/detector_model/intensifier_gain_50")
result = assoc_50.associate(relax=1.5)  # Returns DataFrame
plots = assoc_50.plot_stats()  # Works like single folder
```

### Example 4: Try Different Relax Values

```python
import neutron_event_analyzer as nea

assoc = nea.Analyse("archive/pencilbeam/detector_model/intensifier_gain_50")

for relax_val in [1.0, 1.5, 2.0, 3.0]:
    result = assoc.associate(relax=relax_val, verbosity=0)
    match_rate = result['assoc_photon_id'].notna().sum() / len(result) * 100
    print(f"relax={relax_val}: {match_rate:.1f}% pixel→photon match rate")
```

---

## Advanced: Checking Structure Type

```python
import neutron_event_analyzer as nea

assoc = nea.Analyse("your/data/path")

if assoc.is_groupby:
    print(f"Grouped structure with {len(assoc.groupby_subdirs)} groups:")
    print(assoc.groupby_subdirs)
else:
    print("Single folder structure")
```

---

## Tips & Best Practices

1. **Start with default `relax=1.0`**, increase if matching is poor:
   ```python
   result = assoc.associate()  # Try default first
   match_rate = result['assoc_photon_id'].notna().sum() / len(result) * 100
   if match_rate < 60:
       result = assoc.associate(relax=1.5)  # Try more relaxed
   ```

2. **Use `verbosity=1` for progress bars**, `2` for detailed info:
   ```python
   assoc.associate(relax=1.5, verbosity=1)  # See progress
   ```

3. **For grouped data, check all results** before plotting:
   ```python
   results = assoc.associate(relax=1.5)
   # Inspect results first
   for name, df in results.items():
       print(f"{name}: {len(df)} rows")
   # Then plot
   plots = assoc.plot_stats()
   ```

4. **Specific group analysis** - two ways:
   ```python
   # Method 1: Load group directly
   assoc = nea.Analyse("data/grouped/group1")

   # Method 2: Process all, plot specific
   assoc = nea.Analyse("data/grouped")
   assoc.associate(relax=1.5)
   assoc.plot_stats(group='group1')
   ```

---

## Migration from Old Code

### Before (Manual Processing)

```python
# Had to manually loop through groups
results = {}
for folder in ['group1', 'group2', 'group3']:
    assoc = nea.Analyse(f"data/{folder}", export_dir="/path/to/empir")
    assoc.associate()
    results[folder] = assoc.associated_df
    assoc.plot_stats()
```

### After (Automatic)

```python
# Set EMPIR_PATH once
os.environ['EMPIR_PATH'] = "/path/to/empir"

# One call does it all
assoc = nea.Analyse("data")
results = assoc.associate(relax=1.5)
plots = assoc.plot_stats()
```

**Benefits:**
- ✅ Simpler code
- ✅ Automatic detection
- ✅ Progress bars
- ✅ Sequential processing (no pickling issues)
- ✅ Consistent API

---

## Summary

The unified API means you write the **same code** whether you have:
- A single experiment folder
- Multiple grouped experiment folders

The library automatically:
- Detects the structure
- Processes appropriately
- Returns the right type
- Shows progress

**Key methods that work for both:**
- `associate()` - Run association
- `plot_stats()` - Generate plots
- `save_associated_data()` - Save results

Just call them the same way, and the library handles the rest! 🎉
