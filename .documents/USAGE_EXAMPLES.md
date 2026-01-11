# Usage Examples for New Features

## Quick Start

The `associate()` method now **automatically works for both single folders and grouped structures**!

### Example 1: Single Folder (Standard Workflow)

```python
import neutron_event_analyzer as nea
import os

# Set EMPIR_PATH once (no need to specify export_dir anymore!)
os.environ['EMPIR_PATH'] = '/work/nuclear/lumacam/lumacam_measurementcontrol'

# Load and associate with relaxed parameters
assoc = nea.Analyse("archive/pencilbeam/detector_model/intensifier_gain_50/")
result = assoc.associate(relax=1.5, verbosity=2)

# result is a DataFrame
print(f"Associated {len(result)} pixels")
print(f"Pixel→Photon match rate: {result['assoc_photon_id'].notna().sum() / len(result) * 100:.1f}%")
```

### Example 2: Grouped Folders (Automatic Parallel Processing)

```python
import neutron_event_analyzer as nea
import os

# Set EMPIR_PATH
os.environ['EMPIR_PATH'] = '/work/nuclear/lumacam/lumacam_measurementcontrol'

# Point to parent folder - automatically detects groups
assoc = nea.Analyse("archive/pencilbeam/detector_model")

# Same .associate() method automatically processes all groups in parallel!
results = assoc.associate(relax=1.5, method='simple', verbosity=1)

# results is a dict mapping group names to DataFrames
for group_name, df in results.items():
    pix_phot = df['assoc_photon_id'].notna().sum() / len(df) * 100
    phot_evt = df['assoc_event_id'].notna().sum() / len(df) * 100
    print(f"{group_name}:")
    print(f"  Pixel→Photon: {pix_phot:.1f}%")
    print(f"  Photon→Event: {phot_evt:.1f}%")
```

### Example 3: Comparing Different Relax Values

```python
import neutron_event_analyzer as nea

assoc = nea.Analyse("archive/pencilbeam/detector_model/intensifier_gain_50/")

# Try different relax values
for relax_val in [1.0, 1.5, 2.0]:
    result = assoc.associate(relax=relax_val, verbosity=0)
    match_rate = result['assoc_photon_id'].notna().sum() / len(result) * 100
    print(f"relax={relax_val}: {match_rate:.1f}% pixel→photon match rate")
```

### Example 4: Using in Jupyter Notebooks

```python
# In a Jupyter cell:
%env EMPIR_PATH /work/nuclear/lumacam/lumacam_measurementcontrol

import neutron_event_analyzer as nea

# Works the same whether single or grouped!
assoc = nea.Analyse("archive/pencilbeam/detector_model")
results = assoc.associate(relax=1.5, verbosity=1)

# If grouped, results is a dict
if isinstance(results, dict):
    print(f"Processed {len(results)} groups")
    for name, df in results.items():
        print(f"  {name}: {len(df)} rows")
else:
    # Single folder, results is a DataFrame
    print(f"Processed single folder: {len(results)} rows")
```

### Example 5: Analyzing Specific Groups from a Grouped Structure

```python
import neutron_event_analyzer as nea

# Process all groups
assoc_all = nea.Analyse("archive/pencilbeam/detector_model")
all_results = assoc_all.associate(relax=1.5, verbosity=1)

# Or process just one specific group
assoc_50 = nea.Analyse("archive/pencilbeam/detector_model/intensifier_gain_50")
result_50 = assoc_50.associate(relax=1.5, verbosity=2)

# Both work with the same associate() method!
```

## Key Points

1. **Same method for both**: Use `.associate()` whether you have a single folder or grouped folders
2. **Automatic detection**: No need to specify if it's grouped - it's detected automatically
3. **Return type differs**:
   - Single folder → Returns `DataFrame`
   - Grouped folder → Returns `dict` of `{group_name: DataFrame}`
4. **EMPIR_PATH**: Set once as environment variable, no need to pass `export_dir`
5. **relax parameter**: Scale all association parameters with one simple parameter

## Troubleshooting

### Poor matching rates?
Try increasing `relax`:
```python
result = assoc.associate(relax=2.0, verbosity=2)  # 100% more relaxed
```

### Want to see what parameters are being used?
Set `verbosity=2`:
```python
result = assoc.associate(relax=1.5, verbosity=2)
# Shows:
# Relaxation factor applied: 1.5x
#   Pixel-photon: max_dist=3.00px, max_time=150.0ns
#   Photon-event: dSpace=75.00px, max_time=750.0ns
```

### EMPIR_PATH not working?
Check if it's set:
```python
import os
print(os.environ.get('EMPIR_PATH'))  # Should show your path
```

### Groupby not detected?
Check folder structure:
```python
from neutron_event_analyzer import Analyse
is_groupby, subdirs = Analyse._is_groupby_folder("your/path")
print(f"Is groupby: {is_groupby}")
print(f"Subdirs: {subdirs}")
```
