# HTML Representation & Clean Output Guide

## What's New

### 1. Beautiful HTML Display in Jupyter Notebooks ✅
When you run association in Jupyter, you now get a beautifully formatted HTML table showing results!

### 2. Cleaned Up Verbosity Levels ✅
Warnings and debug output now properly respect verbosity levels - no clutter at default verbosity!

---

## HTML Display in Jupyter Notebooks

### Automatic Display

Simply type the variable name in a Jupyter cell and you get a formatted HTML table:

```python
import neutron_event_analyzer as nea

assoc = nea.Analyse("data/")
assoc.associate(relax=1.5)

# Just type the variable name
assoc
```

**Output for Single Folder:**
```
╔══════════════════════════════════════════════════════════════╗
║ ✅ Association Results                                       ║
╠══════════════════════════════════════════════════════════════╣
║ Pixel → Photon:    3,456 / 4,459    [ 77.5% ]              ║
║ Photon → Event:    2,987 / 3,456    [ 86.4% ]              ║
╚══════════════════════════════════════════════════════════════╝
```
*(Actual display is nicely colored HTML with green/orange/red badges based on match rates)*

**Output for Grouped Folders:**
```
╔══════════════════════════════════════════════════════════════════════╗
║ 📊 Groupby Association Results                                       ║
╠═══════════════════════╦══════════╦════════════╦════════════════════╣
║ Group                 ║ Pixels   ║ Pix→Phot   ║ Phot→Evt          ║
╠═══════════════════════╬══════════╬════════════╬════════════════════╣
║ intensifier           ║ 4,459    ║  77.5%     ║  86.4%            ║
║ intensifier_gain_50   ║ 4,459    ║  82.3%     ║  91.2%            ║
║ intensifier_gain_500  ║ 4,459    ║  88.1%     ║  95.7%            ║
║ intensifier_gain_5000 ║ 4,459    ║  91.4%     ║  97.8%            ║
╚═══════════════════════╩══════════╩════════════╩════════════════════╝
💡 Tip: Use .plot_stats() to visualize, .plot_stats(group='name') for specific group
```
*(Nicely colored with conditional formatting - green for >70%, orange for 50-70%, red for <50%)*

### Programmatic Access

Get statistics as a dictionary:

```python
import neutron_event_analyzer as nea

assoc = nea.Analyse("data/")
assoc.associate(relax=1.5)

# Get stats as dict
stats = assoc.get_association_stats()
print(stats)
# Output: {'pixels_total': 4459, 'pixels_matched': 3456, 'pixel_photon_rate': 0.775, ...}
```

For grouped data:

```python
assoc = nea.Analyse("data/grouped")
assoc.associate(relax=1.5)

# Get stats for all groups
all_stats = assoc.get_association_stats()
for group_name, stats in all_stats.items():
    print(f"{group_name}: {stats['pixel_photon_rate']*100:.1f}% pixel→photon")
```

---

## Verbosity Levels

The output is now properly controlled by verbosity level:

### Verbosity 0 - Silent

**No output except errors:**

```python
assoc = nea.Analyse("data/", verbosity=0)
result = assoc.associate(relax=1.5, verbosity=0)
# Completely silent - only errors shown
```

**Use case:** Automated scripts, batch processing

### Verbosity 1 - Normal (Default)

**Progress bars and key milestones:**

```python
assoc = nea.Analyse("data/")  # verbosity=1 by default
result = assoc.associate(relax=1.5)

# Output:
# Processing groups: 100%|██████████| 5/5 [02:30<00:00]
# 💾 Auto-saved results for 5/5 groups
```

**What you see:**
- ✅ Progress bars (tqdm)
- ✅ Auto-save confirmations
- ✅ Final summary
- ❌ No warnings
- ❌ No debug info
- ❌ No detailed parameters

**Use case:** Interactive Jupyter notebooks (default)

### Verbosity 2 - Detailed

**Full information including warnings and parameters:**

```python
assoc = nea.Analyse("data/", verbosity=2)
result = assoc.associate(relax=1.5, verbosity=2)

# Output:
# ======================================================================
# Starting Full Multi-Tier Association
# ======================================================================
# Using parameters from settings file
# Relaxation factor applied: 1.5x
#   Pixel-photon: max_dist=3.00px, max_time=150.0ns
#   Photon-event: dSpace=75.00px, max_time=750.0ns
#
# Step 1/2: Associating pixels to photons...
# ✅ Matched 3456 of 4459 pixels to photons (77.5%)
#
# Step 2/2: Associating photons to events...
# ✅ Matched 2987 of 3456 photons (86.4%)
#
# ======================================================================
# Full Association Complete
# ======================================================================
# Final combined dataframe has 4459 rows
# Columns: ['x', 'y', 't', 'tot', 'tof', 'assoc_photon_id', ...]
#
# Warning: Some photons could not be matched to events
# 💾 Auto-saved results to: data/AssociatedPixels/associated_data.csv
```

**What you see:**
- ✅ All progress bars
- ✅ Parameter details
- ✅ Step-by-step progress
- ✅ Warnings
- ✅ Debug information
- ✅ Column listings

**Use case:** Debugging, understanding what's happening, troubleshooting

---

## Comparison: Before vs After

### Before (Cluttered Output)

```python
assoc = nea.Analyse("data/")
result = assoc.associate(relax=1.5)

# Output (messy):
# Warning: empir_export_events binary not found...
# Warning: Settings file not found...
# Loading pixels: 100%|██████████| 1/1
# Loaded 4459 pixels in total.
# Using existing CSV: data/ExportedEvents/traced_data_0.csv
# Warning: Unexpected event CSV format...
# 2026-01-15 10:23:45,123 - Using existing CSV...
# 2026-01-15 10:23:45,456 - Converting...
# Warning: NaN values detected...
# ✅ Matched 3456 of 4459 pixels to photons (77.5%)
# ...lots more output...
```

### After (Clean Output)

```python
assoc = nea.Analyse("data/")  # verbosity=1 by default
result = assoc.associate(relax=1.5)

# Output (clean):
# Processing groups: 100%|██████████| 5/5 [02:30<00:00]
# 💾 Auto-saved results for 5/5 groups

# Then in a new cell, just type:
assoc

# Beautiful HTML table appears! 📊
```

---

## Examples

### Example 1: Quick Analysis with HTML Output

```python
import neutron_event_analyzer as nea

# Load and associate (clean output)
assoc = nea.Analyse("archive/pencilbeam/detector_model")
results = assoc.associate(relax=1.5)

# Display beautiful HTML summary
assoc
```

Output shows a nice HTML table with all groups and their match rates!

### Example 2: Silent Batch Processing

```python
import neutron_event_analyzer as nea

groups = ["group1", "group2", "group3"]
all_results = {}

for group in groups:
    assoc = nea.Analyse(f"data/{group}", verbosity=0)
    result = assoc.associate(relax=1.5, verbosity=0)
    all_results[group] = assoc.get_association_stats()

# Completely silent processing
```

### Example 3: Debugging with Full Details

```python
import neutron_event_analyzer as nea

# Enable detailed output
assoc = nea.Analyse("data/", verbosity=2)
result = assoc.associate(relax=1.5, verbosity=2)

# Shows all warnings, parameters, debug info
```

### Example 4: Compare Groups Visually

```python
import neutron_event_analyzer as nea

assoc = nea.Analyse("archive/pencilbeam/detector_model")
results = assoc.associate(relax=1.5)

# Just display the object - HTML table appears
assoc
```

The HTML table makes it easy to visually compare match rates across groups!

---

## HTML Color Coding

The HTML display uses color-coded badges for quick visual assessment:

| Match Rate | Color  | Meaning |
|-----------|--------|---------|
| > 70%     | 🟢 Green | Good quality |
| 50-70%    | 🟠 Orange | Acceptable, consider relaxing parameters |
| < 50%     | 🔴 Red | Poor quality, definitely need to relax parameters |

**Quick visual assessment:**
- Lots of green = Good associations
- Mostly orange = Try `relax=1.5` or `relax=2.0`
- Any red = Definitely increase `relax` parameter

---

## Accessing HTML Programmatically

### Get HTML String

```python
assoc = nea.Analyse("data/")
assoc.associate(relax=1.5)

# Get HTML as string
html_output = assoc._repr_html_()

# Save to file
with open("results.html", "w") as f:
    f.write(f"<html><body>{html_output}</body></html>")
```

### Display in IPython/Jupyter

```python
from IPython.display import HTML, display

assoc = nea.Analyse("data/")
assoc.associate(relax=1.5)

# Display explicitly
display(assoc)  # Same as just typing 'assoc'

# Or get HTML and display
display(HTML(assoc._repr_html_()))
```

---

## Customizing Verbosity

### Set Default Verbosity

```python
# Set at initialization
assoc = nea.Analyse("data/", verbosity=2)  # Always verbose

# All subsequent operations use verbosity=2
assoc.associate(relax=1.5)  # Uses verbosity=2
assoc.plot_stats()           # Uses verbosity=2
```

### Override Per-Call

```python
# Default is verbosity=1
assoc = nea.Analyse("data/", verbosity=1)

# But can override for specific calls
assoc.associate(relax=1.5, verbosity=0)  # Silent this time
assoc.plot_stats(verbosity=2)            # Detailed this time
```

---

## Tips & Best Practices

### 1. Use Default Verbosity for Interactive Work

```python
# In Jupyter - default verbosity=1 is perfect
assoc = nea.Analyse("data/")
assoc.associate(relax=1.5)
assoc  # Beautiful HTML display
```

### 2. Use Verbosity=0 for Scripts

```python
#!/usr/bin/env python3
import neutron_event_analyzer as nea

# Silent for automated processing
assoc = nea.Analyse("data/", verbosity=0)
result = assoc.associate(relax=1.5, verbosity=0)

# Log only final stats
stats = assoc.get_association_stats()
print(f"Match rate: {stats['pixel_photon_rate']*100:.1f}%")
```

### 3. Use Verbosity=2 for Troubleshooting

```python
# When something goes wrong
assoc = nea.Analyse("data/", verbosity=2)
result = assoc.associate(relax=1.5, verbosity=2)

# See all warnings, parameters, debug info
```

### 4. HTML Display is Automatic in Notebooks

```python
# No need to call print() or display()
assoc = nea.Analyse("data/")
assoc.associate(relax=1.5)

# Just type the variable name
assoc  # ← Automatically renders as beautiful HTML
```

---

## Summary

**Key Features:**

1. ✅ **Beautiful HTML display** in Jupyter notebooks
   - Color-coded match rates (green/orange/red)
   - Formatted tables with all groups
   - Automatic display when you type variable name

2. ✅ **Clean verbosity levels**
   - `verbosity=0`: Silent (errors only)
   - `verbosity=1`: Normal (progress bars, no warnings) ← Default
   - `verbosity=2`: Detailed (everything including warnings/debug)

3. ✅ **Programmatic access**
   - `get_association_stats()` returns dict
   - `_repr_html_()` returns HTML string
   - Works for both single and grouped data

**Usage:**

```python
# Interactive Jupyter (default verbosity=1)
assoc = nea.Analyse("data/")
assoc.associate(relax=1.5)
assoc  # Beautiful HTML table appears!

# Silent scripts (verbosity=0)
assoc = nea.Analyse("data/", verbosity=0)
result = assoc.associate(relax=1.5, verbosity=0)

# Debugging (verbosity=2)
assoc = nea.Analyse("data/", verbosity=2)
result = assoc.associate(relax=1.5, verbosity=2)
```

🎨 Clean, beautiful, and informative!
