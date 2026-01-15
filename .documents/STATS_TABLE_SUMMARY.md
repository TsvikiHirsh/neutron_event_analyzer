# Association Statistics Table - Summary

## What Changed

### 1. HTML Table Format ✅
- **Metrics are columns**: Pix Match, Phot Match, Evt Match, CoM Exact/Good for both px2ph and ph2ev
- **Groups are rows**: Each group (or single dataset) is a row
- **Color coding**:
  - Red (indianred) for <50%
  - Orange for 50-70%
  - Light green for >70%

### 2. Return Value Changed ✅
- `associate()` now returns `IPython.display.HTML` with the stats table
- `associate_groupby()` also returns HTML stats table
- Raw dataframes still accessible via `assoc.associated_df` and `assoc.groupby_results`

### 3. Fixed `get_association_stats()` ✅
- Now returns actual stats from `last_assoc_stats` and `last_photon_event_stats`
- For grouped data, returns `groupby_stats` dictionary
- No longer returns empty dict

### 4. Comparison Groups ✅
- Comparison groups have orange background row
- Small "cmp" badge shown next to comparison group names

## Table Columns

| Column | Description |
|--------|-------------|
| Group | Group/dataset name (with "cmp" badge for comparisons) |
| Pix Match | Percentage of pixels matched to photons |
| Phot Match | Percentage of photons matched (px2ph stage) |
| Evt Match | Percentage of events matched (ph2ev stage) |
| CoM Exact (px2ph) | % with exact CoM match (≤0.1px) in pixel-photon |
| CoM Good (px2ph) | % with good CoM match (≤30% radius) in pixel-photon |
| CoM Exact (ph2ev) | % with exact CoM match (≤0.1px) in photon-event |
| CoM Good (ph2ev) | % with good CoM match (≤30% radius) in photon-event |

## Example Usage

```python
import neutron_event_analyzer as nea

# Single dataset
assoc = nea.Analyse("data/")
table = assoc.associate(relax=1.5)
# Displays nice HTML table with one row

# Access raw data
df = assoc.associated_df
stats = assoc.get_association_stats()  # Now returns actual dict!

# Grouped data
assoc = nea.Analyse("data/grouped/")
table = assoc.associate(relax=1.5)
# Displays HTML table with multiple rows (one per group)

# Access raw data
all_dfs = assoc.groupby_results
stats = assoc.get_association_stats()  # Returns dict with all group stats
```

## Example Table Output

```
┌──────────────┬───────────┬────────────┬───────────┬─────────────────┬────────────────┬─────────────────┬────────────────┐
│ Group        │ Pix Match │ Phot Match │ Evt Match │ CoM Exact (px2ph)│ CoM Good (px2ph)│ CoM Exact (ph2ev)│ CoM Good (ph2ev)│
├──────────────┼───────────┼────────────┼───────────┼─────────────────┼────────────────┼─────────────────┼────────────────┤
│ Dataset      │   14.8%   │   99.4%    │  100.0%   │      20.3%      │     78.3%      │     100.0%      │      0.0%      │
└──────────────┴───────────┴────────────┴───────────┴─────────────────┴────────────────┴─────────────────┴────────────────┘
```

With color coding:
- 14.8% (Pix Match) - RED background (low)
- 99.4% (Phot Match) - GREEN background (good)
- 100.0% (Evt Match) - GREEN background (good)
- 20.3% (CoM Exact px2ph) - RED background (low)
- 78.3% (CoM Good px2ph) - GREEN background (good)
- 100.0% (CoM Exact ph2ev) - GREEN background (good)

## Files Cleaned Up

Removed pollution:
- ✅ apply_html_improvements.py
- ✅ updated_repr_html_single_method.py
- ✅ HTML_IMPROVEMENTS_SUMMARY.md
- ✅ analyser.py.backup
- ✅ analyser.py.backup2

## Implementation Details

### Key Methods

1. **`_create_stats_html_table()`**: Main method that generates the HTML table
2. **`_extract_row_metrics()`**: Extracts metrics from stats dict for a row
3. **`get_association_stats()`**: Fixed to return actual stats
4. **`associate_groupby()`**: Now stores per-group stats in `self.groupby_stats`

### Stats Storage

- **Single dataset**: Uses `self.last_assoc_stats` and `self.last_photon_event_stats`
- **Grouped data**: Uses `self.groupby_stats` (dict mapping group names to combined stats)

All working cleanly now!
