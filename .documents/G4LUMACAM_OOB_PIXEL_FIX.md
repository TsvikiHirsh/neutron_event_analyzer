# G4LumaCam Agent Instructions — Fix Out-of-Bounds Pixel Encoding

## Problem

`TracedPhotons` CSV files contain pixel hits with `pixel_x` or `pixel_y` outside
the valid Timepix3 sensor range **[0, 255]** (e.g. `x=173`, `y=-30`, `y=-2`).
These arise when a simulated photon hits outside the detector's active area.

The method `_write_tpx3` (around line 1704-1722 in `optics.py`) already silently
drops these out-of-bounds (OOB) rows from the `.tpx3` binary file via a `valid_mask`
filter. However, `TracedPhotons` is saved to CSV **before** that filter runs
(line 1282-1283), so the `in_tpx3` flag in `TracedPhotons` is **incorrectly `True`**
for OOB pixels — they appear to have survived but are absent from the TPX3 file.

This causes a row-count mismatch between `TracedPhotons` (too many rows) and
`ExportedPixels` (only in-bounds pixels exported by EMPIR), breaking the
`pixel_id` join used in `build_combined`.

---

## Root Cause

In the hits workflow (around line 1210-1298 in `optics.py`), the execution order is:

1. `result_df['in_tpx3'] = False` — initialise flag
2. `saturate_photons(...)` — marks survivors with `in_tpx3 = True`
3. **`result_df.to_csv(output_file)` — TracedPhotons saved here (still has OOB rows with in_tpx3=True)**
4. `tpx3_data = result_df[result_df['in_tpx3']]` — subset for TPX3
5. `_write_tpx3(tpx3_data)` — OOB filter applied silently here, rows dropped

The fix must mark OOB pixels `in_tpx3 = False` **before step 3**.

---

## Fix — `optics.py` hits workflow

After the saturation block updates `in_tpx3` and before saving `TracedPhotons`
to CSV, insert an OOB filter. Find the section that looks like this:

```python
# Sort by time to restore chronological order
result_df = result_df.sort_values('toa2').reset_index(drop=True)

# Remove temporary index column
if '_original_index' in result_df.columns:
    result_df = result_df.drop(columns=['_original_index'])
```

Immediately **after** the `drop(columns=['_original_index'])` block and
**before** the `desired_columns` filtering / CSV save, add:

```python
# Mark out-of-bounds pixels as not written to TPX3.
# _write_tpx3 silently drops pixels outside [0, 255]; TracedPhotons must agree
# so that its row count matches ExportedPixels exactly.
SENSOR_SIZE = 256
if 'pixel_x' in result_df.columns and 'pixel_y' in result_df.columns:
    oob_mask = (
        result_df['pixel_x'].notna() & result_df['pixel_y'].notna() & (
            (result_df['pixel_x'] < 0) | (result_df['pixel_x'] >= SENSOR_SIZE) |
            (result_df['pixel_y'] < 0) | (result_df['pixel_y'] >= SENSOR_SIZE)
        )
    )
    if oob_mask.any():
        result_df.loc[oob_mask, 'in_tpx3'] = False
        if verbosity >= VerbosityLevel.DETAILED:
            print(f"  Marked {int(oob_mask.sum())} out-of-bounds pixels as in_tpx3=False")
```

---

## Why not clip or wrap OOB coordinates?

- **Clipping** (e.g. `x=-2 → x=0`) would place photons at the wrong detector location,
  corrupting EMPIR's centroid calculation.
- **Wrapping** (e.g. `y=-2 → y=254` via uint8 overflow) is the current accidental
  behaviour — EMPIR sees a pixel at row 254 which has nothing to do with the
  simulated photon. This is incorrect.
- **Dropping** is the physically correct choice: a photon that hit outside the
  detector active area was never detected. EMPIR will not reconstruct it.

---

## Verification

After the fix, for every `TracedPhotons` part file:

```python
traced = pd.read_csv("TracedPhotons/traced_sim_data_N.csv")
# All in_tpx3=True rows must have valid coordinates
in_tpx3 = traced[traced['in_tpx3']]
assert (in_tpx3['pixel_x'] >= 0).all()
assert (in_tpx3['pixel_x'] < 256).all()
assert (in_tpx3['pixel_y'] >= 0).all()
assert (in_tpx3['pixel_y'] < 256).all()
print(f"in_tpx3=True: {in_tpx3.shape[0]} rows, all in bounds ✓")
```

The count of `in_tpx3=True` rows across all part files should now equal the
number of rows in the corresponding `ExportedPixels` part files (same pulse range).

---

## Impact on downstream analysis

| Step | Before fix | After fix |
|---|---|---|
| `pixel_id` join in `build_combined` | Fails for OOB pixels (no ExportedPixels match) | Works: only in-bounds pixels are joined |
| `TracedPhotons` row count | > `ExportedPixels` row count | Equal (after filtering `in_tpx3=True`) |
| Photon reconstruction | EMPIR sees corrupted pixels at wrong (wrapped) coordinates | EMPIR only sees valid pixels; photons with OOB seeds are absent from ExportedPhotons |
| `nea-assoc` seed search | Photons with OOB seeds have no matching pixel in ExportedPixels → seed not found, min TOA diff > 0 | Photons with OOB seeds not in ExportedPhotons → not processed |

---

## No other files need changing

The `_write_tpx3` OOB filter is already correct and complete.
`neutron_event_analyzer` reads `TracedPhotons` and filters on `in_tpx3=True`
before doing the `pixel_id` join — once the flag is correct, no analyser changes
are needed for the OOB issue.
