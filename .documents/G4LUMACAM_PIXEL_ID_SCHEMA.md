# G4LumaCam Agent Instructions — `pixel_id` in TracedPhotons

This document describes how to add a `pixel_id` column to `TracedPhotons` so that
`build_combined` can perform a clean integer-ID join instead of a coordinate lookup.

---

## Background

`AssociatedResults` (the output of `nea-assoc`) inherits its row structure from
`ExportedPixels` — one row per detected pixel.
`TracedPhotons` contains the simulated pixel hits that were written to the `tpx3`
stream.
A pixel_id that equals the **row index of the corresponding `ExportedPixels` row**
creates a direct, unambiguous join key between `TracedPhotons` and
`AssociatedResults`.

### Why coordinates + TOA work as a join key

Empirical validation on `openbeam_ptb_hs/out_of_focus` shows:

| Test | Result |
|---|---|
| Coordinates match exactly (`pixel_x == EP.x`, `pixel_y == EP.y`) | 100 % within 100 ns window |
| Exact Timepix3-tick TOA match (no offset, no timewalk correction needed) | 96 % |
| After +16 tick (25 ns) fallback for coarse-clock-boundary hits | **99.99 %** |
| Remaining < 0.01 % | Genuine edge cases |

The 4 % that need the fallback are caused by a **Timepix3 coarse ToA clock
artefact**: when the fast TDC measurement crosses a 25 ns coarse clock boundary,
EMPIR places the hit in the *previous* clock window, shifting its TOA 25 ns (16
ticks × 1.5625 ns) earlier than the raw simulation value.  This is a detector
hardware effect, not a simulation bug.

**Coordinate system:** `TracedPhotons.pixel_x/pixel_y` and `ExportedPixels.x/y` share
the same 0-indexed pixel coordinate frame — **no offset needed**.

**TOA quantisation:** `ExportedPixels.t [s]` is already Timepix3-clock-quantised
(multiples of 1.5625 ns).  `TracedPhotons.toa2` is a raw float from the simulation.
When both are rounded to the nearest 1.5625 ns tick they agree to within **±0 ticks
for 96 % of pixels** — no timewalk correction is required for this detector model.

---

## What G4LumaCam must do

After EMPIR produces `ExportedPixels`, run the following post-processing step
**before** finalising `TracedPhotons`:

### Step 1 — Quantise `toa2` in TracedPhotons

```python
TICK_NS = 1.5625  # Timepix3 clock period in nanoseconds

traced['toa2_tick'] = (traced['toa2'] / TICK_NS).round().astype('int64')
```

### Step 2 — Build a lookup from ExportedPixels

Concatenate all `ExportedPixels` part-files for this run into one DataFrame.
Convert the time column to nanoseconds and quantise to ticks:

```python
import glob, pandas as pd

ep_files = sorted(glob.glob(f'{run_dir}/ExportedPixels/*.csv'))
ep = pd.concat([pd.read_csv(f, skipinitialspace=True) for f in ep_files],
               ignore_index=True)

# Rename columns to canonical names (strip units/spaces)
ep = ep.rename(columns={
    'x [px]':              'ep_x',
    'y [px]':              'ep_y',
    't [s]':               'ep_t_s',
    'tot [a.u.]':          'ep_tot',
    't_relToExtTrigger [s]': 'ep_t_rel',
})
ep['ep_x']    = ep['ep_x'].astype(int)
ep['ep_y']    = ep['ep_y'].astype(int)
ep['toa_tick'] = (ep['ep_t_s'] * 1e9 / TICK_NS).round().astype('int64')

# Row index = pixel_id that AssociatedResults will inherit
ep['pixel_id'] = ep.index  # 0-based row number across all concatenated part-files
```

### Step 3 — Join TracedPhotons → ExportedPixels on (x, y, toa_tick)

> **EMPIR coarse-clock correction:**  EMPIR can be configured to apply a +25 ns
> correction to hits that cross a Timepix3 coarse-clock boundary.  Check whether
> this correction is enabled for your run before deciding whether the fallback
> below is needed:
>
> - **Correction enabled** (recommended): `ExportedPixels.t` has already been
>   corrected — the exact join below will match ~100 % of rows with no fallback.
> - **Correction disabled** (legacy data): `ExportedPixels.t` for ~4 % of hits
>   is 25 ns (16 ticks) earlier than `toa2`.  Use the fallback block below to
>   recover them.

```python
ep_idx = ep.set_index(['ep_x', 'ep_y', 'toa_tick'])[['pixel_id']]

traced = traced.join(ep_idx, on=['pixel_x', 'pixel_y', 'toa2_tick'], how='left')

# ── 25 ns fallback (only needed when EMPIR coarse-clock correction is OFF) ────
# When a hit's fast-TDC measurement crosses a coarse clock boundary
# (25 ns = 16 × 1.5625 ns ticks), EMPIR without correction assigns the hit to
# the *previous* coarse clock window, shifting ~4 % of EP rows 16 ticks earlier
# relative to the raw simulation TOA.  A single +16-tick retry recovers them.
mask = traced['pixel_id'].isna()
if mask.any():
    traced.loc[mask, 'toa2_tick_fb'] = traced.loc[mask, 'toa2_tick'] - 16
    fb_idx = ep_idx.copy()
    fb_idx.index = fb_idx.index.set_levels(
        fb_idx.index.levels[2] + 16, level=2   # shift EP toa_tick up by 16 to meet TP
    )
    fallback = traced.loc[mask, ['pixel_x', 'pixel_y', 'toa2_tick_fb']].join(
        ep_idx.rename_axis(['pixel_x', 'pixel_y', 'toa2_tick_fb']),
        on=['pixel_x', 'pixel_y', 'toa2_tick_fb'],
        how='left',
    )['pixel_id']
    traced.loc[mask, 'pixel_id'] = fallback.values

traced.drop(columns=['toa2_tick'], inplace=True)
```

> **Simpler alternative** if the MultiIndex manipulation is awkward: build two
> dictionaries `{(x,y,t): pixel_id}` and `{(x,y,t+16): pixel_id}` and look up
> each TracedPhotons row in both.

Rows with `pixel_id` still NaN after the fallback are genuine misses (< 0.01 %).

### Step 4 — Save TracedPhotons with `pixel_id`

The final `TracedPhotons` CSV must include the `pixel_id` column.
The full required schema after this change:

| Column | Type | Unit | Description |
|---|---|---|---|
| `pixel_x` | `int` | px | Detector pixel column (0-indexed) |
| `pixel_y` | `int` | px | Detector pixel row (0-indexed) |
| `toa2` | `float64` | ns | Raw simulation time of arrival |
| `time_diff` | `float64` | ns | Time-over-threshold — energy proxy |
| `photon_count` | `int` | — | Photons merged into this pixel hit |
| `id` | `int` | — | Geant4 track ID (= `SimPhotons.id`) |
| `sim_id` | `int` | — | Alias of `id` — primary join key to `SimPhotons` |
| `neutron_id` | `int` | — | Neutron event index within the pulse |
| `pulse_id` | `int` | — | Pulse (trigger) index |
| `pulse_time_ns` | `float64` | ns | Absolute pulse start time |
| **`pixel_id`** | `int` (or `Int64`) | — | **Row index of the matching `ExportedPixels` row — join key to `AssociatedResults`** |

---

## How `build_combined` will use `pixel_id`

Once `TracedPhotons` has `pixel_id`, the Step 2 join in `build_combined` becomes
a direct integer-key merge instead of a coordinate lookup:

```python
# Step 2: AssociatedResults → TracedPhotons on pixel_id == AssociatedResults row index
combined = assoc.join(
    trace_with_sim.set_index('pixel_id')[TRACE_CARRY],
    how='left',
)
```

`assoc`'s integer row index (0, 1, 2, …) equals the `pixel_id` in TracedPhotons,
because `AssociatedResults` preserves the `ExportedPixels` row order through
`nea-assoc`.

---

## Verification checklist

After adding `pixel_id`:

- [ ] `TracedPhotons.pixel_id.notna()` fraction ≥ 95 % (expect ~96 %, 4 % are noise pixels)
- [ ] `(traced.set_index('pixel_id')['pixel_x'] == ep.set_index('pixel_id')['ep_x']).all()` — coordinates agree
- [ ] After `build_combined`, `(combined['px/x'].astype(int) == combined['sim/pixel_x'].astype(int)).mean() > 0.95`
- [ ] Step 2 match rate printed by `build_combined --verbose` ≥ 95 %

---

## File layout

```
<archive>/
├── SimPhotons/
│   └── sim_data_<N>.csv
├── TracedPhotons/
│   └── traced_sim_data_<N>.csv    ← now includes pixel_id column
└── <preset>/                      ← e.g. in_focus/, out_of_focus/
    ├── ExportedPixels/
    │   └── traced_data_<N>_part*.csv
    └── AssociatedResults/
        └── associated_data.csv    ← row index == pixel_id
```
