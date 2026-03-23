# G4LumaCam Agent Instructions — Coarse-Clock Wraparound Flag

## Background: the -25 ns satellite in time resolution

When the user computes `ph/toa - sim/toa` on `combined` data from `nea-assoc
--merge-sim`, they observe a satellite peak at approximately **-25 ns** in
addition to the main peak at 0.  This satellite contains roughly 6 % of events
and prevents accurate time-resolution measurements.

### Root cause

Timepix3 encodes pixel arrival time using a 14-bit **coarse** counter (25 ns
period, 40 MHz) plus a 4-bit **fine** counter (1.5625 ns resolution):

```
toa_ns = coarse_toa * 25 ns - ftoa_raw * 1.5625 ns
       = coarse_toa * 25 ns + (15 - ftoa_raw) * 1.5625 ns
```

(EMPIR uses the second form internally.)

A **coarse-clock wraparound** happens when a pixel fires at the very last
fine-time bin of a coarse period, i.e.:

```python
toa_ticks = round(toa_ns / 1.5625)
wrap_condition = (toa_ticks & 0xF) == 15   # fine bin 15 — last in period
```

At this moment `ftoa_raw = 15 - 15 = 0` is stored in the packet.  In real
hardware, the coarse counter may already have incremented to the *next* period
before the pixel latches.  The result: the packet stores `coarse_toa = C`
while the true period was `C - 1`, so the reconstructed time is 25 ns too
late.  EMPIR detects most of these and corrects them, but for approximately
6 % of such pixels it does not apply the fix.  In simulation G4LumaCam does
not currently introduce this artefact at all, so the satellite cannot be
studied or corrected without ground-truth labelling.

### Why `seed_dt_ns = 0` does not distinguish wrapped photons

`ph/seed_dt_ns` records the TOA offset used by NEA to find the seed pixel
(0, ±25, ±50 ns).  A photon with an uncorrected wrapped seed pixel has:

- `seed_dt_ns = 0`  — NEA found the seed exactly at `ph/toa`
- `ph/toa = true_toa - 25 ns` — EMPIR set `ph/toa` to the wrapped pixel time

Events with correctly reconstructed timing **also** have `seed_dt_ns = 0`, so
the flag alone cannot identify bad events.  A `coarse_clock_wrap` column in
TracedPhotons — indicating which pixels were simulated with the wrap bug —
provides the missing discriminant.

---

## Changes required in G4LumaCam (`optics.py`)

There are **two independent edits**.

### Edit 1 — simulate the wraparound in `_write_tpx3`

**Location:** inside `_write_tpx3`, immediately after the fine-time
decomposition (currently around lines 1784–1791):

```python
# Convert ToA to 1.5625ns ticks
toa_ticks = np.round(toa_ns / TICK_NS).astype(np.int64)

# Decompose ToA into packet fields
spidr_time = ((toa_ticks >> 18) & 0xFFFF).astype(np.int64)
coarse_toa = ((toa_ticks >> 4) & 0x3FFF).astype(np.int64)
ftoa = (15 - (toa_ticks & 0xF)).astype(np.int64)
ftoa = np.clip(ftoa, 0, 15)
```

**After** the `ftoa = np.clip(...)` line, add:

```python
# --- Coarse-clock wraparound simulation -----------------------------------
# Pixels whose fine-time bin is 15 (ftoa_raw == 0) are at the very last
# fine slot of a coarse period.  In real TPX3 hardware the coarse counter
# may have already incremented when the pixel latches, giving coarse_toa
# one period too high.  Simulate this artefact so that downstream analysis
# can study and correct it.
#
# Mathematical result: EMPIR reconstructs  (C-1)*25 + (15-0)*1.5625
#                                         = true_toa - 25 ns  (exactly)
# regardless of the fine-time value within the period.
_wrap_mask = (ftoa == 0)          # same as (toa_ticks & 0xF) == 15
coarse_toa = coarse_toa.copy()    # avoid mutating the original array
coarse_toa[_wrap_mask] -= 1       # simulate hardware reading previous period
# -------------------------------------------------------------------------
```

> **Important:** `_wrap_mask` computed here is the per-row boolean array for
> the rows passed to `_write_tpx3` (i.e., `tpx3_data`, not all of `result_df`).
> See Edit 2 for how to store it back into the TracedPhotons table.

### Edit 2 — add `coarse_clock_wrap` column to TracedPhotons

**Location:** in the hits-workflow loop, **after** `result_df` is built and
saturated but **before** the `desired_columns` filter (currently around
lines 1291–1296):

```python
# Filter columns to keep for hits workflow
desired_columns = ['pixel_x', 'pixel_y', 'toa2', 'photon_count', 'time_diff',
                'id', 'sim_id', 'neutron_id', 'pulse_id', 'pulse_time_ns', 'in_tpx3']

columns_to_keep = [col for col in desired_columns if col in result_df.columns]
result_df = result_df[columns_to_keep]
```

Replace with:

```python
# --- Coarse-clock wrap flag -----------------------------------------------
# Mark pixels that will be written with a coarse_toa decremented by 1 (see
# _write_tpx3).  Uses the same wrap condition: fine bin 15 (ftoa_raw == 0).
# Only rows that are actually sent to _write_tpx3 (in_tpx3 == True) can be
# wrapped; out-of-bounds rows keep False.
_toa_ticks_all = np.round(
    result_df['toa2'].to_numpy().astype(float) / TICK_NS
).astype(np.int64)
result_df['coarse_clock_wrap'] = ((15 - (_toa_ticks_all & 0xF)) == 0)
# Out-of-bounds pixels are not written to TPX3, so they cannot be wrapped.
if 'in_tpx3' in result_df.columns:
    result_df.loc[~result_df['in_tpx3'].astype(bool), 'coarse_clock_wrap'] = False
# -------------------------------------------------------------------------

# Filter columns to keep for hits workflow
desired_columns = ['pixel_x', 'pixel_y', 'toa2', 'photon_count', 'time_diff',
                'id', 'sim_id', 'neutron_id', 'pulse_id', 'pulse_time_ns',
                'in_tpx3', 'coarse_clock_wrap']    # ← added coarse_clock_wrap

columns_to_keep = [col for col in desired_columns if col in result_df.columns]
result_df = result_df[columns_to_keep]
```

> **Note:** `TICK_NS = 1.5625` is already defined at the top of `_write_tpx3`
> as a local constant.  The edit above is placed *outside* that function, so
> use the same literal value `1.5625` or define a module-level constant.

---

## Changes required in NEA (`analyser.py`, `build_combined`)

### Edit 3 — carry `coarse_clock_wrap` through the join

**Location:** the `TRACE_CARRY` list, currently:

```python
TRACE_CARRY = [_id_col]
for _c in ['neutron_id', 'pulse_id', 'pulse_time_ns', 'pixel_x', 'pixel_y', 'toa2']:
    if _c in trace_with_sim.columns:
        TRACE_CARRY.append(_c)
```

Change to:

```python
TRACE_CARRY = [_id_col]
for _c in ['neutron_id', 'pulse_id', 'pulse_time_ns',
           'pixel_x', 'pixel_y', 'toa2', 'coarse_clock_wrap']:
    if _c in trace_with_sim.columns:
        TRACE_CARRY.append(_c)
```

### Edit 4 — rename to `sim/ccw` in the output

**Location:** the column-rename block near the end of `build_combined`,
currently:

```python
_sim_src = set(_DEFAULT_SIM_COLS) | {'neutron_id', 'pulse_id', 'pulse_time_ns', 'pixel_x', 'pixel_y', 'toa2'}
_rename = {c: f'sim/{c}' for c in combined.columns if c in _sim_src}
_rename['sim_id'] = 'sim/id'
combined = combined.rename(columns=_rename)
```

After `combined = combined.rename(columns=_rename)`, add:

```python
if 'coarse_clock_wrap' in combined.columns:
    combined = combined.rename(columns={'coarse_clock_wrap': 'sim/ccw'})
```

---

## How to use `sim/ccw` for time-resolution correction

After running `nea-assoc --merge-sim`, the `combined` CSV contains:

| column | meaning |
|--------|---------|
| `ph/toa` | EMPIR-reconstructed photon time (seconds) |
| `ph/seed_dt_ns` | TOA offset NEA used to find the seed pixel (0, ±25, ±50 ns) |
| `sim/toa2` | True simulation arrival time (ns) |
| `sim/ccw` | `True` if the seed pixel was written with the -25 ns wrap bug |

### Filter for time resolution

```python
import pandas as pd
import numpy as np

df = pd.read_csv('AssociatedResults/combined.csv')

# 1. Correct ph/toa for photons whose seed was an uncorrected wrapped pixel.
#    These have sim/ccw = True AND seed_dt_ns = 0 (NEA found seed at ph/toa
#    exactly, not at ph/toa ± 25 ns).  EMPIR did NOT correct this pixel, so
#    ph/toa = true_toa - 25 ns → add 25 ns.
needs_correction = (
    df.get('sim/ccw', False) &
    (df['ph/seed_dt_ns'].fillna(999) == 0)
)
df['ph/toa_corrected'] = df['ph/toa'].copy()
df.loc[needs_correction, 'ph/toa_corrected'] += 25e-9   # seconds

# 2. Exclude photons whose seed was out-of-bounds (no pixel at ph/toa in-chip),
#    identified by seed_dt_ns = +25 ns.
in_resolution = df['ph/seed_dt_ns'].isin([0, -25])

# 3. Compute time residual (ns) for the resolution plot.
df['dt_ns'] = (df['ph/toa_corrected'] - df['sim/toa2'] * 1e-9) * 1e9

resolution_df = df[in_resolution]
```

### Expected result

- Main peak at 0 ns — correctly reconstructed photons.
- The -25 ns satellite **disappears** after applying the correction above.
- Photons with `seed_dt_ns = -25` (EMPIR corrected internally) contribute
  to the main peak at 0 ns and should be included.

---

## Summary of files to edit

| File | Edit | Purpose |
|------|------|---------|
| `G4LumaCam/src/lumacam/optics.py` | Edit 1 (inside `_write_tpx3`) | Simulate -25 ns wrap in TPX3 output |
| `G4LumaCam/src/lumacam/optics.py` | Edit 2 (hits-workflow loop) | Write `coarse_clock_wrap` to TracedPhotons |
| `neutron_event_analyzer/src/neutron_event_analyzer/analyser.py` | Edit 3 (`TRACE_CARRY`) | Carry column through the join |
| `neutron_event_analyzer/src/neutron_event_analyzer/analyser.py` | Edit 4 (rename block) | Expose as `sim/ccw` in combined output |
