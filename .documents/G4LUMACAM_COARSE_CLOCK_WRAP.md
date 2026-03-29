# G4LumaCam Agent Instructions — Coarse-Clock Wraparound Flag

## Background: the -25 ns satellite in time resolution

When the user computes `ph/toa - sim/toa` on `combined` data from `nea-assoc
--merge-sim`, they observe a satellite peak at approximately **-25 ns** in
addition to the main peak at 0.  This satellite contains roughly 6 % of events
and prevents accurate time-resolution measurements.

> **Key finding (from empirical testing):** the G4LumaCam implementation also
> sets `toa2 = wrapped pixel time` (= `sim/toa - 25 ns`) for wrapped pixels.
> This means `ph/toa - sim/toa2` gives a clean distribution with **no satellite**
> and is the correct reference for time-resolution measurement.
> `ph/toa - sim/toa` retains the -25 ns satellite because `sim/toa` (SimPhotons)
> is the continuous Geant4 true time, while `sim/toa2` (TracedPhotons) is aligned
> to the 25 ns TPX3 coarse-clock grid — exactly the same 25 ns time-bin width
> documented in the TPX3Cam manual (§7 Global Time Extension).

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

## Why `sim/ccw` alone is not sufficient for the correction

`sim/ccw = True` flags every pixel that G4LumaCam wrote with a decremented
coarse clock (all fine-bin-15 pixels).  However EMPIR successfully corrects
**most** of these: the corrected pixel lands back at `true_toa` in
ExportedPixels, and ph/toa = `true_toa` — correct, no adjustment needed.
Only the small fraction that EMPIR **fails** to correct produce the -25 ns
satellite.

The problem is that both cases (EMPIR corrected and not corrected) produce
`seed_dt_ns = 0` in NEA, because in both cases NEA finds the seed pixel
exactly at `ph/toa`.  Applying `+= 25 ns` to all `sim/ccw=True, seed_dt_ns=0`
events corrupts the majority that are already correct.

Empirical confirmation (from a typical PTB dataset):

```
sim/ccw=True, seed_dt_ns=0 → mean(ph/toa − sim/toa) ≈ −8.5 ns (not −25 ns)
std ≈ 14 ns  →  only ~34 % are in the satellite; ~66 % have correct ph/toa
```

### Correct approach: use the per-pixel toa residual

The uncorrected-wrapped pixel is the one where the pixel toa in ExportedPixels
is **exactly 25 ns early** relative to the true simulation toa:

```
px/toa_ns  ≈  sim/toa2 − 25   (uncorrected)
px/toa_ns  ≈  sim/toa2        (EMPIR corrected — do not touch)
```

This is a precise, zero-false-positive filter that does not rely on `sim/ccw`
or `seed_dt_ns`.

## How to use `sim/toa2` for time-resolution correction

After running `nea-assoc --merge-sim`, the combined file contains:

| column | meaning | units |
|--------|---------|-------|
| `ph/toa` | EMPIR-reconstructed photon time | seconds |
| `px/toa` | seed pixel reconstructed time | seconds |
| `ph/seed_dt_ns` | NEA seed-search offset (0, ±25, ±50 ns) | ns |
| `sim/toa2` | True simulation pixel arrival time | ns |
| `sim/ccw` | Pixel was at fine-bin-15 (may or may not be miscorrected) | bool |

### Correction function

```python
def correction(df):
    """
    Correct ph/toa and ev/toa for pixels that EMPIR failed to un-wrap.

    Logic: if a seed pixel's reconstructed toa (px/toa) is 25 ns earlier
    than its true simulation toa (sim/toa2), EMPIR did not apply the
    coarse-clock correction → ph/toa is 25 ns too early → add 25 ns.
    """
    # px/toa in seconds → ns; sim/toa2 already in ns
    px_ns = df['px/toa'] * 1e9
    pixel_offset = px_ns - df['sim/toa2']          # ≈ 0 normal, ≈ -25 uncorrected wrap

    is_uncorrected_wrap = (pixel_offset + 25).abs() < 2.0   # ±2 ns tolerance
    is_seed = (df['px/toa'] - df['ph/toa']).abs() < 0.5e-9  # this pixel set ph/toa

    mask = is_uncorrected_wrap & is_seed

    orig_ph = df['ph/toa'].copy()
    df.loc[mask, 'ph/toa'] += 25e-9

    # ev/toa: if the corrected photon was the first photon in its event,
    # ev/toa was also set from this pixel and needs the same +25 ns.
    first_ph_of_event = (orig_ph - df['ev/toa']).abs() < 0.5e-9
    events_to_fix = df.loc[mask & first_ph_of_event, 'ev/toa'].unique()
    df.loc[df['ev/toa'].isin(events_to_fix), 'ev/toa'] += 25e-9
```

### Expected result

- `mask` selects only the genuinely uncorrected-wrapped seed pixels (~3-5 % of events).
- Main peak at 0 ns — correctly reconstructed photons unchanged.
- The -25 ns satellite disappears.
- No new artifact at +25 ns (EMPIR-corrected wrapped pixels are not touched).
- Photons with `seed_dt_ns = -25` contribute to the main peak and are included.

### Role of `sim/ccw` after this correction

`sim/ccw` is no longer needed for the per-event correction.  It is still
useful as a **diagnostic**: `df[df['sim/ccw']]['px/toa']*1e9 - df[df['sim/ccw']]['sim/toa2']`
should be bimodal at {0, −25} ns, confirming G4LumaCam's wrap simulation
and EMPIR's partial correction rate.

---

## Summary of files to edit

| File | Edit | Purpose |
|------|------|---------|
| `G4LumaCam/src/lumacam/optics.py` | Edit 1 (inside `_write_tpx3`) | Simulate -25 ns wrap in TPX3 output |
| `G4LumaCam/src/lumacam/optics.py` | Edit 2 (hits-workflow loop) | Write `coarse_clock_wrap` to TracedPhotons |
| `neutron_event_analyzer/src/neutron_event_analyzer/analyser.py` | Edit 3 (`TRACE_CARRY`) | Carry column through the join |
| `neutron_event_analyzer/src/neutron_event_analyzer/analyser.py` | Edit 4 (rename block) | Expose as `sim/ccw` in combined output |
