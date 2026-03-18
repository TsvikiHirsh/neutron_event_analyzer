# G4LumaCam Agent Instructions — TracedPhotons & tpx3 Schema

This document specifies the required output schema for the two tables that
G4LumaCam must produce — **TracedPhotons** and **tpx3** — and explains the
`sim/id` field that must be propagated through both so that downstream
analysis can join all tables exactly without coordinate-based approximation.

---

## Background: the join problem

The analysis pipeline contains three datasets that must be linked:

```
SimPhotons          ← Geant4 truth (photon origin, energy, parentage)
    ↓ (1-to-many: one SimPhoton → many detector pixels)
TracedPhotons       ← G4LumaCam optical tracing + detector model
    ↓ (read by EMPIR → written to tpx3)
tpx3                ← Binary pixel stream processed by EMPIR
    ↓
ExportedPixels / ExportedPhotons / ExportedEvents
    ↓
AssociatedResults   ← NEA output (px/* ph/* ev/* columns)
```

`build_combined` must reassemble this chain.  The **only** reliable join key
between AssociatedResults and SimPhotons is a stable integer ID that flows
from the Geant4 truth table all the way through to the per-pixel rows in
AssociatedResults.

### Join key

`TracedPhotons.id` is the Geant4 track ID of the SimPhoton that produced the
pixel — it is identical to `SimPhotons.id`.

```
TracedPhotons.id       ==  SimPhotons.id
TracedPhotons.pulse_id ==  SimPhotons.pulse_id
```

Together `(id, pulse_id)` form a globally unique key across all files and
pulses.  No new column is required; `build_combined` internally aliases `id`
to `sim_id` for clarity but the CSV files do not need to change.

---

## TracedPhotons — full column specification

Saved as `TracedPhotons/traced_sim_data_<N>.csv` (one file per run/pulse
batch, `N` is the file index).

| Column | Type | Unit | Description |
|---|---|---|---|
| `pixel_x` | `int` | px | Detector pixel column (0-indexed) |
| `pixel_y` | `int` | px | Detector pixel row (0-indexed) |
| `toa2` | `float64` | ns | Time of arrival of this pixel hit on the detector |
| `time_diff` | `float64` | ns | Time-over-threshold (ToT) — energy proxy |
| `photon_count` | `int` | — | Number of photons contributing to this pixel (usually 1) |
| `id` | `int` | — | Geant4 track ID — equals `SimPhotons.id`, the primary join key |
| `neutron_id` | `int` | — | Neutron event index within the pulse |
| `pulse_id` | `int` | — | Pulse (trigger) index, shared with SimPhotons |
| `pulse_time_ns` | `float64` | ns | Absolute pulse start time |

---

## tpx3 — required fields

The `tpx3Files/` binary stream is written by `lumacam.Lens._write_tpx3(df)`
and is the input consumed by EMPIR.  EMPIR then exports
`ExportedPixels / ExportedPhotons / ExportedEvents`.

EMPIR does **not** propagate custom metadata columns — it only exports the
reconstructed cluster quantities (position, TOA, n-pixels, etc.).  Therefore
`sim_id` cannot travel through EMPIR; it must be recovered via the
TracedPhotons CSV at join time.

The tpx3 binary must faithfully encode the following fields so that EMPIR's
exported pixel coordinates and TOA exactly match the TracedPhotons CSV:

| tpx3 field | Source column | Notes |
|---|---|---|
| x | `pixel_x` | Integer pixel column |
| y | `pixel_y` | Integer pixel row |
| toa | `toa2` | Nanoseconds, converted to Timepix3 clock ticks by `_write_tpx3` |
| tot | `time_diff` | Nanoseconds, converted to clock ticks |

The pixel coordinates and TOA written to tpx3 **must be identical** (before
clock-tick rounding) to those stored in TracedPhotons.  This is what makes
the exact index join possible downstream.

---

## Downstream join chain (after this change)

With `sim_id` present in TracedPhotons, `build_combined` can use two exact
joins and no approximations:

```
Step 1 — AssociatedResults → TracedPhotons
    Key: (int(px/x), int(px/y), round(px/toa * 1e9))
         == (pixel_x, pixel_y, round(toa2))
    Result: each pixel row gains  sim_id, pulse_id

Step 2 — TracedPhotons → SimPhotons
    Key: (sim_id, pulse_id)  ==  (SimPhotons.id, SimPhotons.pulse_id)
    Result: each pixel row gains all Geant4 truth columns
```

```python
# Step 1: exact lookup via MultiIndex (no merge_asof)
trace_idx = (
    trace
    .assign(
        _px_x    = lambda d: d['pixel_x'].astype(int),
        _px_y    = lambda d: d['pixel_y'].astype(int),
        _toa_key = lambda d: d['toa2'].round().astype('int64'),
    )
    .drop_duplicates(subset=['_px_x', '_px_y', '_toa_key'])
    .set_index(['_px_x', '_px_y', '_toa_key'])
    [['sim_id', 'pulse_id']]
)

assoc['_px_x']    = assoc['px/x'].astype(int)
assoc['_px_y']    = assoc['px/y'].astype(int)
assoc['_toa_key'] = (assoc['px/toa'] * 1e9).round().astype('int64')

combined = assoc.join(trace_idx, on=['_px_x', '_px_y', '_toa_key'], how='left')
combined.drop(columns=['_px_x', '_px_y', '_toa_key'], inplace=True)

# Step 2: exact merge to SimPhotons
combined = combined.merge(sim[SIM_COLS], left_on=['sim_id', 'pulse_id'],
                          right_on=['id', 'pulse_id'], how='left')
```

---

## Verification checklist

After implementing `sim_id`, validate the following:

- [ ] Every row in `TracedPhotons` has a non-null, non-zero `sim_id`.
- [ ] Every `sim_id` value appears in `SimPhotons.id` for the same `pulse_id`.
- [ ] `round(TracedPhotons.toa2)` matches `round(ExportedPixels.t * 1e9)`
      within ±1 ns for every pixel (confirms the tpx3 round-trip is lossless).
- [ ] After `build_combined`, the fraction of rows with a matched `sim_id`
      equals the known pixel-detection efficiency (expected ~95–100 % for
      simulation without noise pixels).
- [ ] No `sim_id` collision: within a single `(pulse_id, sim_id)` group,
      all `pixel_x / pixel_y / toa2` tuples are unique.

---

## File layout reference

```
<run_dir>/
├── SimPhotons/
│   └── sim_data_<N>.csv          # Geant4 truth — id is the join key
├── TracedPhotons/
│   └── traced_sim_data_<N>.csv   # Must contain sim_id column
├── tpx3Files/
│   └── *.tpx3                    # Written by _write_tpx3(); read by EMPIR
└── <preset>/                     # EMPIR output (fast_neutrons, in_focus, …)
    ├── ExportedPixels/
    ├── ExportedPhotons/
    ├── ExportedEvents/
    └── AssociatedResults/        # NEA output — join target
```
