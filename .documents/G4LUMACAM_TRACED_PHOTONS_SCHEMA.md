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

### Why the current `id` column is insufficient

| Table | `id` value (example) | Meaning |
|---|---|---|
| SimPhotons | `7737` | Geant4 **track ID** — assigned by the G4 kernel |
| TracedPhotons | `1044` | G4LumaCam **internal cluster index** — re-numbered sequentially |

These are different numbering schemes.  A join on `(id, pulse_id)` across the
two tables will silently produce all-NaN rows.

---

## Required change: add `sim/id` to TracedPhotons and tpx3

### What `sim/id` is

`sim/id` is the Geant4 track ID of the SimPhoton that produced a given set of
detector pixels.  It equals `SimPhotons.id` for the corresponding row.

```
TracedPhotons.sim_id  ==  SimPhotons.id
TracedPhotons.pulse_id  ==  SimPhotons.pulse_id
```

Together `(sim_id, pulse_id)` form a globally unique key across all files and
pulses, identical to the key already used in SimPhotons.

> **Note on naming:** the column is called `sim_id` in CSV files (underscore,
> no slash) because `/` is not a safe CSV column name character.  The analysis
> code refers to it as `sim/id` after loading.

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
| `id` | `int` | — | G4LumaCam internal cluster index — **keep for backward compat** |
| **`sim_id`** | **`int`** | — | **NEW — Geant4 track ID from SimPhotons (`SimPhotons.id`)** |
| `neutron_id` | `int` | — | Neutron event index within the pulse |
| `pulse_id` | `int` | — | Pulse (trigger) index, shared with SimPhotons |
| `pulse_time_ns` | `float64` | ns | Absolute pulse start time |

### How to populate `sim_id`

When G4LumaCam traces an optical photon through the lens model to the
detector, it already knows which Geant4 track produced that photon.  At the
point where a pixel hit is registered, store `track->GetTrackID()` (or its
Python equivalent from the simulation bookkeeping) as `sim_id`.

```python
# Pseudocode inside G4LumaCam pixel-hit recording
pixel_hit = {
    "pixel_x":      hit.pixel_x,
    "pixel_y":      hit.pixel_y,
    "toa2":         hit.toa_ns,
    "time_diff":    hit.tot_ns,
    "photon_count": 1,
    "id":           cluster_index,          # existing internal index
    "sim_id":       photon.geant4_track_id, # ← NEW: SimPhotons.id
    "neutron_id":   photon.neutron_id,
    "pulse_id":     photon.pulse_id,
    "pulse_time_ns": pulse.start_time_ns,
}
```

`sim_id` must be set **before** any re-numbering or de-duplication steps so
that the value is always traceable back to the original Geant4 track.

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
