# Changelog

## [0.6.1] - 2026-10-05

### Added
- `Analyse.exact_photons_df` after `associate_exact()`: one row per event and photon, every photon
  cluster of every event, with `ph/pid`, the photon id of the pixel table (`pixels=True`).
- `observables.select_exact()` and `observables.extract_distributions_exact()`: the five per-event
  distributions from an exact association. Cluster size (EMPIR's pixel count), clusters per event
  (EMPIR's count) and cluster distances come from the photon rows, so clusters that the pixel step
  leaves without pixels still count; the pixel residual and arrival time come from the pixel rows.

### Changed
- Pixel-photon step: with EMPIR's pixel count per photon in the exports (EMPIR >= 1.0.1), the pixel
  subset of exactly that size is preferred, so that a bright photon does not take the pixels of a
  small neighbour whose centroid it can absorb.
- `associate_exact(pixels=True)` attaches the event columns in place (memory of large pixel tables).

## [0.6.0] - 2026-10-04

### Added
- **`exact` module** and `Analyse.associate_exact()`: exact photon-to-event association of an
  EMPIR reconstruction from its standard exports. The earliest photon of each event is taken from
  an earliest-photon reconstruction of the same photons (`..._firstPhotonPos_direct`), and the other
  photons are the set within the event duration whose mean reproduces the event position. Every
  event is reproduced (measured PTB data, early and late files of a 30-min run: 100%), photons shared
  by two events are kept, and the largest-cluster position uses EMPIR's pixel count per photon.
  Burst events beyond the search limits fall back to the photons nearest the event position
  (`ev/status`).
- `associate_exact(first_events=None)`: without an earliest-photon reconstruction the earliest
  photon is one of the photons on the event's clock tick; when several share it, the one for which
  the other photons admit an exact subset is taken. Same membership (simulated and measured PTB
  data: 100% of the events reproduced).
- `associate_exact(pixels=True)`: the pixel table of `associate()` (pixel-photon step) with the
  exact photon-event step, for the pixel-level observables. The photon id of every pixel row is
  re-derived from the stored cluster time and position, which corrects the about 1% of pixel rows
  whose id pointed to a neighbouring cluster.

## [0.5.1] - 2026-10-02

### Added
- **`truth` module** (`join_truth`, `load_truth`): attaches the G4LumaCam simulation
  truth (TracedPhotons and SimPhotons: optical truth, emission point, parent particle
  and vertex, neutron energy, last interaction and entry point) to every pixel row of
  an associated table. Pixels are matched on the exact key (x, y, clock tick), which
  does not depend on the order of the exported rows.

## [0.5.0] - 2026-10-02

### Added
- **`observables` module**: the five per-event distributions used to calibrate a
  detector model against measured data (pixels per photon cluster, clusters per
  event, pixel-to-event x residual, pixel arrival time within the event, distance of
  each extra cluster from the largest cluster of its event) and their weighted
  symmetric chi-squared comparison (`extract_distributions`, `compare`). Selections by
  event size and by neutron energy from the time of flight (with an optional
  time-of-flight calibration, nres convention), per-event weights and
  `spectrum_weights()` to compare two data sets at the same neutron-energy mix.
- Event time of flight in the associated table (`ev/tof`).
- EMPIR >= 1.0.1 photon exports: the pixel count (`ph/npx`) and intensity
  (`ph/intensity`) of every photon are read and carried into the associated table.
- `load(xy_offset='auto')`: EMPIR >= 1.0.1 places the photons and events of a
  single-chip camera at a chip offset (+260 px in x) while the pixel export stays in
  chip coordinates; the offset is detected from pixel-photon pairs and removed.

## [0.4.0] - 2026-07-05

### Added
- **Satellite-aware event-position estimators**: `associate()` now adds
  `ev/x_cog`/`ev/y_cog` (photon mean), `ev/x_first`/`ev/y_first` (earliest
  photon) and `ev/x_largest`/`ev/y_largest` (largest cluster, robust against
  intensifier-afterpulse satellites) to the associated output. Also exposed
  as `Analyse.compute_event_positions()`, which supports index-only
  association tables by reading photon coordinates from `ExportedPhotons/`.
- **`config.BEST_DETECTOR_MODEL`**: the calibrated G4LumaCam
  `gaussian_probabilistic` detector model (PTB per-event calibration:
  blob 0.405 px, P47 decay 16.5 ns, n_secondaries 9, photon_keep_fraction
  0.241, afterpulse satellites at the literature rate, Mahon et al. 2024).
- `position_mode: "largest"` recorded in the `out_of_focus` settings preset
  as the recommended event-position choice for multi-photon reconstructions.

### Notes
- Verified on the PTB air45 out-of-focus dataset: 119,271 events, median
  photon-mean to largest-cluster shift of 3.54 px (1.64 mm) for multi-photon
  events, consistent with the afterpulse satellite pull reported in the
  accompanying manuscript.
