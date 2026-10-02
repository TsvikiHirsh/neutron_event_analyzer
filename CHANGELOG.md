# Changelog

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
