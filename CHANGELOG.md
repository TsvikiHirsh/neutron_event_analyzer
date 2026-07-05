# Changelog

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
