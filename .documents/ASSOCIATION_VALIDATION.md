# Association Validation Tests

This directory contains comprehensive tests for validating the association algorithms in the neutron event analyzer using synthetic data.

## Overview

The `test_association_validation.py` file implements a complete validation workflow:

1. **Create synthetic data** with known ground truth (photon/event IDs)
2. **Generate data files** in the expected format (CSV files + dummy binary files)
3. **Run association algorithms** using NEA
4. **Verify correctness** by comparing results to ground truth

## Test Structure

### Helper Functions

- `create_synthetic_pixel_data()` - Generate pixel hits for known photons
- `create_synthetic_photon_data()` - Generate photons for known events
- `create_synthetic_event_data()` - Generate ground truth events
- `write_csv_files()` - Write data in NEA-compatible format
- `write_tpx3_file()` - Write TPX3 binary format (for future use with G4LumaCam)

### Test Fixtures

Pre-configured scenarios for testing:

- `simple_scenario` - 3 well-separated events (baseline test)
- `temporal_proximity_scenario` - Events close in time (50ns apart)
- `spatial_clustering_scenario` - Events close in space (10px apart)
- `single_pixel_photons_scenario` - Minimum pixel clustering
- `multi_pixel_photons_scenario` - Large photons with 50+ pixels

### Test Classes

#### `TestConfigurationPresets`
Tests all 4 configuration presets from `config.py`:
- `in_focus` - Tight spatial matching (0.001 px)
- `out_of_focus` - Loose spatial matching (60 px)
- `fast_neutrons` - Moderate parameters (2 px, 100 ns)
- `hitmap` - Minimal matching (0.001 px, 0 ns)

#### `TestAssociationMethods`
Tests all association algorithms:
- `simple` - Fast forward-window algorithm with CoG check
- `kdtree` - Full KDTree on normalized space-time
- `window` - Windowed KDTree for time-sorted data
- `lumacam` - Uses lumacamTesting library (if available)

#### `TestEdgeCases`
Tests challenging scenarios:
- Temporal proximity (events close in time)
- Spatial clustering (events close in space)
- Single-pixel photons (minimal clustering)
- Multi-pixel photons (large photons)

#### `TestParameterSensitivity`
Tests parameter effects:
- Spatial threshold sensitivity
- Temporal threshold sensitivity

#### `TestFullPipeline`
Tests complete 3-tier association:
- Pixels → Photons → Events
- Association statistics computation
- CoM distance calculation

## Running the Tests

### Run all association validation tests:
```bash
pytest tests/test_association_validation.py -v
```

### Run specific test class:
```bash
pytest tests/test_association_validation.py::TestConfigurationPresets -v
```

### Run specific test:
```bash
pytest tests/test_association_validation.py::TestAssociationMethods::test_photon_event_association_method -v
```

### Run with specific configuration:
```bash
pytest tests/test_association_validation.py -v -k "fast_neutrons"
```

## Important Notes

### Association Parameters vs Synthetic Data

The synthetic data generation parameters must be compatible with the association parameters being tested.

**Example:**
- If `fast_neutrons` preset uses `dSpace_px: 2` (2 pixel threshold)
- Then synthetic data should have `photon_spread_spatial ≤ 2.0` pixels
- Otherwise photons will be outside the association window

**Current defaults in synthetic data:**
- `photon_spread_spatial`: 10.0 pixels (may be too large for tight presets)
- `photon_spread_temporal`: 20.0 ns
- `pixel_spread`: 1.5 pixels

**Recommendation:**
- Use `out_of_focus` preset (60 px threshold) for general testing
- Or adjust synthetic data parameters to match specific presets
- Or override association parameters in tests with looser values

### Directory Structure

The tests create this structure for NEA compatibility:

```
test_dir/
├── eventFiles/
│   └── traced_data_0.empirevent (dummy file)
├── photonFiles/
│   └── traced_data_0.empirphot (dummy file)
├── ExportedEvents/
│   └── traced_data_0.csv (actual data)
├── ExportedPhotons/
│   └── traced_data_0.csv (actual data)
└── ExportedPixels/
    └── exported_traced_data_0.csv (actual data)
```

The dummy binary files are needed because NEA's loader scans for `.empirevent` and `.empirphot` files to find pairs, then looks for corresponding CSVs in the `Exported*` directories.

## Integration with G4LumaCam

For full end-to-end testing with empir pipeline:

1. **Generate synthetic tables** using test fixtures
2. **Write TPX3 files** using `write_tpx3_file()` function
3. **Run empir processing** using `G4LumaCam.Analysis.process()`
4. **Run NEA association** on empir outputs
5. **Verify** against ground truth

The `write_tpx3_file()` function creates valid TPX3 binary files that can be processed by empir.

## Future Enhancements

- [ ] Add pixel-to-photon association verification
- [ ] Add tests for different sensor sizes (256 vs 512)
- [ ] Add tests with lumacamTesting integration
- [ ] Add performance benchmarks
- [ ] Add tests for edge cases (empty data, single photon, etc.)
- [ ] Integrate with actual G4LumaCam workflow
- [ ] Add visualization of association results
- [ ] Add statistical analysis of association quality

## Troubleshooting

### "No associations made"
- Check that association parameters match synthetic data spread
- Verify CSV files are created correctly
- Check that dummy binary files exist

### "Row count mismatch"
- Verify that photon/event counts match expected values
- Check that CSV column names are correct (x, y, toa, tof, etc.)

### "TypeError: unexpected keyword argument 'verbosity'"
- The `Analyse` class doesn't have a `verbosity` parameter
- Use `load(verbosity=0)` instead of passing to constructor

## References

- NEA association algorithms: `src/neutron_event_analyzer/analyser.py`
- Configuration presets: `src/neutron_event_analyzer/config.py`
- Existing test patterns: `tests/test_csv_loading.py`, `tests/test_pixel_loading.py`
- G4LumaCam: https://github.com/TsvikiHirsh/G4LumaCam
