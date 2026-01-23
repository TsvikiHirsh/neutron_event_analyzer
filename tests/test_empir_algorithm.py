#!/usr/bin/env python3
"""
Test suite for investigating EMPIR algorithm behavior and comparing with nea association methods.

This script creates synthetic pixel data with known properties, processes it through
EMPIR using lumacam, and compares the results with nea.Analyse association methods.

Test cases:
1. CoG calculation verification (ToT-weighted vs simple mean)
2. TOA selection (first pixel vs something else)
3. Distance metric (Euclidean vs Manhattan)
4. Time window behavior (dTime parameter)
5. Spatial clustering (dSpace parameter)
6. Comparison between EMPIR and nea association methods
"""

import os
import sys
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
import unittest

# Add paths
sys.path.insert(0, '/work/nuclear/G4LumaCam/src')
sys.path.insert(0, '/work/nuclear/neutron_event_analyzer/src')

import lumacam
import neutron_event_analyzer as nea

# EMPIR path
EMPIR_PATH = '/work/Programs/lumacam_measurementcontrol'


def create_test_directory(test_name: str) -> Path:
    """Create a clean test directory with required structure."""
    test_dir = Path(f"/work/nuclear/neutron_event_analyzer/tests/test_data/{test_name}")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True)

    # Create required subdirectories for lumacam
    (test_dir / "TracedPhotons").mkdir()
    (test_dir / "tpx3Files").mkdir()

    return test_dir


def create_traced_photons_df(pixels: list) -> pd.DataFrame:
    """
    Create a TracedPhotons DataFrame from pixel specifications.

    Each pixel is a dict with: x, y, toa_ns, tot_ns, photon_id, neutron_id, pulse_id
    """
    rows = []
    for i, px in enumerate(pixels):
        rows.append({
            'pixel_x': float(px['x']),
            'pixel_y': float(px['y']),
            'toa2': float(px.get('toa_ns', 1000.0)),  # Time in ns
            'time_diff': float(px.get('tot_ns', 50.0)),  # ToT in ns
            'photon_count': int(px.get('photon_count', 1)),
            'id': int(px.get('photon_id', i)),
            'neutron_id': int(px.get('neutron_id', 0)),
            'pulse_id': int(px.get('pulse_id', 0)),
            'pulse_time_ns': float(px.get('pulse_time_ns', 0.0)),
        })
    return pd.DataFrame(rows)


def save_traced_photons(df: pd.DataFrame, test_dir: Path, filename: str = "traced_sim_data_0.csv"):
    """Save traced photons CSV with proper header."""
    filepath = test_dir / "TracedPhotons" / filename

    # Write with header comment
    with open(filepath, 'w') as f:
        f.write("# Synthetic test data for EMPIR algorithm investigation\n")
        df.to_csv(f, index=False)

    return filepath


def process_with_empir(test_dir: Path, preset: str = "fast_neutrons", verbosity: int = 0):
    """Process test data through EMPIR pipeline."""
    # Create Lens and write TPX3
    lens = lumacam.Lens(str(test_dir))

    # Load the traced photons
    df = pd.read_csv(test_dir / "TracedPhotons" / "traced_sim_data_0.csv", comment='#')

    # Saturate and write TPX3 (bypass saturation model, use data directly)
    lens._write_tpx3(df)

    # Run EMPIR analysis
    analysis = lumacam.Analysis(str(test_dir), empir_dirpath=EMPIR_PATH)
    analysis.process(
        preset,
        export_pixels=True,
        export_photons=True,
        export_events=True,
        suffix=preset,
        verbosity=lumacam.VerbosityLevel.QUIET if verbosity == 0 else lumacam.VerbosityLevel.BASIC
    )

    return test_dir / preset


def load_empir_results(results_dir: Path):
    """Load EMPIR exported results."""
    results = {}

    # Load pixels
    pixels_dir = results_dir / "ExportedPixels"
    if pixels_dir.exists():
        pixel_files = list(pixels_dir.glob("*.csv"))
        if pixel_files:
            results['pixels'] = pd.read_csv(pixel_files[0], comment='#', skipinitialspace=True)

    # Load photons
    photons_dir = results_dir / "ExportedPhotons"
    if photons_dir.exists():
        photon_files = list(photons_dir.glob("*.csv"))
        if photon_files:
            results['photons'] = pd.read_csv(photon_files[0], comment='#', skipinitialspace=True)

    # Load events
    events_dir = results_dir / "ExportedEvents"
    if events_dir.exists():
        event_files = list(events_dir.glob("*.csv"))
        if event_files:
            results['events'] = pd.read_csv(event_files[0], comment='#', skipinitialspace=True)

    return results


def print_results(test_name: str, input_pixels: list, results: dict):
    """Print test results in a readable format."""
    print(f"\n{'='*70}")
    print(f"TEST: {test_name}")
    print(f"{'='*70}")

    print("\n--- INPUT PIXELS ---")
    input_df = create_traced_photons_df(input_pixels)
    print(input_df[['pixel_x', 'pixel_y', 'toa2', 'time_diff', 'id', 'neutron_id']].to_string(index=False))

    if 'pixels' in results:
        print("\n--- EMPIR EXPORTED PIXELS ---")
        # Clean column names (remove leading spaces)
        results['pixels'].columns = results['pixels'].columns.str.strip()
        print(results['pixels'].to_string(index=False))

    if 'photons' in results:
        print("\n--- EMPIR EXPORTED PHOTONS ---")
        results['photons'].columns = results['photons'].columns.str.strip()
        print(results['photons'].to_string(index=False))

    if 'events' in results:
        print("\n--- EMPIR EXPORTED EVENTS ---")
        results['events'].columns = results['events'].columns.str.strip()
        print(results['events'].to_string(index=False))


# =============================================================================
# TEST CASES
# =============================================================================

class TestCoGCalculation(unittest.TestCase):
    """Test 1: Verify CoG calculation is ToT-weighted."""

    def test_cog_calculation(self):
        """
        Create pixels with different ToT values at symmetric positions.
        If ToT-weighted: CoG will be pulled toward higher ToT pixel.
        If simple mean: CoG will be at geometric center.
        """
        test_name = "test_cog_calculation"
        test_dir = create_test_directory(test_name)

        # Two pixels: (100, 100) with ToT=100, (102, 100) with ToT=200
        # Simple mean: x = 101
        # ToT-weighted: x = (100*100 + 102*200) / 300 = 30400/300 ≈ 101.33
        pixels = [
            {'x': 100, 'y': 100, 'toa_ns': 1000, 'tot_ns': 100, 'photon_id': 1, 'neutron_id': 0},
            {'x': 102, 'y': 100, 'toa_ns': 1010, 'tot_ns': 200, 'photon_id': 1, 'neutron_id': 0},
        ]

        df = create_traced_photons_df(pixels)
        save_traced_photons(df, test_dir)

        results_dir = process_with_empir(test_dir, "fast_neutrons")
        results = load_empir_results(results_dir)

        print_results(test_name, pixels, results)

        # Analysis
        if 'photons' in results and len(results['photons']) > 0:
            photon_x = results['photons']['x'].iloc[0]
            simple_mean = 101.0
            tot_weighted = (100*100 + 102*200) / 300

            print(f"\n--- ANALYSIS ---")
            print(f"Photon X position: {photon_x:.4f}")
            print(f"Simple mean would be: {simple_mean:.4f}")
            print(f"ToT-weighted mean would be: {tot_weighted:.4f}")

            # Test assertion: should be closer to ToT-weighted
            diff_simple = abs(photon_x - simple_mean)
            diff_weighted = abs(photon_x - tot_weighted)

            if diff_weighted < diff_simple:
                print("RESULT: CoG appears to be ToT-weighted ✓")
            else:
                print("RESULT: CoG appears to be simple mean")

            self.assertLess(diff_weighted, diff_simple,
                           "CoG should be ToT-weighted (closer to weighted mean)")


class TestTOASelection(unittest.TestCase):
    """Test 2: Verify which pixel's TOA becomes the photon TOA."""

    def test_toa_selection(self):
        """
        Create pixels with clearly different TOA values.
        Check if photon TOA = min(pixel TOA) in cluster.
        """
        test_name = "test_toa_selection"
        test_dir = create_test_directory(test_name)

        # Three pixels at nearby positions with different TOAs
        pixels = [
            {'x': 100, 'y': 100, 'toa_ns': 1050, 'tot_ns': 50, 'photon_id': 1, 'neutron_id': 0},
            {'x': 101, 'y': 100, 'toa_ns': 1000, 'tot_ns': 50, 'photon_id': 1, 'neutron_id': 0},  # earliest
            {'x': 100, 'y': 101, 'toa_ns': 1100, 'tot_ns': 50, 'photon_id': 1, 'neutron_id': 0},
        ]

        df = create_traced_photons_df(pixels)
        save_traced_photons(df, test_dir)

        results_dir = process_with_empir(test_dir, "fast_neutrons")
        results = load_empir_results(results_dir)

        print_results(test_name, pixels, results)

        if 'photons' in results and len(results['photons']) > 0:
            photon_toa = results['photons']['toa'].iloc[0]
            min_toa = 1000 / 1e9  # Convert to seconds

            print(f"\n--- ANALYSIS ---")
            print(f"Photon TOA: {photon_toa:.12e} s")
            print(f"Min pixel TOA: {min_toa:.12e} s")
            print(f"Difference: {abs(photon_toa - min_toa)*1e9:.4f} ns")

            # Within 10 ns tolerance
            self.assertAlmostEqual(photon_toa, min_toa, delta=10e-9,
                                  msg="Photon TOA should equal min(pixel TOA)")
            print("RESULT: Photon TOA = min(pixel TOA) - CONFIRMED ✓")


class TestDistanceMetric(unittest.TestCase):
    """Test 3: Verify distance metric is Euclidean (not Manhattan)."""

    def test_distance_metric(self):
        """
        Create two pixels at diagonal positions.
        With dSpace=2: Euclidean dist = sqrt(2) ≈ 1.41, Manhattan dist = 2
        If Euclidean: pixels should cluster into 1 photon
        """
        test_name = "test_distance_metric"
        test_dir = create_test_directory(test_name)

        # Two pixels diagonal: (100,100) and (101,101)
        pixels = [
            {'x': 100, 'y': 100, 'toa_ns': 1000, 'tot_ns': 50, 'photon_id': 1, 'neutron_id': 0},
            {'x': 101, 'y': 101, 'toa_ns': 1010, 'tot_ns': 50, 'photon_id': 1, 'neutron_id': 0},
        ]

        df = create_traced_photons_df(pixels)
        save_traced_photons(df, test_dir)

        results_dir = process_with_empir(test_dir, "fast_neutrons")
        results = load_empir_results(results_dir)

        print_results(test_name, pixels, results)

        if 'photons' in results:
            n_photons = len(results['photons'])
            print(f"\n--- ANALYSIS ---")
            print(f"Number of photons created: {n_photons}")
            print(f"Euclidean distance: {np.sqrt(2):.4f} (should cluster if < dSpace=2)")
            print(f"Manhattan distance: 2 (at boundary)")

            self.assertEqual(n_photons, 1,
                           "Diagonal pixels should cluster (Euclidean distance used)")
            print("RESULT: Pixels clustered → Euclidean distance used ✓")


class TestTimeWindow(unittest.TestCase):
    """Test 5: Test dTime parameter behavior."""

    def test_time_window(self):
        """
        fast_neutrons: dTime = 50 ns
        Create pixels within and outside this window.
        """
        test_name = "test_time_window"
        test_dir = create_test_directory(test_name)

        # Pixel 1 at t=1000ns
        # Pixel 2 at t=1030ns (within 50ns window)
        # Pixel 3 at t=1100ns (outside 50ns window from pixel 1)
        pixels = [
            {'x': 100, 'y': 100, 'toa_ns': 1000, 'tot_ns': 50, 'photon_id': 1, 'neutron_id': 0},
            {'x': 101, 'y': 100, 'toa_ns': 1030, 'tot_ns': 50, 'photon_id': 1, 'neutron_id': 0},
            {'x': 102, 'y': 100, 'toa_ns': 1100, 'tot_ns': 50, 'photon_id': 2, 'neutron_id': 0},
        ]

        df = create_traced_photons_df(pixels)
        save_traced_photons(df, test_dir)

        results_dir = process_with_empir(test_dir, "fast_neutrons")
        results = load_empir_results(results_dir)

        print_results(test_name, pixels, results)

        if 'photons' in results:
            n_photons = len(results['photons'])
            print(f"\n--- ANALYSIS ---")
            print(f"dTime = 50 ns (fast_neutrons preset)")
            print(f"Pixel 1-2 time diff: 30 ns (should cluster)")
            print(f"Pixel 1-3 time diff: 100 ns (should NOT cluster)")
            print(f"Number of photons: {n_photons}")

            # With nPxMin=2, we expect photons only from clusters >= 2 pixels
            print(f"Note: nPxMin=2 for fast_neutrons, so single pixel won't form photon")


class TestNPxMin(unittest.TestCase):
    """Test 6: Test nPxMin parameter (minimum pixels per photon)."""

    def test_npxmin(self):
        """
        fast_neutrons: nPxMin = 2
        Create a single isolated pixel - should NOT form a photon.
        """
        test_name = "test_npxmin"
        test_dir = create_test_directory(test_name)

        # Single pixel - should not form photon with nPxMin=2
        pixels = [
            {'x': 100, 'y': 100, 'toa_ns': 1000, 'tot_ns': 50, 'photon_id': 1, 'neutron_id': 0},
        ]

        df = create_traced_photons_df(pixels)
        save_traced_photons(df, test_dir)

        results_dir = process_with_empir(test_dir, "fast_neutrons")
        results = load_empir_results(results_dir)

        print_results(test_name, pixels, results)

        if 'photons' in results:
            n_photons = len(results['photons'])
            print(f"\n--- ANALYSIS ---")
            print(f"nPxMin = 2 (fast_neutrons preset)")
            print(f"Input: 1 pixel")
            print(f"Output photons: {n_photons}")

            self.assertEqual(n_photons, 0,
                           "Single pixel should not form photon with nPxMin=2")
            print("RESULT: Single pixel correctly rejected (nPxMin=2) ✓")


class TestEMPIRvsNEA(unittest.TestCase):
    """Test: Compare EMPIR results with nea.Analyse association methods."""

    def test_empir_vs_nea_simple(self):
        """Compare EMPIR association with nea simple method."""
        test_name = "test_empir_vs_nea"
        test_dir = create_test_directory(test_name)

        # Create a simple cluster
        pixels = [
            {'x': 100, 'y': 100, 'toa_ns': 1000, 'tot_ns': 100, 'photon_id': 1, 'neutron_id': 0},
            {'x': 101, 'y': 100, 'toa_ns': 1005, 'tot_ns': 100, 'photon_id': 1, 'neutron_id': 0},
            {'x': 100, 'y': 101, 'toa_ns': 1010, 'tot_ns': 100, 'photon_id': 1, 'neutron_id': 0},
        ]

        df = create_traced_photons_df(pixels)
        save_traced_photons(df, test_dir)

        # Process with EMPIR
        results_dir = process_with_empir(test_dir, "fast_neutrons")
        empir_results = load_empir_results(results_dir)

        print_results(test_name, pixels, empir_results)

        # Now load with nea.Analyse and run association
        analyser = nea.Analyse(
            str(results_dir),
            verbosity=2,
            pixels=True, photons=True, events=True
        )

        print(f"\n--- NEA.ANALYSE LOADED DATA ---")
        print(f"Pixels: {len(analyser.pixels_df) if analyser.pixels_df is not None else 0}")
        print(f"Photons: {len(analyser.photons_df) if analyser.photons_df is not None else 0}")
        print(f"Events: {len(analyser.events_df) if analyser.events_df is not None else 0}")

        # Run association
        if analyser.pixels_df is not None and len(analyser.pixels_df) > 0:
            analyser.associate(method='simple', pixel_max_dist_px=5, pixel_max_time_ns=50)

            print(f"\n--- NEA ASSOCIATION RESULTS ---")
            # Check associated_df for results
            if hasattr(analyser, 'associated_df') and analyser.associated_df is not None:
                assoc_df = analyser.associated_df
                print(f"Associated DataFrame columns: {assoc_df.columns.tolist()}")
                print(f"Rows in associated_df: {len(assoc_df)}")

                if 'ph/id' in assoc_df.columns:
                    matched_count = (assoc_df['ph/id'] >= 0).sum()
                    print(f"Pixels with ph/id assigned: {matched_count}")

                    # Get the computed photon CoG from nea
                    if 'ph/x' in assoc_df.columns and 'ph/y' in assoc_df.columns:
                        nea_photon_x = assoc_df['ph/x'].iloc[0]
                        nea_photon_y = assoc_df['ph/y'].iloc[0]
                        print(f"NEA photon position: ({nea_photon_x:.4f}, {nea_photon_y:.4f})")

            # Compare EMPIR photon position with expected
            if 'photons' in empir_results and len(empir_results['photons']) > 0:
                empir_x = empir_results['photons']['x'].iloc[0]
                empir_y = empir_results['photons']['y'].iloc[0]

                print(f"\nEMPIR photon position: ({empir_x:.4f}, {empir_y:.4f})")

                # Calculate expected ToT-weighted CoG from input pixels
                tot_sum = sum(p['tot_ns'] for p in pixels)
                expected_x = sum(p['x'] * p['tot_ns'] for p in pixels) / tot_sum
                expected_y = sum(p['y'] * p['tot_ns'] for p in pixels) / tot_sum
                print(f"Expected ToT-weighted CoG: ({expected_x:.4f}, {expected_y:.4f})")


def run_all_tests():
    """Run all test cases."""
    print("\n" + "="*70)
    print("EMPIR ALGORITHM INVESTIGATION TEST SUITE")
    print("="*70)

    # Run tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestCoGCalculation))
    suite.addTests(loader.loadTestsFromTestCase(TestTOASelection))
    suite.addTests(loader.loadTestsFromTestCase(TestDistanceMetric))
    suite.addTests(loader.loadTestsFromTestCase(TestTimeWindow))
    suite.addTests(loader.loadTestsFromTestCase(TestNPxMin))
    suite.addTests(loader.loadTestsFromTestCase(TestEMPIRvsNEA))

    # Run with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == "__main__":
    run_all_tests()
