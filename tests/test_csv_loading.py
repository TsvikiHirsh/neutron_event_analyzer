"""
Tests for CSV loading functionality in neutron_event_analyzer.

This module tests the ability to load data from pre-exported CSV files
without requiring empir binaries, as well as fallback to empir when needed.
"""

import os
import pytest
import pandas as pd
import numpy as np

# Import the package
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import neutron_event_analyzer as nea


class TestCSVLoading:
    """Test suite for CSV file loading functionality."""

    def test_load_with_existing_csv_files(self, temp_data_dir):
        """Test that the analyzer loads from CSV files when they exist."""
        # Initialize analyzer without export_dir (empir not needed)
        analyser = nea.Analyse(
            data_folder=temp_data_dir,
            export_dir="./nonexistent",  # Intentionally non-existent
            n_threads=1
        )

        # Load data - should use CSV files
        analyser.load()

        # Verify data was loaded
        assert analyser.events_df is not None, "Events dataframe should not be None"
        assert analyser.photons_df is not None, "Photons dataframe should not be None"
        assert len(analyser.events_df) > 0, "Events dataframe should contain data"
        assert len(analyser.photons_df) > 0, "Photons dataframe should contain data"

        # Verify column structure
        expected_event_cols = ['x', 'y', 't', 'n', 'PSD', 'tof']
        assert list(analyser.events_df.columns) == expected_event_cols, \
            f"Event columns should be {expected_event_cols}"

        expected_photon_cols = ['x', 'y', 't', 'tof']
        assert list(analyser.photons_df.columns) == expected_photon_cols, \
            f"Photon columns should be {expected_photon_cols}"

    def test_csv_data_types(self, temp_data_dir):
        """Test that loaded CSV data has correct data types."""
        analyser = nea.Analyse(data_folder=temp_data_dir, n_threads=1)
        analyser.load()

        # Check event data types
        assert analyser.events_df['x'].dtype == float
        assert analyser.events_df['y'].dtype == float
        assert analyser.events_df['t'].dtype == float
        assert analyser.events_df['PSD'].dtype == float
        assert analyser.events_df['tof'].dtype == float

        # Check photon data types
        assert analyser.photons_df['x'].dtype == float
        assert analyser.photons_df['y'].dtype == float
        assert analyser.photons_df['t'].dtype == float

    def test_association_with_csv_data(self, temp_data_dir):
        """Test that photon-event association works with CSV-loaded data."""
        analyser = nea.Analyse(data_folder=temp_data_dir, n_threads=1)
        analyser.load()

        # Perform association
        analyser.associate(
            time_norm_ns=1.0,
            spatial_norm_px=5.0,
            dSpace_px=50.0,
            max_time_ns=500,
            verbosity=0,
            method='simple'
        )

        # Verify association results
        assert analyser.associated_df is not None, "Associated dataframe should not be None"
        assert len(analyser.associated_df) > 0, "Associated dataframe should contain data"

        # Check for association columns
        assert 'assoc_event_id' in analyser.associated_df.columns
        assert 'assoc_x' in analyser.associated_df.columns
        assert 'assoc_y' in analyser.associated_df.columns
        assert 'assoc_t' in analyser.associated_df.columns

    def test_load_with_query_filter(self, temp_data_dir):
        """Test loading data with query filtering."""
        analyser = nea.Analyse(data_folder=temp_data_dir, n_threads=1)

        # Load with query filter
        analyser.load(query="n >= 1")

        # All events should pass this filter (n >= 1)
        assert len(analyser.events_df) > 0, "Should have events with n >= 1"
        assert all(analyser.events_df['n'] >= 1), "All events should have n >= 1"

    def test_load_with_limit(self, temp_data_dir):
        """Test loading data with row limit."""
        analyser = nea.Analyse(data_folder=temp_data_dir, n_threads=1)

        # Load with limit
        analyser.load(limit=5)

        # Should respect the limit
        assert len(analyser.events_df) <= 5, "Events should be limited to 5 rows"
        assert len(analyser.photons_df) <= 5, "Photons should be limited to 5 rows"

    def test_paired_files_matching(self, temp_data_dir):
        """Test that event and photon files are correctly paired by basename."""
        analyser = nea.Analyse(data_folder=temp_data_dir, n_threads=1)
        analyser.load()

        # Check that pair_files was populated correctly
        assert analyser.pair_files is not None, "Pair files should be identified"
        assert len(analyser.pair_files) > 0, "Should find at least one file pair"

        # Verify that basenames match
        for event_file, photon_file in analyser.pair_files:
            event_base = os.path.basename(event_file).rsplit('.', 1)[0]
            photon_base = os.path.basename(photon_file).rsplit('.', 1)[0]
            assert event_base == photon_base, \
                f"Basenames should match: {event_base} vs {photon_base}"

    def test_compute_ellipticity(self, temp_data_dir):
        """Test ellipticity computation on CSV-loaded data."""
        analyser = nea.Analyse(data_folder=temp_data_dir, n_threads=1)
        analyser.load()
        analyser.associate(
            dSpace_px=50.0,
            max_time_ns=500,
            verbosity=0,
            method='simple'
        )

        # Compute ellipticity
        analyser.compute_ellipticity(verbosity=0)

        # Check for ellipticity columns
        assert 'ellipticity' in analyser.associated_df.columns
        assert 'angle_deg' in analyser.associated_df.columns
        assert 'major_x' in analyser.associated_df.columns
        assert 'major_y' in analyser.associated_df.columns

    def test_get_combined_dataframe(self, temp_data_dir):
        """Test retrieving the combined dataframe."""
        analyser = nea.Analyse(data_folder=temp_data_dir, n_threads=1)
        analyser.load()
        analyser.associate(dSpace_px=50.0, verbosity=0, method='simple')

        # Get combined dataframe
        combined_df = analyser.get_combined_dataframe()

        assert combined_df is not None, "Combined dataframe should not be None"
        assert isinstance(combined_df, pd.DataFrame), "Should return a DataFrame"
        assert len(combined_df) > 0, "Combined dataframe should contain data"

    def test_multiple_association_methods(self, temp_data_dir):
        """Test different association methods with CSV-loaded data."""
        methods = ['kdtree', 'window', 'simple']

        for method in methods:
            analyser = nea.Analyse(data_folder=temp_data_dir, n_threads=1)
            analyser.load()

            # Test association with this method
            analyser.associate(
                time_norm_ns=1.0,
                spatial_norm_px=5.0,
                dSpace_px=50.0,
                max_time_ns=500,
                verbosity=0,
                method=method
            )

            assert analyser.associated_df is not None, \
                f"Association should work with method '{method}'"
            assert len(analyser.associated_df) > 0, \
                f"Should have associated data with method '{method}'"


class TestCSVFileFormat:
    """Test suite for CSV file format validation."""

    def test_event_csv_format(self, temp_data_dir):
        """Test that event CSV files have the correct format."""
        event_csv = os.path.join(temp_data_dir, "ExportedEvents", "traced_data_0.csv")
        assert os.path.exists(event_csv), "Event CSV should exist"

        df = pd.read_csv(event_csv)

        # Should have either empir format (' PSD value' column) or pre-processed format
        # The code now handles both formats
        required_cols = ['x', 'y', 't', 'n', 'PSD', 'tof']
        has_empir_format = ' PSD value' in df.columns
        has_preprocessed_format = all(col in df.columns for col in required_cols)

        assert has_empir_format or has_preprocessed_format, \
            f"Event CSV should have either empir format or pre-processed format. Got columns: {df.columns.tolist()}"

    def test_photon_csv_format(self, temp_data_dir):
        """Test that photon CSV files have the correct format."""
        photon_csv = os.path.join(temp_data_dir, "ExportedPhotons", "traced_data_0.csv")
        assert os.path.exists(photon_csv), "Photon CSV should exist"

        df = pd.read_csv(photon_csv)

        # After processing by conftest fixture, should have 't' not 'toa'
        assert 'x' in df.columns, "Photon CSV should have 'x' column"
        assert 'y' in df.columns, "Photon CSV should have 'y' column"
        assert 't' in df.columns, "Photon CSV should have 't' column"
        assert 'tof' in df.columns, "Photon CSV should have 'tof' column"
        assert 'toa' not in df.columns, "Photon CSV should not have 'toa' column"


class TestErrorHandling:
    """Test suite for error handling."""

    def test_missing_csv_and_missing_empir(self, temp_data_dir_no_csv):
        """Test error handling when CSV files are missing and empir is not available."""
        analyser = nea.Analyse(
            data_folder=temp_data_dir_no_csv,
            export_dir="/nonexistent/path",
            n_threads=1
        )

        # Load should fail gracefully
        analyser.load()

        # Should return empty dataframes when conversion fails
        assert analyser.events_df is not None
        assert analyser.photons_df is not None
        # Either empty or error during load should result in empty dfs
        assert len(analyser.events_df) == 0 or analyser.pair_dfs == []

    def test_get_combined_before_association(self, temp_data_dir):
        """Test that getting combined dataframe before association raises error."""
        analyser = nea.Analyse(data_folder=temp_data_dir, n_threads=1)
        analyser.load()

        # Should raise ValueError when trying to get combined df before association
        with pytest.raises(ValueError, match="Associate photons and events first"):
            analyser.get_combined_dataframe()


class TestDataIntegrity:
    """Test suite for data integrity checks."""

    def test_no_data_loss_during_load(self, temp_data_dir):
        """Test that loading from CSV preserves all data."""
        # Read CSV directly
        event_csv = os.path.join(temp_data_dir, "ExportedEvents", "traced_data_0.csv")
        photon_csv = os.path.join(temp_data_dir, "ExportedPhotons", "traced_data_0.csv")

        event_df_direct = pd.read_csv(event_csv).query("` PSD value` >= 0")
        photon_df_direct = pd.read_csv(photon_csv)

        # Load through analyzer
        analyser = nea.Analyse(data_folder=temp_data_dir, n_threads=1)
        analyser.load()

        # Compare row counts (should match after filtering)
        assert len(analyser.events_df) == len(event_df_direct), \
            "Event count should match direct CSV read"
        assert len(analyser.photons_df) == len(photon_df_direct), \
            "Photon count should match direct CSV read"

    def test_csv_filename_matching(self, temp_data_dir):
        """Test that CSV filenames match the binary file basenames."""
        event_files = os.listdir(os.path.join(temp_data_dir, "eventFiles"))
        photon_files = os.listdir(os.path.join(temp_data_dir, "photonFiles"))

        for event_file in event_files:
            if event_file.endswith('.empirevent'):
                basename = event_file.rsplit('.', 1)[0]
                csv_file = os.path.join(temp_data_dir, "ExportedEvents", f"{basename}.csv")
                assert os.path.exists(csv_file), \
                    f"CSV file should exist for {event_file}: {csv_file}"

        for photon_file in photon_files:
            if photon_file.endswith('.empirphot'):
                basename = photon_file.rsplit('.', 1)[0]
                csv_file = os.path.join(temp_data_dir, "ExportedPhotons", f"{basename}.csv")
                assert os.path.exists(csv_file), \
                    f"CSV file should exist for {photon_file}: {csv_file}"
