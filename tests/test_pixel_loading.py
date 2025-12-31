"""
Tests for pixel loading functionality in neutron_event_analyzer.

This module tests the ability to load pixel data from TPX3 files or pre-exported CSV files,
and verify the data structure and integration with photon/event loading.
"""

import os
import pytest
import pandas as pd
import numpy as np

# Import the package
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import neutron_event_analyzer as nea


class TestPixelLoading:
    """Test suite for pixel file loading functionality."""

    def test_load_pixels_only(self, temp_data_dir):
        """Test loading only pixel data."""
        analyser = nea.Analyse(data_folder=temp_data_dir, n_threads=1)
        analyser.load(load_events=False, load_photons=False, load_pixels=True, verbosity=1)

        assert analyser.pixels_df is not None, "Pixels dataframe should not be None"
        assert len(analyser.pixels_df) > 0, "Pixels dataframe should contain data"

        # Verify column structure
        expected_pixel_cols = ['x', 'y', 't', 'tot', 'tof']
        assert list(analyser.pixels_df.columns) == expected_pixel_cols, \
            f"Pixel columns should be {expected_pixel_cols}"

    def test_load_pixels_with_photons(self, temp_data_dir):
        """Test loading pixels and photons together."""
        analyser = nea.Analyse(data_folder=temp_data_dir, n_threads=1)
        analyser.load(load_events=False, load_photons=True, load_pixels=True, verbosity=1)

        assert analyser.pixels_df is not None, "Pixels dataframe should not be None"
        assert analyser.photons_df is not None, "Photons dataframe should not be None"
        assert len(analyser.pixels_df) > 0, "Pixels dataframe should contain data"
        assert len(analyser.photons_df) > 0, "Photons dataframe should contain data"

        # Pixels should be more numerous than photons (clustering effect)
        assert len(analyser.pixels_df) >= len(analyser.photons_df), \
            "Pixels should be more numerous than photons"

    def test_load_all_three_types(self, temp_data_dir):
        """Test loading pixels, photons, and events together."""
        analyser = nea.Analyse(data_folder=temp_data_dir, n_threads=1)
        analyser.load(load_events=True, load_photons=True, load_pixels=True, verbosity=1)

        assert analyser.pixels_df is not None, "Pixels dataframe should not be None"
        assert analyser.photons_df is not None, "Photons dataframe should not be None"
        assert analyser.events_df is not None, "Events dataframe should not be None"
        assert len(analyser.pixels_df) > 0
        assert len(analyser.photons_df) > 0
        assert len(analyser.events_df) > 0

    def test_pixel_data_types(self, temp_data_dir):
        """Test that loaded pixel data has correct data types."""
        analyser = nea.Analyse(data_folder=temp_data_dir, n_threads=1)
        analyser.load(load_events=False, load_photons=False, load_pixels=True)

        assert analyser.pixels_df['x'].dtype == float
        assert analyser.pixels_df['y'].dtype == float
        assert analyser.pixels_df['t'].dtype == float
        # tot and tof can be float or int, just check they're numeric
        assert np.issubdtype(analyser.pixels_df['tot'].dtype, np.number)
        assert np.issubdtype(analyser.pixels_df['tof'].dtype, np.number)

    def test_pixel_loading_with_limit(self, temp_data_dir):
        """Test loading pixels with row limit."""
        analyser = nea.Analyse(data_folder=temp_data_dir, n_threads=1)
        analyser.load(load_events=False, load_photons=False, load_pixels=True, limit=100)

        assert len(analyser.pixels_df) <= 100, "Pixels should be limited to 100 rows"


class TestPixelPhotonAssociation:
    """Test suite for pixel-photon association."""

    def test_associate_pixels_to_photons(self, temp_data_dir):
        """Test basic pixel-photon association."""
        analyser = nea.Analyse(data_folder=temp_data_dir, n_threads=1)
        analyser.load(events=False, photons=True, pixels=True, verbosity=0)

        # Perform 2-tier association (pixels → photons)
        result_df = analyser.associate(
            pixel_max_dist_px=10.0,
            pixel_max_time_ns=1000,
            verbosity=1
        )

        assert result_df is not None, "Association result should not be None"
        assert len(result_df) > 0, "Association result should contain data"

        # Check for pixel-photon association columns
        assert 'assoc_photon_id' in result_df.columns
        assert 'assoc_phot_x' in result_df.columns
        assert 'assoc_phot_y' in result_df.columns
        assert 'assoc_phot_t' in result_df.columns
        assert 'pixel_time_diff_ns' in result_df.columns
        assert 'pixel_spatial_diff_px' in result_df.columns

        # Some pixels should be associated
        associated_count = result_df['assoc_photon_id'].notna().sum()
        assert associated_count > 0, "Some pixels should be associated to photons"

    def test_full_three_tier_association(self, temp_data_dir):
        """Test full three-tier association: pixels → photons → events."""
        analyser = nea.Analyse(data_folder=temp_data_dir, n_threads=1)
        analyser.load(events=True, photons=True, pixels=True, verbosity=0)

        # Perform 3-tier association
        result_df = analyser.associate(
            pixel_max_dist_px=10.0,
            pixel_max_time_ns=1000,
            photon_dSpace_px=50.0,
            max_time_ns=500,
            verbosity=1,
            method='simple'
        )

        assert result_df is not None
        assert len(result_df) > 0

        # Should have both pixel-photon and photon-event association columns
        assert 'assoc_photon_id' in result_df.columns  # Pixel → Photon
        assert 'assoc_event_id' in result_df.columns  # Photon → Event

        # Some pixels should be associated through the full chain
        pixels_with_events = result_df['assoc_event_id'].notna().sum()
        if pixels_with_events > 0:
            print(f"✅ {pixels_with_events} pixels associated through full chain to events")


class TestSaveAssociations:
    """Test suite for saving association results."""

    def test_save_associations_csv(self, temp_data_dir, tmp_path):
        """Test saving association results as CSV."""
        analyser = nea.Analyse(data_folder=temp_data_dir, n_threads=1)
        analyser.load(photons=True, pixels=True, verbosity=0)
        analyser.associate(verbosity=0)

        # Save to temporary directory
        output_path = analyser.save_associations(
            output_dir=str(tmp_path),
            filename="test_associations.csv",
            verbosity=1
        )

        assert os.path.exists(output_path), "Output file should exist"

        # Verify we can read it back
        df_loaded = pd.read_csv(output_path)
        assert len(df_loaded) == len(analyser.associated_df)
        assert list(df_loaded.columns) == list(analyser.associated_df.columns)

    def test_save_without_association_raises_error(self, temp_data_dir):
        """Test that saving without association raises an error."""
        analyser = nea.Analyse(data_folder=temp_data_dir, n_threads=1)
        analyser.load(load_photons=True, verbosity=0)

        with pytest.raises(ValueError, match="No association data to save"):
            analyser.save_associations()
