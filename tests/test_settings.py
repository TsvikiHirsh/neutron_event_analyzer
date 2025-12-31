"""
Tests for settings loading functionality in neutron_event_analyzer.
"""

import os
import pytest
import json
import tempfile

# Import the package
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import neutron_event_analyzer as nea


class TestSettingsLoading:
    """Test suite for settings file/dict loading."""

    def test_load_settings_from_dict(self):
        """Test loading settings from dictionary."""
        settings = {
            "pixel2photon": {
                "dSpace": 10.0,
                "dTime": 1000e-09,
            },
            "photon2event": {
                "dSpace_px": 50.0,
                "dTime_s": 500e-09,
            }
        }

        analyser = nea.Analyse(data_folder='./tests/data/neutrons', settings=settings)

        assert analyser.settings == settings

        # Test parameter extraction
        defaults = analyser._get_association_defaults()
        assert defaults['pixel_max_dist_px'] == 10.0
        assert defaults['pixel_max_time_ns'] == 1000.0  # converted to ns
        assert defaults['photon_dSpace_px'] == 50.0
        assert defaults['max_time_ns'] == 500.0  # converted to ns

    def test_load_settings_from_json_file(self):
        """Test loading settings from JSON file."""
        settings = {
            "pixel2photon": {
                "dSpace": 5.0,
                "dTime": 200e-09,
            },
            "photon2event": {
                "dSpace_px": 25.0,
                "dTime_s": 100e-09,
            }
        }

        # Create temporary JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(settings, f)
            temp_path = f.name

        try:
            analyser = nea.Analyse(data_folder='./tests/data/neutrons', settings=temp_path)

            assert analyser.settings == settings

            defaults = analyser._get_association_defaults()
            assert defaults['pixel_max_dist_px'] == 5.0
            assert defaults['pixel_max_time_ns'] == 200.0
            assert defaults['photon_dSpace_px'] == 25.0
            assert defaults['max_time_ns'] == 100.0
        finally:
            os.unlink(temp_path)

    def test_settings_none(self):
        """Test that None settings works (no settings)."""
        analyser = nea.Analyse(data_folder='./tests/data/neutrons', settings=None)

        assert analyser.settings == {}
        defaults = analyser._get_association_defaults()
        assert defaults == {}

    def test_settings_missing_file(self):
        """Test handling of missing settings file."""
        analyser = nea.Analyse(data_folder='./tests/data/neutrons', settings='/nonexistent/file.json')

        assert analyser.settings == {}

    def test_settings_invalid_type(self):
        """Test handling of invalid settings type."""
        analyser = nea.Analyse(data_folder='./tests/data/neutrons', settings=123)

        assert analyser.settings == {}

    def test_partial_settings(self):
        """Test that partial settings work correctly."""
        settings = {
            "pixel2photon": {
                "dSpace": 7.0,
            }
        }

        analyser = nea.Analyse(data_folder='./tests/data/neutrons', settings=settings)

        defaults = analyser._get_association_defaults()
        assert defaults['pixel_max_dist_px'] == 7.0
        assert 'pixel_max_time_ns' not in defaults
        assert 'photon_dSpace_px' not in defaults


class TestSettingsInAssociation:
    """Test that settings are used in association methods."""

    def test_associate_full_uses_settings(self, temp_data_dir):
        """Test that associate uses settings defaults."""
        settings = {
            "pixel2photon": {
                "dSpace": 15.0,
                "dTime": 2000e-09,
            },
            "photon2event": {
                "dSpace_px": 60.0,
                "dTime_s": 600e-09,
            }
        }

        analyser = nea.Analyse(data_folder=temp_data_dir, settings=settings, n_threads=1)
        analyser.load(pixels=True, photons=True, events=True, verbosity=0)

        # Call associate without parameters - should use settings
        result = analyser.associate(verbosity=0)

        # Just verify it runs without error and returns results
        assert result is not None
        assert len(result) > 0

    def test_associate_full_override_settings(self, temp_data_dir):
        """Test that explicit parameters override settings."""
        settings = {
            "pixel2photon": {
                "dSpace": 15.0,
                "dTime": 2000e-09,
            }
        }

        analyser = nea.Analyse(data_folder=temp_data_dir, settings=settings, n_threads=1)
        analyser.load(pixels=True, photons=True, events=True, verbosity=0)

        # Call with explicit parameter - should override settings
        result = analyser.associate(pixel_max_dist_px=5.0, verbosity=0)

        assert result is not None
        assert len(result) > 0
