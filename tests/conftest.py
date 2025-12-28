"""Pytest configuration and fixtures for neutron_event_analyzer tests."""

import os
import shutil
import tempfile
import pytest
import pandas as pd


@pytest.fixture
def test_data_dir():
    """Return the path to the test data directory."""
    return os.path.join(os.path.dirname(__file__), "data", "neutrons")


@pytest.fixture
def temp_data_dir(test_data_dir):
    """
    Create a temporary copy of test data with properly formatted CSV files.

    This fixture:
    1. Copies the test data to a temp directory
    2. Fixes the photon CSV filename to match the binary file basename
    3. Fixes the photon CSV column name (toa -> t)
    4. Ensures event CSV has the correct format with ' PSD value' column
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy structure
        temp_neutrons = os.path.join(tmpdir, "neutrons")
        shutil.copytree(test_data_dir, temp_neutrons)

        # Fix photon CSV - rename file to match binary basename
        old_photon_csv = os.path.join(temp_neutrons, "ExportedPhotons", "exported_traced_data_0.csv")
        new_photon_csv = os.path.join(temp_neutrons, "ExportedPhotons", "traced_data_0.csv")

        if os.path.exists(old_photon_csv):
            # Read and fix column names
            df = pd.read_csv(old_photon_csv)
            if 'toa' in df.columns:
                df = df.rename(columns={'toa': 't'})
            df.to_csv(new_photon_csv, index=False)
            os.remove(old_photon_csv)

        # Event CSV should already be in correct format
        # No modification needed - the code now handles both formats

        yield temp_neutrons


@pytest.fixture
def temp_data_dir_no_csv(test_data_dir):
    """
    Create a temporary copy of test data WITHOUT CSV files.

    This fixture is used to test the fallback to empir binaries.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy structure
        temp_neutrons = os.path.join(tmpdir, "neutrons")
        shutil.copytree(test_data_dir, temp_neutrons)

        # Remove CSV files to force empir usage
        exported_events = os.path.join(temp_neutrons, "ExportedEvents")
        exported_photons = os.path.join(temp_neutrons, "ExportedPhotons")

        for csv_file in os.listdir(exported_events):
            os.remove(os.path.join(exported_events, csv_file))

        for csv_file in os.listdir(exported_photons):
            os.remove(os.path.join(exported_photons, csv_file))

        yield temp_neutrons


@pytest.fixture
def mock_export_dir():
    """Create a temporary directory for mock empir binaries."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir
