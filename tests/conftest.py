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
    2. CSV files are already correctly named and formatted in the source data
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy structure
        temp_neutrons = os.path.join(tmpdir, "neutrons")
        shutil.copytree(test_data_dir, temp_neutrons)

        # CSV files in source data are already correctly named and formatted
        # - traced_data_0.csv in ExportedPhotons matches traced_data_0.empirphot
        # - traced_data_0.csv in ExportedEvents matches traced_data_0.empirevent

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
