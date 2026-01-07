"""
Tests for the parameter suggester module.

This module tests the ParameterSuggester class which analyzes
association results and suggests improved parameters.
"""

import sys
import tempfile
import pytest
import pandas as pd
from pathlib import Path

# Add tests directory to path
sys.path.insert(0, str(Path(__file__).parent))

import neutron_event_analyzer as nea
from neutron_event_analyzer.parameter_suggester import (
    ParameterSuggester,
    AssociationQualityMetrics,
    suggest_parameters_from_data
)
from test_association_validation import (
    create_synthetic_photon_data,
    create_synthetic_event_data,
    write_csv_files
)


@pytest.fixture
def basic_scenario():
    """Create a basic test scenario with synthetic data."""
    event_configs = [
        {
            'event_id': 0,
            'center_x': 100.0,
            'center_y': 100.0,
            't_ns': 1000.0,
            'n_photons': 8,
            'photon_spread_spatial': 15.0,
            'photon_spread_temporal': 50.0
        },
        {
            'event_id': 1,
            'center_x': 200.0,
            'center_y': 150.0,
            't_ns': 10000.0,
            'n_photons': 8,
            'photon_spread_spatial': 15.0,
            'photon_spread_temporal': 50.0
        }
    ]

    photon_df = create_synthetic_photon_data(event_configs)
    event_df = create_synthetic_event_data(event_configs)

    return {
        'photons': photon_df,
        'events': event_df,
        'configs': event_configs
    }


class TestParameterSuggester:
    """Test the ParameterSuggester class."""

    def test_analyze_quality(self, basic_scenario):
        """Test quality analysis on synthetic data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir)

            # Write data
            write_csv_files(
                None,
                basic_scenario['photons'],
                basic_scenario['events'],
                data_path,
                file_index=0
            )

            # Load and associate
            analyser = nea.Analyse(data_folder=str(data_path))
            analyser.load(verbosity=0)
            analyser.associate_photons_events(
                method='simple',
                dSpace_px=20.0,
                max_time_ns=100.0
            )

            # Analyze quality
            suggester = ParameterSuggester(analyser, verbosity=0)
            metrics = suggester.analyze_quality()

            # Verify metrics structure
            assert isinstance(metrics, AssociationQualityMetrics)
            assert metrics.total_photons > 0
            assert 0 <= metrics.association_rate <= 1.0
            assert metrics.total_events >= 0
            assert metrics.mean_photons_per_event >= 0

    def test_suggest_parameters(self, basic_scenario):
        """Test parameter suggestion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir)

            write_csv_files(
                None,
                basic_scenario['photons'],
                basic_scenario['events'],
                data_path,
                file_index=0
            )

            analyser = nea.Analyse(data_folder=str(data_path))
            analyser.load(verbosity=0)
            analyser.associate_photons_events(
                method='simple',
                dSpace_px=20.0,
                max_time_ns=100.0
            )

            suggester = ParameterSuggester(analyser, verbosity=0)
            suggester.analyze_quality()
            suggestion = suggester.suggest_parameters(
                current_spatial_px=20.0,
                current_temporal_ns=100.0
            )

            # Verify suggestion structure
            assert suggestion.current_spatial_px == 20.0
            assert suggestion.current_temporal_ns == 100.0
            assert suggestion.suggested_spatial_px > 0
            assert suggestion.suggested_temporal_ns > 0
            assert len(suggestion.reasoning) > 0
            assert suggestion.confidence in ['low', 'medium', 'high']

    def test_save_suggested_parameters(self, basic_scenario):
        """Test saving suggested parameters to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "data"
            data_path.mkdir()
            output_path = Path(tmpdir) / "suggested_params.json"

            write_csv_files(
                None,
                basic_scenario['photons'],
                basic_scenario['events'],
                data_path,
                file_index=0
            )

            analyser = nea.Analyse(data_folder=str(data_path))
            analyser.load(verbosity=0)
            analyser.associate_photons_events(
                method='simple',
                dSpace_px=20.0,
                max_time_ns=100.0
            )

            suggester = ParameterSuggester(analyser, verbosity=0)
            suggester.analyze_quality()
            suggester.suggest_parameters(20.0, 100.0)
            suggester.save_suggested_parameters(str(output_path))

            # Verify file was created and has correct format
            assert output_path.exists()

            import json
            with open(output_path) as f:
                params = json.load(f)

            assert 'photon2event' in params
            assert 'dSpace_px' in params['photon2event']
            assert 'dTime_s' in params['photon2event']

    def test_convenience_function(self, basic_scenario):
        """Test the convenience function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "data"
            output_path = Path(tmpdir) / "params.json"
            data_path.mkdir()

            write_csv_files(
                None,
                basic_scenario['photons'],
                basic_scenario['events'],
                data_path,
                file_index=0
            )

            # Use convenience function
            suggestion = suggest_parameters_from_data(
                data_folder=str(data_path),
                current_spatial_px=20.0,
                current_temporal_ns=100.0,
                output_path=str(output_path),
                verbosity=0
            )

            assert suggestion is not None
            assert output_path.exists()


class TestQualityMetrics:
    """Test quality metrics calculation."""

    def test_tight_clustering_detection(self):
        """Test that tight clustering is detected."""
        event_configs = [
            {
                'event_id': 0,
                'center_x': 100.0,
                'center_y': 100.0,
                't_ns': 1000.0,
                'n_photons': 10,
                'photon_spread_spatial': 5.0,  # Very tight
                'photon_spread_temporal': 20.0
            }
        ]

        photon_df = create_synthetic_photon_data(event_configs)
        event_df = create_synthetic_event_data(event_configs)

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir)
            write_csv_files(None, photon_df, event_df, data_path, file_index=0)

            analyser = nea.Analyse(data_folder=str(data_path))
            analyser.load(verbosity=0)
            analyser.associate_photons_events(
                method='simple',
                dSpace_px=50.0,  # Very loose parameters
                max_time_ns=500.0
            )

            suggester = ParameterSuggester(analyser, verbosity=0)
            metrics = suggester.analyze_quality()

            # Should have small spatial spread
            assert metrics.median_spatial_spread < 10.0

    def test_loose_clustering_detection(self):
        """Test that loose clustering is detected."""
        event_configs = [
            {
                'event_id': 0,
                'center_x': 100.0,
                'center_y': 100.0,
                't_ns': 1000.0,
                'n_photons': 10,
                'photon_spread_spatial': 50.0,  # Very loose
                'photon_spread_temporal': 200.0
            }
        ]

        photon_df = create_synthetic_photon_data(event_configs)
        event_df = create_synthetic_event_data(event_configs)

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir)
            write_csv_files(None, photon_df, event_df, data_path, file_index=0)

            analyser = nea.Analyse(data_folder=str(data_path))
            analyser.load(verbosity=0)
            analyser.associate_photons_events(
                method='simple',
                dSpace_px=100.0,
                max_time_ns=500.0
            )

            suggester = ParameterSuggester(analyser, verbosity=0)
            metrics = suggester.analyze_quality()

            # Should have large spatial spread
            assert metrics.median_spatial_spread > 20.0

    def test_under_association_detection(self):
        """Test detection of under-association (parameters too tight)."""
        event_configs = [
            {
                'event_id': 0,
                'center_x': 100.0,
                'center_y': 100.0,
                't_ns': 1000.0,
                'n_photons': 10,
                'photon_spread_spatial': 30.0,
                'photon_spread_temporal': 100.0
            }
        ]

        photon_df = create_synthetic_photon_data(event_configs)
        event_df = create_synthetic_event_data(event_configs)

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir)
            write_csv_files(None, photon_df, event_df, data_path, file_index=0)

            analyser = nea.Analyse(data_folder=str(data_path))
            analyser.load(verbosity=0)
            analyser.associate_photons_events(
                method='simple',
                dSpace_px=5.0,  # Very tight - will miss associations
                max_time_ns=10.0
            )

            suggester = ParameterSuggester(analyser, verbosity=0)
            metrics = suggester.analyze_quality()

            # Should detect under-association
            assert metrics.association_rate < 0.6 or metrics.likely_under_associated


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
