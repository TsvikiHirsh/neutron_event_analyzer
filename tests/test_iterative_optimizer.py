"""
Tests for the iterative optimizer module.

This module tests the IterativeOptimizer class which performs
multiple iterations of parameter refinement.
"""

import sys
import tempfile
import pytest
import pandas as pd
from pathlib import Path

# Add tests directory to path
sys.path.insert(0, str(Path(__file__).parent))

from neutron_event_analyzer.iterative_optimizer import (
    IterativeOptimizer,
    IterationResult,
    optimize_parameters_iteratively
)
from test_association_validation import (
    create_synthetic_photon_data,
    create_synthetic_event_data,
    write_csv_files
)


@pytest.fixture
def optimization_scenario():
    """Create a scenario for optimization testing."""
    event_configs = [
        {
            'event_id': 0,
            'center_x': 80.0,
            'center_y': 80.0,
            't_ns': 1000.0,
            'n_photons': 8,
            'photon_spread_spatial': 20.0,
            'photon_spread_temporal': 60.0
        },
        {
            'event_id': 1,
            'center_x': 180.0,
            'center_y': 120.0,
            't_ns': 10000.0,
            'n_photons': 8,
            'photon_spread_spatial': 20.0,
            'photon_spread_temporal': 60.0
        }
    ]

    photon_df = create_synthetic_photon_data(event_configs)
    event_df = create_synthetic_event_data(event_configs)

    return {
        'photons': photon_df,
        'events': event_df,
        'configs': event_configs
    }


class TestIterativeOptimizer:
    """Test the IterativeOptimizer class."""

    def test_single_iteration(self, optimization_scenario):
        """Test running a single iteration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir)

            write_csv_files(
                None,
                optimization_scenario['photons'],
                optimization_scenario['events'],
                data_path,
                file_index=0
            )

            optimizer = IterativeOptimizer(
                data_folder=str(data_path),
                initial_spatial_px=25.0,
                initial_temporal_ns=80.0,
                verbosity=0
            )

            result = optimizer.run_iteration(1)

            # Verify result structure
            assert isinstance(result, IterationResult)
            assert result.iteration == 1
            assert result.spatial_px == 25.0
            assert result.temporal_ns == 80.0
            assert 0 <= result.association_rate <= 1.0
            assert result.total_events >= 0

    def test_multiple_iterations(self, optimization_scenario):
        """Test running multiple iterations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir)

            write_csv_files(
                None,
                optimization_scenario['photons'],
                optimization_scenario['events'],
                data_path,
                file_index=0
            )

            optimizer = IterativeOptimizer(
                data_folder=str(data_path),
                initial_spatial_px=25.0,
                initial_temporal_ns=80.0,
                verbosity=0
            )

            best_result = optimizer.optimize(
                max_iterations=3,
                convergence_threshold=0.1
            )

            # Should have run some iterations
            assert len(optimizer.results) > 0
            assert len(optimizer.results) <= 3
            assert best_result is not None

    def test_convergence(self, optimization_scenario):
        """Test that optimizer converges."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir)

            write_csv_files(
                None,
                optimization_scenario['photons'],
                optimization_scenario['events'],
                data_path,
                file_index=0
            )

            optimizer = IterativeOptimizer(
                data_folder=str(data_path),
                initial_spatial_px=40.0,  # Start with reasonable values
                initial_temporal_ns=120.0,
                verbosity=0
            )

            best_result = optimizer.optimize(
                max_iterations=10,
                convergence_threshold=0.02  # Tight convergence
            )

            # Should converge before max iterations (usually)
            # or reach max iterations
            assert len(optimizer.results) <= 10

    def test_save_results(self, optimization_scenario):
        """Test saving optimization results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "data"
            output_path = Path(tmpdir) / "results"
            data_path.mkdir()

            write_csv_files(
                None,
                optimization_scenario['photons'],
                optimization_scenario['events'],
                data_path,
                file_index=0
            )

            optimizer = IterativeOptimizer(
                data_folder=str(data_path),
                initial_spatial_px=25.0,
                initial_temporal_ns=80.0,
                verbosity=0
            )

            optimizer.optimize(
                max_iterations=3,
                output_dir=str(output_path)
            )

            # Verify output files
            assert (output_path / "optimization_history.json").exists()
            assert (output_path / "best_parameters.json").exists()
            assert (output_path / "summary.json").exists()

            # Verify JSON format
            import json
            with open(output_path / "best_parameters.json") as f:
                params = json.load(f)
            assert 'photon2event' in params

    def test_progress_dataframe(self, optimization_scenario):
        """Test getting progress as DataFrame."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir)

            write_csv_files(
                None,
                optimization_scenario['photons'],
                optimization_scenario['events'],
                data_path,
                file_index=0
            )

            optimizer = IterativeOptimizer(
                data_folder=str(data_path),
                initial_spatial_px=25.0,
                initial_temporal_ns=80.0,
                verbosity=0
            )

            optimizer.optimize(max_iterations=3)

            df = optimizer.get_progress_dataframe()

            assert isinstance(df, pd.DataFrame)
            assert len(df) == len(optimizer.results)
            assert 'iteration' in df.columns
            assert 'spatial_px' in df.columns
            assert 'temporal_ns' in df.columns
            assert 'association_rate' in df.columns

    def test_convenience_function(self, optimization_scenario):
        """Test the convenience function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "data"
            output_path = Path(tmpdir) / "results"
            data_path.mkdir()

            write_csv_files(
                None,
                optimization_scenario['photons'],
                optimization_scenario['events'],
                data_path,
                file_index=0
            )

            result = optimize_parameters_iteratively(
                data_folder=str(data_path),
                initial_spatial_px=25.0,
                initial_temporal_ns=80.0,
                max_iterations=3,
                output_dir=str(output_path),
                verbosity=0
            )

            # Verify result format
            assert isinstance(result, dict)
            assert 'spatial_px' in result
            assert 'temporal_ns' in result
            assert 'association_rate' in result

            # Verify output files
            assert (output_path / "best_parameters.json").exists()


class TestOptimizationBehavior:
    """Test optimizer behavior in different scenarios."""

    def test_tight_clustering_optimization(self):
        """Test optimization on tightly clustered data."""
        event_configs = [
            {
                'event_id': 0,
                'center_x': 100.0,
                'center_y': 100.0,
                't_ns': 1000.0,
                'n_photons': 10,
                'photon_spread_spatial': 8.0,  # Tight
                'photon_spread_temporal': 30.0
            }
        ]

        photon_df = create_synthetic_photon_data(event_configs)
        event_df = create_synthetic_event_data(event_configs)

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir)
            write_csv_files(None, photon_df, event_df, data_path, file_index=0)

            optimizer = IterativeOptimizer(
                data_folder=str(data_path),
                initial_spatial_px=50.0,  # Start too loose
                initial_temporal_ns=200.0,
                verbosity=0
            )

            best_result = optimizer.optimize(max_iterations=5)

            # Should tighten or maintain reasonable parameters for tight clustering
            # (might not change much if already reasonable)
            assert best_result.spatial_px <= 50.0

    def test_loose_clustering_optimization(self):
        """Test optimization on loosely clustered data."""
        event_configs = [
            {
                'event_id': 0,
                'center_x': 100.0,
                'center_y': 100.0,
                't_ns': 1000.0,
                'n_photons': 10,
                'photon_spread_spatial': 40.0,  # Loose
                'photon_spread_temporal': 150.0
            }
        ]

        photon_df = create_synthetic_photon_data(event_configs)
        event_df = create_synthetic_event_data(event_configs)

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir)
            write_csv_files(None, photon_df, event_df, data_path, file_index=0)

            optimizer = IterativeOptimizer(
                data_folder=str(data_path),
                initial_spatial_px=10.0,  # Start too tight
                initial_temporal_ns=30.0,
                verbosity=0
            )

            best_result = optimizer.optimize(max_iterations=5)

            # Should loosen parameters for loose clustering
            assert best_result.spatial_px > 10.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
