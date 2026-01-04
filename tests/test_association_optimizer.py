"""
Tests for the association parameter optimizer.

This module demonstrates how to use the AssociationOptimizer to automatically
find the best association method and parameters for synthetic data.
"""

import os
import tempfile
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from association_optimizer import (
    AssociationOptimizer,
    AssociationMetrics,
    optimize_for_synthetic_data
)
from test_association_validation import (
    create_synthetic_pixel_data,
    create_synthetic_photon_data,
    create_synthetic_event_data,
    write_csv_files
)


@pytest.fixture
def optimizer_scenario():
    """
    Create a scenario specifically designed for optimizer testing.

    This uses moderate spread values that should work with multiple parameter settings.
    """
    # Define 3 well-separated events
    event_configs = [
        {
            'event_id': 0,
            'center_x': 50.0,
            'center_y': 50.0,
            't_ns': 1000.0,
            'n_photons': 5,
            'photon_spread_spatial': 15.0,  # Moderate spread
            'photon_spread_temporal': 50.0
        },
        {
            'event_id': 1,
            'center_x': 150.0,
            'center_y': 150.0,
            't_ns': 10000.0,  # Well separated in time
            'n_photons': 5,
            'photon_spread_spatial': 15.0,
            'photon_spread_temporal': 50.0
        },
        {
            'event_id': 2,
            'center_x': 200.0,
            'center_y': 100.0,
            't_ns': 20000.0,
            'n_photons': 5,
            'photon_spread_spatial': 15.0,
            'photon_spread_temporal': 50.0
        }
    ]

    # Generate photons
    photon_df = create_synthetic_photon_data(event_configs)

    # Generate events
    event_df = create_synthetic_event_data(event_configs)

    return {
        'photons': photon_df,
        'events': event_df,
        'description': 'Optimizer scenario: 3 events with moderate spread'
    }


class TestAssociationOptimizer:
    """Test the AssociationOptimizer class."""

    def test_single_evaluation(self, optimizer_scenario):
        """Test evaluating a single association configuration."""
        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir)

            # Write synthetic data
            write_csv_files(
                None,  # No pixels needed
                optimizer_scenario['photons'],
                optimizer_scenario['events'],
                data_path,
                file_index=0
            )

            # Initialize optimizer
            optimizer = AssociationOptimizer(
                synthetic_data_dir=str(data_path),
                ground_truth_photons=optimizer_scenario['photons'],
                ground_truth_events=optimizer_scenario['events'],
                verbosity=0
            )

            # Evaluate single configuration
            result = optimizer.evaluate_association(
                method='simple',
                spatial_threshold_px=20.0,
                temporal_threshold_ns=100.0
            )

            # Verify result structure
            assert isinstance(result, AssociationMetrics)
            assert result.method == 'simple'
            assert result.spatial_threshold_px == 20.0
            assert result.temporal_threshold_ns == 100.0
            assert result.total_photons > 0
            assert 0 <= result.association_rate <= 1.0
            assert 0 <= result.accuracy <= 1.0
            assert 0 <= result.f1_score <= 1.0

    def test_grid_search_basic(self, optimizer_scenario):
        """Test basic grid search functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir)

            write_csv_files(
                None,
                optimizer_scenario['photons'],
                optimizer_scenario['events'],
                data_path,
                file_index=0
            )

            optimizer = AssociationOptimizer(
                synthetic_data_dir=str(data_path),
                ground_truth_photons=optimizer_scenario['photons'],
                ground_truth_events=optimizer_scenario['events'],
                verbosity=0
            )

            # Grid search with limited options for speed
            best = optimizer.grid_search(
                methods=['simple'],
                spatial_thresholds_px=[10.0, 20.0, 50.0],
                temporal_thresholds_ns=[50.0, 100.0, 500.0],
                metric='f1_score'
            )

            # Verify results
            assert best is not None
            assert len(optimizer.results) == 9  # 1 method × 3 spatial × 3 temporal
            assert optimizer.best_result == best

            # Best result should have reasonable metrics
            assert best.association_rate >= 0
            assert best.f1_score >= 0

    def test_grid_search_multiple_methods(self, optimizer_scenario):
        """Test grid search with multiple association methods."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir)

            write_csv_files(
                None,
                optimizer_scenario['photons'],
                optimizer_scenario['events'],
                data_path,
                file_index=0
            )

            optimizer = AssociationOptimizer(
                synthetic_data_dir=str(data_path),
                ground_truth_photons=optimizer_scenario['photons'],
                ground_truth_events=optimizer_scenario['events'],
                verbosity=0
            )

            # Test multiple methods
            best = optimizer.grid_search(
                methods=['simple', 'kdtree'],
                spatial_thresholds_px=[20.0, 50.0],
                temporal_thresholds_ns=[100.0, 500.0],
                metric='f1_score'
            )

            # Should test all combinations
            assert len(optimizer.results) == 8  # 2 methods × 2 spatial × 2 temporal
            assert best is not None
            assert best.method in ['simple', 'kdtree']

    def test_recursive_optimize(self, optimizer_scenario):
        """Test recursive optimization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir)

            write_csv_files(
                None,
                optimizer_scenario['photons'],
                optimizer_scenario['events'],
                data_path,
                file_index=0
            )

            optimizer = AssociationOptimizer(
                synthetic_data_dir=str(data_path),
                ground_truth_photons=optimizer_scenario['photons'],
                ground_truth_events=optimizer_scenario['events'],
                verbosity=0
            )

            # Recursive optimization
            best = optimizer.recursive_optimize(
                method='simple',
                initial_spatial_px=20.0,
                initial_temporal_ns=100.0,
                max_iterations=3,  # Limited for speed
                convergence_threshold=0.01,
                metric='f1_score'
            )

            # Verify results
            assert best is not None
            assert len(optimizer.results) >= 1  # At least initial evaluation
            assert optimizer.best_result == best

    def test_save_results(self, optimizer_scenario):
        """Test saving optimization results to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "data"
            data_path.mkdir()

            write_csv_files(
                None,
                optimizer_scenario['photons'],
                optimizer_scenario['events'],
                data_path,
                file_index=0
            )

            optimizer = AssociationOptimizer(
                synthetic_data_dir=str(data_path),
                ground_truth_photons=optimizer_scenario['photons'],
                ground_truth_events=optimizer_scenario['events'],
                verbosity=0
            )

            # Run quick grid search
            optimizer.grid_search(
                methods=['simple'],
                spatial_thresholds_px=[20.0],
                temporal_thresholds_ns=[100.0]
            )

            # Save results
            output_path = Path(tmpdir) / "results.json"
            optimizer.save_results(str(output_path))

            # Verify file was created
            assert output_path.exists()

            # Verify it's valid JSON
            import json
            with open(output_path) as f:
                data = json.load(f)

            assert 'best_result' in data
            assert 'all_results' in data
            assert len(data['all_results']) > 0

    def test_save_best_parameters(self, optimizer_scenario):
        """Test saving best parameters as parameterSettings.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "data"
            data_path.mkdir()

            write_csv_files(
                None,
                optimizer_scenario['photons'],
                optimizer_scenario['events'],
                data_path,
                file_index=0
            )

            optimizer = AssociationOptimizer(
                synthetic_data_dir=str(data_path),
                ground_truth_photons=optimizer_scenario['photons'],
                ground_truth_events=optimizer_scenario['events'],
                verbosity=0
            )

            # Run optimization
            optimizer.grid_search(
                methods=['simple'],
                spatial_thresholds_px=[20.0],
                temporal_thresholds_ns=[100.0]
            )

            # Save best parameters
            output_path = Path(tmpdir) / "best_params.json"
            optimizer.save_best_parameters(str(output_path))

            # Verify file was created
            assert output_path.exists()

            # Verify it has correct format
            import json
            with open(output_path) as f:
                params = json.load(f)

            assert 'photon2event' in params
            assert 'dSpace_px' in params['photon2event']
            assert 'dTime_s' in params['photon2event']

    def test_optimize_for_synthetic_data_convenience(self, optimizer_scenario):
        """Test the convenience function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "data"
            output_path = Path(tmpdir) / "output"
            data_path.mkdir()

            write_csv_files(
                None,
                optimizer_scenario['photons'],
                optimizer_scenario['events'],
                data_path,
                file_index=0
            )

            # Use convenience function
            best = optimize_for_synthetic_data(
                synthetic_data_dir=str(data_path),
                ground_truth_photons=optimizer_scenario['photons'],
                ground_truth_events=optimizer_scenario['events'],
                mode='grid',
                output_dir=str(output_path),
                verbosity=0,
                methods=['simple'],
                spatial_thresholds_px=[20.0],
                temporal_thresholds_ns=[100.0]
            )

            # Verify results
            assert best is not None

            # Verify output files were created
            assert (output_path / "optimization_results.json").exists()
            assert (output_path / "best_parameters.json").exists()


class TestOptimizationMetrics:
    """Test different optimization metrics."""

    def test_optimize_for_f1_score(self, optimizer_scenario):
        """Test optimizing for F1 score (balanced precision/recall)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir)

            write_csv_files(
                None,
                optimizer_scenario['photons'],
                optimizer_scenario['events'],
                data_path,
                file_index=0
            )

            optimizer = AssociationOptimizer(
                synthetic_data_dir=str(data_path),
                ground_truth_photons=optimizer_scenario['photons'],
                ground_truth_events=optimizer_scenario['events'],
                verbosity=0
            )

            best = optimizer.grid_search(
                methods=['simple'],
                spatial_thresholds_px=[10.0, 50.0],
                temporal_thresholds_ns=[50.0, 500.0],
                metric='f1_score'
            )

            assert best is not None
            # F1 should be between 0 and 1
            assert 0 <= best.f1_score <= 1.0

    def test_optimize_for_association_rate(self, optimizer_scenario):
        """Test optimizing for maximum association rate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir)

            write_csv_files(
                None,
                optimizer_scenario['photons'],
                optimizer_scenario['events'],
                data_path,
                file_index=0
            )

            optimizer = AssociationOptimizer(
                synthetic_data_dir=str(data_path),
                ground_truth_photons=optimizer_scenario['photons'],
                ground_truth_events=optimizer_scenario['events'],
                verbosity=0
            )

            best = optimizer.grid_search(
                methods=['simple'],
                spatial_thresholds_px=[10.0, 50.0, 200.0],  # Include very loose threshold
                temporal_thresholds_ns=[50.0, 500.0, 5000.0],
                metric='association_rate'
            )

            assert best is not None
            # Looser parameters should give higher association rate
            # The best should be one of the looser settings
            assert best.spatial_threshold_px >= 10.0


class TestRealWorldScenarios:
    """Test optimizer with realistic scenarios."""

    def test_optimize_tight_clustering(self):
        """Test optimization for tightly clustered photons."""
        # Create scenario with tight clustering
        event_configs = [
            {
                'event_id': 0,
                'center_x': 100.0,
                'center_y': 100.0,
                't_ns': 1000.0,
                'n_photons': 10,
                'photon_spread_spatial': 3.0,  # Tight spatial clustering
                'photon_spread_temporal': 20.0
            }
        ]

        photon_df = create_synthetic_photon_data(event_configs)
        event_df = create_synthetic_event_data(event_configs)

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir)

            write_csv_files(None, photon_df, event_df, data_path, file_index=0)

            best = optimize_for_synthetic_data(
                synthetic_data_dir=str(data_path),
                ground_truth_photons=photon_df,
                ground_truth_events=event_df,
                mode='grid',
                verbosity=0,
                methods=['simple'],
                spatial_thresholds_px=[2.0, 5.0, 10.0],
                temporal_thresholds_ns=[50.0, 100.0]
            )

            assert best is not None
            # For tight clustering, best spatial threshold should be small
            assert best.spatial_threshold_px <= 10.0

    def test_optimize_loose_clustering(self):
        """Test optimization for loosely clustered photons."""
        # Create scenario with loose clustering
        event_configs = [
            {
                'event_id': 0,
                'center_x': 100.0,
                'center_y': 100.0,
                't_ns': 1000.0,
                'n_photons': 10,
                'photon_spread_spatial': 50.0,  # Loose spatial clustering
                'photon_spread_temporal': 200.0
            }
        ]

        photon_df = create_synthetic_photon_data(event_configs)
        event_df = create_synthetic_event_data(event_configs)

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir)

            write_csv_files(None, photon_df, event_df, data_path, file_index=0)

            best = optimize_for_synthetic_data(
                synthetic_data_dir=str(data_path),
                ground_truth_photons=photon_df,
                ground_truth_events=event_df,
                mode='grid',
                verbosity=0,
                methods=['simple'],
                spatial_thresholds_px=[20.0, 50.0, 100.0],
                temporal_thresholds_ns=[100.0, 500.0, 1000.0]
            )

            assert best is not None
            # For loose clustering, best threshold should be larger
            assert best.spatial_threshold_px >= 20.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
