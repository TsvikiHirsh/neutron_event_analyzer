"""
Example: Using the Association Parameter Optimizer

This script demonstrates how to use the AssociationOptimizer to find
the best association parameters for your synthetic data.
"""

import sys
from pathlib import Path
import tempfile

from neutron_event_analyzer.optimizer import AssociationOptimizer, optimize_for_synthetic_data
from test_association_validation import (
    create_synthetic_pixel_data,
    create_synthetic_photon_data,
    create_synthetic_event_data,
    write_csv_files
)


def example_1_grid_search():
    """
    Example 1: Basic grid search across methods and parameters.

    This finds the best association method and parameters from a set of options.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Grid Search Optimization")
    print("="*70)

    # Create synthetic data
    print("\n1. Creating synthetic data...")
    event_configs = [
        {
            'event_id': 0,
            'center_x': 50.0,
            'center_y': 50.0,
            't_ns': 1000.0,
            'n_photons': 5,
            'photon_spread_spatial': 15.0,
            'photon_spread_temporal': 50.0
        },
        {
            'event_id': 1,
            'center_x': 150.0,
            'center_y': 150.0,
            't_ns': 10000.0,
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

    photon_df = create_synthetic_photon_data(event_configs)
    event_df = create_synthetic_event_data(event_configs)

    print(f"   Created {len(event_configs)} events with {len(photon_df)} photons")

    # Write to temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir)
        write_csv_files(None, photon_df, event_df, data_path, file_index=0)

        print(f"\n2. Initializing optimizer...")
        optimizer = AssociationOptimizer(
            synthetic_data_dir=str(data_path),
            ground_truth_photons=photon_df,
            ground_truth_events=event_df,
            verbosity=1  # Normal output
        )

        print(f"\n3. Running grid search...")
        best_result = optimizer.grid_search(
            methods=['simple', 'kdtree', 'window'],
            spatial_thresholds_px=[10.0, 20.0, 50.0],
            temporal_thresholds_ns=[50.0, 100.0, 500.0],
            metric='f1_score'
        )

        print(f"\n4. Best parameters found:")
        print(f"   Method: {best_result.method}")
        print(f"   Spatial: {best_result.spatial_threshold_px} px")
        print(f"   Temporal: {best_result.temporal_threshold_ns} ns")
        print(f"   F1 Score: {best_result.f1_score:.4f}")
        print(f"   Association Rate: {best_result.association_rate:.2%}")

        # Get parameters in empir format
        print(f"\n5. Parameters in empir format:")
        params = optimizer.get_best_parameters_json()
        import json
        print(json.dumps(params, indent=2))


def example_2_recursive_optimization():
    """
    Example 2: Recursive optimization for fine-tuning.

    This performs hill-climbing to find locally optimal parameters.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Recursive Optimization")
    print("="*70)

    # Create synthetic data
    print("\n1. Creating synthetic data...")
    event_configs = [
        {
            'event_id': 0,
            'center_x': 100.0,
            'center_y': 100.0,
            't_ns': 1000.0,
            'n_photons': 8,
            'photon_spread_spatial': 12.0,
            'photon_spread_temporal': 40.0
        }
    ]

    photon_df = create_synthetic_photon_data(event_configs)
    event_df = create_synthetic_event_data(event_configs)

    print(f"   Created {len(event_configs)} events with {len(photon_df)} photons")

    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir)
        write_csv_files(None, photon_df, event_df, data_path, file_index=0)

        print(f"\n2. Initializing optimizer...")
        optimizer = AssociationOptimizer(
            synthetic_data_dir=str(data_path),
            ground_truth_photons=photon_df,
            ground_truth_events=event_df,
            verbosity=1
        )

        print(f"\n3. Running recursive optimization...")
        best_result = optimizer.recursive_optimize(
            method='simple',
            initial_spatial_px=20.0,
            initial_temporal_ns=100.0,
            max_iterations=5,
            metric='f1_score'
        )

        print(f"\n4. Optimized parameters:")
        print(f"   Spatial: {best_result.spatial_threshold_px:.2f} px")
        print(f"   Temporal: {best_result.temporal_threshold_ns:.2f} ns")
        print(f"   F1 Score: {best_result.f1_score:.4f}")


def example_3_save_results():
    """
    Example 3: Run optimization and save results to files.

    This shows how to save both detailed results and best parameters.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Optimization with File Output")
    print("="*70)

    # Create synthetic data
    print("\n1. Creating synthetic data...")
    event_configs = [
        {
            'event_id': i,
            'center_x': 50.0 + i * 80.0,
            'center_y': 50.0 + i * 60.0,
            't_ns': 1000.0 + i * 5000.0,
            'n_photons': 6,
            'photon_spread_spatial': 15.0,
            'photon_spread_temporal': 45.0
        }
        for i in range(3)
    ]

    photon_df = create_synthetic_photon_data(event_configs)
    event_df = create_synthetic_event_data(event_configs)

    print(f"   Created {len(event_configs)} events with {len(photon_df)} photons")

    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / "data"
        output_path = Path(tmpdir) / "results"
        data_path.mkdir()

        write_csv_files(None, photon_df, event_df, data_path, file_index=0)

        print(f"\n2. Running optimization with file output...")

        best_result = optimize_for_synthetic_data(
            synthetic_data_dir=str(data_path),
            ground_truth_photons=photon_df,
            ground_truth_events=event_df,
            mode='grid',
            output_dir=str(output_path),
            verbosity=1,
            methods=['simple'],
            spatial_thresholds_px=[10.0, 20.0, 30.0],
            temporal_thresholds_ns=[50.0, 100.0, 200.0]
        )

        print(f"\n3. Results saved to: {output_path}")
        print(f"   - optimization_results.json: Detailed results for all combinations")
        print(f"   - best_parameters.json: Best parameters in empir format")

        # Show what was saved
        print(f"\n4. Files created:")
        for file in output_path.iterdir():
            print(f"   ✓ {file.name}")


def example_4_optimize_from_empir_params():
    """
    Example 4: Use existing parameterSettings.json as starting point.

    This shows how to load existing parameters and optimize from there.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Optimize from Existing Parameters")
    print("="*70)

    # Create synthetic data
    print("\n1. Creating synthetic data...")
    event_configs = [
        {
            'event_id': 0,
            'center_x': 100.0,
            'center_y': 100.0,
            't_ns': 1000.0,
            'n_photons': 7,
            'photon_spread_spatial': 18.0,
            'photon_spread_temporal': 60.0
        }
    ]

    photon_df = create_synthetic_photon_data(event_configs)
    event_df = create_synthetic_event_data(event_configs)

    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir)
        write_csv_files(None, photon_df, event_df, data_path, file_index=0)

        # Simulate existing parameterSettings.json
        print(f"\n2. Loading existing parameterSettings.json...")
        existing_params = {
            "photon2event": {
                "dSpace_px": 25.0,
                "dTime_s": 150e-9  # 150 ns
            }
        }
        print(f"   Existing spatial: {existing_params['photon2event']['dSpace_px']} px")
        print(f"   Existing temporal: {existing_params['photon2event']['dTime_s'] * 1e9} ns")

        # Use as starting point for recursive optimization
        print(f"\n3. Optimizing from these starting values...")
        optimizer = AssociationOptimizer(
            synthetic_data_dir=str(data_path),
            ground_truth_photons=photon_df,
            ground_truth_events=event_df,
            verbosity=1
        )

        best_result = optimizer.recursive_optimize(
            method='simple',
            initial_spatial_px=existing_params['photon2event']['dSpace_px'],
            initial_temporal_ns=existing_params['photon2event']['dTime_s'] * 1e9,
            max_iterations=5,
            metric='f1_score'
        )

        print(f"\n4. Improvement:")
        print(f"   Old spatial: {existing_params['photon2event']['dSpace_px']} px")
        print(f"   New spatial: {best_result.spatial_threshold_px:.2f} px")
        print(f"   Old temporal: {existing_params['photon2event']['dTime_s'] * 1e9} ns")
        print(f"   New temporal: {best_result.temporal_threshold_ns:.2f} ns")
        print(f"   F1 Score: {best_result.f1_score:.4f}")


def example_5_compare_metrics():
    """
    Example 5: Compare optimization for different metrics.

    This shows how results differ when optimizing for different objectives.
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Comparing Different Optimization Metrics")
    print("="*70)

    # Create synthetic data
    print("\n1. Creating synthetic data...")
    event_configs = [
        {
            'event_id': i,
            'center_x': 50.0 + i * 70.0,
            'center_y': 50.0 + i * 70.0,
            't_ns': 1000.0 + i * 3000.0,
            'n_photons': 5,
            'photon_spread_spatial': 12.0,
            'photon_spread_temporal': 40.0
        }
        for i in range(3)
    ]

    photon_df = create_synthetic_photon_data(event_configs)
    event_df = create_synthetic_event_data(event_configs)

    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir)
        write_csv_files(None, photon_df, event_df, data_path, file_index=0)

        # Test different metrics
        metrics_to_test = ['f1_score', 'accuracy', 'association_rate']
        results = {}

        for metric in metrics_to_test:
            print(f"\n2. Optimizing for: {metric}...")

            optimizer = AssociationOptimizer(
                synthetic_data_dir=str(data_path),
                ground_truth_photons=photon_df,
                ground_truth_events=event_df,
                verbosity=0  # Silent for comparison
            )

            best = optimizer.grid_search(
                methods=['simple'],
                spatial_thresholds_px=[10.0, 20.0, 50.0],
                temporal_thresholds_ns=[50.0, 100.0, 500.0],
                metric=metric
            )

            results[metric] = best

        # Compare results
        print(f"\n3. Comparison of different optimization objectives:\n")
        print(f"{'Metric':<20} {'Spatial (px)':<15} {'Temporal (ns)':<15} {'F1':<10} {'Accuracy':<12} {'Assoc Rate':<12}")
        print("-" * 94)

        for metric, result in results.items():
            print(f"{metric:<20} {result.spatial_threshold_px:<15.1f} "
                  f"{result.temporal_threshold_ns:<15.1f} "
                  f"{result.f1_score:<10.3f} "
                  f"{result.accuracy:<12.2%} "
                  f"{result.association_rate:<12.2%}")

        print("\n4. Observations:")
        print("   - 'f1_score' balances precision and recall")
        print("   - 'accuracy' optimizes for correctness of associations")
        print("   - 'association_rate' maximizes number of associations")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("Association Parameter Optimizer - Examples")
    print("="*70)
    print("\nThis script demonstrates different ways to use the optimizer")
    print("to find optimal association parameters for your data.")

    # Run examples
    example_1_grid_search()
    example_2_recursive_optimization()
    example_3_save_results()
    example_4_optimize_from_empir_params()
    example_5_compare_metrics()

    print("\n" + "="*70)
    print("Examples Complete!")
    print("="*70)
    print("\nYou can now use these patterns to optimize parameters for your own data.")
    print("\n")


if __name__ == '__main__':
    main()
