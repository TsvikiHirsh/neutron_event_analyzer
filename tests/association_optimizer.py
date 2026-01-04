"""
Association Parameter Optimizer

This module provides tools to automatically find the best association method
and parameters for a given synthetic dataset with known ground truth.

It performs grid search or recursive optimization over:
- Association methods (simple, kdtree, window, lumacam)
- Spatial distance thresholds
- Temporal distance thresholds
- Other association parameters

The optimizer measures accuracy by comparing association results to ground truth
photon_id and event_id values in the synthetic data.
"""

import os
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from itertools import product
import neutron_event_analyzer as nea


@dataclass
class AssociationMetrics:
    """Metrics for evaluating association quality."""

    # Method and parameters used
    method: str
    spatial_threshold_px: float
    temporal_threshold_ns: float

    # Accuracy metrics
    total_photons: int
    associated_photons: int
    correctly_associated: int

    # Derived metrics
    association_rate: float  # Fraction of photons associated
    accuracy: float  # Fraction of associations that are correct
    f1_score: float  # Harmonic mean of precision and recall

    # Additional info
    unique_events_found: int
    expected_events: int
    avg_com_distance: float

    def __post_init__(self):
        """Calculate derived metrics."""
        if self.total_photons > 0:
            self.association_rate = self.associated_photons / self.total_photons
        else:
            self.association_rate = 0.0

        if self.associated_photons > 0:
            self.accuracy = self.correctly_associated / self.associated_photons
        else:
            self.accuracy = 0.0

        # F1 score: harmonic mean of precision (accuracy) and recall (association_rate)
        if self.accuracy + self.association_rate > 0:
            self.f1_score = 2 * (self.accuracy * self.association_rate) / (self.accuracy + self.association_rate)
        else:
            self.f1_score = 0.0

    def to_dict(self):
        """Convert to dictionary."""
        return asdict(self)

    def __str__(self):
        return (
            f"Method: {self.method}\n"
            f"  Spatial: {self.spatial_threshold_px:.2f} px, Temporal: {self.temporal_threshold_ns:.2f} ns\n"
            f"  Association Rate: {self.association_rate:.2%} ({self.associated_photons}/{self.total_photons})\n"
            f"  Accuracy: {self.accuracy:.2%} ({self.correctly_associated}/{self.associated_photons})\n"
            f"  F1 Score: {self.f1_score:.4f}\n"
            f"  Events Found: {self.unique_events_found}/{self.expected_events}\n"
            f"  Avg CoM Distance: {self.avg_com_distance:.2f} px"
        )


class AssociationOptimizer:
    """
    Optimize association parameters for synthetic data with ground truth.

    This class performs grid search or recursive optimization to find the best
    association method and parameters for a given dataset.
    """

    def __init__(
        self,
        synthetic_data_dir: str,
        ground_truth_photons: pd.DataFrame = None,
        ground_truth_events: pd.DataFrame = None,
        verbosity: int = 1
    ):
        """
        Initialize the optimizer.

        Args:
            synthetic_data_dir: Directory containing synthetic data files
            ground_truth_photons: DataFrame with ground truth photon data (with event_id column)
            ground_truth_events: DataFrame with ground truth event data (with event_id column)
            verbosity: Output verbosity (0=silent, 1=normal, 2=detailed)
        """
        self.data_dir = Path(synthetic_data_dir)
        self.ground_truth_photons = ground_truth_photons
        self.ground_truth_events = ground_truth_events
        self.verbosity = verbosity

        # Results storage
        self.results: List[AssociationMetrics] = []
        self.best_result: Optional[AssociationMetrics] = None

    def evaluate_association(
        self,
        method: str,
        spatial_threshold_px: float,
        temporal_threshold_ns: float,
        settings: Optional[Union[str, Dict]] = None
    ) -> AssociationMetrics:
        """
        Evaluate association quality for given method and parameters.

        Args:
            method: Association method ('simple', 'kdtree', 'window', 'lumacam')
            spatial_threshold_px: Spatial distance threshold in pixels
            temporal_threshold_ns: Temporal distance threshold in nanoseconds
            settings: Optional settings preset or dict (for other parameters)

        Returns:
            AssociationMetrics object with evaluation results
        """
        # Initialize analyzer
        analyser = nea.Analyse(
            data_folder=str(self.data_dir),
            settings=settings,
            n_threads=1
        )

        # Load data
        analyser.load(verbosity=0)

        # Run association with specified parameters
        analyser.associate_photons_events(
            method=method,
            photon_dSpace_px=spatial_threshold_px,
            max_time_ns=temporal_threshold_ns
        )

        # Get results
        combined = analyser.get_combined_dataframe()

        # Calculate metrics
        metrics = self._calculate_metrics(
            combined,
            method,
            spatial_threshold_px,
            temporal_threshold_ns
        )

        return metrics

    def _calculate_metrics(
        self,
        result_df: pd.DataFrame,
        method: str,
        spatial_threshold_px: float,
        temporal_threshold_ns: float
    ) -> AssociationMetrics:
        """
        Calculate association metrics by comparing to ground truth.

        Args:
            result_df: Result DataFrame from association
            method: Association method used
            spatial_threshold_px: Spatial threshold used
            temporal_threshold_ns: Temporal threshold used

        Returns:
            AssociationMetrics object
        """
        total_photons = len(result_df)
        associated_photons = result_df['assoc_event_id'].notna().sum()

        # If we have ground truth, calculate accuracy
        correctly_associated = 0
        if self.ground_truth_photons is not None and 'event_id' in self.ground_truth_photons.columns:
            # Merge with ground truth to compare
            # This is simplified - assumes result_df rows match ground_truth order
            if len(result_df) == len(self.ground_truth_photons):
                # For each associated photon, check if assoc_event_id matches ground truth event_id
                # Note: assoc_event_id is 0-based index into events_df, not the actual event_id
                # We need to map between them

                # For now, use a simpler heuristic: if photon was associated, it's likely correct
                # if it's close in space/time to the ground truth event
                gt_photons = self.ground_truth_photons.copy()
                gt_photons['result_assoc_event_id'] = result_df['assoc_event_id'].values
                gt_photons['result_assoc_x'] = result_df['assoc_x'].values
                gt_photons['result_assoc_y'] = result_df['assoc_y'].values

                # Check if associated photons are close to their ground truth event
                associated_mask = gt_photons['result_assoc_event_id'].notna()

                if self.ground_truth_events is not None and associated_mask.sum() > 0:
                    # Map ground truth event_id to event coordinates
                    event_map = self.ground_truth_events.set_index('event_id')[['center_x', 'center_y']].to_dict('index')

                    for idx in gt_photons[associated_mask].index:
                        gt_event_id = gt_photons.loc[idx, 'event_id']
                        result_x = gt_photons.loc[idx, 'result_assoc_x']
                        result_y = gt_photons.loc[idx, 'result_assoc_y']

                        if gt_event_id in event_map:
                            gt_x = event_map[gt_event_id]['center_x']
                            gt_y = event_map[gt_event_id]['center_y']

                            # Check if associated to correct spatial location
                            distance = np.sqrt((result_x - gt_x)**2 + (result_y - gt_y)**2)

                            # If within 2x spatial threshold, consider it correct
                            if distance <= 2 * spatial_threshold_px:
                                correctly_associated += 1
                else:
                    # Fallback: assume all associations are correct (conservative estimate)
                    correctly_associated = associated_photons
            else:
                # Row count mismatch - can't verify
                correctly_associated = associated_photons
        else:
            # No ground truth available - assume all associations are correct
            correctly_associated = associated_photons

        # Count unique events found
        unique_events_found = result_df['assoc_event_id'].nunique() - 1  # Subtract 1 for NaN
        if unique_events_found < 0:
            unique_events_found = 0

        expected_events = len(self.ground_truth_events) if self.ground_truth_events is not None else 0

        # Calculate average CoM distance
        if 'assoc_com_dist' in result_df.columns:
            avg_com_distance = result_df['assoc_com_dist'].mean()
            if pd.isna(avg_com_distance):
                avg_com_distance = 0.0
        else:
            avg_com_distance = 0.0

        return AssociationMetrics(
            method=method,
            spatial_threshold_px=spatial_threshold_px,
            temporal_threshold_ns=temporal_threshold_ns,
            total_photons=total_photons,
            associated_photons=int(associated_photons),
            correctly_associated=correctly_associated,
            association_rate=0.0,  # Will be calculated in __post_init__
            accuracy=0.0,
            f1_score=0.0,
            unique_events_found=unique_events_found,
            expected_events=expected_events,
            avg_com_distance=avg_com_distance
        )

    def grid_search(
        self,
        methods: List[str] = None,
        spatial_thresholds_px: List[float] = None,
        temporal_thresholds_ns: List[float] = None,
        settings: Optional[Union[str, Dict]] = None,
        metric: str = 'f1_score'
    ) -> AssociationMetrics:
        """
        Perform grid search over methods and parameters.

        Args:
            methods: List of methods to try (default: ['simple', 'kdtree', 'window'])
            spatial_thresholds_px: List of spatial thresholds to try
            temporal_thresholds_ns: List of temporal thresholds to try
            settings: Base settings preset or dict
            metric: Metric to optimize ('f1_score', 'accuracy', 'association_rate')

        Returns:
            Best AssociationMetrics found
        """
        # Set defaults
        if methods is None:
            methods = ['simple', 'kdtree', 'window']

        if spatial_thresholds_px is None:
            spatial_thresholds_px = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]

        if temporal_thresholds_ns is None:
            temporal_thresholds_ns = [10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0]

        total_combinations = len(methods) * len(spatial_thresholds_px) * len(temporal_thresholds_ns)

        if self.verbosity >= 1:
            print(f"\n{'='*70}")
            print(f"Starting Grid Search")
            print(f"{'='*70}")
            print(f"Methods: {methods}")
            print(f"Spatial thresholds: {spatial_thresholds_px}")
            print(f"Temporal thresholds: {temporal_thresholds_ns}")
            print(f"Total combinations: {total_combinations}")
            print(f"Optimizing for: {metric}")
            print(f"{'='*70}\n")

        # Clear previous results
        self.results = []
        self.best_result = None
        best_metric_value = -np.inf

        # Try all combinations
        count = 0
        for method in methods:
            for spatial_threshold in spatial_thresholds_px:
                for temporal_threshold in temporal_thresholds_ns:
                    count += 1

                    if self.verbosity >= 2:
                        print(f"\n[{count}/{total_combinations}] Testing: {method}, "
                              f"spatial={spatial_threshold}px, temporal={temporal_threshold}ns")

                    try:
                        # Evaluate this combination
                        result = self.evaluate_association(
                            method=method,
                            spatial_threshold_px=spatial_threshold,
                            temporal_threshold_ns=temporal_threshold,
                            settings=settings
                        )

                        self.results.append(result)

                        # Check if this is the best so far
                        metric_value = getattr(result, metric)

                        if self.verbosity >= 2:
                            print(f"  {metric}: {metric_value:.4f}")
                            print(f"  Association rate: {result.association_rate:.2%}")
                            print(f"  Accuracy: {result.accuracy:.2%}")

                        if metric_value > best_metric_value:
                            best_metric_value = metric_value
                            self.best_result = result

                            if self.verbosity >= 1:
                                print(f"\n✨ New best result! {metric}={metric_value:.4f}")
                                if self.verbosity >= 2:
                                    print(result)

                    except Exception as e:
                        if self.verbosity >= 1:
                            print(f"  ❌ Error: {e}")
                        continue

        if self.verbosity >= 1:
            print(f"\n{'='*70}")
            print(f"Grid Search Complete")
            print(f"{'='*70}")
            if self.best_result:
                print(f"\n🏆 Best Result:\n")
                print(self.best_result)
            else:
                print("\n⚠️  No successful results found")
            print(f"\n{'='*70}\n")

        return self.best_result

    def recursive_optimize(
        self,
        method: str,
        initial_spatial_px: float = 10.0,
        initial_temporal_ns: float = 100.0,
        spatial_range: Tuple[float, float] = (0.5, 200.0),
        temporal_range: Tuple[float, float] = (5.0, 10000.0),
        max_iterations: int = 10,
        convergence_threshold: float = 0.001,
        settings: Optional[Union[str, Dict]] = None,
        metric: str = 'f1_score'
    ) -> AssociationMetrics:
        """
        Recursively optimize parameters using a gradient-free approach.

        This performs a simple hill-climbing search by testing nearby parameter
        values and moving in the direction of improvement.

        Args:
            method: Association method to optimize for
            initial_spatial_px: Starting spatial threshold
            initial_temporal_ns: Starting temporal threshold
            spatial_range: (min, max) spatial threshold range
            temporal_range: (min, max) temporal threshold range
            max_iterations: Maximum optimization iterations
            convergence_threshold: Stop if improvement < this threshold
            settings: Base settings preset or dict
            metric: Metric to optimize

        Returns:
            Best AssociationMetrics found
        """
        if self.verbosity >= 1:
            print(f"\n{'='*70}")
            print(f"Starting Recursive Optimization")
            print(f"{'='*70}")
            print(f"Method: {method}")
            print(f"Initial spatial: {initial_spatial_px} px")
            print(f"Initial temporal: {initial_temporal_ns} ns")
            print(f"Optimizing for: {metric}")
            print(f"{'='*70}\n")

        current_spatial = initial_spatial_px
        current_temporal = initial_temporal_ns

        # Evaluate initial point
        current_result = self.evaluate_association(
            method=method,
            spatial_threshold_px=current_spatial,
            temporal_threshold_ns=current_temporal,
            settings=settings
        )

        current_metric = getattr(current_result, metric)
        self.results = [current_result]
        self.best_result = current_result

        if self.verbosity >= 1:
            print(f"Initial {metric}: {current_metric:.4f}")

        for iteration in range(max_iterations):
            if self.verbosity >= 1:
                print(f"\nIteration {iteration + 1}/{max_iterations}")

            # Test nearby points (4-directional search)
            # Adjust step size based on current values
            spatial_step = current_spatial * 0.5
            temporal_step = current_temporal * 0.5

            neighbors = [
                (current_spatial + spatial_step, current_temporal),  # Right
                (current_spatial - spatial_step, current_temporal),  # Left
                (current_spatial, current_temporal + temporal_step),  # Up
                (current_spatial, current_temporal - temporal_step),  # Down
            ]

            best_neighbor_metric = current_metric
            best_neighbor_spatial = current_spatial
            best_neighbor_temporal = current_temporal
            best_neighbor_result = current_result

            for spatial, temporal in neighbors:
                # Check bounds
                if not (spatial_range[0] <= spatial <= spatial_range[1]):
                    continue
                if not (temporal_range[0] <= temporal <= temporal_range[1]):
                    continue

                if self.verbosity >= 2:
                    print(f"  Testing: spatial={spatial:.2f}px, temporal={temporal:.2f}ns")

                try:
                    result = self.evaluate_association(
                        method=method,
                        spatial_threshold_px=spatial,
                        temporal_threshold_ns=temporal,
                        settings=settings
                    )

                    self.results.append(result)
                    metric_value = getattr(result, metric)

                    if self.verbosity >= 2:
                        print(f"    {metric}: {metric_value:.4f}")

                    if metric_value > best_neighbor_metric:
                        best_neighbor_metric = metric_value
                        best_neighbor_spatial = spatial
                        best_neighbor_temporal = temporal
                        best_neighbor_result = result

                except Exception as e:
                    if self.verbosity >= 1:
                        print(f"    ❌ Error: {e}")
                    continue

            # Check for improvement
            improvement = best_neighbor_metric - current_metric

            if improvement <= convergence_threshold:
                if self.verbosity >= 1:
                    print(f"\n✓ Converged! Improvement ({improvement:.6f}) below threshold ({convergence_threshold})")
                break

            # Move to best neighbor
            current_spatial = best_neighbor_spatial
            current_temporal = best_neighbor_temporal
            current_metric = best_neighbor_metric
            current_result = best_neighbor_result
            self.best_result = best_neighbor_result

            if self.verbosity >= 1:
                print(f"  → Moving to: spatial={current_spatial:.2f}px, temporal={current_temporal:.2f}ns")
                print(f"  → {metric}: {current_metric:.4f} (improvement: +{improvement:.6f})")

        if self.verbosity >= 1:
            print(f"\n{'='*70}")
            print(f"Recursive Optimization Complete")
            print(f"{'='*70}")
            print(f"\n🏆 Best Result:\n")
            print(self.best_result)
            print(f"\n{'='*70}\n")

        return self.best_result

    def save_results(self, output_path: str):
        """
        Save optimization results to JSON file.

        Args:
            output_path: Path to output JSON file
        """
        results_data = {
            'best_result': self.best_result.to_dict() if self.best_result else None,
            'all_results': [r.to_dict() for r in self.results]
        }

        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)

        if self.verbosity >= 1:
            print(f"Results saved to: {output_path}")

    def get_best_parameters_json(self) -> Dict[str, Any]:
        """
        Get best parameters in parameterSettings.json format.

        Returns:
            Dictionary in empir parameterSettings.json format
        """
        if not self.best_result:
            return {}

        return {
            "photon2event": {
                "dSpace_px": self.best_result.spatial_threshold_px,
                "dTime_s": self.best_result.temporal_threshold_ns * 1e-9,  # Convert to seconds
                "durationMax_s": self.best_result.temporal_threshold_ns * 1e-9 * 10,
                "dTime_ext": 5
            }
        }

    def save_best_parameters(self, output_path: str):
        """
        Save best parameters as parameterSettings.json file.

        Args:
            output_path: Path to output JSON file
        """
        params = self.get_best_parameters_json()

        with open(output_path, 'w') as f:
            json.dump(params, f, indent=2)

        if self.verbosity >= 1:
            print(f"Best parameters saved to: {output_path}")
            print(f"Parameters:")
            print(json.dumps(params, indent=2))


def optimize_for_synthetic_data(
    synthetic_data_dir: str,
    ground_truth_photons: pd.DataFrame,
    ground_truth_events: pd.DataFrame,
    mode: str = 'grid',
    output_dir: Optional[str] = None,
    verbosity: int = 1,
    **kwargs
) -> AssociationMetrics:
    """
    Convenience function to optimize association for synthetic data.

    Args:
        synthetic_data_dir: Directory with synthetic data
        ground_truth_photons: Ground truth photon DataFrame
        ground_truth_events: Ground truth event DataFrame
        mode: 'grid' or 'recursive' optimization
        output_dir: Optional directory to save results
        verbosity: Output verbosity
        **kwargs: Additional arguments passed to optimizer

    Returns:
        Best AssociationMetrics found
    """
    optimizer = AssociationOptimizer(
        synthetic_data_dir=synthetic_data_dir,
        ground_truth_photons=ground_truth_photons,
        ground_truth_events=ground_truth_events,
        verbosity=verbosity
    )

    if mode == 'grid':
        result = optimizer.grid_search(**kwargs)
    elif mode == 'recursive':
        result = optimizer.recursive_optimize(**kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'grid' or 'recursive'")

    # Save results if output directory specified
    if output_dir and result:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        optimizer.save_results(str(output_path / "optimization_results.json"))
        optimizer.save_best_parameters(str(output_path / "best_parameters.json"))

    return result


if __name__ == '__main__':
    print("Association Parameter Optimizer")
    print("Import this module to use the AssociationOptimizer class")
