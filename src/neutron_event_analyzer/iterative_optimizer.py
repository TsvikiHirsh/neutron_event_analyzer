"""
Iterative Parameter Optimizer

Automatically refine association parameters through iterative analysis of real data.
This tool runs multiple iterations of association, analyzes quality metrics,
and suggests improved parameters for the next iteration.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np

import neutron_event_analyzer as nea
from neutron_event_analyzer.parameter_suggester import (
    ParameterSuggester,
    AssociationQualityMetrics,
    ParameterSuggestion
)


@dataclass
class IterationResult:
    """Results from a single optimization iteration."""

    iteration: int
    spatial_px: float
    temporal_ns: float
    method: str

    # Quality metrics
    association_rate: float
    mean_photons_per_event: float
    total_events: int

    # Optional detailed metrics
    metrics: Optional[AssociationQualityMetrics] = None
    suggestion: Optional[ParameterSuggestion] = None

    def to_dict(self):
        return {
            'iteration': self.iteration,
            'spatial_px': self.spatial_px,
            'temporal_ns': self.temporal_ns,
            'method': self.method,
            'association_rate': self.association_rate,
            'mean_photons_per_event': self.mean_photons_per_event,
            'total_events': self.total_events
        }


class IterativeOptimizer:
    """
    Iteratively refine association parameters on real data.

    This tool runs multiple iterations of association with progressively
    refined parameters, analyzing quality metrics to guide improvements.
    """

    def __init__(
        self,
        data_folder: str,
        initial_spatial_px: float = 20.0,
        initial_temporal_ns: float = 100.0,
        settings: Optional[str] = None,
        method: str = 'simple',
        verbosity: int = 1
    ):
        """
        Initialize the iterative optimizer.

        Args:
            data_folder: Folder containing photon/event data
            initial_spatial_px: Starting spatial threshold
            initial_temporal_ns: Starting temporal threshold
            settings: Settings preset or path to settings file
            method: Association method to use
            verbosity: Output verbosity (0=silent, 1=normal, 2=detailed)
        """
        self.data_folder = Path(data_folder)
        self.initial_spatial_px = initial_spatial_px
        self.initial_temporal_ns = initial_temporal_ns
        self.settings = settings
        self.method = method
        self.verbosity = verbosity

        self.results: List[IterationResult] = []
        self.current_spatial_px = initial_spatial_px
        self.current_temporal_ns = initial_temporal_ns

    def run_iteration(self, iteration: int) -> IterationResult:
        """
        Run a single optimization iteration.

        Args:
            iteration: Iteration number

        Returns:
            IterationResult object
        """
        if self.verbosity >= 1:
            print(f"\n{'='*70}")
            print(f"Iteration {iteration}")
            print(f"{'='*70}")
            print(f"Parameters: spatial={self.current_spatial_px:.2f} px, "
                  f"temporal={self.current_temporal_ns:.2f} ns")

        # Load and associate
        analyser = nea.Analyse(
            data_folder=str(self.data_folder),
            settings=self.settings,
            n_threads=1
        )
        analyser.load(verbosity=0)
        analyser.associate_photons_events(
            method=self.method,
            dSpace_px=self.current_spatial_px,
            max_time_ns=self.current_temporal_ns
        )

        # Analyze quality
        suggester = ParameterSuggester(analyser, verbosity=self.verbosity)
        metrics = suggester.analyze_quality()
        suggestion = suggester.suggest_parameters(
            self.current_spatial_px,
            self.current_temporal_ns
        )

        # Create result
        result = IterationResult(
            iteration=iteration,
            spatial_px=self.current_spatial_px,
            temporal_ns=self.current_temporal_ns,
            method=self.method,
            association_rate=metrics.association_rate,
            mean_photons_per_event=metrics.mean_photons_per_event,
            total_events=metrics.total_events,
            metrics=metrics,
            suggestion=suggestion
        )

        self.results.append(result)

        # Update parameters for next iteration
        self.current_spatial_px = suggestion.suggested_spatial_px
        self.current_temporal_ns = suggestion.suggested_temporal_ns

        return result

    def optimize(
        self,
        max_iterations: int = 5,
        convergence_threshold: float = 0.05,
        output_dir: Optional[str] = None
    ) -> IterationResult:
        """
        Run iterative optimization.

        Args:
            max_iterations: Maximum number of iterations
            convergence_threshold: Stop if parameter changes < this fraction
            output_dir: Optional directory to save results

        Returns:
            Best IterationResult
        """
        if self.verbosity >= 1:
            print(f"\n{'='*70}")
            print(f"Starting Iterative Optimization")
            print(f"{'='*70}")
            print(f"Data folder: {self.data_folder}")
            print(f"Initial parameters: spatial={self.initial_spatial_px:.2f} px, "
                  f"temporal={self.initial_temporal_ns:.2f} ns")
            print(f"Max iterations: {max_iterations}")
            print(f"Convergence threshold: {convergence_threshold:.1%}")
            print(f"{'='*70}")

        for i in range(max_iterations):
            result = self.run_iteration(i + 1)

            # Check convergence
            if i > 0:
                prev_result = self.results[-2]
                spatial_change = abs(result.spatial_px - prev_result.spatial_px) / prev_result.spatial_px
                temporal_change = abs(result.temporal_ns - prev_result.temporal_ns) / prev_result.temporal_ns

                if spatial_change < convergence_threshold and temporal_change < convergence_threshold:
                    if self.verbosity >= 1:
                        print(f"\n✓ Converged! Parameter changes below {convergence_threshold:.1%}")
                    break

        # Find best result
        best_result = max(self.results, key=lambda r: r.association_rate)

        if self.verbosity >= 1:
            print(f"\n{'='*70}")
            print(f"Optimization Complete")
            print(f"{'='*70}")
            print(f"\nBest result from iteration {best_result.iteration}:")
            print(f"  Spatial: {best_result.spatial_px:.2f} px")
            print(f"  Temporal: {best_result.temporal_ns:.2f} ns")
            print(f"  Association rate: {best_result.association_rate:.2%}")
            print(f"  Mean photons/event: {best_result.mean_photons_per_event:.1f}")
            print(f"  Total events: {best_result.total_events}")

        # Save results if requested
        if output_dir:
            self.save_results(output_dir)

        return best_result

    def save_results(self, output_dir: str):
        """
        Save optimization results to files.

        Args:
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save iteration history
        history = [r.to_dict() for r in self.results]
        with open(output_path / 'optimization_history.json', 'w') as f:
            json.dump(history, f, indent=2)

        # Save best parameters
        best_result = max(self.results, key=lambda r: r.association_rate)
        best_params = {
            "photon2event": {
                "dSpace_px": best_result.spatial_px,
                "dTime_s": best_result.temporal_ns * 1e-9,
                "durationMax_s": best_result.temporal_ns * 1e-9 * 10,
                "dTime_ext": 5
            }
        }

        with open(output_path / 'best_parameters.json', 'w') as f:
            json.dump(best_params, f, indent=2)

        # Save summary
        summary = {
            'initial_parameters': {
                'spatial_px': self.initial_spatial_px,
                'temporal_ns': self.initial_temporal_ns
            },
            'best_parameters': {
                'spatial_px': best_result.spatial_px,
                'temporal_ns': best_result.temporal_ns
            },
            'improvement': {
                'spatial_change_pct': ((best_result.spatial_px - self.initial_spatial_px) / self.initial_spatial_px * 100),
                'temporal_change_pct': ((best_result.temporal_ns - self.initial_temporal_ns) / self.initial_temporal_ns * 100)
            },
            'total_iterations': len(self.results),
            'best_iteration': best_result.iteration,
            'final_association_rate': best_result.association_rate,
            'final_mean_photons_per_event': best_result.mean_photons_per_event
        }

        with open(output_path / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        if self.verbosity >= 1:
            print(f"\n✓ Results saved to: {output_path}")
            print(f"  - optimization_history.json: Full iteration history")
            print(f"  - best_parameters.json: Best parameterSettings.json")
            print(f"  - summary.json: Optimization summary")

    def get_progress_dataframe(self) -> pd.DataFrame:
        """
        Get optimization progress as a DataFrame.

        Returns:
            DataFrame with iteration progress
        """
        return pd.DataFrame([r.to_dict() for r in self.results])

    def plot_progress(self, save_path: Optional[str] = None):
        """
        Plot optimization progress.

        Args:
            save_path: Optional path to save plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available for plotting")
            return

        df = self.get_progress_dataframe()

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Iterative Optimization Progress', fontsize=16)

        # Association rate
        axes[0, 0].plot(df['iteration'], df['association_rate'] * 100, 'o-', linewidth=2)
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Association Rate (%)')
        axes[0, 0].set_title('Association Rate')
        axes[0, 0].grid(True, alpha=0.3)

        # Spatial threshold
        axes[0, 1].plot(df['iteration'], df['spatial_px'], 'o-', linewidth=2, color='green')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Spatial Threshold (px)')
        axes[0, 1].set_title('Spatial Threshold')
        axes[0, 1].grid(True, alpha=0.3)

        # Temporal threshold
        axes[1, 0].plot(df['iteration'], df['temporal_ns'], 'o-', linewidth=2, color='red')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Temporal Threshold (ns)')
        axes[1, 0].set_title('Temporal Threshold')
        axes[1, 0].grid(True, alpha=0.3)

        # Photons per event
        axes[1, 1].plot(df['iteration'], df['mean_photons_per_event'], 'o-', linewidth=2, color='purple')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Mean Photons per Event')
        axes[1, 1].set_title('Event Size')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            if self.verbosity >= 1:
                print(f"✓ Plot saved to: {save_path}")
        else:
            plt.show()


def optimize_parameters_iteratively(
    data_folder: str,
    initial_spatial_px: float = 20.0,
    initial_temporal_ns: float = 100.0,
    max_iterations: int = 5,
    settings: Optional[str] = None,
    method: str = 'simple',
    output_dir: Optional[str] = None,
    verbosity: int = 1
) -> Dict[str, Any]:
    """
    Convenience function for iterative parameter optimization.

    Args:
        data_folder: Folder containing data files
        initial_spatial_px: Starting spatial threshold
        initial_temporal_ns: Starting temporal threshold
        max_iterations: Maximum iterations
        settings: Settings preset or file path
        method: Association method
        output_dir: Optional output directory
        verbosity: Output verbosity

    Returns:
        Dictionary with best parameters
    """
    optimizer = IterativeOptimizer(
        data_folder=data_folder,
        initial_spatial_px=initial_spatial_px,
        initial_temporal_ns=initial_temporal_ns,
        settings=settings,
        method=method,
        verbosity=verbosity
    )

    best_result = optimizer.optimize(
        max_iterations=max_iterations,
        output_dir=output_dir
    )

    return {
        'spatial_px': best_result.spatial_px,
        'temporal_ns': best_result.temporal_ns,
        'association_rate': best_result.association_rate,
        'mean_photons_per_event': best_result.mean_photons_per_event,
        'total_events': best_result.total_events
    }


if __name__ == '__main__':
    print("Iterative Parameter Optimizer")
    print("Import this module to use the IterativeOptimizer class")
