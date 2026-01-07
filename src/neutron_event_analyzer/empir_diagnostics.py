"""
EMPIR Parameter Diagnostics and Optimization

This module implements the EMPIR Parameter Optimization Framework, which uses
intrinsic distribution shapes from reconstruction stages to optimize parameters
without requiring event association or ground truth.

The framework analyzes:
- Stage 1: Pixel-to-Photon clustering (dTime, dSpace, nPxMin, nPxMax)
- Stage 2: Photon-to-Event clustering (dTime_s, dSpace_px, durationMax_s)

Each parameter is optimized by analyzing specific diagnostic distributions that
reveal the statistical signatures of good vs. poor parameter choices.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from scipy import stats
from scipy.optimize import curve_fit
import warnings


# =============================================================================
# Diagnostic Distribution Extractors
# =============================================================================

class EMPIRDiagnostics:
    """Extract diagnostic distributions from EMPIR reconstructed data."""

    def __init__(self, photons_df: Optional[pd.DataFrame] = None,
                 events_df: Optional[pd.DataFrame] = None,
                 pixels_df: Optional[pd.DataFrame] = None,
                 associated_df: Optional[pd.DataFrame] = None):
        """
        Initialize diagnostics extractor.

        Args:
            photons_df: Photon-level data (from EMPIR photon reconstruction)
            events_df: Event-level data (from EMPIR event reconstruction)
            pixels_df: Pixel-level data (raw TPX3 data)
            associated_df: Associated data (pixels/photons linked to photons/events)
        """
        self.photons_df = photons_df
        self.events_df = events_df
        self.pixels_df = pixels_df
        self.associated_df = associated_df

    # -------------------------------------------------------------------------
    # Stage 1: Pixel-to-Photon Diagnostics
    # -------------------------------------------------------------------------

    def intra_photon_time_diffs(self) -> np.ndarray:
        """
        Extract all Δt values within photon clusters.

        For each photon, computes pairwise time differences between all
        pixels in that photon. This reveals the temporal clustering quality.

        Returns:
            Array of time differences in nanoseconds
        """
        if self.associated_df is None or 'assoc_photon_id' not in self.associated_df.columns:
            raise ValueError("Need associated pixel-photon data")

        delta_t_list = []

        # Group pixels by photon
        for photon_id, group in self.associated_df.groupby('assoc_photon_id'):
            if pd.isna(photon_id):
                continue

            toa_values = group['toa_ns'].values

            # Compute all pairwise differences
            for i in range(len(toa_values)):
                for j in range(i + 1, len(toa_values)):
                    delta_t_list.append(abs(toa_values[j] - toa_values[i]))

        return np.array(delta_t_list)

    def intra_photon_spatial_spread(self) -> np.ndarray:
        """
        Extract radial distances of pixels from photon centroids.

        For each photon, computes distance of each pixel from the photon's
        center of mass. This reveals spatial clustering quality.

        Returns:
            Array of radial distances in pixels
        """
        if self.associated_df is None or 'assoc_photon_id' not in self.associated_df.columns:
            raise ValueError("Need associated pixel-photon data")

        radial_distances = []

        # Group pixels by photon
        for photon_id, group in self.associated_df.groupby('assoc_photon_id'):
            if pd.isna(photon_id) or len(group) < 2:
                continue

            # Compute centroid
            centroid_x = group['x'].mean()
            centroid_y = group['y'].mean()

            # Compute distances from centroid
            for _, row in group.iterrows():
                r = np.sqrt((row['x'] - centroid_x)**2 + (row['y'] - centroid_y)**2)
                radial_distances.append(r)

        return np.array(radial_distances)

    def photon_cluster_sizes(self) -> np.ndarray:
        """
        Get distribution of pixels per photon.

        Returns:
            Array of cluster sizes (number of pixels per photon)
        """
        if self.associated_df is None or 'assoc_photon_id' not in self.associated_df.columns:
            raise ValueError("Need associated pixel-photon data")

        cluster_sizes = self.associated_df.groupby('assoc_photon_id').size().values
        return cluster_sizes[cluster_sizes > 0]  # Remove any zero-sized groups

    # -------------------------------------------------------------------------
    # Stage 2: Photon-to-Event Diagnostics
    # -------------------------------------------------------------------------

    def inter_photon_intervals(self) -> np.ndarray:
        """
        Extract time intervals between consecutive photons.

        Sorts all photons by time and computes Δt between consecutive photons.
        This reveals temporal correlation between photons from same event.

        Returns:
            Array of inter-photon intervals in nanoseconds
        """
        if self.photons_df is None:
            raise ValueError("Need photon data")

        # Sort photons by time
        toa_sorted = np.sort(self.photons_df['toa_ns'].values)

        # Compute differences
        intervals = np.diff(toa_sorted)

        return intervals[intervals > 0]  # Remove any zero or negative intervals

    def intra_event_photon_spread(self) -> np.ndarray:
        """
        Extract radial distances of photons from event centroids.

        For each event, computes distance of each photon from the event's
        center of mass. This reveals spatial clustering quality at event level.

        Returns:
            Array of radial distances in pixels
        """
        if self.associated_df is None or 'assoc_event_id' not in self.associated_df.columns:
            raise ValueError("Need associated photon-event data")

        radial_distances = []

        # Group photons by event
        for event_id, group in self.associated_df.groupby('assoc_event_id'):
            if pd.isna(event_id) or len(group) < 2:
                continue

            # Compute centroid
            centroid_x = group['x'].mean()
            centroid_y = group['y'].mean()

            # Compute distances from centroid
            for _, row in group.iterrows():
                r = np.sqrt((row['x'] - centroid_x)**2 + (row['y'] - centroid_y)**2)
                radial_distances.append(r)

        return np.array(radial_distances)

    def event_durations(self) -> np.ndarray:
        """
        Extract time span of events (first to last photon).

        For each event, computes the time difference between the first
        and last photon. This reveals event temporal extent.

        Returns:
            Array of event durations in nanoseconds
        """
        if self.associated_df is None or 'assoc_event_id' not in self.associated_df.columns:
            raise ValueError("Need associated photon-event data")

        durations = []

        # Group photons by event
        for event_id, group in self.associated_df.groupby('assoc_event_id'):
            if pd.isna(event_id) or len(group) < 2:
                continue

            toa_values = group['toa_ns'].values
            duration = toa_values.max() - toa_values.min()
            durations.append(duration)

        return np.array(durations)

    def event_multiplicities(self) -> np.ndarray:
        """
        Get distribution of photons per event.

        Returns:
            Array of event multiplicities (number of photons per event)
        """
        if self.associated_df is None or 'assoc_event_id' not in self.associated_df.columns:
            raise ValueError("Need associated photon-event data")

        multiplicities = self.associated_df.groupby('assoc_event_id').size().values
        return multiplicities[multiplicities > 0]


# =============================================================================
# Statistical Distribution Analyzer
# =============================================================================

class DistributionAnalyzer:
    """Analyze distributions and extract statistical properties."""

    @staticmethod
    def fit_gaussian(data: np.ndarray, bins: int = 100) -> Dict[str, float]:
        """
        Fit Gaussian to data.

        Args:
            data: Input data array
            bins: Number of histogram bins

        Returns:
            Dict with 'mu', 'sigma', 'amplitude', 'r_squared'
        """
        if len(data) == 0:
            return {'mu': 0, 'sigma': 0, 'amplitude': 0, 'r_squared': 0}

        # Create histogram
        hist, bin_edges = np.histogram(data, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Initial guess
        mu_guess = np.mean(data)
        sigma_guess = np.std(data)
        amp_guess = hist.max()

        # Gaussian function
        def gaussian(x, amp, mu, sigma):
            return amp * np.exp(-(x - mu)**2 / (2 * sigma**2))

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(
                    gaussian, bin_centers, hist,
                    p0=[amp_guess, mu_guess, sigma_guess],
                    maxfev=1000
                )

            amp, mu, sigma = popt

            # Calculate R²
            y_pred = gaussian(bin_centers, *popt)
            ss_res = np.sum((hist - y_pred)**2)
            ss_tot = np.sum((hist - np.mean(hist))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            return {
                'mu': float(mu),
                'sigma': float(abs(sigma)),
                'amplitude': float(amp),
                'r_squared': float(r_squared)
            }
        except:
            # Fallback to simple statistics
            return {
                'mu': float(np.mean(data)),
                'sigma': float(np.std(data)),
                'amplitude': float(hist.max()),
                'r_squared': 0.0
            }

    @staticmethod
    def fit_exponential(data: np.ndarray, bins: int = 100) -> Dict[str, float]:
        """
        Fit exponential decay to data.

        Args:
            data: Input data array
            bins: Number of histogram bins

        Returns:
            Dict with 'tau', 'amplitude', 'r_squared'
        """
        if len(data) == 0:
            return {'tau': 0, 'amplitude': 0, 'r_squared': 0}

        # Create histogram
        hist, bin_edges = np.histogram(data, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Remove zeros for log fitting
        mask = hist > 0
        if mask.sum() < 3:
            return {'tau': np.mean(data), 'amplitude': hist.max(), 'r_squared': 0}

        # Exponential function
        def exponential(x, amp, tau):
            return amp * np.exp(-x / tau)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(
                    exponential, bin_centers[mask], hist[mask],
                    p0=[hist.max(), np.mean(data)],
                    maxfev=1000
                )

            amp, tau = popt

            # Calculate R²
            y_pred = exponential(bin_centers[mask], *popt)
            ss_res = np.sum((hist[mask] - y_pred)**2)
            ss_tot = np.sum((hist[mask] - np.mean(hist[mask]))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            return {
                'tau': float(abs(tau)),
                'amplitude': float(amp),
                'r_squared': float(r_squared)
            }
        except:
            return {
                'tau': float(np.mean(data)),
                'amplitude': float(hist.max()),
                'r_squared': 0.0
            }

    @staticmethod
    def fit_two_component_exponential(data: np.ndarray, bins: int = 100) -> Dict[str, float]:
        """
        Fit two-component exponential model: signal + background.

        Model: N(Δt) = A₁·exp(-Δt/τ_signal) + A₂·exp(-Δt/τ_background)

        Args:
            data: Input data array
            bins: Number of histogram bins

        Returns:
            Dict with 'tau_signal', 'tau_background', 'amp_signal', 'amp_background',
            'crossover_point', 'r_squared'
        """
        if len(data) == 0:
            return {
                'tau_signal': 0, 'tau_background': 0,
                'amp_signal': 0, 'amp_background': 0,
                'crossover_point': 0, 'r_squared': 0
            }

        # Create histogram
        hist, bin_edges = np.histogram(data, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Two-component exponential
        def two_exp(x, a1, tau1, a2, tau2):
            return a1 * np.exp(-x / tau1) + a2 * np.exp(-x / tau2)

        # Initial guess: fast component (signal) + slow component (background)
        p25 = np.percentile(data, 25)
        p75 = np.percentile(data, 75)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(
                    two_exp, bin_centers, hist,
                    p0=[hist.max(), p25, hist.max() * 0.1, p75],
                    bounds=([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf]),
                    maxfev=2000
                )

            a1, tau1, a2, tau2 = popt

            # Ensure tau1 < tau2 (signal is faster)
            if tau1 > tau2:
                a1, tau1, a2, tau2 = a2, tau2, a1, tau1

            # Find crossover point where signal = background
            # a1*exp(-x/tau1) = a2*exp(-x/tau2)
            # Solve: x = (tau1*tau2) / (tau2 - tau1) * ln(a1*tau2 / a2*tau1)
            if tau2 > tau1 and a1 > 0 and a2 > 0:
                crossover = (tau1 * tau2) / (tau2 - tau1) * np.log((a1 * tau2) / (a2 * tau1))
                crossover = max(0, crossover)
            else:
                crossover = tau1 * 3

            # Calculate R²
            y_pred = two_exp(bin_centers, *popt)
            ss_res = np.sum((hist - y_pred)**2)
            ss_tot = np.sum((hist - np.mean(hist))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            return {
                'tau_signal': float(tau1),
                'tau_background': float(tau2),
                'amp_signal': float(a1),
                'amp_background': float(a2),
                'crossover_point': float(crossover),
                'r_squared': float(r_squared)
            }
        except:
            # Fallback to single exponential
            single_fit = DistributionAnalyzer.fit_exponential(data, bins)
            return {
                'tau_signal': single_fit['tau'],
                'tau_background': single_fit['tau'] * 10,
                'amp_signal': single_fit['amplitude'],
                'amp_background': 0,
                'crossover_point': single_fit['tau'] * 3,
                'r_squared': single_fit['r_squared']
            }

    @staticmethod
    def find_percentile_threshold(data: np.ndarray, percentile: float = 95) -> float:
        """
        Find value containing X% of distribution.

        Args:
            data: Input data array
            percentile: Percentile to find (0-100)

        Returns:
            Value at specified percentile
        """
        if len(data) == 0:
            return 0.0
        return float(np.percentile(data, percentile))

    @staticmethod
    def calculate_signal_purity(data: np.ndarray, threshold: float,
                                signal_at_small_values: bool = True) -> float:
        """
        Calculate fraction that is signal below/above threshold.

        Args:
            data: Input data array
            threshold: Threshold value
            signal_at_small_values: If True, signal is at small values

        Returns:
            Purity fraction (0-1)
        """
        if len(data) == 0:
            return 0.0

        if signal_at_small_values:
            return float((data <= threshold).sum() / len(data))
        else:
            return float((data >= threshold).sum() / len(data))

    @staticmethod
    def find_distribution_knee(data: np.ndarray, bins: int = 100) -> float:
        """
        Find "knee" in cumulative distribution (point of maximum curvature).

        Args:
            data: Input data array
            bins: Number of bins for histogram

        Returns:
            Value at knee point
        """
        if len(data) < 3:
            return float(np.median(data)) if len(data) > 0 else 0.0

        # Create cumulative distribution
        sorted_data = np.sort(data)
        cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

        # Compute curvature (second derivative)
        # Use numerical differentiation
        if len(sorted_data) < 10:
            return float(np.percentile(data, 75))

        # Smooth the cumulative curve first
        from scipy.ndimage import uniform_filter1d
        smooth_cumulative = uniform_filter1d(cumulative, size=max(3, len(cumulative) // 20))

        # First derivative
        dy = np.gradient(smooth_cumulative)
        dx = np.gradient(sorted_data)
        first_deriv = dy / (dx + 1e-10)

        # Second derivative (curvature)
        curvature = np.abs(np.gradient(first_deriv))

        # Find maximum curvature
        knee_idx = np.argmax(curvature)

        return float(sorted_data[knee_idx])


# =============================================================================
# Parameter Suggestions based on Distributions
# =============================================================================

@dataclass
class EMPIRParameterSuggestion:
    """Suggested EMPIR parameters with reasoning."""

    # Current parameters
    current_params: Dict[str, Any]

    # Suggested parameters
    suggested_params: Dict[str, Any]

    # Change factors
    changes: Dict[str, float]

    # Reasoning for each parameter
    reasoning: Dict[str, List[str]]

    # Confidence levels for each parameter
    confidence: Dict[str, str]

    # Diagnostic metrics used
    diagnostics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def get_parameter_json(self) -> Dict[str, Any]:
        """Get suggested parameters in parameterSettings.json format."""
        return {
            "pixel2photon": {
                "dSpace": self.suggested_params.get('dSpace', 2.0),
                "dTime": self.suggested_params.get('dTime', 100e-9),
                "nPxMin": int(self.suggested_params.get('nPxMin', 8)),
                "nPxMax": int(self.suggested_params.get('nPxMax', 100)),
                "TDC1": True
            },
            "photon2event": {
                "dSpace_px": self.suggested_params.get('dSpace_px', 50.0),
                "dTime_s": self.suggested_params.get('dTime_s', 50e-9),
                "durationMax_s": self.suggested_params.get('durationMax_s', 500e-9),
                "dTime_ext": 5
            }
        }

    def __str__(self) -> str:
        """Format as readable string."""
        lines = ["", "="*70, "EMPIR Parameter Suggestions", "="*70, ""]

        for stage in ['pixel2photon', 'photon2event']:
            lines.append(f"{stage.upper()}:")
            lines.append("-" * 70)

            for param in self.suggested_params:
                if (stage == 'pixel2photon' and param in ['dSpace', 'dTime', 'nPxMin', 'nPxMax']) or \
                   (stage == 'photon2event' and param in ['dSpace_px', 'dTime_s', 'durationMax_s']):

                    current = self.current_params.get(param, 'N/A')
                    suggested = self.suggested_params.get(param, 'N/A')
                    change = self.changes.get(param, 1.0)
                    conf = self.confidence.get(param, 'medium')

                    change_pct = (change - 1) * 100
                    arrow = "↑" if change > 1.02 else "↓" if change < 0.98 else "→"

                    lines.append(f"  {param:15} {current:12} → {suggested:12} {arrow} "
                               f"({change_pct:+.1f}%) [{conf}]")

                    if param in self.reasoning:
                        for reason in self.reasoning[param]:
                            lines.append(f"    • {reason}")

            lines.append("")

        lines.append("="*70)
        return "\n".join(lines)


if __name__ == '__main__':
    print("EMPIR Diagnostics and Optimization Framework")
    print("Import this module to use EMPIRDiagnostics and DistributionAnalyzer")
