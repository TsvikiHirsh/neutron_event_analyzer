"""
EMPIR Parameter Optimizer

Implements the EMPIR Parameter Optimization Framework for suggesting improved
reconstruction parameters based on intrinsic distribution analysis.

This module provides high-level optimization functions that analyze diagnostic
distributions and suggest improved parameters for both pixel-to-photon and
photon-to-event reconstruction stages.
"""

import numpy as np
import pandas as pd
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import neutron_event_analyzer as nea

from .empir_diagnostics import (
    EMPIRDiagnostics,
    DistributionAnalyzer,
    EMPIRParameterSuggestion
)


class EMPIRParameterOptimizer:
    """
    Optimize EMPIR reconstruction parameters using intrinsic distributions.

    This optimizer doesn't require ground truth or event association. Instead,
    it analyzes the statistical signatures in reconstruction outputs to
    identify optimal parameters.
    """

    def __init__(self, analyser: nea.Analyse, verbosity: int = 1):
        """
        Initialize the EMPIR parameter optimizer.

        Args:
            analyser: NEA Analyse object with loaded data
            verbosity: Output verbosity (0=silent, 1=normal, 2=detailed)
        """
        self.analyser = analyser
        self.verbosity = verbosity
        self.diagnostics = None
        self.analyzer = DistributionAnalyzer()

    def optimize_pixel2photon(
        self,
        current_dSpace: float = 2.0,
        current_dTime: float = 100e-9,
        current_nPxMin: int = 8,
        current_nPxMax: int = 100
    ) -> EMPIRParameterSuggestion:
        """
        Optimize pixel-to-photon clustering parameters.

        Sequential optimization:
        1. dTime (temporal clustering) - most critical
        2. dSpace (spatial clustering)
        3. nPxMin/nPxMax (cluster size limits)

        Args:
            current_dSpace: Current spatial clustering radius (mm)
            current_dTime: Current temporal clustering window (seconds)
            current_nPxMin: Current minimum pixels per photon
            current_nPxMax: Current maximum pixels per photon

        Returns:
            EMPIRParameterSuggestion object
        """
        if self.verbosity >= 1:
            print("\n" + "="*70)
            print("EMPIR Pixel-to-Photon Parameter Optimization")
            print("="*70)

        # Get associated data
        df = self.analyser.get_combined_dataframe()
        if df is None or 'assoc_photon_id' not in df.columns:
            raise ValueError("Need pixel-photon association. Run associate() first.")

        # Extract diagnostics
        self.diagnostics = EMPIRDiagnostics(associated_df=df)

        # Initialize results
        suggested = {
            'dSpace': current_dSpace,
            'dTime': current_dTime,
            'nPxMin': current_nPxMin,
            'nPxMax': current_nPxMax
        }
        reasoning = {param: [] for param in suggested.keys()}
        confidence = {param: 'medium' for param in suggested.keys()}
        diagnostic_metrics = {}

        # 1. Optimize dTime (temporal clustering)
        if self.verbosity >= 1:
            print("\n[1/3] Analyzing temporal clustering (dTime)...")

        try:
            delta_t = self.diagnostics.intra_photon_time_diffs()
            if len(delta_t) > 10:
                # Fit Gaussian to signal peak
                fit = self.analyzer.fit_gaussian(delta_t)
                diagnostic_metrics['temporal_fit'] = fit

                if fit['r_squared'] > 0.5:
                    # Good fit - use 3σ rule
                    suggested_dTime = 3 * fit['sigma'] * 1e-9  # Convert to seconds
                    reasoning['dTime'].append(
                        f"Fitted Gaussian: μ={fit['mu']:.1f} ns, σ={fit['sigma']:.1f} ns (R²={fit['r_squared']:.3f})"
                    )
                    reasoning['dTime'].append(
                        f"Setting dTime = 3σ = {suggested_dTime*1e9:.1f} ns for signal capture"
                    )
                    confidence['dTime'] = 'high'
                else:
                    # Poor fit - use percentile-based approach
                    p95 = self.analyzer.find_percentile_threshold(delta_t, 95)
                    suggested_dTime = p95 * 1.1 * 1e-9  # 10% margin, convert to seconds
                    reasoning['dTime'].append(
                        f"Gaussian fit poor (R²={fit['r_squared']:.3f}), using 95th percentile"
                    )
                    reasoning['dTime'].append(
                        f"P95 = {p95:.1f} ns → setting dTime = {suggested_dTime*1e9:.1f} ns"
                    )
                    confidence['dTime'] = 'medium'

                suggested['dTime'] = suggested_dTime
            else:
                reasoning['dTime'].append("Insufficient data for temporal analysis")
                confidence['dTime'] = 'low'
        except Exception as e:
            reasoning['dTime'].append(f"Analysis failed: {str(e)}")
            confidence['dTime'] = 'low'

        # 2. Optimize dSpace (spatial clustering)
        if self.verbosity >= 1:
            print("[2/3] Analyzing spatial clustering (dSpace)...")

        try:
            radii = self.diagnostics.intra_photon_spatial_spread()
            if len(radii) > 10:
                # Find 95th and 99th percentiles
                r_95 = self.analyzer.find_percentile_threshold(radii, 95)
                r_99 = self.analyzer.find_percentile_threshold(radii, 99)

                diagnostic_metrics['spatial_percentiles'] = {
                    'p95': r_95, 'p99': r_99
                }

                # Calculate outlier fraction with current dSpace
                outlier_frac = (radii > current_dSpace).sum() / len(radii)

                if outlier_frac > 0.05:
                    # Too tight
                    suggested['dSpace'] = r_99
                    reasoning['dSpace'].append(
                        f"Outlier fraction {outlier_frac:.1%} > 5% (too tight)"
                    )
                    reasoning['dSpace'].append(
                        f"Setting dSpace = P99 = {r_99:.2f} px"
                    )
                    confidence['dSpace'] = 'high'
                elif outlier_frac < 0.01:
                    # Too loose
                    suggested['dSpace'] = r_95
                    reasoning['dSpace'].append(
                        f"Outlier fraction {outlier_frac:.1%} < 1% (too loose)"
                    )
                    reasoning['dSpace'].append(
                        f"Setting dSpace = P95 = {r_95:.2f} px"
                    )
                    confidence['dSpace'] = 'high'
                else:
                    # Good
                    suggested['dSpace'] = current_dSpace
                    reasoning['dSpace'].append(
                        f"Outlier fraction {outlier_frac:.1%} in optimal range (1-5%)"
                    )
                    reasoning['dSpace'].append("Current dSpace is appropriate")
                    confidence['dSpace'] = 'high'
            else:
                reasoning['dSpace'].append("Insufficient data for spatial analysis")
                confidence['dSpace'] = 'low'
        except Exception as e:
            reasoning['dSpace'].append(f"Analysis failed: {str(e)}")
            confidence['dSpace'] = 'low'

        # 3. Optimize nPxMin and nPxMax
        if self.verbosity >= 1:
            print("[3/3] Analyzing cluster size limits (nPxMin, nPxMax)...")

        try:
            cluster_sizes = self.diagnostics.photon_cluster_sizes()
            if len(cluster_sizes) > 10:
                # Analyze distribution
                f_small = (cluster_sizes <= 5).sum() / len(cluster_sizes)
                p_95 = self.analyzer.find_percentile_threshold(cluster_sizes, 95)
                p_99 = self.analyzer.find_percentile_threshold(cluster_sizes, 99)
                peak_size = np.median(cluster_sizes)

                diagnostic_metrics['cluster_size_dist'] = {
                    'f_small': f_small,
                    'p95': p_95,
                    'p99': p_99,
                    'peak': peak_size
                }

                # Optimize nPxMin
                if f_small > 0.3:
                    suggested['nPxMin'] = 5
                    reasoning['nPxMin'].append(
                        f"Small clusters ({f_small:.1%}) indicate moderate noise"
                    )
                    reasoning['nPxMin'].append("Setting nPxMin = 5")
                    confidence['nPxMin'] = 'medium'
                elif f_small > 0.1:
                    suggested['nPxMin'] = 3
                    reasoning['nPxMin'].append(
                        f"Small clusters ({f_small:.1%}) indicate low noise"
                    )
                    reasoning['nPxMin'].append("Setting nPxMin = 3")
                    confidence['nPxMin'] = 'medium'
                else:
                    suggested['nPxMin'] = max(3, int(peak_size * 0.3))
                    reasoning['nPxMin'].append(
                        f"Very clean data ({f_small:.1%} small clusters)"
                    )
                    reasoning['nPxMin'].append(
                        f"Setting nPxMin = {suggested['nPxMin']} (30% of peak)"
                    )
                    confidence['nPxMin'] = 'high'

                # Optimize nPxMax
                suggested['nPxMax'] = int(p_95 * 1.2)  # 20% margin above P95
                rejection_rate = (cluster_sizes > suggested['nPxMax']).sum() / len(cluster_sizes)

                reasoning['nPxMax'].append(
                    f"P95 = {p_95:.0f}, P99 = {p_99:.0f}"
                )
                reasoning['nPxMax'].append(
                    f"Setting nPxMax = {suggested['nPxMax']} (P95 × 1.2)"
                )
                reasoning['nPxMax'].append(
                    f"Expected rejection rate: {rejection_rate:.1%}"
                )
                confidence['nPxMax'] = 'high' if rejection_rate < 0.05 else 'medium'
            else:
                reasoning['nPxMin'].append("Insufficient data")
                reasoning['nPxMax'].append("Insufficient data")
                confidence['nPxMin'] = 'low'
                confidence['nPxMax'] = 'low'
        except Exception as e:
            reasoning['nPxMin'].append(f"Analysis failed: {str(e)}")
            reasoning['nPxMax'].append(f"Analysis failed: {str(e)}")
            confidence['nPxMin'] = 'low'
            confidence['nPxMax'] = 'low'

        # Create suggestion object
        current = {
            'dSpace': current_dSpace,
            'dTime': current_dTime,
            'nPxMin': current_nPxMin,
            'nPxMax': current_nPxMax
        }

        changes = {
            param: suggested[param] / current[param] if current[param] > 0 else 1.0
            for param in suggested.keys()
        }

        return EMPIRParameterSuggestion(
            current_params=current,
            suggested_params=suggested,
            changes=changes,
            reasoning=reasoning,
            confidence=confidence,
            diagnostics=diagnostic_metrics
        )

    def optimize_photon2event(
        self,
        current_dSpace_px: float = 50.0,
        current_dTime_s: float = 50e-9,
        current_durationMax_s: float = 500e-9
    ) -> EMPIRParameterSuggestion:
        """
        Optimize photon-to-event clustering parameters.

        Sequential optimization:
        1. dTime_s (temporal clustering)
        2. dSpace_px (spatial clustering)
        3. durationMax_s (event duration limit)

        Args:
            current_dSpace_px: Current spatial clustering radius (pixels)
            current_dTime_s: Current temporal clustering window (seconds)
            current_durationMax_s: Current maximum event duration (seconds)

        Returns:
            EMPIRParameterSuggestion object
        """
        if self.verbosity >= 1:
            print("\n" + "="*70)
            print("EMPIR Photon-to-Event Parameter Optimization")
            print("="*70)

        # Get data
        df = self.analyser.get_combined_dataframe()
        if df is None:
            raise ValueError("Need associated data. Run associate() first.")

        # Extract diagnostics
        self.diagnostics = EMPIRDiagnostics(
            photons_df=self.analyser.photons_df,
            associated_df=df
        )

        # Initialize results
        suggested = {
            'dSpace_px': current_dSpace_px,
            'dTime_s': current_dTime_s,
            'durationMax_s': current_durationMax_s
        }
        reasoning = {param: [] for param in suggested.keys()}
        confidence = {param: 'medium' for param in suggested.keys()}
        diagnostic_metrics = {}

        # 1. Optimize dTime_s (temporal clustering)
        if self.verbosity >= 1:
            print("\n[1/3] Analyzing inter-photon temporal correlation (dTime_s)...")

        try:
            intervals = self.diagnostics.inter_photon_intervals()
            if len(intervals) > 20:
                # Fit two-component exponential
                fit = self.analyzer.fit_two_component_exponential(intervals)
                diagnostic_metrics['temporal_correlation'] = fit

                if fit['r_squared'] > 0.6:
                    # Good fit - use signal component
                    suggested_dTime = fit['tau_signal'] * 3 * 1e-9  # 3τ, convert to seconds
                    purity_at_3tau = self.analyzer.calculate_signal_purity(
                        intervals, fit['tau_signal'] * 3
                    )

                    reasoning['dTime_s'].append(
                        f"Two-component fit: τ_signal={fit['tau_signal']:.1f} ns, "
                        f"τ_background={fit['tau_background']:.1f} ns (R²={fit['r_squared']:.3f})"
                    )
                    reasoning['dTime_s'].append(
                        f"Setting dTime = 3τ_signal = {suggested_dTime*1e9:.1f} ns"
                    )
                    reasoning['dTime_s'].append(
                        f"Estimated purity at this threshold: {purity_at_3tau:.1%}"
                    )
                    confidence['dTime_s'] = 'high'
                    suggested['dTime_s'] = suggested_dTime
                else:
                    # Poor fit - use knee finding
                    knee = self.analyzer.find_distribution_knee(intervals)
                    suggested_dTime = knee * 2 * 1e-9  # 2× knee, convert to seconds

                    reasoning['dTime_s'].append(
                        f"Two-component fit poor (R²={fit['r_squared']:.3f})"
                    )
                    reasoning['dTime_s'].append(
                        f"Using distribution knee at {knee:.1f} ns"
                    )
                    reasoning['dTime_s'].append(
                        f"Setting dTime = 2 × knee = {suggested_dTime*1e9:.1f} ns"
                    )
                    confidence['dTime_s'] = 'medium'
                    suggested['dTime_s'] = suggested_dTime
            else:
                reasoning['dTime_s'].append("Insufficient data for temporal analysis")
                confidence['dTime_s'] = 'low'
        except Exception as e:
            reasoning['dTime_s'].append(f"Analysis failed: {str(e)}")
            confidence['dTime_s'] = 'low'

        # 2. Optimize dSpace_px (spatial clustering)
        if self.verbosity >= 1:
            print("[2/3] Analyzing spatial clustering and event multiplicity (dSpace_px)...")

        try:
            if 'assoc_event_id' in df.columns:
                radii = self.diagnostics.intra_event_photon_spread()
                multiplicities = self.diagnostics.event_multiplicities()

                if len(radii) > 10 and len(multiplicities) > 5:
                    r_95 = self.analyzer.find_percentile_threshold(radii, 95)
                    mean_mult = np.mean(multiplicities)
                    single_photon_frac = (multiplicities == 1).sum() / len(multiplicities)

                    diagnostic_metrics['spatial_event'] = {
                        'r_95': r_95,
                        'mean_multiplicity': mean_mult,
                        'single_photon_fraction': single_photon_frac
                    }

                    # Check if too tight (many single-photon events)
                    if single_photon_frac > 0.3:
                        suggested['dSpace_px'] = current_dSpace_px * 1.5
                        reasoning['dSpace_px'].append(
                            f"High single-photon fraction ({single_photon_frac:.1%}) - parameters too tight"
                        )
                        reasoning['dSpace_px'].append(
                            f"Increasing dSpace by 50% to {suggested['dSpace_px']:.1f} px"
                        )
                        confidence['dSpace_px'] = 'high'
                    # Check if too loose (very high multiplicity)
                    elif mean_mult > 15:
                        suggested['dSpace_px'] = r_95 * 0.8
                        reasoning['dSpace_px'].append(
                            f"High mean multiplicity ({mean_mult:.1f}) - parameters too loose"
                        )
                        reasoning['dSpace_px'].append(
                            f"Setting dSpace = 80% of P95 = {suggested['dSpace_px']:.1f} px"
                        )
                        confidence['dSpace_px'] = 'high'
                    else:
                        # Good range
                        suggested['dSpace_px'] = r_95 * 1.2
                        reasoning['dSpace_px'].append(
                            f"Multiplicity ({mean_mult:.1f}) in good range"
                        )
                        reasoning['dSpace_px'].append(
                            f"Setting dSpace = 1.2 × P95 = {suggested['dSpace_px']:.1f} px"
                        )
                        confidence['dSpace_px'] = 'high'
                else:
                    reasoning['dSpace_px'].append("Insufficient event data")
                    confidence['dSpace_px'] = 'low'
            else:
                reasoning['dSpace_px'].append("No event association available")
                confidence['dSpace_px'] = 'low'
        except Exception as e:
            reasoning['dSpace_px'].append(f"Analysis failed: {str(e)}")
            confidence['dSpace_px'] = 'low'

        # 3. Optimize durationMax_s
        if self.verbosity >= 1:
            print("[3/3] Analyzing event duration (durationMax_s)...")

        try:
            if 'assoc_event_id' in df.columns:
                durations = self.diagnostics.event_durations()

                if len(durations) > 10:
                    # Fit exponential
                    fit = self.analyzer.fit_exponential(durations)
                    p_95 = self.analyzer.find_percentile_threshold(durations, 95)
                    p_99 = self.analyzer.find_percentile_threshold(durations, 99)

                    diagnostic_metrics['event_duration'] = {
                        'tau': fit['tau'],
                        'p95': p_95,
                        'p99': p_99
                    }

                    # Use conservative approach: 5τ or P99
                    tau_based = fit['tau'] * 5 * 1e-9  # Convert to seconds
                    percentile_based = p_99 * 1e-9

                    suggested_duration = min(tau_based, percentile_based)

                    reasoning['durationMax_s'].append(
                        f"Exponential fit: τ = {fit['tau']:.1f} ns (R²={fit['r_squared']:.3f})"
                    )
                    reasoning['durationMax_s'].append(
                        f"P95 = {p_95:.1f} ns, P99 = {p_99:.1f} ns"
                    )
                    reasoning['durationMax_s'].append(
                        f"Setting durationMax = min(5τ, P99) = {suggested_duration*1e9:.1f} ns"
                    )

                    rejection_rate = (durations > suggested_duration * 1e9).sum() / len(durations)
                    reasoning['durationMax_s'].append(
                        f"Expected rejection rate: {rejection_rate:.1%}"
                    )

                    suggested['durationMax_s'] = suggested_duration
                    confidence['durationMax_s'] = 'high' if fit['r_squared'] > 0.7 else 'medium'
                else:
                    reasoning['durationMax_s'].append("Insufficient duration data")
                    confidence['durationMax_s'] = 'low'
            else:
                reasoning['durationMax_s'].append("No event association available")
                confidence['durationMax_s'] = 'low'
        except Exception as e:
            reasoning['durationMax_s'].append(f"Analysis failed: {str(e)}")
            confidence['durationMax_s'] = 'low'

        # Create suggestion object
        current = {
            'dSpace_px': current_dSpace_px,
            'dTime_s': current_dTime_s,
            'durationMax_s': current_durationMax_s
        }

        changes = {
            param: suggested[param] / current[param] if current[param] > 0 else 1.0
            for param in suggested.keys()
        }

        return EMPIRParameterSuggestion(
            current_params=current,
            suggested_params=suggested,
            changes=changes,
            reasoning=reasoning,
            confidence=confidence,
            diagnostics=diagnostic_metrics
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def optimize_empir_parameters(
    data_folder: str,
    stage: str = 'both',  # 'pixel2photon', 'photon2event', or 'both'
    current_params: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None,
    verbosity: int = 1,
    empir_binaries: Optional[str] = None
) -> Dict[str, EMPIRParameterSuggestion]:
    """
    Simple API to optimize EMPIR parameters.

    Args:
        data_folder: Path to folder with data
        stage: Which stage to optimize ('pixel2photon', 'photon2event', or 'both')
        current_params: Current parameter values (optional, will use defaults)
        output_path: Path to save suggested parameters JSON (optional)
        verbosity: Output verbosity level
        empir_binaries: Path to EMPIR binaries directory (optional, defaults to './export')

    Returns:
        Dictionary with suggestions for requested stages

    Example:
        >>> suggestions = optimize_empir_parameters(
        ...     data_folder='/path/to/data',
        ...     stage='both',
        ...     verbosity=1
        ... )
        >>> print(suggestions['photon2event'])
    """
    from pathlib import Path
    from .empir_runner import EMPIRRunner, get_default_params

    # Load data
    export_dir = empir_binaries if empir_binaries is not None else './export'
    data_path = Path(data_folder)

    # Check if we need to run EMPIR reconstruction
    tpx3_dir = data_path / "tpx3Files"
    photon_dir = data_path / "photonFiles"
    event_dir = data_path / "eventFiles"

    needs_pixel2photon = False
    needs_photon2event = False

    # Check what files exist and what we need
    if tpx3_dir.exists() and list(tpx3_dir.glob("*.tpx3")):
        # Have TPX3 files
        if not photon_dir.exists() or not list(photon_dir.glob("*.empirphot")):
            needs_pixel2photon = True
            if verbosity >= 1:
                print("\n" + "="*70)
                print("Missing photon files - will run pixel2photon reconstruction")
                print("="*70)

    # Check if we'll need event files (after potentially creating photon files)
    if stage in ['photon2event', 'both']:
        # We will need event files - check if they exist
        if not event_dir.exists() or not list(event_dir.glob("*.empirevent")):
            # Check if we will have photon files (either existing or about to create)
            will_have_photons = (photon_dir.exists() and list(photon_dir.glob("*.empirphot"))) or needs_pixel2photon
            if will_have_photons:
                needs_photon2event = True
                if verbosity >= 1:
                    print("\n" + "="*70)
                    print("Missing event files - will run photon2event reconstruction")
                    print("="*70)

    # Run EMPIR reconstruction if needed
    if needs_pixel2photon or needs_photon2event:
        runner = EMPIRRunner(empir_binaries_dir=export_dir, verbosity=verbosity)

        # Get or create reconstruction parameters
        if current_params is None:
            recon_params = get_default_params()
        else:
            recon_params = current_params.copy()
            # Ensure we have the required structure
            if 'pixel2photon' not in recon_params:
                recon_params['pixel2photon'] = get_default_params()['pixel2photon']
            if 'photon2event' not in recon_params:
                recon_params['photon2event'] = get_default_params()['photon2event']

        # Run pixel2photon if needed
        if needs_pixel2photon:
            photon_dir.mkdir(parents=True, exist_ok=True)
            success = runner.run_pixel2photon(
                tpx3_dir=tpx3_dir,
                output_dir=photon_dir,
                params=recon_params,
                n_threads=4
            )
            if not success:
                raise RuntimeError("Failed to run pixel2photon reconstruction")

            # Export photons and pixels to CSV for analyser
            exported_photons_dir = data_path / "ExportedPhotons"
            runner.run_export_photons(photon_dir, exported_photons_dir)

            # Export pixels for pixel-photon association
            try:
                exported_pixels_dir = data_path / "ExportedPixels"
                if 'export_pixels' in runner.binaries:
                    exported_pixels_dir.mkdir(parents=True, exist_ok=True)
                    for tpx3_file in tpx3_dir.glob("*.tpx3"):
                        output_file = exported_pixels_dir / f"exported_{tpx3_file.stem}.csv"
                        cmd = [
                            str(runner.binaries['export_pixels']),
                            str(tpx3_file),
                            str(output_file),
                            "csv"
                        ]
                        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    if verbosity >= 1:
                        print(f"✓ Exported pixel files to {exported_pixels_dir}")
            except Exception as e:
                if verbosity >= 1:
                    print(f"Warning: Could not export pixels: {e}")

        # Run photon2event if needed
        if needs_photon2event:
            event_dir.mkdir(parents=True, exist_ok=True)
            success = runner.run_photon2event(
                photon_dir=photon_dir,
                output_dir=event_dir,
                params=recon_params,
                n_threads=4
            )
            if not success:
                raise RuntimeError("Failed to run photon2event reconstruction")

            # Export events to CSV for analyser
            exported_events_dir = data_path / "ExportedEvents"
            runner.run_export_events(event_dir, exported_events_dir)

    # Now load the data
    analyser = nea.Analyse(data_folder=data_folder, export_dir=export_dir)
    analyser.load(verbosity=verbosity)

    # Associate pixels with photons if needed for pixel2photon optimization
    # This requires exported pixel files to be available
    has_exported_pixels = (data_path / "ExportedPixels").exists() and list((data_path / "ExportedPixels").glob("*.csv"))

    if stage in ['pixel2photon', 'both'] and has_exported_pixels:
        if verbosity >= 1:
            print("\nAssociating pixels with photons for pixel2photon optimization...")
        try:
            analyser.associate(method='simple', verbosity=0)
        except Exception as e:
            if verbosity >= 1:
                print(f"Warning: Pixel-photon association failed: {e}")
                print("Skipping pixel2photon optimization (requires pixel export binaries)")
            stage = 'photon2event' if stage == 'both' else None

    # Associate photons with events if needed for photon2event optimization
    if stage in ['photon2event', 'both']:
        if verbosity >= 1:
            print("\nAssociating photons with events for photon2event optimization...")
        analyser.associate_photons_events(method='simple', verbosity=0)

    # Create optimizer
    optimizer = EMPIRParameterOptimizer(analyser, verbosity=verbosity)

    # Default parameters
    if current_params is None:
        current_params = {}

    results = {}

    # Optimize requested stages
    if stage in ['pixel2photon', 'both'] and has_exported_pixels:
        p2p_params = current_params.get('pixel2photon', {})
        results['pixel2photon'] = optimizer.optimize_pixel2photon(
            current_dSpace=p2p_params.get('dSpace', 2.0),
            current_dTime=p2p_params.get('dTime', 100e-9),
            current_nPxMin=p2p_params.get('nPxMin', 8),
            current_nPxMax=p2p_params.get('nPxMax', 100)
        )
    elif stage in ['pixel2photon', 'both'] and not has_exported_pixels:
        if verbosity >= 1:
            print("\n⚠️  Skipping pixel2photon optimization: pixel export binaries not available")
            print("   To enable: ensure empir_export_pixelActivations is in your EMPIR binaries directory")

    if stage in ['photon2event', 'both']:
        p2e_params = current_params.get('photon2event', {})
        results['photon2event'] = optimizer.optimize_photon2event(
            current_dSpace_px=p2e_params.get('dSpace_px', 50.0),
            current_dTime_s=p2e_params.get('dTime_s', 50e-9),
            current_durationMax_s=p2e_params.get('durationMax_s', 500e-9)
        )

    # Save if requested
    if output_path:
        import json
        combined_params = {}
        for stage_name, suggestion in results.items():
            combined_params.update(suggestion.get_parameter_json())

        with open(output_path, 'w') as f:
            json.dump(combined_params, f, indent=2)

        if verbosity >= 1:
            print(f"\n✓ Suggested parameters saved to: {output_path}")

    return results


if __name__ == '__main__':
    print("EMPIR Parameter Optimizer")
    print("Import this module to use EMPIRParameterOptimizer")
