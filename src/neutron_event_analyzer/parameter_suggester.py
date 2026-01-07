"""
Parameter Suggester for Association Analysis

This module analyzes association results from real data and suggests
improved parameters for better event classification.

Unlike the optimizer (which requires ground truth), this tool works with
real data by analyzing quality metrics and suggesting parameter adjustments.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import neutron_event_analyzer as nea


@dataclass
class AssociationQualityMetrics:
    """Quality metrics for association results (no ground truth needed)."""

    # Basic counts
    total_photons: int
    associated_photons: int
    unassociated_photons: int
    total_events: int

    # Rates
    association_rate: float  # Fraction of photons associated

    # Event characteristics
    mean_photons_per_event: float
    median_photons_per_event: float
    std_photons_per_event: float
    min_photons_per_event: int
    max_photons_per_event: int

    # Spatial characteristics
    mean_spatial_spread: float  # Average spatial spread within events (px)
    median_spatial_spread: float
    std_spatial_spread: float

    # Temporal characteristics
    mean_temporal_spread: float  # Average temporal spread within events (ns)
    median_temporal_spread: float
    std_temporal_spread: float

    # Center of mass metrics
    mean_com_distance: float  # Average distance from photons to event CoM
    median_com_distance: float

    # Quality indicators
    likely_over_associated: bool  # Events might be too large (parameters too loose)
    likely_under_associated: bool  # Too many unassociated photons (parameters too tight)
    spatial_outliers_detected: bool  # Photons far from event centers
    temporal_outliers_detected: bool  # Photons with large time spreads

    def to_dict(self):
        """Convert to dictionary."""
        return asdict(self)

    def __str__(self):
        return f"""
Association Quality Metrics:
{'='*70}
Photon Statistics:
  Total Photons: {self.total_photons}
  Associated: {self.associated_photons} ({self.association_rate:.1%})
  Unassociated: {self.unassociated_photons} ({(1-self.association_rate):.1%})

Event Statistics:
  Total Events: {self.total_events}
  Photons per Event: {self.mean_photons_per_event:.1f} ± {self.std_photons_per_event:.1f}
  Range: [{self.min_photons_per_event}, {self.max_photons_per_event}]

Spatial Characteristics:
  Mean Spread: {self.mean_spatial_spread:.2f} px
  Median Spread: {self.median_spatial_spread:.2f} px
  Mean CoM Distance: {self.mean_com_distance:.2f} px

Temporal Characteristics:
  Mean Spread: {self.mean_temporal_spread:.2f} ns
  Median Spread: {self.median_temporal_spread:.2f} ns

Quality Indicators:
  Over-associated (too loose): {self.likely_over_associated}
  Under-associated (too tight): {self.likely_under_associated}
  Spatial Outliers: {self.spatial_outliers_detected}
  Temporal Outliers: {self.temporal_outliers_detected}
{'='*70}
"""


@dataclass
class ParameterSuggestion:
    """Suggested parameter changes."""

    current_spatial_px: float
    current_temporal_ns: float

    suggested_spatial_px: float
    suggested_temporal_ns: float

    spatial_change_factor: float
    temporal_change_factor: float

    reasoning: List[str]
    confidence: str  # 'high', 'medium', 'low'

    def to_dict(self):
        """Convert to dictionary."""
        return asdict(self)

    def get_parameter_json(self) -> Dict[str, Any]:
        """Get suggested parameters in parameterSettings.json format."""
        return {
            "photon2event": {
                "dSpace_px": self.suggested_spatial_px,
                "dTime_s": self.suggested_temporal_ns * 1e-9,  # Convert to seconds
                "durationMax_s": self.suggested_temporal_ns * 1e-9 * 10,
                "dTime_ext": 5
            }
        }

    def __str__(self):
        change_direction_spatial = "↑" if self.spatial_change_factor > 1 else "↓" if self.spatial_change_factor < 1 else "→"
        change_direction_temporal = "↑" if self.temporal_change_factor > 1 else "↓" if self.temporal_change_factor < 1 else "→"

        return f"""
Parameter Suggestions (Confidence: {self.confidence.upper()}):
{'='*70}
Spatial Threshold:
  Current:   {self.current_spatial_px:.2f} px
  Suggested: {self.suggested_spatial_px:.2f} px {change_direction_spatial}
  Change:    {((self.spatial_change_factor - 1) * 100):+.1f}%

Temporal Threshold:
  Current:   {self.current_temporal_ns:.2f} ns
  Suggested: {self.suggested_temporal_ns:.2f} ns {change_direction_temporal}
  Change:    {((self.temporal_change_factor - 1) * 100):+.1f}%

Reasoning:
{chr(10).join(f'  • {reason}' for reason in self.reasoning)}
{'='*70}
"""


class ParameterSuggester:
    """
    Analyze association results and suggest improved parameters.

    This class examines the quality of associations in real data and
    suggests parameter adjustments to improve event classification.
    """

    def __init__(self, analyser: nea.Analyse, verbosity: int = 1):
        """
        Initialize the parameter suggester.

        Args:
            analyser: NEA Analyse object with loaded and associated data
            verbosity: Output verbosity (0=silent, 1=normal, 2=detailed)
        """
        self.analyser = analyser
        self.verbosity = verbosity
        self.metrics: Optional[AssociationQualityMetrics] = None
        self.suggestion: Optional[ParameterSuggestion] = None

    def analyze_quality(self) -> AssociationQualityMetrics:
        """
        Analyze the quality of current associations.

        Returns:
            AssociationQualityMetrics object
        """
        # Get combined dataframe
        df = self.analyser.get_combined_dataframe()

        if df is None or len(df) == 0:
            raise ValueError("No data available. Make sure to load() and associate_photons_events() first.")

        # Basic counts
        total_photons = len(df)
        associated_mask = df['assoc_event_id'].notna()
        associated_photons = associated_mask.sum()
        unassociated_photons = total_photons - associated_photons
        association_rate = associated_photons / total_photons if total_photons > 0 else 0

        # Get events dataframe
        events_df = self.analyser.get_events_dataframe()
        total_events = len(events_df) if events_df is not None else 0

        # Photons per event
        if associated_photons > 0:
            photons_per_event = df[associated_mask].groupby('assoc_event_id').size()
            mean_photons_per_event = photons_per_event.mean()
            median_photons_per_event = photons_per_event.median()
            std_photons_per_event = photons_per_event.std()
            min_photons_per_event = int(photons_per_event.min())
            max_photons_per_event = int(photons_per_event.max())
        else:
            mean_photons_per_event = 0
            median_photons_per_event = 0
            std_photons_per_event = 0
            min_photons_per_event = 0
            max_photons_per_event = 0

        # Spatial characteristics
        if associated_photons > 0 and 'assoc_com_dist' in df.columns:
            com_distances = df[associated_mask]['assoc_com_dist']
            mean_com_distance = com_distances.mean()
            median_com_distance = com_distances.median()

            # Calculate spatial spread per event
            def calc_spatial_spread(group):
                if len(group) < 2:
                    return 0
                x_std = group['x'].std()
                y_std = group['y'].std()
                return np.sqrt(x_std**2 + y_std**2)

            spatial_spreads = df[associated_mask].groupby('assoc_event_id').apply(calc_spatial_spread)
            mean_spatial_spread = spatial_spreads.mean()
            median_spatial_spread = spatial_spreads.median()
            std_spatial_spread = spatial_spreads.std()
        else:
            mean_com_distance = 0
            median_com_distance = 0
            mean_spatial_spread = 0
            median_spatial_spread = 0
            std_spatial_spread = 0

        # Temporal characteristics
        if associated_photons > 0 and 'toa_ns' in df.columns:
            def calc_temporal_spread(group):
                if len(group) < 2:
                    return 0
                return group['toa_ns'].std()

            temporal_spreads = df[associated_mask].groupby('assoc_event_id').apply(calc_temporal_spread)
            mean_temporal_spread = temporal_spreads.mean()
            median_temporal_spread = temporal_spreads.median()
            std_temporal_spread = temporal_spreads.std()
        else:
            mean_temporal_spread = 0
            median_temporal_spread = 0
            std_temporal_spread = 0

        # Quality indicators
        # Over-associated: very large events or high spatial/temporal spread
        likely_over_associated = (
            (mean_photons_per_event > 20) or  # Unusually large events
            (mean_spatial_spread > 100) or     # Very large spatial spread
            (mean_temporal_spread > 5000)      # Very large temporal spread
        )

        # Under-associated: low association rate or many small events
        likely_under_associated = (
            (association_rate < 0.5) or                           # Less than half associated
            (median_photons_per_event < 3 and total_events > 10)  # Many tiny events
        )

        # Outliers: check if there are photons far from event centers
        spatial_outliers_detected = False
        temporal_outliers_detected = False

        if associated_photons > 0:
            if 'assoc_com_dist' in df.columns:
                # Outliers are photons > 3 sigma from mean CoM distance
                com_threshold = mean_com_distance + 3 * com_distances.std()
                spatial_outliers_detected = (com_distances > com_threshold).any()

            if mean_temporal_spread > 0:
                temporal_outliers_detected = (temporal_spreads > mean_temporal_spread + 3 * std_temporal_spread).any()

        self.metrics = AssociationQualityMetrics(
            total_photons=total_photons,
            associated_photons=int(associated_photons),
            unassociated_photons=int(unassociated_photons),
            total_events=total_events,
            association_rate=association_rate,
            mean_photons_per_event=mean_photons_per_event,
            median_photons_per_event=median_photons_per_event,
            std_photons_per_event=std_photons_per_event,
            min_photons_per_event=min_photons_per_event,
            max_photons_per_event=max_photons_per_event,
            mean_spatial_spread=mean_spatial_spread,
            median_spatial_spread=median_spatial_spread,
            std_spatial_spread=std_spatial_spread,
            mean_temporal_spread=mean_temporal_spread,
            median_temporal_spread=median_temporal_spread,
            std_temporal_spread=std_temporal_spread,
            mean_com_distance=mean_com_distance,
            median_com_distance=median_com_distance,
            likely_over_associated=likely_over_associated,
            likely_under_associated=likely_under_associated,
            spatial_outliers_detected=spatial_outliers_detected,
            temporal_outliers_detected=temporal_outliers_detected
        )

        if self.verbosity >= 1:
            print(self.metrics)

        return self.metrics

    def suggest_parameters(
        self,
        current_spatial_px: float = None,
        current_temporal_ns: float = None
    ) -> ParameterSuggestion:
        """
        Suggest improved parameters based on quality analysis.

        Args:
            current_spatial_px: Current spatial threshold (if known)
            current_temporal_ns: Current temporal threshold (if known)

        Returns:
            ParameterSuggestion object
        """
        if self.metrics is None:
            self.analyze_quality()

        # If current parameters not provided, estimate from data
        if current_spatial_px is None:
            current_spatial_px = self.metrics.median_spatial_spread * 2
        if current_temporal_ns is None:
            current_temporal_ns = self.metrics.median_temporal_spread * 2

        reasoning = []
        suggested_spatial = current_spatial_px
        suggested_temporal = current_temporal_ns
        confidence = 'medium'

        # Analyze and suggest adjustments

        # 1. Check association rate
        if self.metrics.association_rate < 0.3:
            # Very low association - parameters likely too tight
            suggested_spatial *= 1.5
            suggested_temporal *= 1.5
            reasoning.append(f"Association rate very low ({self.metrics.association_rate:.1%}) - increasing thresholds by 50%")
            confidence = 'high'
        elif self.metrics.association_rate < 0.6:
            # Low association - parameters somewhat tight
            suggested_spatial *= 1.2
            suggested_temporal *= 1.2
            reasoning.append(f"Association rate low ({self.metrics.association_rate:.1%}) - increasing thresholds by 20%")
            confidence = 'medium'

        # 2. Check if likely over-associated
        if self.metrics.likely_over_associated:
            if self.metrics.mean_photons_per_event > 20:
                suggested_spatial *= 0.7
                suggested_temporal *= 0.7
                reasoning.append(f"Events very large (avg {self.metrics.mean_photons_per_event:.1f} photons) - decreasing thresholds by 30%")
                confidence = 'high'
            else:
                suggested_spatial *= 0.85
                suggested_temporal *= 0.85
                reasoning.append("Events appear over-associated - decreasing thresholds by 15%")

        # 3. Check spatial spread
        if self.metrics.mean_spatial_spread > 0:
            # Suggest spatial threshold based on spatial spread
            # Good rule of thumb: threshold should be ~2x the typical spread
            target_spatial = self.metrics.median_spatial_spread * 2.0

            if abs(current_spatial_px - target_spatial) / current_spatial_px > 0.2:
                # More than 20% different - suggest adjustment
                suggested_spatial = target_spatial
                reasoning.append(
                    f"Spatial spread is {self.metrics.median_spatial_spread:.1f} px - "
                    f"suggesting threshold of {target_spatial:.1f} px (2x spread)"
                )

        # 4. Check temporal spread
        if self.metrics.mean_temporal_spread > 0:
            # Similar logic for temporal
            target_temporal = self.metrics.median_temporal_spread * 2.0

            if abs(current_temporal_ns - target_temporal) / current_temporal_ns > 0.2:
                suggested_temporal = target_temporal
                reasoning.append(
                    f"Temporal spread is {self.metrics.median_temporal_spread:.1f} ns - "
                    f"suggesting threshold of {target_temporal:.1f} ns (2x spread)"
                )

        # 5. Check for outliers
        if self.metrics.spatial_outliers_detected:
            suggested_spatial *= 0.9
            reasoning.append("Spatial outliers detected - tightening spatial threshold by 10%")

        if self.metrics.temporal_outliers_detected:
            suggested_temporal *= 0.9
            reasoning.append("Temporal outliers detected - tightening temporal threshold by 10%")

        # 6. Check if parameters look good
        if (0.6 <= self.metrics.association_rate <= 0.95 and
            5 <= self.metrics.mean_photons_per_event <= 15 and
            not self.metrics.likely_over_associated and
            not self.metrics.likely_under_associated):
            # Parameters look reasonable
            if not reasoning:
                reasoning.append("Current parameters appear reasonable - no major changes suggested")
                confidence = 'low'
                # Make minimal adjustments
                suggested_spatial = current_spatial_px
                suggested_temporal = current_temporal_ns

        # Calculate change factors
        spatial_change_factor = suggested_spatial / current_spatial_px if current_spatial_px > 0 else 1.0
        temporal_change_factor = suggested_temporal / current_temporal_ns if current_temporal_ns > 0 else 1.0

        # Round to reasonable precision
        suggested_spatial = round(suggested_spatial, 2)
        suggested_temporal = round(suggested_temporal, 2)

        self.suggestion = ParameterSuggestion(
            current_spatial_px=current_spatial_px,
            current_temporal_ns=current_temporal_ns,
            suggested_spatial_px=suggested_spatial,
            suggested_temporal_ns=suggested_temporal,
            spatial_change_factor=spatial_change_factor,
            temporal_change_factor=temporal_change_factor,
            reasoning=reasoning,
            confidence=confidence
        )

        if self.verbosity >= 1:
            print(self.suggestion)

        return self.suggestion

    def save_suggested_parameters(self, output_path: str):
        """
        Save suggested parameters to JSON file.

        Args:
            output_path: Path to output parameterSettings.json file
        """
        if self.suggestion is None:
            raise ValueError("No suggestions available. Run suggest_parameters() first.")

        params = self.suggestion.get_parameter_json()

        with open(output_path, 'w') as f:
            json.dump(params, f, indent=2)

        if self.verbosity >= 1:
            print(f"\nSuggested parameters saved to: {output_path}")
            print("Contents:")
            print(json.dumps(params, indent=2))

    def generate_matching_synthetic_data(
        self,
        n_events: int = 10,
        output_dir: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate synthetic data matching the characteristics of the real data.

        This can be used with the optimizer to find better parameters.

        Args:
            n_events: Number of synthetic events to generate
            output_dir: Optional directory to write synthetic data files

        Returns:
            Tuple of (photon_df, event_df) with synthetic data
        """
        if self.metrics is None:
            self.analyze_quality()

        # Create event configurations matching real data characteristics
        event_configs = []
        for i in range(n_events):
            config = {
                'event_id': i,
                'center_x': np.random.uniform(50, 200),
                'center_y': np.random.uniform(50, 200),
                't_ns': 1000.0 + i * 5000.0,  # Space them out in time
                'n_photons': max(3, int(np.random.normal(
                    self.metrics.mean_photons_per_event,
                    self.metrics.std_photons_per_event
                ))),
                'photon_spread_spatial': self.metrics.median_spatial_spread,
                'photon_spread_temporal': self.metrics.median_temporal_spread
            }
            event_configs.append(config)

        # Import synthetic data generation (avoiding circular import)
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'tests'))
        from test_association_validation import (
            create_synthetic_photon_data,
            create_synthetic_event_data,
            write_csv_files
        )

        photon_df = create_synthetic_photon_data(event_configs)
        event_df = create_synthetic_event_data(event_configs)

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            write_csv_files(None, photon_df, event_df, output_path, file_index=0)

            if self.verbosity >= 1:
                print(f"\nSynthetic data written to: {output_path}")
                print(f"  Events: {len(event_df)}")
                print(f"  Photons: {len(photon_df)}")

        return photon_df, event_df


def suggest_parameters_from_data(
    data_folder: str,
    current_spatial_px: float = None,
    current_temporal_ns: float = None,
    settings: Optional[str] = None,
    method: str = 'simple',
    output_path: Optional[str] = None,
    verbosity: int = 1
) -> ParameterSuggestion:
    """
    Convenience function to analyze data and suggest parameters.

    Args:
        data_folder: Folder containing photon/event CSV files
        current_spatial_px: Current spatial threshold
        current_temporal_ns: Current temporal threshold
        settings: Settings preset or path to settings file
        method: Association method to use
        output_path: Optional path to save suggested parameters
        verbosity: Output verbosity

    Returns:
        ParameterSuggestion object
    """
    # Load and associate
    analyser = nea.Analyse(
        data_folder=data_folder,
        settings=settings,
        n_threads=1
    )
    analyser.load(verbosity=0)

    # Use provided parameters or defaults
    if current_spatial_px is None:
        current_spatial_px = 20.0
    if current_temporal_ns is None:
        current_temporal_ns = 100.0

    analyser.associate_photons_events(
        method=method,
        dSpace_px=current_spatial_px,
        max_time_ns=current_temporal_ns
    )

    # Analyze and suggest
    suggester = ParameterSuggester(analyser, verbosity=verbosity)
    suggester.analyze_quality()
    suggestion = suggester.suggest_parameters(current_spatial_px, current_temporal_ns)

    # Save if requested
    if output_path:
        suggester.save_suggested_parameters(output_path)

    return suggestion


if __name__ == '__main__':
    print("Parameter Suggester for NEA")
    print("Import this module to use the ParameterSuggester class")
