#!/usr/bin/env python3
"""
Command-line interface for Neutron Event Analyzer.

This module provides two CLI tools:
- nea-assoc: For running pixel-photon-event association
- nea-optimize: For iterative parameter optimization
"""

import argparse
import sys
import os
import json
from pathlib import Path


# =============================================================================
# nea-assoc CLI - Association Tool
# =============================================================================

def detect_settings_file(data_folder):
    """
    Detect settings file in the data folder.

    Checks for .parameterSettings.json or parameterSettings.json.

    Args:
        data_folder (str): Path to data folder.

    Returns:
        str or None: Path to settings file if found, None otherwise.
    """
    # Try both hidden and non-hidden versions
    for filename in ['.parameterSettings.json', 'parameterSettings.json']:
        settings_path = os.path.join(data_folder, filename)
        if os.path.exists(settings_path):
            return settings_path
    return None


def validate_data_folder(data_folder):
    """
    Validate that the data folder exists and has expected structure.

    Args:
        data_folder (str): Path to data folder.

    Returns:
        dict: Information about what data is available.
    """
    if not os.path.exists(data_folder):
        print(f"❌ Error: Data folder not found: {data_folder}")
        sys.exit(1)

    # Check for expected subdirectories
    expected_dirs = {
        'events': ['eventFiles', 'EventFiles', 'ExportedEvents'],
        'photons': ['photonFiles', 'PhotonFiles', 'ExportedPhotons'],
        'pixels': ['tpx3Files', 'Tpx3Files', 'TPX3Files', 'ExportedPixels']
    }

    available = {}
    for data_type, possible_names in expected_dirs.items():
        for name in possible_names:
            path = os.path.join(data_folder, name)
            if os.path.exists(path):
                available[data_type] = True
                break
        else:
            available[data_type] = False

    return available


def create_assoc_parser():
    """Create argument parser for nea-assoc CLI."""
    from .config import DEFAULT_PARAMS

    parser = argparse.ArgumentParser(
        prog='nea-assoc',
        description='Neutron Event Analyzer - Associate pixels, photons, and events',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - analyze folder with auto-detected settings
  nea-assoc /path/to/run112_ZnS

  # Use specific settings preset
  nea-assoc /path/to/data --settings in_focus

  # Custom settings file
  nea-assoc /path/to/data --settings my_settings.json

  # Adjust verbosity and disable pixels
  nea-assoc /path/to/data --no-pixels -v 2

  # Full control over association parameters
  nea-assoc /path/to/data --pixel-max-dist 10 --photon-dspace 60

Available settings presets: in_focus, out_of_focus, fast_neutrons, hitmap

For more information, visit: https://github.com/nuclear/neutron_event_analyzer
        """
    )

    # Positional arguments
    parser.add_argument(
        'data_folder',
        type=str,
        help='Path to data folder containing eventFiles, photonFiles, and/or tpx3Files'
    )

    # Data loading options
    data_group = parser.add_argument_group('data loading options')
    data_group.add_argument(
        '--no-events',
        action='store_true',
        help='Do not load event data'
    )
    data_group.add_argument(
        '--no-photons',
        action='store_true',
        help='Do not load photon data'
    )
    data_group.add_argument(
        '--no-pixels',
        action='store_true',
        help='Do not load pixel data'
    )
    data_group.add_argument(
        '--limit',
        type=int,
        metavar='N',
        help='Limit number of rows loaded per file (for testing)'
    )
    data_group.add_argument(
        '--query',
        type=str,
        metavar='EXPR',
        help='Pandas query expression to filter data (e.g., "PSD > 0.5")'
    )

    # Settings and configuration
    settings_group = parser.add_argument_group('settings and configuration')
    settings_group.add_argument(
        '--settings', '-s',
        type=str,
        metavar='PRESET|FILE',
        help='Settings preset name (in_focus, out_of_focus, etc.) or path to JSON settings file. '
             'If not specified, will auto-detect .parameterSettings.json in data folder.'
    )
    settings_group.add_argument(
        '--empir-bin-dir',
        type=str,
        metavar='DIR',
        help='Directory containing empir export binaries (empir_export_events, empir_export_photons, empir_pixel2photon). '
             'Only needed if CSV files are not pre-exported and binaries are not in default location.'
    )
    settings_group.add_argument(
        '--threads', '-j',
        type=int,
        default=None,
        metavar='N',
        help='Number of threads for parallel processing (default: auto)'
    )

    # Association parameters
    assoc_group = parser.add_argument_group('association parameters')
    assoc_group.add_argument(
        '--pixel-max-dist',
        type=float,
        metavar='PIXELS',
        help='Maximum spatial distance for pixel-photon association (pixels)'
    )
    assoc_group.add_argument(
        '--pixel-max-time',
        type=float,
        metavar='NS',
        help='Maximum time difference for pixel-photon association (nanoseconds)'
    )
    assoc_group.add_argument(
        '--photon-dspace',
        type=float,
        metavar='PIXELS',
        help='Maximum center-of-mass distance for photon-event association (pixels)'
    )
    assoc_group.add_argument(
        '--max-time',
        type=float,
        metavar='NS',
        help='Maximum time window for associations (nanoseconds)'
    )
    assoc_group.add_argument(
        '--method',
        type=str,
        choices=['simple', 'kdtree', 'window', 'auto'],
        default='simple',
        help='Association method for photon-event association (default: simple)'
    )

    # Output options
    output_group = parser.add_argument_group('output options')
    output_group.add_argument(
        '--output-dir', '-o',
        type=str,
        metavar='DIR',
        help='Output directory (default: <data_folder>/AssociatedResults)'
    )
    output_group.add_argument(
        '--output-file', '-f',
        type=str,
        default='associated_data.csv',
        metavar='FILE',
        help='Output filename (default: associated_data.csv)'
    )
    output_group.add_argument(
        '--format',
        type=str,
        choices=['csv', 'parquet'],
        default='csv',
        help='Output file format (default: csv)'
    )

    # Verbosity and display
    display_group = parser.add_argument_group('display options')
    display_group.add_argument(
        '--verbose', '-v',
        action='count',
        default=1,
        help='Increase verbosity (can be repeated: -v, -vv)'
    )
    display_group.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress all output except errors'
    )
    display_group.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0.0'
    )

    return parser


def main_assoc():
    """Main entry point for nea-assoc CLI."""
    from . import Analyse
    from .config import DEFAULT_PARAMS

    parser = create_assoc_parser()
    args = parser.parse_args()

    # Handle quiet mode
    if args.quiet:
        verbosity = 0
    else:
        verbosity = args.verbose

    # Print banner
    if verbosity >= 1:
        print("=" * 70)
        print("Neutron Event Analyzer - Association Tool")
        print("=" * 70)

    # Validate data folder
    if verbosity >= 1:
        print(f"\n📁 Data folder: {args.data_folder}")

    available_data = validate_data_folder(args.data_folder)

    if verbosity >= 1:
        print(f"\n📊 Available data types:")
        print(f"   Events:  {'✓' if available_data['events'] else '✗'}")
        print(f"   Photons: {'✓' if available_data['photons'] else '✗'}")
        print(f"   Pixels:  {'✓' if available_data['pixels'] else '✗'}")

    # Determine what to load
    load_events = not args.no_events and available_data['events']
    load_photons = not args.no_photons and available_data['photons']
    load_pixels = not args.no_pixels and available_data['pixels']

    if not (load_events or load_photons or load_pixels):
        print("\n❌ Error: No data to load. Check your data folder structure.")
        sys.exit(1)

    # Detect or use settings
    settings = args.settings
    if settings is None:
        # Try to auto-detect settings file
        detected_settings = detect_settings_file(args.data_folder)
        if detected_settings:
            settings = detected_settings
            if verbosity >= 1:
                print(f"\n⚙️  Auto-detected settings: {os.path.basename(detected_settings)}")
        else:
            if verbosity >= 1:
                print("\n⚙️  Using default settings")
    else:
        if verbosity >= 1:
            if settings in DEFAULT_PARAMS:
                print(f"\n⚙️  Using settings preset: '{settings}'")
            else:
                print(f"\n⚙️  Using settings file: {settings}")

    # Initialize analyzer
    if verbosity >= 1:
        print(f"\n🔧 Initializing analyzer...")

    # Prepare analyzer kwargs
    analyser_kwargs = {
        'data_folder': args.data_folder,
        'settings': settings,
        'n_threads': args.threads
    }

    # Add empir binary directory if specified
    if args.empir_bin_dir:
        analyser_kwargs['export_dir'] = args.empir_bin_dir
        if verbosity >= 1:
            print(f"   Using empir binaries from: {args.empir_bin_dir}")

    try:
        analyser = Analyse(**analyser_kwargs)
    except Exception as e:
        print(f"\n❌ Error initializing analyzer: {e}")
        sys.exit(1)

    # Load data
    if verbosity >= 1:
        print(f"\n📥 Loading data...")
        print(f"   Events:  {'enabled' if load_events else 'disabled'}")
        print(f"   Photons: {'enabled' if load_photons else 'disabled'}")
        print(f"   Pixels:  {'enabled' if load_pixels else 'disabled'}")

    try:
        analyser.load(
            events=load_events,
            photons=load_photons,
            pixels=load_pixels,
            limit=args.limit,
            query=args.query,
            verbosity=verbosity
        )
    except Exception as e:
        print(f"\n❌ Error loading data: {e}")
        sys.exit(1)

    # Perform association
    if verbosity >= 1:
        print(f"\n🔗 Performing association...")

    # Build association parameters
    assoc_params = {
        'verbosity': verbosity,
        'method': args.method
    }

    if args.pixel_max_dist is not None:
        assoc_params['pixel_max_dist_px'] = args.pixel_max_dist
    if args.pixel_max_time is not None:
        assoc_params['pixel_max_time_ns'] = args.pixel_max_time
    if args.photon_dspace is not None:
        assoc_params['photon_dSpace_px'] = args.photon_dspace
    if args.max_time is not None:
        assoc_params['max_time_ns'] = args.max_time

    try:
        if load_pixels and load_photons and load_events:
            # Full 3-tier association
            result = analyser.associate(**assoc_params)
        elif load_photons and load_events:
            # Photon-event association only
            result = analyser.associate_photons_events(**assoc_params)
        elif load_pixels and load_photons:
            # Pixel-photon association only
            result = analyser.associate(**assoc_params)
        else:
            print("\n❌ Error: Not enough data types loaded for association.")
            print("   Association requires at least two data types (pixels+photons or photons+events)")
            sys.exit(1)

        if result is None or len(result) == 0:
            print("\n⚠️  Warning: Association produced no results")
            sys.exit(0)

    except Exception as e:
        print(f"\n❌ Error during association: {e}")
        import traceback
        if verbosity >= 2:
            traceback.print_exc()
        sys.exit(1)

    # Save results
    if verbosity >= 1:
        print(f"\n💾 Saving results...")

    try:
        output_path = analyser.save_associations(
            output_dir=args.output_dir,
            filename=args.output_file,
            format=args.format,
            verbosity=verbosity
        )

        if verbosity >= 1:
            print(f"\n✅ Analysis complete!")
            print(f"   Results saved to: {output_path}")
            print(f"   README available at: {os.path.join(os.path.dirname(output_path), 'README.md')}")

    except Exception as e:
        print(f"\n❌ Error saving results: {e}")
        sys.exit(1)

    if verbosity >= 1:
        print("\n" + "=" * 70)


# =============================================================================
# nea-optimize CLI - Parameter Optimization Tool
# =============================================================================

def cmd_optimize(args):
    """Iteratively optimize association parameters on real data."""
    from neutron_event_analyzer.iterative_optimizer import IterativeOptimizer

    print("Neutron Event Analyzer - Parameter Optimization")
    print("=" * 70)

    optimizer = IterativeOptimizer(
        data_folder=args.data_folder,
        initial_spatial_px=args.spatial,
        initial_temporal_ns=args.temporal,
        settings=args.settings,
        method=args.method,
        verbosity=args.verbose
    )

    best_result = optimizer.optimize(
        max_iterations=args.iterations,
        convergence_threshold=args.convergence,
        output_dir=args.output
    )

    # Print final results
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"\nBest parameters (iteration {best_result.iteration}):")
    print(f"  Spatial threshold:  {best_result.spatial_px:.2f} px")
    print(f"  Temporal threshold: {best_result.temporal_ns:.2f} ns")
    print(f"\nQuality metrics:")
    print(f"  Association rate:   {best_result.association_rate:.2%}")
    print(f"  Events found:       {best_result.total_events}")
    print(f"  Photons per event:  {best_result.mean_photons_per_event:.1f}")

    if args.output:
        print(f"\n✓ Results saved to: {args.output}")
        print(f"  Use: {args.output}/best_parameters.json")

    # Show progress table
    if args.verbose >= 1:
        df = optimizer.get_progress_dataframe()
        print(f"\n\nIteration History:")
        print("=" * 70)
        print(f"{'Iter':<6} {'Spatial (px)':<15} {'Temporal (ns)':<15} {'Assoc Rate':<12} {'Events':<8}")
        print("-" * 70)
        for _, row in df.iterrows():
            print(f"{int(row['iteration']):<6} {row['spatial_px']:<15.2f} "
                  f"{row['temporal_ns']:<15.2f} {row['association_rate']:<12.2%} "
                  f"{int(row['total_events']):<8}")

    return 0


def cmd_suggest(args):
    """Analyze data and suggest improved parameters."""
    from neutron_event_analyzer.parameter_suggester import suggest_parameters_from_data

    print("Neutron Event Analyzer - Parameter Suggestion")
    print("=" * 70)

    suggestion = suggest_parameters_from_data(
        data_folder=args.data_folder,
        current_spatial_px=args.spatial,
        current_temporal_ns=args.temporal,
        settings=args.settings,
        method=args.method,
        output_path=args.output,
        verbosity=args.verbose
    )

    if args.output:
        print(f"\n✓ Suggested parameters saved to: {args.output}")

    return 0


def cmd_analyze(args):
    """Analyze association quality without suggesting changes."""
    import neutron_event_analyzer as nea
    from neutron_event_analyzer.parameter_suggester import ParameterSuggester

    print("Neutron Event Analyzer - Association Quality Analysis")
    print("=" * 70)

    # Load and associate
    analyser = nea.Analyse(
        data_folder=args.data_folder,
        settings=args.settings,
        n_threads=1
    )
    analyser.load(verbosity=0)
    analyser.associate_photons_events(
        method=args.method,
        dSpace_px=args.spatial,
        max_time_ns=args.temporal
    )

    # Analyze
    suggester = ParameterSuggester(analyser, verbosity=args.verbose)
    metrics = suggester.analyze_quality()

    # Save if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)
        print(f"\n✓ Metrics saved to: {output_path}")

    return 0


def main_optimize():
    """Main CLI entry point for nea-optimize."""
    parser = argparse.ArgumentParser(
        description='Neutron Event Analyzer - Parameter Optimization Tools',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Optimize parameters iteratively
  nea-optimize optimize data/ --iterations 5 --output results/

  # Quick parameter suggestion
  nea-optimize suggest data/ --spatial 20 --temporal 100

  # Analyze current association quality
  nea-optimize analyze data/ --spatial 20 --temporal 100
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    subparsers.required = True

    # Optimize command
    optimize_parser = subparsers.add_parser(
        'optimize',
        help='Iteratively optimize parameters on real data'
    )
    optimize_parser.add_argument(
        'data_folder',
        help='Folder containing photon/event data'
    )
    optimize_parser.add_argument(
        '--spatial', '-s',
        type=float,
        default=20.0,
        help='Initial spatial threshold (pixels, default: 20.0)'
    )
    optimize_parser.add_argument(
        '--temporal', '-t',
        type=float,
        default=100.0,
        help='Initial temporal threshold (nanoseconds, default: 100.0)'
    )
    optimize_parser.add_argument(
        '--iterations', '-n',
        type=int,
        default=5,
        help='Maximum number of iterations (default: 5)'
    )
    optimize_parser.add_argument(
        '--convergence', '-c',
        type=float,
        default=0.05,
        help='Convergence threshold (default: 0.05)'
    )
    optimize_parser.add_argument(
        '--method', '-m',
        choices=['simple', 'kdtree', 'window', 'lumacam'],
        default='simple',
        help='Association method (default: simple)'
    )
    optimize_parser.add_argument(
        '--settings',
        help='Settings preset or path to settings file'
    )
    optimize_parser.add_argument(
        '--output', '-o',
        help='Output directory for results'
    )
    optimize_parser.add_argument(
        '--verbose', '-v',
        action='count',
        default=1,
        help='Increase verbosity (use -vv for more detail)'
    )
    optimize_parser.set_defaults(func=cmd_optimize)

    # Suggest command
    suggest_parser = subparsers.add_parser(
        'suggest',
        help='Analyze data and suggest improved parameters'
    )
    suggest_parser.add_argument(
        'data_folder',
        help='Folder containing photon/event data'
    )
    suggest_parser.add_argument(
        '--spatial', '-s',
        type=float,
        default=20.0,
        help='Current spatial threshold (pixels, default: 20.0)'
    )
    suggest_parser.add_argument(
        '--temporal', '-t',
        type=float,
        default=100.0,
        help='Current temporal threshold (nanoseconds, default: 100.0)'
    )
    suggest_parser.add_argument(
        '--method', '-m',
        choices=['simple', 'kdtree', 'window', 'lumacam'],
        default='simple',
        help='Association method (default: simple)'
    )
    suggest_parser.add_argument(
        '--settings',
        help='Settings preset or path to settings file'
    )
    suggest_parser.add_argument(
        '--output', '-o',
        help='Output file for suggested parameters (JSON)'
    )
    suggest_parser.add_argument(
        '--verbose', '-v',
        action='count',
        default=1,
        help='Increase verbosity'
    )
    suggest_parser.set_defaults(func=cmd_suggest)

    # Analyze command
    analyze_parser = subparsers.add_parser(
        'analyze',
        help='Analyze association quality metrics'
    )
    analyze_parser.add_argument(
        'data_folder',
        help='Folder containing photon/event data'
    )
    analyze_parser.add_argument(
        '--spatial', '-s',
        type=float,
        default=20.0,
        help='Spatial threshold (pixels, default: 20.0)'
    )
    analyze_parser.add_argument(
        '--temporal', '-t',
        type=float,
        default=100.0,
        help='Temporal threshold (nanoseconds, default: 100.0)'
    )
    analyze_parser.add_argument(
        '--method', '-m',
        choices=['simple', 'kdtree', 'window', 'lumacam'],
        default='simple',
        help='Association method (default: simple)'
    )
    analyze_parser.add_argument(
        '--settings',
        help='Settings preset or path to settings file'
    )
    analyze_parser.add_argument(
        '--output', '-o',
        help='Output file for metrics (JSON)'
    )
    analyze_parser.add_argument(
        '--verbose', '-v',
        action='count',
        default=1,
        help='Increase verbosity'
    )
    analyze_parser.set_defaults(func=cmd_analyze)

    # Parse and execute
    args = parser.parse_args()
    return args.func(args)


# =============================================================================
# nea-empir CLI - EMPIR Parameter Optimization Tool
# =============================================================================

def main_empir():
    """Main CLI entry point for nea-empir (EMPIR parameter optimization)."""
    parser = argparse.ArgumentParser(
        description='Neutron Event Analyzer - EMPIR Parameter Optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EMPIR Parameter Optimization Framework

This tool analyzes intrinsic distribution shapes from EMPIR reconstruction
to suggest optimal parameters WITHOUT requiring ground truth or event association.

Examples:
  # Optimize photon-to-event parameters
  nea-empir /path/to/data --stage photon2event

  # Optimize pixel-to-photon parameters
  nea-empir /path/to/data --stage pixel2photon

  # Optimize both stages
  nea-empir /path/to/data --stage both --output optimized_params.json

  # Use current parameter values as starting point
  nea-empir /path/to/data --current-params settings.json --output new_params.json

Available stages:
  - pixel2photon:  Optimize dSpace, dTime, nPxMin, nPxMax
  - photon2event:  Optimize dSpace_px, dTime_s, durationMax_s
  - both:          Optimize all parameters (default)

How it works:
  This tool analyzes statistical signatures in reconstruction outputs:
  - Temporal clustering quality from intra-cluster time differences
  - Spatial clustering quality from cluster spread distributions
  - Event quality from multiplicity and duration distributions

  Each parameter is optimized independently based on the specific
  distribution it affects, without requiring ground truth data.

For more information, see the EMPIR Parameter Optimization Framework documentation.
        """
    )

    # Positional arguments
    parser.add_argument(
        'data_folder',
        type=str,
        help='Path to folder containing data (pixels, photons, and/or events)'
    )

    # Optimization options
    opt_group = parser.add_argument_group('optimization options')
    opt_group.add_argument(
        '--stage', '-s',
        type=str,
        choices=['pixel2photon', 'photon2event', 'both'],
        default='both',
        help='Which reconstruction stage to optimize (default: both)'
    )
    opt_group.add_argument(
        '--current-params',
        type=str,
        metavar='FILE',
        help='JSON file with current parameter values (for comparison)'
    )
    opt_group.add_argument(
        '--empir-binaries',
        type=str,
        metavar='DIR',
        help='Path to directory containing EMPIR binaries (empir_export_events, etc.). Can also be set via EMPIR_PATH environment variable.'
    )

    # Output options
    output_group = parser.add_argument_group('output options')
    output_group.add_argument(
        '--output', '-o',
        type=str,
        metavar='FILE',
        help='Output file for suggested parameters (JSON format, default: print to console)'
    )

    # Display options
    display_group = parser.add_argument_group('display options')
    display_group.add_argument(
        '--verbose', '-v',
        action='count',
        default=1,
        help='Increase verbosity (use -vv for more detail)'
    )
    display_group.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress all output except results'
    )
    display_group.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0.0'
    )

    args = parser.parse_args()

    # Handle verbosity
    verbosity = 0 if args.quiet else args.verbose

    # Determine EMPIR binaries path: CLI arg > env var > default
    empir_binaries = args.empir_binaries or os.environ.get('EMPIR_PATH', './export')

    # Print banner
    if verbosity >= 1:
        print("=" * 70)
        print("EMPIR Parameter Optimization Framework")
        print("=" * 70)
        print(f"\nData folder: {args.data_folder}")
        print(f"Optimization stage: {args.stage}")
        if args.empir_binaries:
            print(f"EMPIR binaries: {empir_binaries}")
        elif os.environ.get('EMPIR_PATH'):
            print(f"EMPIR binaries: {empir_binaries} (from EMPIR_PATH)")

    # Load current parameters if provided
    current_params = None
    if args.current_params:
        try:
            with open(args.current_params, 'r') as f:
                current_params = json.load(f)
            if verbosity >= 1:
                print(f"Using current parameters from: {args.current_params}")
        except Exception as e:
            print(f"⚠️  Warning: Could not load current parameters: {e}")
            print("    Using default values instead.")

    # Set default output path if not specified
    if not args.output:
        args.output = os.path.join(args.data_folder, ".suggestedSettingsParameters.json")

    # Run optimization
    try:
        from neutron_event_analyzer.empir_optimizer import optimize_empir_parameters

        results = optimize_empir_parameters(
            data_folder=args.data_folder,
            stage=args.stage,
            current_params=current_params,
            output_path=args.output,
            verbosity=verbosity,
            empir_binaries=empir_binaries
        )

        # Display results with statistics
        if verbosity >= 1:
            for stage_name, suggestion in results.items():
                print(suggestion)

                # Display diagnostic statistics if available
                if suggestion.diagnostics:
                    print(f"\n📊 Diagnostic Statistics for {stage_name}:")
                    print("─" * 70)
                    for metric, value in suggestion.diagnostics.items():
                        if isinstance(value, dict):
                            # Display nested dict values with indentation
                            print(f"  {metric}:")
                            for sub_key, sub_value in value.items():
                                if isinstance(sub_value, float):
                                    print(f"    {sub_key:28s}: {sub_value:>12.4g}")
                                elif isinstance(sub_value, int):
                                    print(f"    {sub_key:28s}: {sub_value:>12,d}")
                                else:
                                    print(f"    {sub_key:28s}: {str(sub_value):>12}")
                        elif isinstance(value, float):
                            print(f"  {metric:30s}: {value:>12.4g}")
                        elif isinstance(value, int):
                            print(f"  {metric:30s}: {value:>12,d}")
                        else:
                            print(f"  {metric:30s}: {str(value):>12}")

        if verbosity >= 1:
            print(f"\n✅ Suggested parameters saved to: {args.output}")
            print("\nNext steps:")
            print(f"  1. Review the suggestions in {args.output}")
            print("  2. Use these parameters in your EMPIR reconstruction")
            print("  3. Re-run analysis to verify improvement")

    except Exception as e:
        print(f"\n❌ Error during optimization: {e}")
        if verbosity >= 2:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    if verbosity >= 1:
        print("\n" + "=" * 70)

    return 0


if __name__ == '__main__':
    # Support all entry points when run directly
    prog_name = os.path.basename(sys.argv[0])
    if 'empir' in prog_name:
        sys.exit(main_empir())
    elif 'optimize' in prog_name:
        sys.exit(main_optimize())
    else:
        sys.exit(main_assoc())
