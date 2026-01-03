#!/usr/bin/env python3
"""
Command-line interface for Neutron Event Analyzer (nea-assoc).

This tool performs pixel-photon-event association for neutron detector data.
"""

import argparse
import sys
import os
from pathlib import Path
from . import Analyse
from .config import DEFAULT_PARAMS


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


def create_parser():
    """Create argument parser for nea-assoc CLI."""
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


def main():
    """Main entry point for nea-assoc CLI."""
    parser = create_parser()
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

    try:
        analyser = Analyse(
            data_folder=args.data_folder,
            settings=settings,
            n_threads=args.threads
        )
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


if __name__ == '__main__':
    main()
