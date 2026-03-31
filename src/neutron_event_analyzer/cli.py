#!/usr/bin/env python3
"""
nea-assoc: Associate neutron event camera data.

Assumes ExportedPixels/, ExportedPhotons/, and ExportedEvents/ directories already
exist under the data folder (produced by empirun). Associates the data across tiers
and writes AssociatedResults/associated_data[_suffix].csv.
"""

import argparse
import sys
import os
import json
from pathlib import Path

# The EMPIR pixel-photon bit-matrix operations are small (<<1000 elements).
# NumPy's BLAS/OpenBLAS spawns worker threads that spend more time waiting
# on locks than doing useful work.  Force single-threaded numpy to eliminate
# the overhead (~20s saved on a typical PTB dataset).
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')


# =============================================================================
# Simple / Advanced help toggle
# =============================================================================

def _is_advanced():
    """Check whether --advanced flag is present in sys.argv."""
    return '--advanced' in sys.argv


_SIMPLE_EPILOG = """
Examples:
  nea-assoc ./run112_ZnS
  nea-assoc ./data --settings in_focus
  nea-assoc ./data --method ml
  nea-assoc ./data --suffix run1

Settings presets: in_focus, out_of_focus, fast_neutrons, hitmap
Association methods: empir (default), simple, kdtree, window, mystic, ml

Run 'nea-assoc --advanced --help' to see all options.
"""

_ADVANCED_EPILOG = """
Examples:
  nea-assoc ./data --method ml --suffix run1
  nea-assoc ./data --photon-dspace 60 --max-time 500
  nea-assoc ./data --relax 1.5 --no-pixels --format parquet
  nea-assoc ./data --limit 5000 --query "PSD > 0.5"

Settings presets: in_focus, out_of_focus, fast_neutrons, hitmap
Association methods: simple, kdtree, window, mystic, ml
"""


class _SmartHelpFormatter(argparse.HelpFormatter):
    """
    Custom formatter that hides advanced options when ``--advanced`` is absent.

    Advanced actions are marked with ``action._advanced = True`` after
    ``parser.add_argument()``.
    """

    def add_arguments(self, actions):
        if not _is_advanced():
            actions = [a for a in actions if not getattr(a, '_advanced', False)]
        super().add_arguments(actions)

    def _format_usage(self, usage, actions, groups, prefix):
        if not _is_advanced():
            actions = [a for a in actions if not getattr(a, '_advanced', False)]
        return super()._format_usage(usage, actions, groups, prefix)


def _mark_advanced(action):
    """Mark an argparse Action as advanced (hidden from simple help)."""
    action._advanced = True
    return action


# =============================================================================
# Parser
# =============================================================================

def create_assoc_parser():
    """Build the argument parser for nea-assoc."""
    epilog = _ADVANCED_EPILOG if _is_advanced() else _SIMPLE_EPILOG

    parser = argparse.ArgumentParser(
        prog='nea-assoc',
        description=(
            'Associate neutron event camera data: pixels -> photons -> events.\n'
            'Reads from ExportedPixels/, ExportedPhotons/, ExportedEvents/ directories.'
        ),
        formatter_class=_SmartHelpFormatter,
        epilog=epilog,
    )

    # ---- Positional --------------------------------------------------------
    parser.add_argument(
        'data',
        type=str,
        help='Path to data folder',
    )

    # ---- Core options (always visible) -------------------------------------
    parser.add_argument(
        '--settings', '-s',
        type=str,
        metavar='PRESET|FILE',
        help=(
            'Settings preset (in_focus, out_of_focus, fast_neutrons, hitmap) '
            'or path to parameterSettings JSON. Auto-detected if omitted.'
        ),
    )
    parser.add_argument(
        '--method', '-m',
        type=str,
        choices=['simple', 'kdtree', 'window', 'mystic', 'ml', 'empir'],
        default='empir',
        help='Association method (default: empir)',
    )
    parser.add_argument(
        '--suffix',
        type=str,
        metavar='TEXT',
        default=None,
        help="Suffix for output files, e.g. 'run1' → associated_data_run1.csv",
    )
    parser.add_argument(
        '--verbose', '-v',
        action='count',
        default=1,
        help='Increase verbosity (-vv for debug output)',
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress all output except errors',
    )
    parser.add_argument(
        '--tag',
        action='store_true',
        default=False,
        help='Compute and append photon/event topology tags (ph/tags, ev/tags)',
    )
    parser.add_argument(
        '--advanced',
        action='store_true',
        default=False,
        help='Show advanced options in --help',
    )

    # ---- Data loading (advanced) -------------------------------------------
    _mark_advanced(parser.add_argument(
        '--no-events',
        action='store_true',
        help='Skip loading event data',
    ))
    _mark_advanced(parser.add_argument(
        '--no-photons',
        action='store_true',
        help='Skip loading photon data',
    ))
    _mark_advanced(parser.add_argument(
        '--no-pixels',
        action='store_true',
        help='Skip loading pixel data',
    ))
    _mark_advanced(parser.add_argument(
        '--limit',
        type=int,
        metavar='N',
        help='Limit rows loaded per CSV file (useful for quick tests)',
    ))
    _mark_advanced(parser.add_argument(
        '--query',
        type=str,
        metavar='EXPR',
        help='Pandas query expression to filter loaded data (e.g. "PSD > 0.5")',
    ))

    # ---- Association parameters (advanced) ---------------------------------
    _mark_advanced(parser.add_argument(
        '--pixel-max-dist',
        type=float,
        metavar='PIXELS',
        help='Max spatial distance for pixel-photon association (pixels)',
    ))
    _mark_advanced(parser.add_argument(
        '--pixel-max-time',
        type=float,
        metavar='NS',
        help='Max time window for pixel-photon association (nanoseconds)',
    ))
    _mark_advanced(parser.add_argument(
        '--photon-dspace',
        type=float,
        metavar='PIXELS',
        help='Max CoM distance for photon-event association (pixels)',
    ))
    _mark_advanced(parser.add_argument(
        '--max-time',
        type=float,
        metavar='NS',
        help='Max time window for photon-event association (nanoseconds)',
    ))
    _mark_advanced(parser.add_argument(
        '--min-pixels',
        type=int,
        metavar='N',
        default=None,
        help='Minimum pixels per photon for empir method (default: from settings or 1)',
    ))
    _mark_advanced(parser.add_argument(
        '--relax',
        type=float,
        metavar='FACTOR',
        default=None,
        help='Scale all association parameters by this factor (e.g. 1.5 = 50%% more relaxed)',
    ))

    # ---- Simulation truth merge --------------------------------------------
    parser.add_argument(
        '--merge-sim',
        action='store_true',
        default=False,
        help=(
            'After association, join AssociatedResults → TracedPhotons → SimPhotons. '
            'Uses the parent of the data folder as the archive root (where '
            'TracedPhotons/ and SimPhotons/ live). '
            'Saves combined[_suffix].csv alongside associated_data[_suffix].csv.'
        ),
    )
    parser.add_argument(
        '--merge-only',
        action='store_true',
        default=False,
        help=(
            'Skip association entirely; only run the sim truth merge on an existing '
            'AssociatedResults/associated_data[_suffix].csv. Implies --merge-sim.'
        ),
    )

    # ---- Output (advanced) -------------------------------------------------
    _mark_advanced(parser.add_argument(
        '--output-dir', '-o',
        type=str,
        metavar='DIR',
        help='Output directory (default: <data>/AssociatedResults/)',
    ))
    _mark_advanced(parser.add_argument(
        '--format',
        type=str,
        choices=['csv', 'parquet'],
        default='csv',
        help='Output file format (default: csv)',
    ))

    # ---- Performance (advanced) --------------------------------------------
    _mark_advanced(parser.add_argument(
        '--threads', '-j',
        type=int,
        default=None,
        metavar='N',
        help='Number of threads for parallel processing (default: 10)',
    ))

    return parser


# =============================================================================
# Main entry point
# =============================================================================

def main_assoc():
    """Entry point for the ``nea-assoc`` CLI command."""
    from .analyser import Analyse

    parser = create_assoc_parser()
    args = parser.parse_args()

    verbosity = 0 if args.quiet else args.verbose

    # ------------------------------------------------------------------
    # Banner
    # ------------------------------------------------------------------
    if verbosity >= 1:
        print("=" * 60)
        print("Neutron Event Analyzer  |  nea-assoc")
        print("=" * 60)
        print(f"Data: {args.data}")
        if args.settings:
            print(f"Settings: {args.settings}")
        print(f"Method: {args.method}")
        if args.suffix:
            print(f"Suffix: {args.suffix}")

    if not args.merge_only:
        # ------------------------------------------------------------------
        # Initialise analyser
        # ------------------------------------------------------------------
        try:
            analyser = Analyse(
                data_folder=args.data,
                settings=args.settings,
                n_threads=args.threads or 10,
                verbosity=verbosity,
                auto_load=False,
            )
        except Exception as e:
            print(f"Error: {e}")
            if verbosity >= 2:
                import traceback
                traceback.print_exc()
            sys.exit(1)

        # ------------------------------------------------------------------
        # Load data
        # ------------------------------------------------------------------
        try:
            analyser.load(
                events=not args.no_events,
                photons=not args.no_photons,
                pixels=not args.no_pixels,
                limit=args.limit,
                query=args.query,
                verbosity=verbosity,
            )
        except Exception as e:
            print(f"Error loading data: {e}")
            if verbosity >= 2:
                import traceback
                traceback.print_exc()
            sys.exit(1)

        # ------------------------------------------------------------------
        # Associate
        # ------------------------------------------------------------------
        assoc_kwargs = {
            'method': args.method,
            'verbosity': verbosity,
            'suffix': args.suffix,
        }
        if args.relax is not None:
            assoc_kwargs['relax'] = args.relax
        if args.pixel_max_dist is not None:
            assoc_kwargs['pixel_max_dist_px'] = args.pixel_max_dist
        if args.pixel_max_time is not None:
            assoc_kwargs['pixel_max_time_ns'] = args.pixel_max_time
        if args.photon_dspace is not None:
            assoc_kwargs['photon_dSpace_px'] = args.photon_dspace
        if args.max_time is not None:
            assoc_kwargs['max_time_ns'] = args.max_time
        if args.min_pixels is not None:
            assoc_kwargs['min_pixels'] = args.min_pixels

        try:
            analyser.associate(**assoc_kwargs)
        except Exception as e:
            print(f"Error during association: {e}")
            if verbosity >= 2:
                import traceback
                traceback.print_exc()
            sys.exit(1)

        # ------------------------------------------------------------------
        # Topology tagging (optional)
        # ------------------------------------------------------------------
        if args.tag:
            df = analyser.associated_df
            _PX_COLS = {"px/x", "px/y", "px/toa", "px/tot"}
            if df is None or len(df) == 0:
                print("Warning: --tag skipped — association produced no data")
            elif not _PX_COLS.issubset(df.columns):
                missing = _PX_COLS - set(df.columns)
                print(
                    f"Warning: --tag skipped — pixel columns missing: {sorted(missing)}. "
                    "Tagging requires pixel-level data (re-run without --no-pixels)."
                )
            else:
                from .tagging import add_topology_tags
                if verbosity >= 1:
                    print("Computing topology tags...")
                try:
                    analyser.associated_df = add_topology_tags(analyser.associated_df)
                    # Re-save so the auto-saved file includes the tag columns.
                    analyser.save_associations(
                        format=args.format,
                        suffix=args.suffix,
                        verbosity=verbosity,
                    )
                    if verbosity >= 1:
                        print("Tags added: ph/tags, ev/tags")
                except Exception as e:
                    print(f"Error during tagging: {e}")
                    if verbosity >= 2:
                        import traceback
                        traceback.print_exc()
                    sys.exit(1)

    # ------------------------------------------------------------------
    # Save to user-specified output dir (if given)
    # associate() already auto-saves to AssociatedResults/ by default.
    # ------------------------------------------------------------------
    if args.output_dir and not args.merge_only:
        try:
            out = analyser.save_associations(
                output_dir=args.output_dir,
                format=args.format,
                suffix=args.suffix,
                verbosity=verbosity,
            )
            if verbosity >= 1:
                print(f"Also saved to: {out}")
        except Exception as e:
            print(f"Error saving to {args.output_dir}: {e}")
            sys.exit(1)

    # ------------------------------------------------------------------
    # Merge simulation truth tables if requested
    # ------------------------------------------------------------------
    if args.merge_sim or args.merge_only:
        from .analyser import build_combined
        archive = Path(args.data).parent
        if verbosity >= 1:
            print(f"Merging sim truth from: {archive}")
        try:
            combined = build_combined(
                run_dir=Path(args.data),
                archive=archive,
                suffix=args.suffix or '',
                verbose=verbosity >= 1,
            )
            stem = f"associated_data_{args.suffix}" if args.suffix else "associated_data"
            ext  = '.parquet' if args.format == 'parquet' else '.csv'
            out_path = Path(args.output_dir) if args.output_dir else Path(args.data) / 'AssociatedResults'
            out_path.mkdir(parents=True, exist_ok=True)
            out_file = out_path / (stem + ext)
            if args.format == 'parquet':
                combined.to_parquet(out_file, index=False)
            else:
                combined.to_csv(out_file, index=False)
            if verbosity >= 1:
                print(f"Saved → {out_file}  ({len(combined):,} rows × {len(combined.columns)} cols)")
        except Exception as e:
            print(f"Error during sim merge: {e}")
            if verbosity >= 2:
                import traceback
                traceback.print_exc()

    if verbosity >= 1:
        print("=" * 60)


if __name__ == '__main__':
    sys.exit(main_assoc())
