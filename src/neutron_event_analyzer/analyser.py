import os
import glob
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from scipy.spatial import cKDTree
import logging
import json
from pathlib import Path
from .config import DEFAULT_PARAMS

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class Analyse:
    def __init__(self, data_folder, settings=None, n_threads=10, verbosity=1):
        """
        Initialize the Analyse object.

        Assumes ExportedPixels, ExportedPhotons, and ExportedEvents directories already
        exist under data_folder (produced by empirun or EMPIR binaries). The role of
        this class is purely to associate those exports and produce an Associated table.

        Args:
            data_folder (str): Path to the data folder containing ExportedEvents,
                               ExportedPhotons, and/or ExportedPixels subdirectories.
            settings (str or dict, optional): Path to parameterSettings JSON file,
                                              named preset ('in_focus', 'out_of_focus',
                                              'fast_neutrons', 'hitmap'), or dict.
                                              Auto-detected from data_folder if None.
            n_threads (int): Number of threads for parallel association (default: 10).
            verbosity (int): 0=silent, 1=progress bars, 2=detailed. Default: 1.
        """
        self.data_folder = data_folder
        self.n_threads = n_threads
        self.verbosity = verbosity

        self.events_df = None
        self.photons_df = None
        self.pixels_df = None
        self.associated_df = None
        self.assoc_method = None
        self.last_assoc_stats = None
        self.last_photon_event_stats = None
        self._ml_association_model = None

        # Auto-detect settings
        if settings is None:
            settings = self._detect_settings_file()
        self.settings = self._load_settings(settings)
        self.settings_source = self._get_settings_source(settings)

        if verbosity >= 2 and self.settings:
            print(f"Settings: {self.settings_source}")

        # Auto-load pre-existing association results
        assoc_file = os.path.join(data_folder, "AssociatedResults", "associated_data.csv")
        stats_file = os.path.join(data_folder, "AssociatedResults", "association_stats.json")
        if os.path.exists(assoc_file):
            try:
                self.associated_df = pd.read_csv(assoc_file)
                if os.path.exists(stats_file):
                    with open(stats_file) as f:
                        stats_dict = json.load(f)
                    self.last_assoc_stats = stats_dict.get('pixel_photon')
                    self.last_photon_event_stats = stats_dict.get('photon_event')
                if verbosity >= 1:
                    print(f"Auto-loaded {len(self.associated_df):,} rows from AssociatedResults/")
            except Exception as e:
                if verbosity >= 2:
                    print(f"Could not load AssociatedResults: {e}")

        # Auto-load pre-existing ML model
        ml_model_file = os.path.join(data_folder, "AssociatedResults", "ml_association_model.joblib")
        if os.path.exists(ml_model_file):
            try:
                import joblib
                self._ml_association_model = joblib.load(ml_model_file)
                if verbosity >= 1:
                    print(f"Auto-loaded ML model from AssociatedResults/")
            except Exception:
                pass

        # Load raw data
        self.load(verbosity=verbosity)

    # =========================================================================
    # Settings helpers
    # =========================================================================

    def _load_settings(self, settings):
        if settings is None:
            return {}
        if isinstance(settings, dict):
            return settings
        if isinstance(settings, str):
            if settings in DEFAULT_PARAMS:
                return DEFAULT_PARAMS[settings]
            if os.path.exists(settings):
                try:
                    with open(settings) as f:
                        return json.load(f)
                except Exception as e:
                    print(f"Warning: could not load settings from {settings}: {e}")
            else:
                print(f"Warning: settings not found and not a preset: {settings}")
                print(f"Available presets: {list(DEFAULT_PARAMS.keys())}")
        return {}

    def _detect_settings_file(self):
        for name in ['.parameterSettings.json', 'parameterSettings.json']:
            path = os.path.join(self.data_folder, name)
            if os.path.exists(path):
                return path
        return None

    def _get_settings_source(self, settings):
        if settings is None:
            return "defaults"
        if isinstance(settings, dict):
            return "dictionary"
        if isinstance(settings, str):
            if settings in DEFAULT_PARAMS:
                return f"preset '{settings}'"
            return f"file '{os.path.basename(settings)}'"
        return "unknown"

    def _get_association_defaults(self):
        defaults = {}
        if not self.settings:
            return defaults
        p2p = self.settings.get('pixel2photon', {})
        if 'dSpace' in p2p:
            defaults['pixel_max_dist_px'] = float(p2p['dSpace'])
        if 'dTime' in p2p:
            defaults['pixel_max_time_ns'] = float(p2p['dTime']) * 1e9
        if 'nPxMin' in p2p:
            defaults['min_pixels'] = int(p2p['nPxMin'])
        p2e = self.settings.get('photon2event', {})
        if 'dSpace_px' in p2e:
            defaults['photon_dSpace_px'] = float(p2e['dSpace_px'])
        if 'dTime_s' in p2e:
            defaults['max_time_ns'] = float(p2e['dTime_s']) * 1e9
        return defaults

    # =========================================================================
    # Data loading (ExportedPixels/Photons/Events CSVs only)
    # =========================================================================

    def load(self, events=True, photons=True, pixels=True,
             limit=None, query=None, verbosity=None):
        """
        Load data from ExportedEvents, ExportedPhotons, and/or ExportedPixels directories.

        Expects CSV files produced by empirun or EMPIR export binaries.

        Args:
            events (bool): Load events (default: True).
            photons (bool): Load photons (default: True).
            pixels (bool): Load pixels (default: True).
            limit (int or float): Row limit (int) or max TOA in seconds (float).
            query (str): Pandas query to filter events (e.g. "PSD > 0.5").
            verbosity (int): Override instance verbosity.
        """
        if verbosity is None:
            verbosity = self.verbosity

        if events:
            events_dir = os.path.join(self.data_folder, "ExportedEvents")
            if os.path.isdir(events_dir):
                files = sorted(glob.glob(os.path.join(events_dir, "*.csv")))
                if files:
                    dfs = []
                    for f in tqdm(files, desc="Loading events", disable=(verbosity == 0)):
                        df = self._load_event_csv(f, verbosity)
                        if df is not None and len(df) > 0:
                            dfs.append(df)
                    if dfs:
                        self.events_df = pd.concat(dfs, ignore_index=True).replace(" nan", float("nan"))
                        if verbosity >= 1:
                            print(f"Loaded {len(self.events_df):,} events")
                    else:
                        self.events_df = pd.DataFrame()
                elif verbosity >= 2:
                    print(f"No CSV files in {events_dir}")
            elif verbosity >= 2:
                print(f"ExportedEvents not found at {events_dir}")

        if photons:
            photons_dir = os.path.join(self.data_folder, "ExportedPhotons")
            if os.path.isdir(photons_dir):
                files = sorted(glob.glob(os.path.join(photons_dir, "*.csv")))
                if files:
                    dfs = []
                    for f in tqdm(files, desc="Loading photons", disable=(verbosity == 0)):
                        df = self._load_photon_csv(f, verbosity)
                        if df is not None and len(df) > 0:
                            dfs.append(df)
                    if dfs:
                        self.photons_df = pd.concat(dfs, ignore_index=True).replace(" nan", float("nan"))
                        if verbosity >= 1:
                            print(f"Loaded {len(self.photons_df):,} photons")
                    else:
                        self.photons_df = pd.DataFrame()
                elif verbosity >= 2:
                    print(f"No CSV files in {photons_dir}")
            elif verbosity >= 2:
                print(f"ExportedPhotons not found at {photons_dir}")

        if pixels:
            pixels_dir = os.path.join(self.data_folder, "ExportedPixels")
            if os.path.isdir(pixels_dir):
                files = sorted(glob.glob(os.path.join(pixels_dir, "*.csv")))
                if files:
                    dfs = []
                    for f in tqdm(files, desc="Loading pixels", disable=(verbosity == 0)):
                        df = self._load_pixel_csv(f, verbosity)
                        if df is not None and len(df) > 0:
                            dfs.append(df)
                    if dfs:
                        self.pixels_df = pd.concat(dfs, ignore_index=True).replace(" nan", float("nan"))
                        if verbosity >= 1:
                            print(f"Loaded {len(self.pixels_df):,} pixels")
                    else:
                        self.pixels_df = pd.DataFrame()
                elif verbosity >= 2:
                    print(f"No CSV files in {pixels_dir}")
            elif verbosity >= 2:
                print(f"ExportedPixels not found at {pixels_dir}")

        # Apply query filter to events
        if query is not None and self.events_df is not None and len(self.events_df) > 0:
            n_before = len(self.events_df)
            self.events_df = self.events_df.query(query)
            if verbosity >= 2:
                print(f"Query '{query}': {n_before} -> {len(self.events_df)} events")

        # Apply cascading limits
        if limit is not None:
            self._apply_cascading_limits(limit, verbosity=verbosity)

        # Correct pixel time offset if needed
        if self.pixels_df is not None and len(self.pixels_df) > 0:
            if self.photons_df is not None and len(self.photons_df) > 0:
                self._correct_pixel_time_offset(verbosity=verbosity)

    def _load_event_csv(self, path, verbosity=0):
        try:
            df = pd.read_csv(path)
            if ' PSD value' in df.columns:
                df = df[df[' PSD value'] >= 0]
                df.columns = ["x", "y", "t", "n", "PSD", "tof"]
            elif list(df.columns) == ["x", "y", "t", "n", "PSD", "tof"]:
                df = df[df['PSD'] >= 0]
            elif all(c in df.columns for c in ["x", "y", "t", "n", "PSD"]):
                if "tof" not in df.columns:
                    df["tof"] = np.nan
                df = df[["x", "y", "t", "n", "PSD", "tof"]]
                df = df[df['PSD'] >= 0]
            else:
                if verbosity >= 1:
                    print(f"Warning: unexpected event CSV columns {df.columns.tolist()} in {os.path.basename(path)}")
                return None
            df["tof"] = pd.to_numeric(df["tof"], errors="coerce")
            df["PSD"] = df["PSD"].astype(float)
            return df
        except Exception as e:
            if verbosity >= 1:
                print(f"Error loading {path}: {e}")
            return None

    def _load_photon_csv(self, path, verbosity=0):
        try:
            df = pd.read_csv(path)
            if list(df.columns) == ["x", "y", "toa", "tof"]:
                df.columns = ["x", "y", "t", "tof"]
            elif len(df.columns) == 4:
                df.columns = ["x", "y", "t", "tof"]
            elif all(c in df.columns for c in ["x", "y", "t"]):
                if "tof" not in df.columns:
                    df["tof"] = np.nan
                df = df[["x", "y", "t", "tof"]]
            else:
                if verbosity >= 1:
                    print(f"Warning: unexpected photon CSV columns {df.columns.tolist()} in {os.path.basename(path)}")
                return None
            df[["x", "y", "t"]] = df[["x", "y", "t"]].astype(float)
            df["tof"] = pd.to_numeric(df["tof"], errors="coerce")
            return df
        except Exception as e:
            if verbosity >= 1:
                print(f"Error loading {path}: {e}")
            return None

    def _load_pixel_csv(self, path, verbosity=0):
        try:
            df = pd.read_csv(path)
            df.columns = [col.strip().split('[')[0].strip() for col in df.columns]
            if 't_relToExtTrigger' in df.columns:
                df.rename(columns={'t_relToExtTrigger': 'tof'}, inplace=True)
            expected = ['x', 'y', 't', 'tot', 'tof']
            if not all(c in df.columns for c in expected):
                if verbosity >= 1:
                    print(f"Warning: unexpected pixel CSV columns {df.columns.tolist()} in {os.path.basename(path)}")
                return None
            df = df[expected]
            df[["x", "y", "t"]] = df[["x", "y", "t"]].astype(float)
            df["tot"] = pd.to_numeric(df["tot"], errors="coerce")
            df["tof"] = pd.to_numeric(df["tof"], errors="coerce")
            return df
        except Exception as e:
            if verbosity >= 1:
                print(f"Error loading {path}: {e}")
            return None

    # =========================================================================
    # Cascading limits
    # =========================================================================

    def _apply_cascading_limits(self, limit, relax=1.0, verbosity=0):
        """Apply row or TOA limits, cascading time windows to downstream data."""
        defaults = self._get_association_defaults()
        pixel_time_s = defaults.get('pixel_max_time_ns', 500) * relax / 1e9
        photon_time_s = defaults.get('max_time_ns', 500) * relax / 1e9
        limit_is_time = isinstance(limit, float)

        has_px = self.pixels_df is not None and len(self.pixels_df) > 0
        has_ph = self.photons_df is not None and len(self.photons_df) > 0
        has_ev = self.events_df is not None and len(self.events_df) > 0

        def _trim(df, col='t'):
            if limit_is_time:
                return df[df[col] <= limit].copy()
            return df.sort_values(col).head(int(limit)).copy()

        def _trim_time(df, toa_limit, col='t'):
            return df[df[col] <= toa_limit].copy()

        if has_px and has_ph and has_ev:
            self.pixels_df = _trim(self.pixels_df)
            if len(self.pixels_df) > 0:
                self.photons_df = _trim_time(self.photons_df, self.pixels_df['t'].max() + pixel_time_s)
            if len(self.photons_df) > 0:
                self.events_df = _trim_time(self.events_df, self.photons_df['t'].max() + photon_time_s)
        elif has_px and has_ph:
            self.pixels_df = _trim(self.pixels_df)
            if len(self.pixels_df) > 0:
                self.photons_df = _trim_time(self.photons_df, self.pixels_df['t'].max() + pixel_time_s)
        elif has_ph and has_ev:
            self.photons_df = _trim(self.photons_df)
            if len(self.photons_df) > 0:
                self.events_df = _trim_time(self.events_df, self.photons_df['t'].max() + photon_time_s)
        else:
            if has_px:
                self.pixels_df = _trim(self.pixels_df)
            if has_ph:
                self.photons_df = _trim(self.photons_df)
            if has_ev:
                self.events_df = _trim(self.events_df)

    # =========================================================================
    # Pixel time offset correction
    # =========================================================================

    def _correct_pixel_time_offset(self, verbosity=0):
        """Detect and correct time offset between pixel and photon timestamps."""
        if self.pixels_df is None or len(self.pixels_df) == 0:
            return
        if self.photons_df is None or len(self.photons_df) == 0:
            return

        px_min, px_max = self.pixels_df['t'].min(), self.pixels_df['t'].max()
        ph_min, ph_max = self.photons_df['t'].min(), self.photons_df['t'].max()

        if px_max >= ph_min and ph_max >= px_min:
            return  # Already overlapping

        if verbosity >= 1:
            print(f"   Pixel t [{px_min:.3f}, {px_max:.3f}]s doesn't overlap photon t [{ph_min:.3f}, {ph_max:.3f}]s")
            print(f"   Attempting to align pixel timestamps...")

        offset = self._find_pixel_photon_time_offset(verbosity=verbosity)
        if offset is None:
            if verbosity >= 1:
                print("   Warning: could not determine pixel time offset")
            return

        if verbosity >= 1:
            print(f"   Applying pixel time offset: {offset:.6f}s")
        self.pixels_df['t'] = self.pixels_df['t'] + offset

    def _find_pixel_photon_time_offset(self, n_samples=100, verbosity=0):
        """Find time offset between pixel and photon timestamps by position matching."""
        photons_sorted = self.photons_df.sort_values('t').head(n_samples * 10).copy()
        photons_sorted['x_int'] = photons_sorted['x'].round().astype(int)
        photons_sorted['y_int'] = photons_sorted['y'].round().astype(int)

        pixels_sorted = self.pixels_df.sort_values('t').copy()
        pixels_sorted['x_int'] = pixels_sorted['x'].round().astype(int)
        pixels_sorted['y_int'] = pixels_sorted['y'].round().astype(int)

        offsets = []
        checked = 0
        for _, phot in photons_sorted.iterrows():
            if checked >= n_samples:
                break
            mask = ((pixels_sorted['x_int'] == phot['x_int']) &
                    (pixels_sorted['y_int'] == phot['y_int']))
            matching = pixels_sorted[mask]
            if len(matching) == 0:
                continue
            offsets.append(phot['t'] - matching.iloc[0]['t'])
            checked += 1

        if len(offsets) < 10:
            if verbosity >= 1:
                print(f"   Only {len(offsets)} matches found, cannot determine offset")
            return None

        offsets = np.array(offsets)
        median_offset = np.median(offsets)
        if verbosity >= 2:
            print(f"   {len(offsets)} matches, offset={median_offset:.6f}s (std={np.std(offsets)*1e9:.1f}ns)")
        return median_offset

    # =========================================================================
    # Main association entry point
    # =========================================================================

    def associate(self, pixel_max_dist_px=None, pixel_max_time_ns=None,
                  photon_dSpace_px=None, max_time_ns=None,
                  min_pixels=None,
                  verbosity=None, method='empir', relax=5, suffix=None):
        """
        Perform full association: pixels -> photons -> events (or subsets).

        Automatically handles the data tiers present:
        - Pixels + Photons + Events: 3-tier association
        - Pixels + Photons: pixel-to-photon only
        - Photons + Events: photon-to-event only

        Args:
            pixel_max_dist_px (float): Max spatial distance for pixel-photon association (px).
                                       Default from settings or 5.0.
            pixel_max_time_ns (float): Max time window for pixel-photon association (ns).
                                       Default from settings or 500.
            photon_dSpace_px (float): Max CoM distance for photon-event association (px).
                                      Default from settings or 50.0.
            max_time_ns (float): Max time window for photon-event association (ns).
                                 Default from settings or 500.
            verbosity (int): Override instance verbosity.
            method (str): Association method:
                - 'simple': Fast forward time-window with CoM check (default).
                - 'kdtree': KDTree-based with iterative CoM refinement.
                - 'window': Symmetric sliding time-window KDTree.
                - 'mystic': Constrained optimization (requires mystic package).
                - 'ml': Machine learning (requires trained model; falls back to 'simple').
            relax (float): For empir: starting search-window multiplier (default 5).
                           The algorithm doubles it each iteration until CoG converges
                           (< 0.1 px) or the radius reaches 100 px.
                           For other methods: scales pixel/photon parameters directly.
            suffix (str): Optional suffix for output filename, e.g. 'run1' produces
                          'associated_data_run1.csv'.

        Returns:
            pd.DataFrame: Associated DataFrame.
        """
        if verbosity is None:
            verbosity = self.verbosity

        defaults = self._get_association_defaults()
        if pixel_max_dist_px is None:
            pixel_max_dist_px = defaults.get('pixel_max_dist_px', 5.0)
        if pixel_max_time_ns is None:
            pixel_max_time_ns = defaults.get('pixel_max_time_ns', 500)
        if photon_dSpace_px is None:
            photon_dSpace_px = defaults.get('photon_dSpace_px', 50.0)
        if max_time_ns is None:
            max_time_ns = defaults.get('max_time_ns', 500)
        if min_pixels is None:
            min_pixels = defaults.get('min_pixels', 1)

        # For empir, relax is the search-window multiplier passed directly to the
        # method (defaults to 10); pixel params are not pre-scaled here.
        if method != 'empir':
            pixel_max_dist_px *= relax
            pixel_max_time_ns *= relax
        photon_dSpace_px *= relax
        max_time_ns *= relax

        has_px = self.pixels_df is not None and len(self.pixels_df) > 0
        has_ph = self.photons_df is not None and len(self.photons_df) > 0
        has_ev = self.events_df is not None and len(self.events_df) > 0

        if verbosity >= 2:
            print(f"\nAssociation: pixels={has_px}, photons={has_ph}, events={has_ev}, method={method}")
            if relax != 1.0:
                print(f"Relax factor: {relax}x")

        if has_px and has_ph and has_ev:
            # 3-tier: pixels -> photons -> events
            if verbosity >= 2:
                print("3-tier: Pixels -> Photons -> Events")

            pixels_assoc = self._run_pixel_photon_assoc(
                method, pixel_max_dist_px, pixel_max_time_ns, verbosity,
                min_pixels=min_pixels, relax=relax)

            self._run_photon_event_assoc(
                method, photon_dSpace_px, max_time_ns, verbosity)

            # Merge event info into pixel dataframe
            self.associated_df = self._merge_pixel_photon_event(
                pixels_assoc, self.associated_df, verbosity)

        elif has_px and has_ph:
            # 2-tier: pixels -> photons
            if verbosity >= 2:
                print("2-tier: Pixels -> Photons")
            pixels_assoc = self._run_pixel_photon_assoc(
                method, pixel_max_dist_px, pixel_max_time_ns, verbosity,
                min_pixels=min_pixels, relax=relax)
            self.associated_df = self._standardize_column_names(pixels_assoc, verbosity)

        elif has_ph and has_ev:
            # 2-tier: photons -> events
            if verbosity >= 2:
                print("2-tier: Photons -> Events")
            self._run_photon_event_assoc(
                method, photon_dSpace_px, max_time_ns, verbosity)

        else:
            if verbosity >= 1:
                print("Warning: need at least two data tiers for association")
            self.associated_df = pd.DataFrame()

        # Auto-save
        if self.associated_df is not None and len(self.associated_df) > 0:
            try:
                filename = f"associated_data_{suffix}.csv" if suffix else "associated_data.csv"
                out = self.save_associations(filename=filename, verbosity=verbosity)
                if verbosity >= 1:
                    print(f"Saved to: {out}")
            except Exception as e:
                if verbosity >= 2:
                    print(f"Warning: could not auto-save: {e}")

        try:
            from IPython.display import HTML
            return HTML(self._repr_html_())
        except ImportError:
            return self.associated_df

    def _run_pixel_photon_assoc(self, method, max_dist_px, max_time_ns, verbosity,
                                min_pixels=1, relax=1.0):
        if method == 'kdtree':
            return self._associate_pixels_to_photons_kdtree(
                self.pixels_df, self.photons_df, max_dist_px, max_time_ns, verbosity)
        elif method == 'mystic':
            return self._associate_pixels_to_photons_mystic(
                self.pixels_df, self.photons_df, max_dist_px, max_time_ns, verbosity=verbosity)
        elif method == 'ml':
            return self._associate_pixels_to_photons_ml(
                self.pixels_df, self.photons_df, max_dist_px, max_time_ns, verbosity=verbosity)
        elif method == 'empir':
            return self._associate_pixels_to_photons_empir(
                self.pixels_df, self.photons_df, max_dist_px, max_time_ns,
                min_pixels=min_pixels, relax=relax, verbosity=verbosity)
        else:
            return self._associate_pixels_to_photons_simple(
                self.pixels_df, self.photons_df, max_dist_px, max_time_ns, verbosity)

    def _run_photon_event_assoc(self, method, dSpace_px, max_time_ns, verbosity):
        max_time_s = max_time_ns / 1e9
        if method == 'mystic':
            result = self._associate_photons_to_events_mystic(
                self.photons_df, self.events_df, dSpace_px, max_time_ns, verbosity=verbosity)
        elif method == 'kdtree':
            result = self._associate_photons_to_events_kdtree(
                self.photons_df, self.events_df, 1.0, 1.0, dSpace_px, verbosity)
        elif method == 'window':
            result = self._associate_photons_to_events_window(
                self.photons_df, self.events_df, 1.0, 1.0, dSpace_px, max_time_s, verbosity)
        else:
            result = self._associate_photons_to_events_simple_window(
                self.photons_df, self.events_df, dSpace_px, max_time_s, verbosity)

        # Re-number event IDs globally
        if result is not None and len(result) > 0:
            mask = (result['assoc_x'].notna() & result['assoc_y'].notna() &
                    result['assoc_t'].notna() & result['assoc_n'].notna() &
                    result['assoc_PSD'].notna())
            if mask.any():
                grouped = result.loc[mask].groupby(
                    ['assoc_x', 'assoc_y', 'assoc_t', 'assoc_n', 'assoc_PSD'])
                result.loc[mask, 'assoc_event_id'] = grouped.ngroup() + 1

        self.associated_df = result

    def _merge_pixel_photon_event(self, pixels_assoc, photons_with_events, verbosity):
        """Merge event association info into the pixel-centric dataframe."""
        if photons_with_events is None or len(photons_with_events) == 0:
            return self._standardize_column_names(pixels_assoc, verbosity)

        photons_ev = photons_with_events.copy()
        photons_ev['_mx'] = photons_ev['x'].round(6)
        photons_ev['_my'] = photons_ev['y'].round(6)
        photons_ev['_mt'] = photons_ev['t'].round(12)

        pixels_full = pixels_assoc.copy()
        pixels_full['_mx'] = pixels_full['assoc_phot_x'].round(6)
        pixels_full['_my'] = pixels_full['assoc_phot_y'].round(6)
        pixels_full['_mt'] = pixels_full['assoc_phot_t'].round(12)

        merge_cols = ['_mx', '_my', '_mt', 'assoc_event_id',
                      'assoc_x', 'assoc_y', 'assoc_t', 'assoc_n', 'assoc_PSD']
        if 'assoc_com_dist' in photons_ev.columns:
            merge_cols.append('assoc_com_dist')

        pixels_full = pixels_full.merge(
            photons_ev[merge_cols],
            on=['_mx', '_my', '_mt'],
            how='left',
            suffixes=('', '_event')
        ).drop(columns=['_mx', '_my', '_mt'])

        return self._standardize_column_names(pixels_full, verbosity)

    # =========================================================================
    # Photon-event association methods
    # =========================================================================

    def _associate_photons_to_events_kdtree(
        self, photons_df, events_df, time_norm_ns, spatial_norm_px, dSpace_px, verbosity
    ):
        """Associate photons to events using a full KDTree on normalized coordinates."""
        photons = photons_df.copy()
        events = events_df.copy()
        photons['assoc_event_id'] = np.nan
        photons['assoc_x'] = np.nan
        photons['assoc_y'] = np.nan
        photons['assoc_t'] = np.nan
        photons['assoc_n'] = 0
        photons['assoc_PSD'] = 0
        photons['time_diff_ns'] = np.nan
        photons['spatial_diff_px'] = np.nan
        photons['assoc_com_dist'] = np.nan

        events = events.sort_values('t').reset_index(drop=True)
        events['event_id'] = events.index + 1

        photon_coords = np.vstack([
            photons['x'].to_numpy(),
            photons['y'].to_numpy(),
            photons['t'].to_numpy() * 1e9 / time_norm_ns
        ]).T
        tree = cKDTree(photon_coords)

        for _, ev in tqdm(events.iterrows(), total=len(events),
                          desc="Associating photons to events (kdtree)", disable=(verbosity == 0)):
            n_photons = int(ev['n'])
            ex, ey, et, eid = ev['x'], ev['y'], ev['t'], ev['event_id']
            query_point = np.array([ex, ey, et * 1e9 / time_norm_ns])
            indices = tree.query_ball_point(query_point, 5.0)
            if not indices:
                continue

            cands = photons.iloc[indices]
            time_diff = np.abs(cands['t'] - et) * 1e9
            spatial_diff = np.sqrt((cands['x'] - ex)**2 + (cands['y'] - ey)**2)
            combined = (time_diff / time_norm_ns) + (spatial_diff / spatial_norm_px)

            if n_photons == 1:
                best_i = np.argmin(combined)
                best_idx = indices[best_i]
                if np.isnan(photons.loc[best_idx, 'assoc_event_id']):
                    com_dist = float(spatial_diff.iloc[best_i])
                    if com_dist <= dSpace_px:
                        photons.loc[best_idx, 'assoc_event_id'] = eid
                        photons.loc[best_idx, ['assoc_x', 'assoc_y', 'assoc_t']] = ex, ey, et
                        photons.loc[best_idx, 'assoc_n'] = n_photons
                        photons.loc[best_idx, 'assoc_PSD'] = ev.get('PSD', 0)
                        photons.loc[best_idx, 'time_diff_ns'] = float(time_diff.iloc[best_i])
                        photons.loc[best_idx, 'spatial_diff_px'] = com_dist
                        photons.loc[best_idx, 'assoc_com_dist'] = com_dist
            else:
                top_i = np.argsort(combined)[:n_photons]
                sel_idx = [indices[i] for i in top_i]
                sel_x = cands.iloc[top_i]['x']
                sel_y = cands.iloc[top_i]['y']
                if not (np.any(np.isnan(sel_x)) or np.any(np.isnan(sel_y))):
                    com_dist = np.sqrt((sel_x.mean() - ex)**2 + (sel_y.mean() - ey)**2)
                    if com_dist <= dSpace_px:
                        for idx, di in zip(sel_idx, top_i):
                            if np.isnan(photons.loc[idx, 'assoc_event_id']):
                                photons.loc[idx, 'assoc_event_id'] = eid
                                photons.loc[idx, ['assoc_x', 'assoc_y', 'assoc_t']] = ex, ey, et
                                photons.loc[idx, 'assoc_n'] = n_photons
                                photons.loc[idx, 'assoc_PSD'] = ev.get('PSD', 0)
                                photons.loc[idx, 'time_diff_ns'] = float(time_diff.iloc[di])
                                photons.loc[idx, 'spatial_diff_px'] = float(spatial_diff.iloc[di])
                                photons.loc[idx, 'assoc_com_dist'] = com_dist

        self._store_photon_event_stats(photons, events, dSpace_px, verbosity)
        return photons

    def _associate_photons_to_events_window(
        self, photons_df, events_df, time_norm_ns, spatial_norm_px, dSpace_px, max_time_s, verbosity
    ):
        """Associate photons to events using a symmetric sliding time-window KDTree."""
        if max_time_s is None:
            all_t = np.concatenate((photons_df['t'].to_numpy(), events_df['t'].to_numpy()))
            max_time_s = 3 * (np.std(all_t) if len(all_t) > 1 else 1e-6)

        photons = photons_df.copy()
        events = events_df.copy()
        photons['assoc_event_id'] = np.nan
        photons['assoc_x'] = np.nan
        photons['assoc_y'] = np.nan
        photons['assoc_t'] = np.nan
        photons['assoc_n'] = 0
        photons['assoc_PSD'] = 0
        photons['time_diff_ns'] = np.nan
        photons['spatial_diff_px'] = np.nan
        photons['assoc_com_dist'] = np.nan

        photons = photons.sort_values('t').reset_index(drop=True)
        events = events.sort_values('t').reset_index(drop=True)
        events['event_id'] = events.index + 1

        p_t = photons['t'].to_numpy()
        p_x = photons['x'].to_numpy()
        p_y = photons['y'].to_numpy()
        left = 0
        n_ph = len(photons)

        for _, ev in tqdm(events.iterrows(), total=len(events),
                          desc="Associating photons to events (window)", disable=(verbosity == 0)):
            et, ex, ey, eid, n = ev['t'], ev['x'], ev['y'], ev['event_id'], int(ev['n'])

            while left < n_ph and p_t[left] < et - max_time_s:
                left += 1
            right = left
            while right < n_ph and p_t[right] <= et + max_time_s:
                right += 1
            if right == left:
                continue

            sub_idx = np.arange(left, right)
            sub_coords = np.vstack([
                p_x[sub_idx], p_y[sub_idx],
                p_t[sub_idx] * 1e9 / time_norm_ns
            ]).T
            tree = cKDTree(sub_coords)
            indices = tree.query_ball_point([ex, ey, et * 1e9 / time_norm_ns], r=5.0)
            if not indices:
                continue

            indices = sub_idx[indices]
            cands = photons.iloc[indices]
            time_diff = np.abs(cands['t'] - et) * 1e9
            spatial_diff = np.sqrt((cands['x'] - ex)**2 + (cands['y'] - ey)**2)
            combined = (time_diff / time_norm_ns) + (spatial_diff / spatial_norm_px)

            if n == 1:
                min_i = np.argmin(combined)
                best_idx = indices[min_i]
                if np.isnan(photons.loc[best_idx, 'assoc_event_id']):
                    com_dist = float(spatial_diff.iloc[min_i])
                    if com_dist <= dSpace_px:
                        photons.loc[best_idx, 'assoc_event_id'] = eid
                        photons.loc[best_idx, ['assoc_x', 'assoc_y', 'assoc_t']] = ex, ey, et
                        photons.loc[best_idx, 'assoc_n'] = n
                        photons.loc[best_idx, 'assoc_PSD'] = ev.get('PSD', 0)
                        photons.loc[best_idx, 'time_diff_ns'] = float(time_diff.iloc[min_i])
                        photons.loc[best_idx, 'spatial_diff_px'] = com_dist
                        photons.loc[best_idx, 'assoc_com_dist'] = com_dist
            else:
                top_i = np.argsort(combined)[:n]
                sel_idx = indices[top_i]
                sel_x = cands.iloc[top_i]['x']
                sel_y = cands.iloc[top_i]['y']
                if not (np.any(np.isnan(sel_x)) or np.any(np.isnan(sel_y))):
                    com_dist = np.sqrt((sel_x.mean() - ex)**2 + (sel_y.mean() - ey)**2)
                    if com_dist <= dSpace_px:
                        for i, di in enumerate(top_i):
                            idx = sel_idx[i]
                            if np.isnan(photons.loc[idx, 'assoc_event_id']):
                                photons.loc[idx, 'assoc_event_id'] = eid
                                photons.loc[idx, ['assoc_x', 'assoc_y', 'assoc_t']] = ex, ey, et
                                photons.loc[idx, 'assoc_n'] = n
                                photons.loc[idx, 'assoc_PSD'] = ev.get('PSD', 0)
                                photons.loc[idx, 'time_diff_ns'] = float(time_diff.iloc[di])
                                photons.loc[idx, 'spatial_diff_px'] = float(spatial_diff.iloc[di])
                                photons.loc[idx, 'assoc_com_dist'] = com_dist

        self._store_photon_event_stats(photons, events, dSpace_px, verbosity)
        return photons

    def _associate_photons_to_events_simple_window(
        self, photons_df, events_df, dSpace_px, max_time_s, verbosity
    ):
        """
        Associate photons to events: two-pass forward time-window with conflict resolution.

        Pass 1: For each event find candidate photons and compute CoM distance.
        Pass 2: Resolve conflicts, preferring the event with smallest CoM distance.
        """
        if max_time_s is None:
            max_time_s = 500e-9

        photons = photons_df.copy()
        events = events_df.copy()
        photons['assoc_event_id'] = np.nan
        photons['assoc_x'] = np.nan
        photons['assoc_y'] = np.nan
        photons['assoc_t'] = np.nan
        photons['assoc_n'] = 0
        photons['assoc_PSD'] = 0
        photons['time_diff_ns'] = np.nan
        photons['spatial_diff_px'] = np.nan
        photons['assoc_com_dist'] = np.nan

        photons = photons.sort_values('t').reset_index(drop=True)
        events = events.sort_values('t').reset_index(drop=True)
        events['event_id'] = events.index + 1

        p_t = photons['t'].to_numpy()
        p_x = photons['x'].to_numpy()
        p_y = photons['y'].to_numpy()
        n_total = len(photons)

        # Pass 1: build candidate assignments
        photon_candidates = {}
        left = 0
        for _, ev in tqdm(events.iterrows(), total=len(events),
                          desc="Associating photons to events", disable=(verbosity == 0)):
            et, ex, ey, eid, n = ev['t'], ev['x'], ev['y'], ev['event_id'], int(ev['n'])
            psd = ev.get('PSD', 0)

            while left < n_total and p_t[left] < et:
                left += 1
            right = left
            while right < n_total and p_t[right] <= et + max_time_s:
                right += 1

            if right - left < n:
                continue

            sub_idx = np.arange(left, right)
            spatial_diffs = np.sqrt((p_x[sub_idx] - ex)**2 + (p_y[sub_idx] - ey)**2)
            sort_i = np.argsort(spatial_diffs)[:n]
            sel_x = p_x[sub_idx][sort_i]
            sel_y = p_y[sub_idx][sort_i]

            com_dist = spatial_diffs[sort_i[0]] if n == 1 else \
                np.sqrt((sel_x.mean() - ex)**2 + (sel_y.mean() - ey)**2)

            if com_dist > dSpace_px:
                continue

            global_idx = sub_idx[sort_i]
            ev_data = {'ex': ex, 'ey': ey, 'et': et, 'n': n, 'psd': psd}
            for i, loc_idx in enumerate(global_idx):
                t_diff = (p_t[sub_idx[sort_i[i]]] - et) * 1e9
                sp_diff = spatial_diffs[sort_i[i]]
                if loc_idx not in photon_candidates:
                    photon_candidates[loc_idx] = []
                photon_candidates[loc_idx].append((eid, com_dist, sp_diff, t_diff, ev_data))

        # Pass 2: resolve conflicts
        for loc_idx, candidates in photon_candidates.items():
            best = min(candidates, key=lambda x: x[1])
            eid, com_dist, sp_diff, t_diff, ev_data = best
            photons.loc[loc_idx, 'assoc_event_id'] = eid
            photons.loc[loc_idx, 'assoc_x'] = ev_data['ex']
            photons.loc[loc_idx, 'assoc_y'] = ev_data['ey']
            photons.loc[loc_idx, 'assoc_t'] = ev_data['et']
            photons.loc[loc_idx, 'assoc_n'] = ev_data['n']
            photons.loc[loc_idx, 'assoc_PSD'] = ev_data['psd']
            photons.loc[loc_idx, 'time_diff_ns'] = t_diff
            photons.loc[loc_idx, 'spatial_diff_px'] = sp_diff
            photons.loc[loc_idx, 'assoc_com_dist'] = com_dist

        self._store_photon_event_stats(photons, events, dSpace_px, verbosity)
        return photons

    def _store_photon_event_stats(self, photons, events, dSpace_px, verbosity):
        """Compute and store photon-event association statistics."""
        matched_photons = photons['assoc_event_id'].notna().sum()
        total_photons = len(photons)
        matched_ev_ids = photons[photons['assoc_event_id'].notna()]['assoc_event_id'].unique()
        matched_events = len(matched_ev_ids)
        total_events = len(events)

        quality = {'exact_n': 0, 'n_mismatch': 0, 'exact_com': 0,
                   'good_com': 0, 'acceptable_com': 0, 'poor_com': 0}
        for eid in matched_ev_ids:
            ep = photons[photons['assoc_event_id'] == eid]
            actual_n = len(ep)
            pred_n = int(ep.iloc[0]['assoc_n'])
            if actual_n == pred_n:
                quality['exact_n'] += 1
            else:
                quality['n_mismatch'] += 1
            if 'assoc_com_dist' in ep.columns:
                com_dist = ep.iloc[0]['assoc_com_dist']
                sr = dSpace_px if dSpace_px != np.inf else 50.0
                if com_dist <= 0.1:
                    quality['exact_com'] += 1
                elif com_dist <= sr * 0.3:
                    quality['good_com'] += 1
                elif com_dist <= sr * 0.5:
                    quality['acceptable_com'] += 1
                else:
                    quality['poor_com'] += 1

        self.last_photon_event_stats = {
            'matched_photons': int(matched_photons),
            'total_photons': int(total_photons),
            'matched_events': int(matched_events),
            'total_events': int(total_events),
            'quality': quality
        }

        if verbosity >= 1:
            pct = 100 * matched_photons / total_photons if total_photons > 0 else 0
            print(f"Photon-Event: {matched_photons:,}/{total_photons:,} photons matched ({pct:.1f}%)")
            epct = 100 * matched_events / total_events if total_events > 0 else 0
            print(f"             {matched_events:,}/{total_events:,} events matched ({epct:.1f}%)")

    def _associate_photons_to_events_mystic(self, photons_df, events_df, max_dist_px=10.0,
                                             max_time_ns=500, time_weight=1.0, cog_weight=1.0,
                                             min_photons=1, verbosity=0):
        """Associate photons to events using mystic constrained optimization."""
        try:
            from mystic.solvers import fmin_powell, diffev2
        except ImportError:
            raise ImportError("mystic required. Install with: pip install mystic")

        if photons_df is None or len(photons_df) == 0 or events_df is None or len(events_df) == 0:
            return photons_df

        photons = photons_df.copy()
        events = events_df.copy()
        photons['assoc_event_id'] = np.nan
        photons['assoc_x'] = np.nan
        photons['assoc_y'] = np.nan
        photons['assoc_t'] = np.nan
        photons['assoc_n'] = np.nan
        photons['assoc_PSD'] = np.nan
        photons['assoc_com_dist'] = np.nan

        photons = photons.sort_values('t').reset_index(drop=True)
        events = events.sort_values('t').reset_index(drop=True)
        events['event_id'] = events.index + 1

        photon_tree = cKDTree(np.column_stack([photons['x'].to_numpy(), photons['y'].to_numpy()]))
        max_time_s = max_time_ns / 1e9
        opt_stats = {'success': 0, 'fallback': 0, 'failed': 0}

        for _, ev in tqdm(events.iterrows(), total=len(events),
                          desc="Associating photons to events (mystic)", disable=(verbosity == 0)):
            ev_t, ev_x, ev_y = ev['t'], ev['x'], ev['y']
            ev_n, ev_psd, ev_id = ev['n'], ev['PSD'], ev['event_id']

            cand_idx = photon_tree.query_ball_point([ev_x, ev_y], max_dist_px)
            if not cand_idx:
                continue

            cand_ph = photons.iloc[cand_idx]
            tmask = (cand_ph['t'] >= ev_t) & (cand_ph['t'] <= ev_t + max_time_s)
            if not tmask.any():
                continue

            valid_ph = cand_ph[tmask]
            valid_idx = np.array(cand_idx)[tmask.to_numpy()]
            unassigned = valid_ph['assoc_event_id'].isna().to_numpy()
            if not unassigned.any():
                continue

            unasgn_idx = valid_idx[unassigned]
            ua = valid_ph[unassigned]
            ux, uy, ut = ua['x'].to_numpy(), ua['y'].to_numpy(), ua['t'].to_numpy()
            nc = len(unasgn_idx)

            if nc <= min_photons:
                com_x, com_y = ux.mean(), uy.mean()
                com_dist = np.sqrt((com_x - ev_x)**2 + (com_y - ev_y)**2)
                for pi in unasgn_idx:
                    photons.loc[pi, 'assoc_event_id'] = ev_id
                    photons.loc[pi, ['assoc_x', 'assoc_y', 'assoc_t', 'assoc_n', 'assoc_PSD']] = \
                        ev_x, ev_y, ev_t, ev_n, ev_psd
                    photons.loc[pi, 'assoc_com_dist'] = com_dist
                opt_stats['fallback'] += 1
                continue

            def objective(weights):
                w = np.array(weights)
                ws = w.sum()
                if ws < 0.1:
                    return 1e10
                cx = (w * ux).sum() / ws
                cy = (w * uy).sum() / ws
                cog_dist = np.sqrt((cx - ev_x)**2 + (cy - ev_y)**2)
                wt = (w * (ut - ev_t) * 1e9).sum() / ws
                return cog_weight * cog_dist**2 + time_weight * (wt / max_time_ns)**2

            def constraint(weights):
                w = np.array(weights)
                ws = w.sum()
                if ws < min_photons:
                    w = np.clip(w * min_photons / max(ws, 0.01), 0, 1)
                return w

            x0 = np.full(nc, 0.5)
            bounds = list(zip(np.zeros(nc), np.ones(nc)))
            try:
                sol = fmin_powell(objective, x0, bounds=bounds, constraints=constraint,
                                  disp=False, gtol=1e-4) if nc <= 20 else \
                    diffev2(objective, bounds, constraints=constraint,
                            npop=min(20, nc * 2), disp=False, gtol=50)
                ow = np.array(sol)
                amask = ow >= 0.3
                if amask.sum() < min_photons:
                    top = np.argsort(ow)[-min_photons:]
                    amask = np.zeros(nc, dtype=bool)
                    amask[top] = True
                opt_stats['success'] += 1
            except Exception:
                amask = np.ones(nc, dtype=bool)
                opt_stats['failed'] += 1

            final_idx = unasgn_idx[amask]
            fx, fy = ux[amask], uy[amask]
            if len(final_idx) > 0:
                com_dist = np.sqrt((fx.mean() - ev_x)**2 + (fy.mean() - ev_y)**2)
                for pi in final_idx:
                    photons.loc[pi, 'assoc_event_id'] = ev_id
                    photons.loc[pi, ['assoc_x', 'assoc_y', 'assoc_t', 'assoc_n', 'assoc_PSD']] = \
                        ev_x, ev_y, ev_t, ev_n, ev_psd
                    photons.loc[pi, 'assoc_com_dist'] = com_dist

        self._store_photon_event_stats(photons, events, max_dist_px, verbosity)
        return photons

    # =========================================================================
    # Pixel-photon association methods
    # =========================================================================

    def _associate_pixels_to_photons_simple(self, pixels_df, photons_df,
                                             max_dist_px=5.0, max_time_ns=500, verbosity=0):
        """
        Associate pixels to photons using forward time-window with iterative CoM refinement.
        """
        if pixels_df is None or photons_df is None or len(pixels_df) == 0 or len(photons_df) == 0:
            return pixels_df

        pixels = pixels_df.copy()
        photons = photons_df.copy()
        pixels['assoc_photon_id'] = np.nan
        pixels['assoc_phot_x'] = np.nan
        pixels['assoc_phot_y'] = np.nan
        pixels['assoc_phot_t'] = np.nan
        pixels['pixel_com_dist'] = np.nan

        pixels = pixels.sort_values('t').reset_index(drop=True)
        photons = photons.sort_values('t').reset_index(drop=True)
        photons['photon_id'] = photons.index + 1

        pix_t = pixels['t'].to_numpy()
        pix_x = pixels['x'].to_numpy()
        pix_y = pixels['y'].to_numpy()
        pix_tot = pixels['tot'].to_numpy() if 'tot' in pixels.columns else np.ones(len(pixels))

        max_time_s = max_time_ns / 1e9
        left = 0
        n_px = len(pixels)
        com_quality = {'exact': 0, 'good': 0, 'acceptable': 0, 'poor': 0, 'failed': 0}

        for _, phot in tqdm(photons.iterrows(), total=len(photons),
                            desc="Associating pixels to photons", disable=(verbosity == 0)):
            phot_t, phot_x, phot_y, phot_id = phot['t'], phot['x'], phot['y'], phot['photon_id']

            while left < n_px and pix_t[left] < phot_t:
                left += 1
            right = left
            while right < n_px and pix_t[right] <= phot_t + max_time_s:
                right += 1
            if right == left:
                continue

            sub_idx = np.arange(left, right)
            spatial = np.sqrt((pix_x[sub_idx] - phot_x)**2 + (pix_y[sub_idx] - phot_y)**2)
            valid_mask = spatial <= max_dist_px
            if not valid_mask.any():
                continue

            valid_sub = sub_idx[valid_mask]
            valid_x = pix_x[sub_idx][valid_mask]
            valid_y = pix_y[sub_idx][valid_mask]
            valid_t = pix_t[sub_idx][valid_mask]
            valid_tot = pix_tot[sub_idx][valid_mask]

            unassigned = np.array([np.isnan(pixels.loc[i, 'assoc_photon_id']) for i in valid_sub])
            if not unassigned.any():
                continue

            ua_idx = valid_sub[unassigned]
            ua_x = valid_x[unassigned]
            ua_y = valid_y[unassigned]
            ua_tot = valid_tot[unassigned]

            # Iterative CoM refinement
            keep = np.ones(len(ua_idx), dtype=bool)
            for _ in range(10):
                tot_sum = ua_tot[keep].sum() or 1
                com_x = (ua_x[keep] * ua_tot[keep]).sum() / tot_sum
                com_y = (ua_y[keep] * ua_tot[keep]).sum() / tot_sum
                com_dist = np.sqrt((com_x - phot_x)**2 + (com_y - phot_y)**2)
                if com_dist <= max_dist_px * 0.5:
                    break
                px_dists = np.sqrt((ua_x[keep] - phot_x)**2 + (ua_y[keep] - phot_y)**2)
                if len(px_dists) <= 1:
                    break
                worst = np.argmax(px_dists)
                keep_idx = np.where(keep)[0]
                keep[keep_idx[worst]] = False

            if keep.any():
                ft = ua_tot[keep]
                fts = ft.sum() or 1
                final_com_dist = np.sqrt(
                    ((ua_x[keep] * ft).sum() / fts - phot_x)**2 +
                    ((ua_y[keep] * ft).sum() / fts - phot_y)**2
                )
                if final_com_dist <= 0.1:
                    com_quality['exact'] += 1
                elif final_com_dist <= max_dist_px * 0.3:
                    com_quality['good'] += 1
                elif final_com_dist <= max_dist_px * 0.5:
                    com_quality['acceptable'] += 1
                elif final_com_dist <= max_dist_px:
                    com_quality['poor'] += 1
                else:
                    com_quality['failed'] += 1

                for i in ua_idx[keep]:
                    pixels.loc[i, 'assoc_photon_id'] = phot_id
                    pixels.loc[i, 'assoc_phot_x'] = phot_x
                    pixels.loc[i, 'assoc_phot_y'] = phot_y
                    pixels.loc[i, 'assoc_phot_t'] = phot_t
                    pixels.loc[i, 'pixel_com_dist'] = final_com_dist

        self._store_pixel_photon_stats(pixels, photons, com_quality, verbosity)
        return pixels

    def _associate_pixels_to_photons_empir(self, pixels_df, photons_df,
                                            max_dist_px=2.0, max_time_ns=50,
                                            min_pixels=1, relax=10, verbosity=0):
        """
        Associate pixels to photons via greedy best-subset optimisation with
        adaptive search-window widening.

        For each photon the search window starts at relax × (dSpace, dTime) and
        doubles until the best CoG distance drops below CONVERGE_DIST (0.1 px)
        or the search radius would exceed MAX_SEARCH_PX (100 px).  This keeps
        the window tight for well-matched photons and only widens it for those
        that need more pixels to converge.

        At every relax level all 2^N − 1 non-empty subsets with ≥ min_pixels
        members are evaluated via vectorised bit-matrix arithmetic; the subset
        whose ToT-weighted CoG (truncated to 2 dp, matching EMPIR) is closest
        to (ph/x, ph/y) is retained.  The best result across all relax levels
        is claimed.

        Args:
            relax:       Starting search-window multiplier (default 5).
                         Doubles each iteration until convergence or radius cap.
            min_pixels:  Minimum pixels required in any accepted subset.
        """
        CONVERGE_DIST = 0.1    # CoG distance threshold for "good enough"
        MAX_SEARCH_PX = 100.0  # absolute largest search radius (pixels)
        MAX_CAND      = 15     # 2^15 − 1 = 32 767 subsets, ~2 MB bit-matrix

        if pixels_df is None or photons_df is None or len(pixels_df) == 0 or len(photons_df) == 0:
            return pixels_df

        pixels  = pixels_df.copy()
        photons = photons_df.copy()
        pixels['assoc_photon_id'] = np.nan
        pixels['assoc_phot_x']    = np.nan
        pixels['assoc_phot_y']    = np.nan
        pixels['assoc_phot_t']    = np.nan
        pixels['pixel_com_dist']  = np.nan

        pixels  = pixels.sort_values('t').reset_index(drop=True)
        photons = photons.sort_values('t').reset_index(drop=True)
        photons['photon_id'] = photons.index + 1

        pix_t   = pixels['t'].to_numpy()
        pix_x   = pixels['x'].to_numpy()
        pix_y   = pixels['y'].to_numpy()
        pix_tot = pixels['tot'].to_numpy() if 'tot' in pixels.columns else np.ones(len(pixels))

        base_time_s = max_time_ns / 1e9
        TOL = 1e-12   # guard against float round-trip from CSV

        # Build adaptive relax schedule: relax, 2×relax, 4×relax, …
        # capped so the search radius never exceeds MAX_SEARCH_PX.
        relax_schedule = []
        r = float(relax)
        while True:
            relax_schedule.append(r)
            if r * max_dist_px >= MAX_SEARCH_PX:
                break
            r = min(r * 2, MAX_SEARCH_PX / max_dist_px)

        if verbosity >= 2:
            steps = [f"{int(s)}×" for s in relax_schedule]
            print(f"   empir relax schedule: {' → '.join(steps)} "
                  f"(search radius {relax_schedule[0]*max_dist_px:.0f}–"
                  f"{relax_schedule[-1]*max_dist_px:.0f} px)")

        n_px = len(pixels)
        com_quality = {'exact': 0, 'good': 0, 'acceptable': 0, 'poor': 0, 'failed': 0}

        assoc_id  = np.full(n_px, np.nan)
        assoc_x   = np.full(n_px, np.nan)
        assoc_y   = np.full(n_px, np.nan)
        assoc_t   = np.full(n_px, np.nan)
        assoc_com = np.full(n_px, np.nan)
        claimed   = np.zeros(n_px, dtype=bool)

        for _, phot in tqdm(photons.iterrows(), total=len(photons),
                            desc="Associating pixels to photons", disable=(verbosity == 0)):
            ph_t  = phot['t']
            ph_x  = phot['x']
            ph_y  = phot['y']
            ph_id = phot['photon_id']

            best_com_dist = np.inf
            best_cand_idx = None

            for step_relax in relax_schedule:
                search_time_s = base_time_s * step_relax
                search_dist   = max_dist_px * step_relax

                # Time window: [ph_t, ph_t + search_time_s]
                l = int(np.searchsorted(pix_t, ph_t - TOL))
                r = int(np.searchsorted(pix_t, ph_t + search_time_s + TOL, side='right'))
                if r == l:
                    continue

                # Unclaimed candidates within time window
                cands = np.arange(l, r)
                cands = cands[~claimed[cands]]
                if len(cands) == 0:
                    continue

                # Spatial filter: within search_dist of photon position
                dx = pix_x[cands] - ph_x
                dy = pix_y[cands] - ph_y
                cands = cands[dx * dx + dy * dy <= search_dist * search_dist]
                if len(cands) == 0:
                    continue

                # Cap to MAX_CAND nearest candidates
                if len(cands) > MAX_CAND:
                    dx = pix_x[cands] - ph_x
                    dy = pix_y[cands] - ph_y
                    cands = cands[np.argsort(dx * dx + dy * dy)[:MAX_CAND]]

                # Exhaustive subset search via vectorised bit-matrix
                n_c  = len(cands)
                tots = pix_tot[cands].astype(np.float32)
                wx   = (pix_x[cands] * tots).astype(np.float32)
                wy   = (pix_y[cands] * tots).astype(np.float32)

                masks    = np.arange(1, 1 << n_c, dtype=np.int32)
                bits     = ((masks[:, None] >> np.arange(n_c, dtype=np.int32)) & 1).astype(np.float32)
                counts   = bits.sum(axis=1)

                valid = counts >= min_pixels
                if not valid.any():
                    continue

                tot_sums = bits @ tots
                tot_safe = np.where(tot_sums == 0, 1.0, tot_sums)
                cog_xs   = (bits @ wx) / tot_safe
                cog_ys   = (bits @ wy) / tot_safe

                # EMPIR truncates CoG to 2 decimal places
                trunc_xs = np.floor(cog_xs * 100) / 100
                trunc_ys = np.floor(cog_ys * 100) / 100
                sq_dists = (trunc_xs - ph_x) ** 2 + (trunc_ys - ph_y) ** 2

                sq_dists_v = np.where(valid, sq_dists, np.inf)
                best_i     = int(np.argmin(sq_dists_v))
                com_dist   = float(np.sqrt(sq_dists_v[best_i]))

                if com_dist < best_com_dist:
                    best_com_dist = com_dist
                    best_cand_idx = cands[bits[best_i].astype(bool)]

                if best_com_dist <= CONVERGE_DIST:
                    break   # converged — no need to widen further

            # Record quality and claim the best match found
            if best_cand_idx is None:
                com_quality['failed'] += 1
                continue

            if best_com_dist <= 1e-10:
                com_quality['exact'] += 1
            elif best_com_dist <= 0.2:
                com_quality['good'] += 1
            elif best_com_dist <= 0.5:
                com_quality['acceptable'] += 1
            elif best_com_dist <= max_dist_px:
                com_quality['poor'] += 1
            else:
                com_quality['failed'] += 1

            claimed[best_cand_idx]   = True
            assoc_id[best_cand_idx]  = ph_id
            assoc_x[best_cand_idx]   = ph_x
            assoc_y[best_cand_idx]   = ph_y
            assoc_t[best_cand_idx]   = ph_t
            assoc_com[best_cand_idx] = best_com_dist

        pixels['assoc_photon_id'] = assoc_id
        pixels['assoc_phot_x']    = assoc_x
        pixels['assoc_phot_y']    = assoc_y
        pixels['assoc_phot_t']    = assoc_t
        pixels['pixel_com_dist']  = assoc_com

        self._store_pixel_photon_stats(pixels, photons, com_quality, verbosity)
        return pixels

    def _associate_pixels_to_photons_kdtree(self, pixels_df, photons_df,
                                              max_dist_px=5.0, max_time_ns=500, verbosity=0):
        """Associate pixels to photons using KDTree for spatial queries + CoM refinement."""
        if pixels_df is None or photons_df is None or len(pixels_df) == 0 or len(photons_df) == 0:
            return pixels_df

        pixels = pixels_df.copy()
        photons = photons_df.copy()
        pixels['assoc_photon_id'] = np.nan
        pixels['assoc_phot_x'] = np.nan
        pixels['assoc_phot_y'] = np.nan
        pixels['assoc_phot_t'] = np.nan
        pixels['pixel_com_dist'] = np.nan

        pixels = pixels.sort_values('t').reset_index(drop=True)
        photons = photons.sort_values('t').reset_index(drop=True)
        photons['photon_id'] = photons.index + 1

        px_coords = np.column_stack([pixels['x'].to_numpy(), pixels['y'].to_numpy()])
        pixel_tree = cKDTree(px_coords)
        max_time_s = max_time_ns / 1e9
        com_quality = {'exact': 0, 'good': 0, 'acceptable': 0, 'poor': 0, 'failed': 0}

        for _, phot in tqdm(photons.iterrows(), total=len(photons),
                            desc="Associating pixels to photons (kdtree)", disable=(verbosity == 0)):
            phot_t, phot_x, phot_y, phot_id = phot['t'], phot['x'], phot['y'], phot['photon_id']

            cand_idx = pixel_tree.query_ball_point([phot_x, phot_y], max_dist_px)
            if not cand_idx:
                continue

            cand_px = pixels.iloc[cand_idx]
            tmask = (cand_px['t'] >= phot_t) & (cand_px['t'] <= phot_t + max_time_s)
            if not tmask.any():
                continue

            valid_px = cand_px[tmask]
            valid_idx = np.array(cand_idx)[tmask.to_numpy()]
            unassigned = valid_px['assoc_photon_id'].isna().to_numpy()
            if not unassigned.any():
                continue

            ua_idx = valid_idx[unassigned]
            ua = valid_px[unassigned]
            ua_x = ua['x'].to_numpy()
            ua_y = ua['y'].to_numpy()
            ua_tot = ua['tot'].to_numpy() if 'tot' in ua.columns else np.ones(len(ua))

            # Iterative CoM refinement
            keep = np.ones(len(ua_idx), dtype=bool)
            for _ in range(10):
                tot_sum = ua_tot[keep].sum() or 1
                com_x = (ua_x[keep] * ua_tot[keep]).sum() / tot_sum
                com_y = (ua_y[keep] * ua_tot[keep]).sum() / tot_sum
                com_dist = np.sqrt((com_x - phot_x)**2 + (com_y - phot_y)**2)
                if com_dist <= max_dist_px * 0.5:
                    break
                px_dists = np.sqrt((ua_x[keep] - phot_x)**2 + (ua_y[keep] - phot_y)**2)
                if len(px_dists) <= 1:
                    break
                worst = np.argmax(px_dists)
                keep_idx = np.where(keep)[0]
                keep[keep_idx[worst]] = False

            if keep.any():
                ft = ua_tot[keep]
                fts = ft.sum() or 1
                final_com_dist = np.sqrt(
                    ((ua_x[keep] * ft).sum() / fts - phot_x)**2 +
                    ((ua_y[keep] * ft).sum() / fts - phot_y)**2
                )
                if final_com_dist <= 0.1:
                    com_quality['exact'] += 1
                elif final_com_dist <= max_dist_px * 0.3:
                    com_quality['good'] += 1
                elif final_com_dist <= max_dist_px * 0.5:
                    com_quality['acceptable'] += 1
                elif final_com_dist <= max_dist_px:
                    com_quality['poor'] += 1
                else:
                    com_quality['failed'] += 1

                for i in ua_idx[keep]:
                    pixels.loc[i, 'assoc_photon_id'] = phot_id
                    pixels.loc[i, 'assoc_phot_x'] = phot_x
                    pixels.loc[i, 'assoc_phot_y'] = phot_y
                    pixels.loc[i, 'assoc_phot_t'] = phot_t
                    pixels.loc[i, 'pixel_com_dist'] = final_com_dist

        self._store_pixel_photon_stats(pixels, photons, com_quality, verbosity)
        return pixels

    def _associate_pixels_to_photons_mystic(self, pixels_df, photons_df, max_dist_px=5.0,
                                              max_time_ns=500, time_weight=1.0, cog_weight=1.0,
                                              min_pixels=1, verbosity=0):
        """Associate pixels to photons using mystic constrained optimization."""
        try:
            from mystic.solvers import fmin_powell, diffev2
        except ImportError:
            raise ImportError("mystic required. Install with: pip install mystic")

        if pixels_df is None or photons_df is None or len(pixels_df) == 0 or len(photons_df) == 0:
            return pixels_df

        pixels = pixels_df.copy()
        photons = photons_df.copy()
        pixels['assoc_photon_id'] = np.nan
        pixels['assoc_phot_x'] = np.nan
        pixels['assoc_phot_y'] = np.nan
        pixels['assoc_phot_t'] = np.nan
        pixels['pixel_com_dist'] = np.nan

        pixels = pixels.sort_values('t').reset_index(drop=True)
        photons = photons.sort_values('t').reset_index(drop=True)
        photons['photon_id'] = photons.index + 1

        pixel_tree = cKDTree(np.column_stack([pixels['x'].to_numpy(), pixels['y'].to_numpy()]))
        max_time_s = max_time_ns / 1e9
        com_quality = {'exact': 0, 'good': 0, 'acceptable': 0, 'poor': 0, 'failed': 0}
        opt_stats = {'success': 0, 'fallback': 0, 'failed': 0}

        for _, phot in tqdm(photons.iterrows(), total=len(photons),
                            desc="Associating pixels to photons (mystic)", disable=(verbosity == 0)):
            phot_t, phot_x, phot_y, phot_id = phot['t'], phot['x'], phot['y'], phot['photon_id']

            cand_idx = pixel_tree.query_ball_point([phot_x, phot_y], max_dist_px)
            if not cand_idx:
                continue

            cand_px = pixels.iloc[cand_idx]
            tmask = (cand_px['t'] >= phot_t) & (cand_px['t'] <= phot_t + max_time_s)
            if not tmask.any():
                continue

            valid_px = cand_px[tmask]
            valid_idx = np.array(cand_idx)[tmask.to_numpy()]
            unassigned = valid_px['assoc_photon_id'].isna().to_numpy()
            if not unassigned.any():
                continue

            ua_idx = valid_idx[unassigned]
            ua = valid_px[unassigned]
            ux = ua['x'].to_numpy()
            uy = ua['y'].to_numpy()
            ut = ua['t'].to_numpy()
            utot = ua['tot'].to_numpy() if 'tot' in ua.columns else np.ones(len(ua))
            nc = len(ua_idx)

            if nc <= min_pixels:
                tot_sum = utot.sum() or 1
                com_dist = np.sqrt(
                    ((ux * utot).sum() / tot_sum - phot_x)**2 +
                    ((uy * utot).sum() / tot_sum - phot_y)**2
                )
                for pi in ua_idx:
                    pixels.loc[pi, 'assoc_photon_id'] = phot_id
                    pixels.loc[pi, ['assoc_phot_x', 'assoc_phot_y', 'assoc_phot_t']] = phot_x, phot_y, phot_t
                    pixels.loc[pi, 'pixel_com_dist'] = com_dist
                opt_stats['fallback'] += 1
                continue

            def objective(weights):
                w = np.array(weights)
                ws = w.sum()
                if ws < 0.1:
                    return 1e10
                ew = w * utot
                ews = ew.sum()
                if ews < 0.01:
                    return 1e10
                cx = (ew * ux).sum() / ews
                cy = (ew * uy).sum() / ews
                cog_dist = np.sqrt((cx - phot_x)**2 + (cy - phot_y)**2)
                wt = (w * (ut - phot_t) * 1e9).sum() / ws
                return cog_weight * cog_dist**2 + time_weight * (wt / max_time_ns)**2

            def constraint(weights):
                w = np.array(weights)
                ws = w.sum()
                if ws < min_pixels:
                    w = np.clip(w * min_pixels / max(ws, 0.01), 0, 1)
                return w

            bounds = list(zip(np.zeros(nc), np.ones(nc)))
            try:
                sol = fmin_powell(objective, np.full(nc, 0.5), bounds=bounds,
                                  constraints=constraint, disp=False, gtol=1e-4) if nc <= 20 else \
                    diffev2(objective, bounds, constraints=constraint,
                            npop=min(20, nc * 2), disp=False, gtol=50)
                ow = np.array(sol)
                amask = ow >= 0.3
                if amask.sum() < min_pixels:
                    top = np.argsort(ow)[-min_pixels:]
                    amask = np.zeros(nc, dtype=bool)
                    amask[top] = True
                opt_stats['success'] += 1
            except Exception:
                amask = np.ones(nc, dtype=bool)
                opt_stats['failed'] += 1

            final_idx = ua_idx[amask]
            fx, fy, ft = ux[amask], uy[amask], utot[amask]
            if len(final_idx) > 0:
                fts = ft.sum() or 1
                com_dist = np.sqrt(
                    ((fx * ft).sum() / fts - phot_x)**2 +
                    ((fy * ft).sum() / fts - phot_y)**2
                )
                if com_dist <= 0.1:
                    com_quality['exact'] += 1
                elif com_dist <= max_dist_px * 0.3:
                    com_quality['good'] += 1
                elif com_dist <= max_dist_px * 0.5:
                    com_quality['acceptable'] += 1
                elif com_dist <= max_dist_px:
                    com_quality['poor'] += 1
                else:
                    com_quality['failed'] += 1

                for pi in final_idx:
                    pixels.loc[pi, 'assoc_photon_id'] = phot_id
                    pixels.loc[pi, ['assoc_phot_x', 'assoc_phot_y', 'assoc_phot_t']] = phot_x, phot_y, phot_t
                    pixels.loc[pi, 'pixel_com_dist'] = com_dist

        if verbosity >= 1:
            print(f"Mystic optimization: {opt_stats['success']} success, "
                  f"{opt_stats['fallback']} fallback, {opt_stats['failed']} failed")
        self._store_pixel_photon_stats(pixels, photons, com_quality, verbosity)
        return pixels

    def _associate_pixels_to_photons_ml(self, pixels_df, photons_df,
                                         max_dist_px=5.0, max_time_ns=500,
                                         model=None, model_path=None, verbosity=0):
        """Associate pixels to photons using a trained ML model."""
        if model is None and model_path is not None:
            try:
                import joblib
                model = joblib.load(model_path)
            except Exception as e:
                if verbosity >= 1:
                    print(f"Failed to load model: {e}, falling back to simple")
                return self._associate_pixels_to_photons_simple(
                    pixels_df, photons_df, max_dist_px, max_time_ns, verbosity)

        if model is None and hasattr(self, '_ml_association_model') and self._ml_association_model is not None:
            model = self._ml_association_model

        if model is None:
            if verbosity >= 1:
                print("No ML model available, falling back to simple. "
                      "Train with train_association_model() first.")
            return self._associate_pixels_to_photons_simple(
                pixels_df, photons_df, max_dist_px, max_time_ns, verbosity)

        if pixels_df is None or photons_df is None or len(pixels_df) == 0 or len(photons_df) == 0:
            return pixels_df

        pixels = pixels_df.copy()
        photons = photons_df.copy()
        pixels['assoc_photon_id'] = np.nan
        pixels['assoc_phot_x'] = np.nan
        pixels['assoc_phot_y'] = np.nan
        pixels['assoc_phot_t'] = np.nan
        pixels['pixel_com_dist'] = np.nan
        pixels['ml_confidence'] = np.nan

        pixels = pixels.sort_values('t').reset_index(drop=True)
        photons = photons.sort_values('t').reset_index(drop=True)
        photons['photon_id'] = photons.index + 1

        pixel_tree = cKDTree(np.column_stack([pixels['x'].to_numpy(), pixels['y'].to_numpy()]))
        max_time_s = max_time_ns / 1e9
        com_quality = {'exact': 0, 'good': 0, 'acceptable': 0, 'poor': 0, 'failed': 0}

        for _, phot in tqdm(photons.iterrows(), total=len(photons),
                            desc="Associating pixels to photons (ML)", disable=(verbosity == 0)):
            phot_t, phot_x, phot_y, phot_id = phot['t'], phot['x'], phot['y'], phot['photon_id']

            cand_idx = pixel_tree.query_ball_point([phot_x, phot_y], max_dist_px)
            if not cand_idx:
                continue

            cand_px = pixels.iloc[cand_idx]
            tmask = (cand_px['t'] >= phot_t) & (cand_px['t'] <= phot_t + max_time_s)
            if not tmask.any():
                continue

            valid_px = cand_px[tmask]
            valid_idx = np.array(cand_idx)[tmask.to_numpy()]
            unassigned = valid_px['assoc_photon_id'].isna().to_numpy()
            if not unassigned.any():
                continue

            ua_idx = valid_idx[unassigned]
            ua = valid_px[unassigned]

            features = self._extract_ml_features(ua, phot_x, phot_y, phot_t, max_dist_px, max_time_ns)
            try:
                if hasattr(model, 'predict_proba'):
                    inf_feat = model._scaler.transform(features) if hasattr(model, '_scaler') else features
                    proba = model.predict_proba(inf_feat)
                    probs = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
                elif hasattr(model, 'forward'):
                    import torch
                    model.eval()
                    with torch.no_grad():
                        if hasattr(model, '_X_mean'):
                            features = (features - model._X_mean) / model._X_std
                        probs = torch.sigmoid(model(torch.FloatTensor(features))).numpy().flatten()
                else:
                    probs = np.clip(model.predict(features), 0, 1)
            except Exception as e:
                if verbosity >= 2:
                    print(f"ML prediction failed: {e}")
                continue

            amask = probs >= 0.5
            if not amask.any():
                continue

            final_idx = ua_idx[amask]
            final_probs = probs[amask]
            fx = ua['x'].to_numpy()[amask]
            fy = ua['y'].to_numpy()[amask]
            ftot = ua['tot'].to_numpy()[amask] if 'tot' in ua.columns else np.ones(amask.sum())

            fts = ftot.sum() or 1
            com_dist = np.sqrt(
                ((fx * ftot).sum() / fts - phot_x)**2 +
                ((fy * ftot).sum() / fts - phot_y)**2
            )
            if com_dist <= 0.1:
                com_quality['exact'] += 1
            elif com_dist <= max_dist_px * 0.3:
                com_quality['good'] += 1
            elif com_dist <= max_dist_px * 0.5:
                com_quality['acceptable'] += 1
            elif com_dist <= max_dist_px:
                com_quality['poor'] += 1
            else:
                com_quality['failed'] += 1

            for i, pi in enumerate(final_idx):
                pixels.loc[pi, 'assoc_photon_id'] = phot_id
                pixels.loc[pi, ['assoc_phot_x', 'assoc_phot_y', 'assoc_phot_t']] = phot_x, phot_y, phot_t
                pixels.loc[pi, 'pixel_com_dist'] = com_dist
                pixels.loc[pi, 'ml_confidence'] = final_probs[i]

        self._store_pixel_photon_stats(pixels, photons, com_quality, verbosity)
        return pixels

    def _store_pixel_photon_stats(self, pixels, photons, com_quality, verbosity):
        """Compute and store pixel-photon association statistics."""
        matched_px = pixels['assoc_photon_id'].notna().sum()
        total_px = len(pixels)
        matched_ph_ids = pixels[pixels['assoc_photon_id'].notna()]['assoc_photon_id'].unique()
        matched_ph = len(matched_ph_ids)
        total_ph = len(photons)

        self.last_assoc_stats = {
            'matched_pixels': int(matched_px),
            'total_pixels': int(total_px),
            'matched_photons': int(matched_ph),
            'total_photons': int(total_ph),
            'com_quality': com_quality.copy()
        }

        if verbosity >= 1:
            px_pct = 100 * matched_px / total_px if total_px > 0 else 0
            ph_pct = 100 * matched_ph / total_ph if total_ph > 0 else 0
            print(f"Pixel-Photon: {matched_px:,}/{total_px:,} pixels ({px_pct:.1f}%)")
            print(f"              {matched_ph:,}/{total_ph:,} photons ({ph_pct:.1f}%)")

    # =========================================================================
    # ML model training
    # =========================================================================

    def _extract_ml_features(self, pixels_subset, phot_x, phot_y, phot_t, max_dist_px, max_time_ns):
        """Extract features for ML pixel-photon association prediction."""
        n = len(pixels_subset)
        if n == 0:
            return np.array([]).reshape(0, 9)

        px = pixels_subset['x'].to_numpy()
        py = pixels_subset['y'].to_numpy()
        pt = pixels_subset['t'].to_numpy()
        ptot = pixels_subset['tot'].to_numpy() if 'tot' in pixels_subset.columns else np.ones(n)

        dx = px - phot_x
        dy = py - phot_y
        dist = np.sqrt(dx**2 + dy**2)
        dt_ns = (pt - phot_t) * 1e9
        dist_centroid = np.sqrt((px - px.mean())**2 + (py - py.mean())**2)
        tot_mean = ptot.mean() if ptot.mean() > 0 else 1

        return np.column_stack([
            dist / max_dist_px,
            dx / max_dist_px,
            dy / max_dist_px,
            dt_ns / max_time_ns,
            ptot / (ptot.max() if ptot.max() > 0 else 1),
            dist_centroid / max_dist_px,
            ptot / tot_mean,
            np.full(n, n / 10),
            np.abs(dt_ns) / max_time_ns
        ])

    def train_association_model(self, training_data=None, method='simple', model_type='rf',
                                max_dist_px=5.0, max_time_ns=500, n_samples=10000,
                                save_path=None, verbosity=1):
        """
        Train an ML model for pixel-to-photon association.

        Generates labeled training data using a reference association method,
        then trains a binary classifier predicting pixel-to-photon membership.

        Args:
            training_data (pd.DataFrame, optional): Pre-labeled data with 'label' column.
            method (str): Reference method for generating labels ('simple', 'kdtree', 'mystic').
            model_type (str): 'rf' (Random Forest), 'gb' (Gradient Boosting),
                              'mlp' (Neural Net sklearn), 'torch' (PyTorch).
            max_dist_px (float): Max spatial distance for candidate selection.
            max_time_ns (float): Max time window in nanoseconds.
            n_samples (int): Max training samples.
            save_path (str): Optional additional save path.
            verbosity (int): Verbosity level.

        Returns:
            model: Trained ML model.

        Example:
            assoc = nea.Analyse("data/run1")
            assoc.train_association_model(method='mystic', model_type='rf')
            assoc.associate(method='ml')
        """
        if training_data is None:
            if self.pixels_df is None or self.photons_df is None:
                raise ValueError("No pixels/photons loaded. Load data first.")
            if verbosity >= 1:
                print(f"Generating training data using '{method}' method...")
            if method == 'mystic':
                labeled = self._associate_pixels_to_photons_mystic(
                    self.pixels_df.copy(), self.photons_df.copy(),
                    max_dist_px, max_time_ns, verbosity=max(0, verbosity - 1))
            elif method == 'kdtree':
                labeled = self._associate_pixels_to_photons_kdtree(
                    self.pixels_df.copy(), self.photons_df.copy(),
                    max_dist_px, max_time_ns, verbosity=max(0, verbosity - 1))
            else:
                labeled = self._associate_pixels_to_photons_simple(
                    self.pixels_df.copy(), self.photons_df.copy(),
                    max_dist_px, max_time_ns, verbosity=max(0, verbosity - 1))
            training_data = self._generate_training_data(
                labeled, self.photons_df, max_dist_px, max_time_ns, n_samples, verbosity)

        if len(training_data) == 0:
            raise ValueError("No training data generated.")

        X = training_data.drop(columns=['label']).values
        y = training_data['label'].values

        if verbosity >= 1:
            print(f"Training {model_type} on {len(X):,} samples "
                  f"({100 * y.mean():.1f}% positive)...")

        if model_type == 'rf':
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
            model.fit(X, y)
        elif model_type == 'gb':
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
            model.fit(X, y)
        elif model_type == 'mlp':
            from sklearn.neural_network import MLPClassifier
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
            model.fit(X_scaled, y)
            model._scaler = scaler
        elif model_type == 'torch':
            model = self._train_torch_model(X, y, verbosity)
        else:
            raise ValueError(f"Unknown model_type '{model_type}'. Use 'rf', 'gb', 'mlp', or 'torch'.")

        self._ml_association_model = model

        if verbosity >= 1 and hasattr(model, 'predict_proba'):
            from sklearn.metrics import accuracy_score, f1_score
            X_eval = model._scaler.transform(X) if hasattr(model, '_scaler') else X
            y_pred = model.predict(X_eval)
            print(f"   Training accuracy: {100 * accuracy_score(y, y_pred):.1f}%")
            print(f"   Training F1:       {f1_score(y, y_pred):.3f}")

        # Auto-save
        import joblib
        auto_dir = os.path.join(self.data_folder, "AssociatedResults")
        os.makedirs(auto_dir, exist_ok=True)
        auto_path = os.path.join(auto_dir, "ml_association_model.joblib")
        joblib.dump(model, auto_path)
        if verbosity >= 1:
            print(f"   Model saved to: {auto_path}")
        if save_path:
            joblib.dump(model, save_path)
            if verbosity >= 1:
                print(f"   Model also saved to: {save_path}")

        return model

    def _generate_training_data(self, labeled_pixels, photons_df, max_dist_px,
                                 max_time_ns, n_samples, verbosity):
        """Generate balanced training data from labeled pixel associations."""
        pixel_tree = cKDTree(np.column_stack([
            labeled_pixels['x'].to_numpy(), labeled_pixels['y'].to_numpy()
        ]))
        photons = photons_df.sort_values('t').reset_index(drop=True).copy()
        photons['photon_id'] = photons.index + 1
        max_time_s = max_time_ns / 1e9

        all_features, all_labels = [], []
        spp = max(1, n_samples // len(photons))

        for _, phot in tqdm(photons.iterrows(), total=len(photons),
                            desc="Generating training data", disable=(verbosity == 0)):
            phot_t, phot_x, phot_y, phot_id = phot['t'], phot['x'], phot['y'], phot['photon_id']

            cand_idx = pixel_tree.query_ball_point([phot_x, phot_y], max_dist_px)
            if not cand_idx:
                continue
            cand = labeled_pixels.iloc[cand_idx]
            tmask = (cand['t'] >= phot_t) & (cand['t'] <= phot_t + max_time_s)
            if not tmask.any():
                continue

            valid = cand[tmask]
            features = self._extract_ml_features(valid, phot_x, phot_y, phot_t, max_dist_px, max_time_ns)
            labels = (valid['assoc_photon_id'] == phot_id).astype(int).values
            if labels.sum() == 0:
                continue

            if len(features) > spp:
                pos_i = np.where(labels == 1)[0]
                neg_i = np.where(labels == 0)[0]
                n_pos = min(len(pos_i), spp // 2)
                n_neg = min(len(neg_i), spp // 2)
                sel = np.concatenate([
                    np.random.choice(pos_i, n_pos, replace=False) if n_pos > 0 else np.array([], int),
                    np.random.choice(neg_i, n_neg, replace=False) if n_neg > 0 else np.array([], int)
                ])
                features = features[sel]
                labels = labels[sel]

            all_features.append(features)
            all_labels.append(labels)

            if sum(len(f) for f in all_features) >= n_samples:
                break

        if not all_features:
            return pd.DataFrame()

        X = np.vstack(all_features)
        y = np.concatenate(all_labels)
        cols = ['dist_norm', 'dx_norm', 'dy_norm', 'dt_norm', 'tot_norm',
                'dist_centroid', 'tot_rel', 'cluster_size', 'abs_dt']
        df = pd.DataFrame(X, columns=cols)
        df['label'] = y
        return df

    def _train_torch_model(self, X, y, verbosity):
        """Train a PyTorch neural network for association prediction."""
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            raise ImportError("PyTorch required. Install with: pip install torch")

        class AssociationNet(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.2),
                    nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
                    nn.Linear(32, 16), nn.ReLU(),
                    nn.Linear(16, 1)
                )

            def forward(self, x):
                return self.net(x)

        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0) + 1e-8
        X_norm = (X - X_mean) / X_std

        dataset = TensorDataset(torch.FloatTensor(X_norm), torch.FloatTensor(y).unsqueeze(1))
        loader = DataLoader(dataset, batch_size=256, shuffle=True)
        model = AssociationNet(X.shape[1])
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        model.train()
        for epoch in range(50):
            total_loss = 0
            for bx, by in loader:
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if verbosity >= 2 and (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch+1}/50, Loss: {total_loss/len(loader):.4f}")

        model._X_mean = X_mean
        model._X_std = X_std
        return model

    # =========================================================================
    # Column naming
    # =========================================================================

    def _standardize_column_names(self, df, verbosity=0):
        """Rename internal assoc_* columns to px/ph/ev prefix scheme."""
        rename = {
            'x': 'px/x', 'y': 'px/y', 't': 'px/toa', 'tot': 'px/tot', 'tof': 'px/tof',
            'assoc_photon_id': 'ph/id',
            'assoc_phot_x': 'ph/x', 'assoc_phot_y': 'ph/y', 'assoc_phot_t': 'ph/toa',
            'pixel_com_dist': 'ph/cog',
            'assoc_event_id': 'ev/id',
            'assoc_x': 'ev/x', 'assoc_y': 'ev/y', 'assoc_t': 'ev/toa',
            'assoc_n': 'ev/n', 'assoc_PSD': 'ev/psd', 'assoc_com_dist': 'ev/cog',
        }
        cols = {k: v for k, v in rename.items() if k in df.columns}
        df = df.rename(columns=cols)
        drop = [c for c in ['pixel_time_diff_ns', 'pixel_spatial_diff_px',
                             'time_diff_ns', 'spatial_diff_px'] if c in df.columns]
        if drop:
            df = df.drop(columns=drop)
        if 'ph/id' in df.columns:
            counts = df.groupby('ph/id').size()
            df['ph/n'] = df['ph/id'].map(counts).fillna(0).astype(int)
        return df

    def _rename_columns_for_export(self, df):
        """Rename columns to clean px/ph/ev prefix scheme for CSV export."""
        df_out = df.copy()
        has_pixel = 'tot' in df.columns or 'assoc_photon_id' in df.columns
        rename = {
            'assoc_photon_id': 'ph/id', 'assoc_phot_x': 'ph/x',
            'assoc_phot_y': 'ph/y', 'assoc_phot_t': 'ph/toa', 'pixel_com_dist': 'ph/cog',
            'assoc_event_id': 'ev/id', 'assoc_cluster_id': 'ev/id',
            'assoc_x': 'ev/x', 'assoc_y': 'ev/y', 'assoc_t': 'ev/toa',
            'assoc_n': 'ev/n', 'assoc_PSD': 'ev/psd', 'assoc_com_dist': 'ev/cog',
        }
        if has_pixel:
            rename.update({'x': 'px/x', 'y': 'px/y', 't': 'px/toa', 'tot': 'px/tot', 'tof': 'px/tof'})
        else:
            rename.update({'x': 'ph/x', 'y': 'ph/y', 't': 'ph/toa', 'tof': 'ph/tof'})
        df_out = df_out.rename(columns={k: v for k, v in rename.items() if k in df_out.columns})
        drop = [c for c in ['pixel_time_diff_ns', 'pixel_spatial_diff_px',
                             'time_diff_ns', 'spatial_diff_px'] if c in df_out.columns]
        if drop:
            df_out = df_out.drop(columns=drop)
        return df_out

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_association_stats(self):
        """Return association statistics as a dict."""
        stats = {}
        if self.last_assoc_stats:
            stats.update(self.last_assoc_stats)
        if self.last_photon_event_stats:
            stats['photon_event'] = self.last_photon_event_stats
        return stats

    def _compute_stats_from_dataframe(self, df):
        """Reconstruct association stats from a pre-loaded associated CSV."""
        stats = {}
        has_ph = 'ph/n' in df.columns
        has_ev = 'ev/n' in df.columns

        if has_ph and has_ev:
            event_rows = df[df['ev/id'].notna()] if 'ev/id' in df.columns else df.iloc[0:0]
            matched_ph = len(event_rows)
            total_ph = len(df)
            matched_ev = int(df['ev/id'].nunique()) if 'ev/id' in df.columns else 0

            quality = {'exact_n': 0, 'n_mismatch': 0, 'exact_com': 0,
                       'good_com': 0, 'acceptable_com': 0, 'poor_com': 0}
            if 'ev/n' in event_rows.columns and 'ph/n' in event_rows.columns:
                with_n = event_rows[event_rows['ev/n'].notna() & event_rows['ph/n'].notna()]
                quality['exact_n'] = int((with_n['ev/n'] == with_n['ph/n']).sum())
                quality['n_mismatch'] = int((with_n['ev/n'] != with_n['ph/n']).sum())

            stats['photon_event'] = {
                'matched_photons': int(matched_ph),
                'total_photons': int(total_ph),
                'matched_events': int(matched_ev),
                'total_events': int(matched_ev),
                'quality': quality
            }
        return stats

    def _compute_distributional_stats(self):
        """Compute count/mean/std/p10/p50/p90 for key columns."""
        if self.associated_df is None or len(self.associated_df) == 0:
            return {}

        df = self.associated_df
        stats = {}

        def _col_stats(series):
            s = series.dropna()
            if len(s) == 0:
                return None
            return {
                'count': int(len(s)), 'mean': float(s.mean()), 'std': float(s.std()),
                'p10': float(s.quantile(0.1)), 'p50': float(s.quantile(0.5)),
                'p90': float(s.quantile(0.9)),
            }

        for col in ['ph/x', 'ph/n', 'ph/cog', 'ph/toa',
                    'ev/x', 'ev/n', 'ev/psd', 'ev/cog', 'ev/toa']:
            if col in df.columns:
                result = _col_stats(df[col])
                if result:
                    stats[col] = result

        if 'px/toa' in df.columns and 'ph/toa' in df.columns:
            result = _col_stats((df['px/toa'] - df['ph/toa']).abs())
            if result:
                stats['ph/dt'] = result

        if 'ph/toa' in df.columns and 'ev/toa' in df.columns:
            result = _col_stats((df['ph/toa'] - df['ev/toa']).abs())
            if result:
                stats['ev/dt'] = result

        return stats

    # =========================================================================
    # Save
    # =========================================================================

    def save_associations(self, output_dir=None, filename="associated_data.csv",
                          format='csv', suffix=None, verbosity=1):
        """
        Save associated results to a file.

        Args:
            output_dir (str): Output directory. Default: <data_folder>/AssociatedResults.
            filename (str): Output filename. Default: 'associated_data.csv'.
            format (str): 'csv' or 'parquet'.
            suffix (str): If set, overrides filename to 'associated_data_<suffix>.csv'.
            verbosity (int): 0=silent, 1=info.

        Returns:
            str: Path to saved file.
        """
        if self.associated_df is None or len(self.associated_df) == 0:
            raise ValueError("No association data. Run associate() first.")

        if suffix:
            filename = f"associated_data_{suffix}.{format}"

        df_save = self._rename_columns_for_export(self.associated_df)

        if output_dir is None:
            output_dir = os.path.join(self.data_folder, "AssociatedResults")
        os.makedirs(output_dir, exist_ok=True)

        out_path = os.path.join(output_dir, filename)
        if format.lower() == 'csv':
            df_save.to_csv(out_path, index=False)
        elif format.lower() == 'parquet':
            try:
                df_save.to_parquet(out_path, index=False)
            except ImportError:
                raise ImportError("Parquet requires pyarrow: pip install pyarrow")
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Save stats JSON
        stats_dict = {}
        if self.last_assoc_stats:
            stats_dict['pixel_photon'] = self.last_assoc_stats
        if self.last_photon_event_stats:
            stats_dict['photon_event'] = self.last_photon_event_stats
        dist_stats = self._compute_distributional_stats()
        if dist_stats:
            stats_dict['distributions'] = dist_stats

        if stats_dict:
            def _to_native(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: _to_native(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [_to_native(i) for i in obj]
                return obj

            stats_name = f"association_stats_{suffix}.json" if suffix else "association_stats.json"
            stats_path = os.path.join(output_dir, stats_name)
            with open(stats_path, 'w') as f:
                json.dump(_to_native(stats_dict), f, indent=2)

        if verbosity >= 1:
            size_mb = os.path.getsize(out_path) / (1024 * 1024)
            print(f"Saved {len(df_save):,} rows to {out_path} ({size_mb:.2f} MB)")

        return out_path

    # =========================================================================
    # Analysis utilities
    # =========================================================================

    def compute_ellipticity(self, x_col='x', y_col='y', event_col=None, verbosity=1):
        """
        Compute event ellipticity using PCA of spatial coordinates.

        Updates self.associated_df with columns: major_x, major_y, angle_deg, ellipticity.
        """
        if self.associated_df is None:
            raise ValueError("Run associate() first.")
        if event_col is None:
            event_col = 'assoc_event_id'
        self.associated_df = self._compute_event_ellipticity(
            self.associated_df, x_col, y_col, event_col, verbosity)

    def _compute_event_ellipticity(self, df, x_col, y_col, event_col, verbosity):
        """PCA-based ellipticity for each event group."""
        df = df.copy()
        for col in ['major_x', 'major_y', 'angle_deg', 'ellipticity']:
            df[col] = np.nan

        event_ids = df[event_col].dropna().unique()
        for eid in tqdm(event_ids, desc="Computing ellipticity", disable=(verbosity == 0)):
            grp = df[df[event_col] == eid]
            if len(grp) < 2:
                continue
            coords = grp[[x_col, y_col]].values - grp[[x_col, y_col]].values.mean(axis=0)
            eigvals, eigvecs = np.linalg.eigh(np.cov(coords, rowvar=False))
            major_i = np.argmax(eigvals)
            major_ax = eigvecs[:, major_i]
            ellip = np.sqrt(eigvals[1 - major_i]) / np.sqrt(eigvals[major_i]) \
                if eigvals[major_i] > 0 else 0.0
            df.loc[df[event_col] == eid, 'major_x'] = major_ax[0]
            df.loc[df[event_col] == eid, 'major_y'] = major_ax[1]
            df.loc[df[event_col] == eid, 'angle_deg'] = np.degrees(
                np.arctan2(major_ax[1], major_ax[0]))
            df.loc[df[event_col] == eid, 'ellipticity'] = ellip

        if verbosity >= 1:
            print(f"Computed ellipticity for {len(event_ids)} events.")
        return df

    def get_combined_dataframe(self):
        """Return the associated DataFrame."""
        if self.associated_df is None:
            raise ValueError("Run associate() first.")
        return self.associated_df

    # =========================================================================
    # Jupyter repr
    # =========================================================================

    def _repr_html_(self):
        """Simple HTML summary for Jupyter display."""
        if self.associated_df is None or len(self.associated_df) == 0:
            return "<p><b>Analyse</b> — no association results yet. Run <code>associate()</code>.</p>"

        df = self.associated_df
        lines = [
            "<table style='border-collapse:collapse; font-family:monospace; font-size:12px'>",
            "<tr><th colspan='2' style='text-align:left; padding:4px 8px; "
            "background:#2d2d2d; color:#eee'>Association Summary</th></tr>"
        ]

        def row(label, value, color=''):
            style = f" color:{color};" if color else ""
            return (f"<tr><td style='padding:3px 8px; color:#888'>{label}</td>"
                    f"<td style='padding:3px 8px;{style}'><b>{value}</b></td></tr>")

        lines.append(row("Rows", f"{len(df):,}"))
        lines.append(row("Columns", ", ".join(df.columns.tolist())))
        lines.append(row("Data folder", self.data_folder))
        lines.append(row("Settings", self.settings_source))

        if self.last_photon_event_stats:
            s = self.last_photon_event_stats
            total = s.get('total_photons', 0)
            matched = s.get('matched_photons', 0)
            pct = 100 * matched / total if total > 0 else 0
            color = '#4caf50' if pct > 70 else '#ff9800' if pct > 40 else '#f44336'
            lines.append(row("Photon match rate", f"{matched:,}/{total:,} ({pct:.1f}%)", color))
            ev_total = s.get('total_events', 0)
            ev_matched = s.get('matched_events', 0)
            epct = 100 * ev_matched / ev_total if ev_total > 0 else 0
            lines.append(row("Event match rate", f"{ev_matched:,}/{ev_total:,} ({epct:.1f}%)"))

        if self.last_assoc_stats:
            s = self.last_assoc_stats
            px_total = s.get('total_pixels', 0)
            px_matched = s.get('matched_pixels', 0)
            if px_total > 0:
                ppct = 100 * px_matched / px_total
                lines.append(row("Pixel match rate", f"{px_matched:,}/{px_total:,} ({ppct:.1f}%)"))

        lines.append("</table>")
        return "\n".join(lines)
