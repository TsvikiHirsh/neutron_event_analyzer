import os
import glob
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import tempfile
import uuid
from scipy.spatial import cKDTree
import logging
import json
from pathlib import Path
from .config import DEFAULT_PARAMS

# Check for lumacamTesting availability
try:
    import lumacamTesting as lct
    import yaspin
    LUMACAM_AVAILABLE = True
except ImportError:
    LUMACAM_AVAILABLE = False

# Configure logging (will be adjusted per-instance based on verbosity)
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class Analyse:
    @staticmethod
    def _is_groupby_folder(folder_path):
        """
        Check if a folder is a groupby structure (contains subdirectories with data folders).

        Args:
            folder_path (str): Path to check

        Returns:
            tuple: (is_groupby, subdirs) where is_groupby is bool and subdirs is list of subdirectory names
        """
        folder = Path(folder_path)
        if not folder.exists() or not folder.is_dir():
            return False, []

        # Check for .groupby_metadata.json file
        if (folder / ".groupby_metadata.json").exists():
            subdirs = [d.name for d in folder.iterdir() if d.is_dir() and not d.name.startswith('.')]
            return True, subdirs

        # Check if folder contains subdirectories that have data folders
        subdirs = [d for d in folder.iterdir() if d.is_dir() and not d.name.startswith('.')]

        if len(subdirs) < 2:
            return False, []

        # Check if subdirectories contain typical data folders
        data_folder_names = {'ExportedEvents', 'ExportedPhotons', 'ExportedPixels',
                            'eventFiles', 'photonFiles', 'tpx3Files'}

        has_data_folders = []
        for subdir in subdirs:
            subdir_contents = {d.name for d in subdir.iterdir() if d.is_dir()}
            if subdir_contents & data_folder_names:
                has_data_folders.append(subdir.name)

        # If at least 2 subdirectories have data folders, consider it a groupby structure
        if len(has_data_folders) >= 2:
            return True, has_data_folders

        return False, []

    def __init__(self, data_folder, export_dir="./export", n_threads=10, use_lumacam=False, settings=None,
                 verbosity=1, events=True, photons=True, pixels=True, limit=None, query=None, parent_settings=None):
        """
        Initialize the Analyse object and automatically load data.

        This class provides tools for loading, associating, and analyzing neutron event and photon data
        from paired files. It supports multiple association methods:
        - 'lumacam': Uses lumacamTesting library (requires installation).
        - 'kdtree': Full KDTree-based association on normalized space-time coordinates.
        - 'window': Time-window KDTree for nearly time-sorted data, using symmetric window.
        - 'simple': Simple forward time-window association, selecting closest photons in space,
          with center-of-mass check. Optimized for speed with small windows.

        The class can work with either:
        1. Pre-exported CSV files in 'ExportedEvents', 'ExportedPhotons', and 'ExportedPixels' subdirectories (preferred), or
        2. Original .empirevent, .empirphot, and .tpx3 files (requires empir binaries in export_dir)

        Args:
            data_folder (str): Path to the data folder containing 'photonFiles'/'eventFiles'/'tpx3Files' subdirectories
                              and optionally 'ExportedPhotons'/'ExportedEvents'/'ExportedPixels' subdirectories with CSV files.
            export_dir (str): Path to the directory containing export binaries (empir_export_events, empir_export_photons, empir_pixel2photon).
                             Only required if pre-exported CSV files are not available. Falls back to EMPIR_PATH environment variable if not specified.
            n_threads (int): Number of threads for parallel processing (default: 10).
            use_lumacam (bool): If True, prefer 'lumacam' for association when method='auto' (if available).
            settings (str or dict, optional): Path to settings JSON file or settings dictionary containing empir parameters.
                                             These parameters will be used as defaults for association methods.
            verbosity (int): Verbosity level (0=silent, 1=progress bars only, 2=detailed output). Default is 1.
            events (bool): Whether to load events (default: True).
            photons (bool): Whether to load photons (default: True).
            pixels (bool): Whether to load pixels (default: True).
            limit (int, optional): If provided, limit the number of rows loaded for all data types.
            query (str, optional): If provided, apply a pandas query string to filter the events dataframe.
            parent_settings (dict, optional): Settings inherited from parent groupby folder. Used internally.
        """
        self.data_folder = data_folder
        # Use EMPIR_PATH environment variable if export_dir not provided or is default
        if export_dir == "./export" and 'EMPIR_PATH' in os.environ:
            self.export_dir = os.environ['EMPIR_PATH']
        else:
            self.export_dir = export_dir
        self.n_threads = n_threads
        self.use_lumacam = use_lumacam and LUMACAM_AVAILABLE
        self.verbosity = verbosity
        if use_lumacam and not LUMACAM_AVAILABLE:
            if verbosity >= 2:
                print("Warning: lumacamTesting not installed. Cannot use lumacam association.")
            self.use_lumacam = False
        self.pair_files = None
        self.pair_dfs = None
        self.events_df = None
        self.photons_df = None
        self.pixels_df = None  # NEW: Pixel DataFrame
        self.associated_df = None
        self.assoc_method = None
        self.last_assoc_stats = None  # Statistics from last pixel-photon association
        self.last_photon_event_stats = None  # Statistics from last photon-event association

        # Check if this is a groupby folder structure
        is_groupby, subdirs = self._is_groupby_folder(data_folder)
        self.is_groupby = is_groupby
        self.groupby_subdirs = subdirs if is_groupby else []
        self.groupby_results = {}  # Store results from grouped analyses
        self.limit = limit  # Store limit for passing to child groups

        # Auto-detect settings if not provided (do this even for groupby folders)
        if settings is None:
            # Try to detect settings file in current folder
            detected_settings = self._detect_settings_file()
            if detected_settings:
                settings = detected_settings
            elif parent_settings is not None:
                # Fall back to parent settings if no local settings found
                settings = parent_settings
                if verbosity >= 2:
                    print("⚙️  Inheriting settings from parent folder")

        # Load settings (do this even for groupby folders so they can be passed to child groups)
        self.settings = self._load_settings(settings)
        self.settings_source = self._get_settings_source(settings)

        if is_groupby:
            if verbosity >= 1:
                print(f"📁 Detected groupby folder structure with {len(subdirs)} groups:")
                for subdir in subdirs:
                    print(f"   - {subdir}")
                if self.settings:
                    print(f"⚙️  Will use settings from: {self.settings_source}")

            # Try to load pre-existing association results for all groups
            loaded_groups = []
            self.groupby_stats = {}  # Initialize stats storage
            for subdir in subdirs:
                group_path = os.path.join(data_folder, subdir)
                assoc_file = os.path.join(group_path, "AssociatedResults", "associated_data.csv")
                stats_file = os.path.join(group_path, "AssociatedResults", "association_stats.json")
                if os.path.exists(assoc_file):
                    try:
                        group_df = pd.read_csv(assoc_file)
                        self.groupby_results[subdir] = group_df

                        # Try to load stats from JSON file first (more accurate)
                        if os.path.exists(stats_file):
                            import json
                            with open(stats_file, 'r') as f:
                                self.groupby_stats[subdir] = json.load(f)
                        else:
                            # Fall back to computing stats from DataFrame (less accurate for px2ph)
                            self.groupby_stats[subdir] = self._compute_stats_from_dataframe(group_df)

                        loaded_groups.append(subdir)
                    except Exception as e:
                        if verbosity >= 2:
                            print(f"⚠️  Could not load {subdir}: {e}")

            if loaded_groups:
                if verbosity >= 1:
                    print(f"\n📂 Auto-loaded association results for {len(loaded_groups)} group(s):")
                    for group in loaded_groups:
                        n_rows = len(self.groupby_results[group])
                        print(f"   ✅ {group}: {n_rows:,} rows")
                    print("\nℹ️  You can now use plot_violin(), plot_stats(), etc. without running associate()")
            else:
                if verbosity >= 1:
                    print("\nℹ️  Use .associate() to run association on all groups")
                    print("    or access individual groups using: Analyse(f'{data_folder}/group_name')")

            # Don't auto-load raw data for groupby folders
            return

        # Show settings info if verbosity >= 2
        if verbosity >= 2 and self.settings:
            print(f"⚙️  Using settings: {self.settings_source}")

        # Try to load pre-existing association results
        assoc_file = os.path.join(data_folder, "AssociatedResults", "associated_data.csv")
        stats_file = os.path.join(data_folder, "AssociatedResults", "association_stats.json")
        loaded_assoc_results = False
        if os.path.exists(assoc_file):
            try:
                self.associated_df = pd.read_csv(assoc_file)
                loaded_assoc_results = True

                # Try to load stats from JSON file first (more accurate)
                if os.path.exists(stats_file):
                    import json
                    with open(stats_file, 'r') as f:
                        stats_dict = json.load(f)
                        if 'pixel_photon' in stats_dict:
                            self.last_assoc_stats = stats_dict['pixel_photon']
                        if 'photon_event' in stats_dict:
                            self.last_photon_event_stats = stats_dict['photon_event']
                else:
                    # Fall back to computing stats from DataFrame (less accurate for px2ph)
                    computed_stats = self._compute_stats_from_dataframe(self.associated_df)
                    if 'pixel_photon' in computed_stats:
                        self.last_assoc_stats = computed_stats['pixel_photon']
                    if 'photon_event' in computed_stats:
                        self.last_photon_event_stats = computed_stats['photon_event']

                if verbosity >= 1:
                    print(f"\n📂 Auto-loaded association results: {len(self.associated_df):,} rows")
            except Exception as e:
                if verbosity >= 2:
                    print(f"⚠️  Could not load association results: {e}")

        # Always load raw data (so user can re-run association or use different methods)
        # If any data type is requested (events, photons, or pixels)
        if events or photons or pixels:
            if loaded_assoc_results and verbosity >= 1:
                print("📥 Loading raw data for re-analysis...")
            self.load(events=events, photons=photons, pixels=pixels, limit=limit, query=query, verbosity=verbosity)
        elif loaded_assoc_results and verbosity >= 1:
            print("ℹ️  Raw data not loaded. To re-run association, use .load() to load raw data first.")

    def _load_settings(self, settings):
        """
        Load settings from a named preset, JSON file, or dictionary.

        Args:
            settings (str, dict, or None): Named preset (e.g., 'in_focus'), path to JSON file, or settings dictionary.

        Returns:
            dict: Settings dictionary with empir parameters.
        """
        if settings is None:
            return {}

        if isinstance(settings, dict):
            return settings

        if isinstance(settings, str):
            # Check if it's a named preset first
            if settings in DEFAULT_PARAMS:
                return DEFAULT_PARAMS[settings]

            # Otherwise assume it's a path to a JSON file
            if not os.path.exists(settings):
                print(f"Warning: Settings file not found and not a named preset: {settings}")
                print(f"Available presets: {list(DEFAULT_PARAMS.keys())}")
                return {}
            try:
                with open(settings, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load settings from {settings}: {e}")
                return {}

        print(f"Warning: Invalid settings type: {type(settings)}. Expected str, dict, or None.")
        return {}

    def _detect_settings_file(self):
        """
        Auto-detect settings file in the data folder.

        Checks for .parameterSettings.json or parameterSettings.json in the data folder.
        Hidden file takes precedence.

        Returns:
            str or None: Path to settings file if found, None otherwise.
        """
        for filename in ['.parameterSettings.json', 'parameterSettings.json']:
            settings_path = os.path.join(self.data_folder, filename)
            if os.path.exists(settings_path):
                return settings_path
        return None

    def _get_settings_source(self, settings):
        """
        Determine the source of the settings for display purposes.

        Args:
            settings: The settings parameter passed to __init__ or auto-detected.

        Returns:
            str: Human-readable description of settings source.
        """
        if settings is None:
            return "defaults"
        if isinstance(settings, dict):
            return "dictionary"
        if isinstance(settings, str):
            if settings in DEFAULT_PARAMS:
                return f"preset '{settings}'"
            # It's a file path
            return f"file '{os.path.basename(settings)}'"
        return "unknown"

    def _get_association_defaults(self):
        """
        Extract association parameters from settings.

        Returns:
            dict: Dictionary with default association parameters.
        """
        defaults = {}

        if not self.settings:
            return defaults

        # Extract pixel2photon parameters
        if 'pixel2photon' in self.settings:
            p2p = self.settings['pixel2photon']
            if 'dSpace' in p2p:
                defaults['pixel_max_dist_px'] = float(p2p['dSpace'])
            if 'dTime' in p2p:
                # Convert seconds to nanoseconds
                defaults['pixel_max_time_ns'] = float(p2p['dTime']) * 1e9

        # Extract photon2event parameters
        if 'photon2event' in self.settings:
            p2e = self.settings['photon2event']
            if 'dSpace_px' in p2e:
                defaults['photon_dSpace_px'] = float(p2e['dSpace_px'])
            if 'dTime_s' in p2e:
                # Convert seconds to nanoseconds
                defaults['max_time_ns'] = float(p2e['dTime_s']) * 1e9

        return defaults

    def _process_pair(self, pair, tmp_dir, verbosity=0):
        """
        Process a pair of event and photon files by converting them to CSV and loading into DataFrames.

        Args:
            pair (tuple): Tuple of (event_file, photon_file) paths. Either can be None.
            tmp_dir (str): Path to temporary directory for CSV output.
            verbosity (int): Verbosity level (0=silent, 1=warnings).

        Returns:
            tuple: (event_df, photon_df) if successful, None otherwise.
        """
        event_file, photon_file = pair
        event_df = self._convert_event_file(event_file, tmp_dir, verbosity) if event_file else None
        photon_df = self._convert_photon_file(photon_file, tmp_dir, verbosity) if photon_file else None

        # Return pair if at least one dataframe was loaded successfully
        if event_df is not None or photon_df is not None:
            return event_df, photon_df
        return None

    def load(self, event_glob="[Ee]ventFiles/*.empirevent", photon_glob="[Pp]hotonFiles/*.empirphot",
             pixel_glob="[Tt]px3Files/*.tpx3", events=True, photons=True, pixels=False,
             limit=None, query=None, relax=1.0, verbosity=0,
             # Backward compatibility - deprecated
             load_events=None, load_photons=None, load_pixels=None):
        """
        Load paired event, photon, and optionally pixel files with smart cascading limits.

        This method identifies paired files based on matching base filenames (excluding extensions).
        For each file, it first checks for pre-exported CSV files in ExportedEvents/ExportedPhotons/ExportedPixels folders.
        If CSV files exist, they are used directly. Otherwise, it falls back to converting the original
        files using empir binaries.

        The limit parameter now uses cascading logic to ensure enough downstream data is loaded
        for proper association:

        - **Integer limit**: Interpreted as row count for the primary data type.
          - If pixels loaded: limit applies to pixels, photons loaded up to max_pixel_toa + buffer,
            events loaded up to max_photon_toa + buffer.
          - If only photons+events: limit applies to photons, events loaded up to max_photon_toa + buffer.
          - If only pixels+photons: limit applies to pixels, photons loaded up to max_pixel_toa + buffer.

        - **Float limit**: Interpreted as max time-of-arrival (TOA) in seconds.
          - Primary data filtered to t <= limit, downstream data uses cascading logic.

        Args:
            event_glob (str, optional): Glob pattern relative to data_folder for event files.
            photon_glob (str, optional): Glob pattern relative to data_folder for photon files.
            pixel_glob (str, optional): Glob pattern relative to data_folder for pixel (TPX3) files.
            events (bool, optional): Whether to load events (default: True).
            photons (bool, optional): Whether to load photons (default: True).
            pixels (bool, optional): Whether to load pixels (default: False).
            limit (int or float, optional): If int, limits row count for primary data type.
                                           If float, limits max TOA in seconds.
                                           Downstream data uses cascading limits based on time search windows.
            query (str, optional): If provided, apply a pandas query string to filter the events dataframe (e.g., "n>2").
            relax (float, optional): Relaxation factor for time buffers in cascading limits. Default is 1.0.
                                    Higher values load more downstream data for safety margin.
            verbosity (int, optional): Verbosity level (0=silent, 1=normal, 2=debug). Default is 0.
        """
        # Backward compatibility with old parameter names
        if load_events is not None:
            events = load_events
        if load_photons is not None:
            photons = load_photons
        if load_pixels is not None:
            pixels = load_pixels

        # Configure logging based on verbosity
        if verbosity >= 2:
            logger.setLevel(logging.DEBUG)
        elif verbosity >= 1:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)

        def get_key(f):
            """Get key for file matching, stripping exported_ prefix if present."""
            basename = os.path.basename(f).rsplit('.', 1)[0]
            # Strip 'exported_' prefix that lumacam adds
            if basename.startswith('exported_'):
                basename = basename[9:]  # len('exported_') = 9
            return basename

        # Load event-photon pairs if requested
        if events or photons:
            # First check for already exported CSV files
            exported_events_dir = os.path.join(self.data_folder, "ExportedEvents")
            exported_photons_dir = os.path.join(self.data_folder, "ExportedPhotons")

            # Build file lists - prefer exported CSVs if they exist
            event_files = []
            photon_files = []

            if events:
                if os.path.exists(exported_events_dir):
                    exported_event_csvs = glob.glob(os.path.join(exported_events_dir, "*.csv"))
                    if exported_event_csvs:
                        event_files = exported_event_csvs
                        if verbosity >= 2:
                            print(f"Using {len(event_files)} exported event CSV files from ExportedEvents/")
                if not event_files:
                    # Fall back to binary files
                    event_files = glob.glob(os.path.join(self.data_folder, event_glob))
                    if verbosity >= 2 and event_files:
                        print(f"Using {len(event_files)} binary event files")

            if photons:
                if os.path.exists(exported_photons_dir):
                    exported_photon_csvs = glob.glob(os.path.join(exported_photons_dir, "*.csv"))
                    if exported_photon_csvs:
                        photon_files = exported_photon_csvs
                        if verbosity >= 2:
                            print(f"Using {len(photon_files)} exported photon CSV files from ExportedPhotons/")
                if not photon_files:
                    # Fall back to binary files
                    photon_files = glob.glob(os.path.join(self.data_folder, photon_glob))
                    if verbosity >= 2 and photon_files:
                        print(f"Using {len(photon_files)} binary photon files")

            event_dict = {get_key(f): f for f in event_files}
            photon_dict = {get_key(f): f for f in photon_files}

            if events and photons:
                common_keys = sorted(set(event_dict) & set(photon_dict))
                self.pair_files = [(event_dict[k], photon_dict[k]) for k in common_keys]
                if verbosity >= 2:
                    print(f"Found {len(self.pair_files)} paired event-photon files.")
            elif events:
                self.pair_files = [(event_dict[k], None) for k in sorted(event_dict.keys())]
                if verbosity >= 2:
                    print(f"Found {len(self.pair_files)} event files.")
            else:  # photons only
                self.pair_files = [(None, photon_dict[k]) for k in sorted(photon_dict.keys())]
                if verbosity >= 2:
                    print(f"Found {len(self.pair_files)} photon files.")

            with tempfile.TemporaryDirectory(dir="/tmp") as tmp_dir:
                with ProcessPoolExecutor(max_workers=self.n_threads) as executor:
                    futures = [executor.submit(self._process_pair, pair, tmp_dir, verbosity) for pair in self.pair_files]
                    self.pair_dfs = []
                    for future in tqdm(as_completed(futures), total=len(futures), desc="Loading event-photon pairs", disable=(verbosity == 0)):
                        result = future.result()
                        if result is not None:
                            self.pair_dfs.append(result)

            # Concatenate for full DataFrames
            if self.pair_dfs and events:
                event_dfs = [edf for edf, pdf in self.pair_dfs if edf is not None]
                if event_dfs:
                    self.events_df = pd.concat(event_dfs, ignore_index=True).replace(" nan", float("nan"))
                else:
                    logger.error("No event data could be loaded. Check that ExportedEvents folder exists or empir binaries are available.")
                    self.events_df = pd.DataFrame()
            else:
                self.events_df = pd.DataFrame()

            if self.pair_dfs and photons:
                photon_dfs = [pdf for edf, pdf in self.pair_dfs if pdf is not None]
                if photon_dfs:
                    self.photons_df = pd.concat(photon_dfs, ignore_index=True).replace(" nan", float("nan"))
                else:
                    logger.error("No photon data could be loaded. Check that ExportedPhotons folder exists or empir binaries are available.")
                    self.photons_df = pd.DataFrame()
            else:
                self.photons_df = pd.DataFrame()

            # Apply query filter to events if provided
            if query is not None and events and len(self.events_df) > 0:
                original_events_len = len(self.events_df)
                self.events_df = self.events_df.query(query)
                if verbosity >= 2:
                    print(f"Applied query '{query}': {original_events_len} -> {len(self.events_df)} events")

            # NOTE: Limit application is deferred until after all data is loaded
            # to enable cascading limits (see end of load() method)

        # Load pixels if requested
        if pixels:
            # First check for exported CSV files in ExportedPixels
            exported_pixels_dir = os.path.join(self.data_folder, "ExportedPixels")
            pixel_files = []

            if os.path.exists(exported_pixels_dir):
                # Look for CSV files in ExportedPixels
                csv_files = glob.glob(os.path.join(exported_pixels_dir, "*.csv"))
                if csv_files:
                    pixel_files = csv_files
                    if verbosity >= 2:
                        print(f"Found {len(pixel_files)} exported pixel CSV files in ExportedPixels/")

            # If no exported CSVs, fall back to looking for .tpx3 files
            if not pixel_files:
                pixel_files = glob.glob(os.path.join(self.data_folder, pixel_glob))
                if verbosity >= 2:
                    print(f"Found {len(pixel_files)} .tpx3 pixel files.")

            pixel_dict = {get_key(f): f for f in pixel_files}

            with tempfile.TemporaryDirectory(dir="/tmp") as tmp_dir:
                with ProcessPoolExecutor(max_workers=self.n_threads) as executor:
                    futures = {executor.submit(self._convert_pixel_file, pfile, tmp_dir, verbosity): key
                              for key, pfile in pixel_dict.items()}
                    pixel_dfs = []
                    for future in tqdm(as_completed(futures), total=len(futures), desc="Loading pixels", disable=(verbosity == 0)):
                        result = future.result()
                        if result is not None:
                            pixel_dfs.append(result)

            if pixel_dfs:
                self.pixels_df = pd.concat(pixel_dfs, ignore_index=True).replace(" nan", float("nan"))

                # NOTE: Limit application is deferred until after all data is loaded
                # to enable cascading limits (see end of load() method)

                if verbosity >= 1:
                    print(f"Loaded {len(self.pixels_df)} pixels in total.")
            else:
                logger.error("No pixel data could be loaded. Check that ExportedPixels folder exists or empir binaries are available.")
                self.pixels_df = pd.DataFrame()

        # Apply cascading limits after all data is loaded
        self._apply_cascading_limits(limit=limit, relax=relax, verbosity=verbosity)

        # Update pair_dfs to reflect the filtered data for association
        if limit is not None or query is not None:
            if events and photons:
                self.pair_dfs = [(self.events_df, self.photons_df)]

    def _apply_cascading_limits(self, limit, relax=1.0, verbosity=0):
        """
        Apply cascading limits to loaded data based on data types and time windows.

        This method implements smart limit logic:
        - Integer limit: Row count for primary data, time-based cascading for downstream
        - Float limit: Max TOA for primary data, time-based cascading for downstream

        The cascading ensures downstream data extends beyond primary data's max TOA
        by the appropriate time search window to enable proper association.

        Args:
            limit (int, float, or None): Row limit (int) or max TOA in seconds (float)
            relax (float): Relaxation factor for time buffers (default 1.0)
            verbosity (int): Verbosity level
        """
        if limit is None:
            return

        # Get association time parameters from settings
        defaults = self._get_association_defaults()
        pixel_max_time_ns = defaults.get('pixel_max_time_ns', 500) * relax
        photon_max_time_ns = defaults.get('max_time_ns', 500) * relax

        # Convert to seconds for time comparisons
        pixel_time_buffer_s = pixel_max_time_ns / 1e9
        photon_time_buffer_s = photon_max_time_ns / 1e9

        # Determine if limit is row-based (int) or time-based (float)
        limit_is_time = isinstance(limit, float)

        # Determine what data we have
        has_pixels = self.pixels_df is not None and len(self.pixels_df) > 0
        has_photons = self.photons_df is not None and len(self.photons_df) > 0
        has_events = self.events_df is not None and len(self.events_df) > 0

        if verbosity >= 2:
            limit_type = "time (TOA)" if limit_is_time else "rows"
            print(f"\n📊 Applying cascading limits (limit={limit}, type={limit_type}, relax={relax})")

        # Case 1: Full 3-tier (pixels → photons → events)
        if has_pixels and has_photons and has_events:
            # Pixels are primary - apply limit directly
            if limit_is_time:
                original_len = len(self.pixels_df)
                self.pixels_df = self.pixels_df[self.pixels_df['t'] <= limit].copy()
                if verbosity >= 2:
                    print(f"   Pixels: {original_len:,} → {len(self.pixels_df):,} (t <= {limit:.6f}s)")
            else:
                original_len = len(self.pixels_df)
                self.pixels_df = self.pixels_df.head(int(limit)).copy()
                if verbosity >= 2:
                    print(f"   Pixels: {original_len:,} → {len(self.pixels_df):,} (first {int(limit)} rows)")

            # Cascade to photons: use max pixel TOA + buffer
            if len(self.pixels_df) > 0:
                max_pixel_toa = self.pixels_df['t'].max()
                photon_toa_limit = max_pixel_toa + pixel_time_buffer_s
                original_len = len(self.photons_df)
                self.photons_df = self.photons_df[self.photons_df['t'] <= photon_toa_limit].copy()
                if verbosity >= 2:
                    print(f"   Photons: {original_len:,} → {len(self.photons_df):,} (t <= {photon_toa_limit:.6f}s, max_pixel_toa + {pixel_time_buffer_s*1e9:.0f}ns)")

            # Cascade to events: use max photon TOA + buffer
            if len(self.photons_df) > 0:
                max_photon_toa = self.photons_df['t'].max()
                event_toa_limit = max_photon_toa + photon_time_buffer_s
                original_len = len(self.events_df)
                self.events_df = self.events_df[self.events_df['t'] <= event_toa_limit].copy()
                if verbosity >= 2:
                    print(f"   Events: {original_len:,} → {len(self.events_df):,} (t <= {event_toa_limit:.6f}s, max_photon_toa + {photon_time_buffer_s*1e9:.0f}ns)")

        # Case 2: Pixels → Photons only
        elif has_pixels and has_photons:
            # Pixels are primary
            if limit_is_time:
                original_len = len(self.pixels_df)
                self.pixels_df = self.pixels_df[self.pixels_df['t'] <= limit].copy()
                if verbosity >= 2:
                    print(f"   Pixels: {original_len:,} → {len(self.pixels_df):,} (t <= {limit:.6f}s)")
            else:
                original_len = len(self.pixels_df)
                self.pixels_df = self.pixels_df.head(int(limit)).copy()
                if verbosity >= 2:
                    print(f"   Pixels: {original_len:,} → {len(self.pixels_df):,} (first {int(limit)} rows)")

            # Cascade to photons
            if len(self.pixels_df) > 0:
                max_pixel_toa = self.pixels_df['t'].max()
                photon_toa_limit = max_pixel_toa + pixel_time_buffer_s
                original_len = len(self.photons_df)
                self.photons_df = self.photons_df[self.photons_df['t'] <= photon_toa_limit].copy()
                if verbosity >= 2:
                    print(f"   Photons: {original_len:,} → {len(self.photons_df):,} (t <= {photon_toa_limit:.6f}s)")

        # Case 3: Photons → Events only
        elif has_photons and has_events:
            # Photons are primary
            if limit_is_time:
                original_len = len(self.photons_df)
                self.photons_df = self.photons_df[self.photons_df['t'] <= limit].copy()
                if verbosity >= 2:
                    print(f"   Photons: {original_len:,} → {len(self.photons_df):,} (t <= {limit:.6f}s)")
            else:
                original_len = len(self.photons_df)
                self.photons_df = self.photons_df.head(int(limit)).copy()
                if verbosity >= 2:
                    print(f"   Photons: {original_len:,} → {len(self.photons_df):,} (first {int(limit)} rows)")

            # Cascade to events
            if len(self.photons_df) > 0:
                max_photon_toa = self.photons_df['t'].max()
                event_toa_limit = max_photon_toa + photon_time_buffer_s
                original_len = len(self.events_df)
                self.events_df = self.events_df[self.events_df['t'] <= event_toa_limit].copy()
                if verbosity >= 2:
                    print(f"   Events: {original_len:,} → {len(self.events_df):,} (t <= {event_toa_limit:.6f}s)")

        # Case 4: Single data type - apply limit directly
        else:
            if has_pixels:
                if limit_is_time:
                    self.pixels_df = self.pixels_df[self.pixels_df['t'] <= limit].copy()
                else:
                    self.pixels_df = self.pixels_df.head(int(limit)).copy()
            if has_photons:
                if limit_is_time:
                    self.photons_df = self.photons_df[self.photons_df['t'] <= limit].copy()
                else:
                    self.photons_df = self.photons_df.head(int(limit)).copy()
            if has_events:
                if limit_is_time:
                    self.events_df = self.events_df[self.events_df['t'] <= limit].copy()
                else:
                    self.events_df = self.events_df.head(int(limit)).copy()

    def _convert_event_file(self, eventfile, tmp_dir, verbosity=0):
        """
        Convert an event file to CSV and load it into a DataFrame.

        First checks for an already exported CSV file in the ExportedEvents subfolder.
        If found, uses that file directly. Otherwise, falls back to using the
        empir_export_events binary to convert the .empirevent file.

        Args:
            eventfile (str): Path to the event file.
            tmp_dir (str): Temporary directory for output CSV (used only if conversion is needed).
            verbosity (int): Verbosity level (0=silent, 1=warnings).

        Returns:
            pd.DataFrame: Loaded event DataFrame, or None on error.
        """
        # Get the base filename without extension
        basename = os.path.splitext(os.path.basename(eventfile))[0]

        # Check for already exported CSV in ExportedEvents folder
        # Try both with and without "exported_" prefix (lumacam adds this prefix)
        exported_csv = os.path.join(self.data_folder, "ExportedEvents", f"{basename}.csv")
        exported_csv_with_prefix = os.path.join(self.data_folder, "ExportedEvents", f"exported_{basename}.csv")

        if os.path.exists(exported_csv):
            # Use already exported CSV
            logger.debug(f"Using existing CSV: {exported_csv}")
            csv_file = exported_csv
        elif os.path.exists(exported_csv_with_prefix):
            # Use CSV with exported_ prefix
            logger.debug(f"Using existing CSV: {exported_csv_with_prefix}")
            csv_file = exported_csv_with_prefix
        else:
            # Fall back to empir binary conversion
            export_bin = os.path.join(self.export_dir, "empir_export_events")
            if not os.path.exists(export_bin):
                logger.error(f"empir_export_events binary not found at {export_bin} and no exported CSV found at {exported_csv}")
                if verbosity >= 2:
                    print(f"Warning: empir_export_events binary not found at {export_bin} and no exported CSV found at {exported_csv}")
                return None

            csv_file = os.path.join(tmp_dir, f"{uuid.uuid4().hex}.csv")
            logger.info(f"Converting {eventfile} using empir_export_events")
            os.system(f"{export_bin} {eventfile} {csv_file} csv")

        try:
            df = pd.read_csv(csv_file)

            if verbosity >= 2:
                print(f"Read CSV with shape {df.shape}, columns: {df.columns.tolist()}")

            # Handle two CSV formats:
            # 1. empir export format: has ' PSD value' column that needs to be filtered
            # 2. Pre-processed format: already has correct columns without ' PSD value'
            if ' PSD value' in df.columns:
                # empir export format - filter and rename
                # Use boolean indexing instead of query to handle backtick-quoted column names
                df = df[df[' PSD value'] >= 0]
                df.columns = ["x", "y", "t", "n", "PSD", "tof"]
                if verbosity >= 2:
                    print(f"After filtering empir format: shape {df.shape}")
            elif df.columns.tolist() == ["x", "y", "t", "n", "PSD", "tof"]:
                # Already in correct format - just filter by PSD >= 0
                initial_len = len(df)
                df = df[df['PSD'] >= 0]
                if verbosity >= 2:
                    print(f"After PSD filter: {initial_len} -> {len(df)} rows")
            else:
                if verbosity >= 1:
                    print(f"Warning: Unexpected event CSV format with columns: {df.columns.tolist()}")
                return None

            df["tof"] = df["tof"].astype(float)
            df["PSD"] = df["PSD"].astype(float)

            if len(df) == 0:
                if verbosity >= 1:
                    print(f"Warning: Event DataFrame is empty after processing {os.path.basename(csv_file)}")

            return df
        except Exception as e:
            logger.error(f"Error processing {csv_file}: {e}")
            if verbosity >= 2:
                print(f"Error processing {csv_file}: {e}")
            return None

    def _convert_photon_file(self, photonfile, tmp_dir, verbosity=0):
        """
        Convert a photon file to CSV and load it into a DataFrame.

        First checks for an already exported CSV file in the ExportedPhotons subfolder.
        If found, uses that file directly. Otherwise, falls back to using the
        empir_export_photons binary to convert the .empirphot file.

        Args:
            photonfile (str): Path to the photon file.
            tmp_dir (str): Temporary directory for output CSV (used only if conversion is needed).
            verbosity (int): Verbosity level (0=silent, 1=warnings).

        Returns:
            pd.DataFrame: Loaded photon DataFrame, or None on error.
        """
        # Get the base filename without extension
        basename = os.path.splitext(os.path.basename(photonfile))[0]

        # Check for already exported CSV in ExportedPhotons folder
        # Try both with and without "exported_" prefix (lumacam adds this prefix)
        exported_csv = os.path.join(self.data_folder, "ExportedPhotons", f"{basename}.csv")
        exported_csv_with_prefix = os.path.join(self.data_folder, "ExportedPhotons", f"exported_{basename}.csv")

        if os.path.exists(exported_csv):
            # Use already exported CSV
            logger.debug(f"Using existing CSV: {exported_csv}")
            csv_file = exported_csv
        elif os.path.exists(exported_csv_with_prefix):
            # Use CSV with exported_ prefix
            logger.debug(f"Using existing CSV: {exported_csv_with_prefix}")
            csv_file = exported_csv_with_prefix
        else:
            # Fall back to empir binary conversion
            export_bin = os.path.join(self.export_dir, "empir_export_photons")
            if not os.path.exists(export_bin):
                logger.error(f"empir_export_photons binary not found at {export_bin} and no exported CSV found at {exported_csv}")
                if verbosity >= 2:
                    print(f"Warning: empir_export_photons binary not found at {export_bin} and no exported CSV found at {exported_csv}")
                return None

            csv_file = os.path.join(tmp_dir, f"{uuid.uuid4().hex}.csv")
            logger.info(f"Converting {photonfile} using empir_export_photons")
            os.system(f"{export_bin} {photonfile} {csv_file} csv")

        try:
            df = pd.read_csv(csv_file)

            if verbosity >= 2:
                print(f"Read photon CSV with shape {df.shape}, columns: {df.columns.tolist()}")

            # Handle different CSV formats
            expected_cols = ["x", "y", "t", "tof"]
            if df.columns.tolist() == ["x", "y", "toa", "tof"]:
                # Lumacam format uses 'toa' instead of 't'
                df.columns = expected_cols
            elif len(df.columns) == 4:
                # Assume it's in the correct order, just rename
                df.columns = expected_cols
            else:
                logger.error(f"Unexpected photon CSV format. Expected 4 columns, got {len(df.columns)}: {df.columns.tolist()}")
                if verbosity >= 2:
                    print(f"Warning: Unexpected photon CSV format with columns: {df.columns.tolist()}")
                return None

            df["x"] = df["x"].astype(float)
            df["y"] = df["y"].astype(float)
            df["t"] = df["t"].astype(float)
            df["tof"] = pd.to_numeric(df["tof"], errors="coerce")

            if len(df) == 0:
                if verbosity >= 1:
                    print(f"Warning: Photon DataFrame is empty after processing {os.path.basename(csv_file)}")

            return df
        except Exception as e:
            logger.error(f"Error processing photon file {csv_file}: {e}")
            if verbosity >= 2:
                print(f"Error processing photon file {csv_file}: {e}")
                import traceback
                traceback.print_exc()
            return None

    def _convert_pixel_file(self, pixelfile, tmp_dir, verbosity=0):
        """
        Convert a pixel file to CSV and load it into a DataFrame.

        First checks for an already exported CSV file in the ExportedPixels subfolder.
        If found, uses that file directly. Otherwise, falls back to using the
        empir_pixel2photon binary to convert the .tpx3 file.

        Args:
            pixelfile (str): Path to the TPX3 pixel file.
            tmp_dir (str): Temporary directory for output CSV (used only if conversion is needed).
            verbosity (int): Verbosity level (0=silent, 1=warnings).

        Returns:
            pd.DataFrame: Loaded pixel DataFrame, or None on error.
        """
        # Get the base filename without extension
        basename = os.path.splitext(os.path.basename(pixelfile))[0]

        # Check for already exported CSV in ExportedPixels folder
        # Try both with and without "exported_" prefix (lumacam adds this prefix)
        exported_csv = os.path.join(self.data_folder, "ExportedPixels", f"{basename}.csv")
        exported_csv_with_prefix = os.path.join(self.data_folder, "ExportedPixels", f"exported_{basename}.csv")

        if os.path.exists(exported_csv):
            # Use already exported CSV
            logger.debug(f"Using existing CSV: {exported_csv}")
            csv_file = exported_csv
        elif os.path.exists(exported_csv_with_prefix):
            # Use CSV with exported_ prefix
            logger.debug(f"Using existing CSV: {exported_csv_with_prefix}")
            csv_file = exported_csv_with_prefix
        else:
            # Fall back to empir binary conversion
            export_bin = os.path.join(self.export_dir, "empir_pixel2photon")
            if not os.path.exists(export_bin):
                if verbosity >= 1:
                    print(f"Warning: empir_pixel2photon binary not found at {export_bin} and no exported CSV found at {exported_csv}")
                return None

            csv_file = os.path.join(tmp_dir, f"{uuid.uuid4().hex}.csv")
            logger.info(f"Converting {pixelfile} using empir_pixel2photon")
            # Note: empir_pixel2photon might need additional parameters
            os.system(f"{export_bin} {pixelfile} {csv_file}")

        try:
            df = pd.read_csv(csv_file)

            if verbosity >= 2:
                print(f"Read pixel CSV with shape {df.shape}, columns: {df.columns.tolist()}")

            # Standardize column names
            # Expected format: "x [px], y [px], t [s], tot [a.u.], t_relToExtTrigger [s]"
            # or simplified: "x, y, t, tot, tof"

            # Strip whitespace and remove units from column names
            df.columns = [col.strip().split('[')[0].strip() for col in df.columns]

            # Rename t_relToExtTrigger to tof for consistency
            if 't_relToExtTrigger' in df.columns:
                df.rename(columns={'t_relToExtTrigger': 'tof'}, inplace=True)

            # Ensure we have the expected columns
            expected_cols = ['x', 'y', 't', 'tot', 'tof']
            if not all(col in df.columns for col in expected_cols):
                if verbosity >= 1:
                    print(f"Warning: Unexpected pixel CSV format. Expected {expected_cols}, got {df.columns.tolist()}")
                return None

            # Select and reorder columns
            df = df[expected_cols]

            # Convert data types
            df["x"] = df["x"].astype(float)
            df["y"] = df["y"].astype(float)
            df["t"] = df["t"].astype(float)
            df["tot"] = pd.to_numeric(df["tot"], errors="coerce")
            df["tof"] = pd.to_numeric(df["tof"], errors="coerce")

            if len(df) == 0:
                if verbosity >= 1:
                    print(f"Warning: Pixel DataFrame is empty after processing {os.path.basename(csv_file)}")

            return df
        except Exception as e:
            if verbosity >= 1:
                print(f"Error processing pixel file {csv_file}: {e}")
            import traceback
            if verbosity >= 2:
                traceback.print_exc()
            return None

    def _associate_pair(self, pair, time_norm_ns, spatial_norm_px, dSpace_px, weight_px_in_s, max_time_s, verbosity, method):
        """
        Associate photons to events for a single pair of event and photon DataFrames using the specified method.

        Args:
            pair (tuple): Tuple of (event_df, photon_df).
            time_norm_ns (float): Time normalization factor (ns) for 'kdtree' and 'window' methods.
            spatial_norm_px (float): Spatial normalization factor (px) for 'kdtree' and 'window' methods.
            dSpace_px (float): Max allowed center-of-mass distance for multiphoton matches.
            weight_px_in_s (float, optional): Weight for pixel-to-second conversion ('lumacam').
            max_time_s (float, optional): Max time duration in seconds ('lumacam', 'window', 'simple').
            verbosity (int): 0=silent, 1=summary, 2=debug.
            method (str): Association method: 'lumacam', 'kdtree', 'window', 'simple'.

        Returns:
            pandas.DataFrame: Associated photon DataFrame, or None if processing fails.
        """
        edf, pdf = pair
        # Validate input DataFrames
        required_event_cols = ['x', 'y', 't', 'n', 'PSD']
        required_photon_cols = ['x', 'y', 't']
        if not all(col in edf.columns for col in required_event_cols):
            logger.error(f"Event DataFrame missing required columns: {required_event_cols}")
            return None
        if not all(col in pdf.columns for col in required_photon_cols):
            logger.error(f"Photon DataFrame missing required columns: {required_photon_cols}")
            return None
        # Check for excessive NaNs
        if edf[required_event_cols].isna().any().any() or pdf[required_photon_cols].isna().any().any():
            logger.warning(f"NaN values detected in event or photon DataFrame for pair")
        if verbosity >= 2:
            logger.info(f"Starting association for pair with {len(edf)} events and {len(pdf)} photons using method '{method}'")
        try:
            if method == 'lumacam':
                result = self._associate_photons_to_events(pdf, edf, weight_px_in_s, max_time_s, verbosity)
            elif method == 'kdtree':
                result = self._associate_photons_to_events_kdtree(pdf, edf, time_norm_ns, spatial_norm_px, dSpace_px, verbosity)
            elif method == 'window':
                result = self._associate_photons_to_events_window(pdf, edf, time_norm_ns, spatial_norm_px, dSpace_px, max_time_s, verbosity)
            elif method == 'simple':
                result = self._associate_photons_to_events_simple_window(pdf, edf, dSpace_px, max_time_s, verbosity)
            else:
                raise ValueError(f"Unknown association method: {method}")
            # Log number of matched photons
            event_col = 'assoc_cluster_id' if method == 'lumacam' else 'assoc_event_id'
            matched = result[event_col].notna().sum()
            if verbosity>1:
                logger.info(f"Finished association for pair with {len(edf)} events, {matched} photons matched")
            return result
        except Exception as e:
            logger.error(f"Error associating pair with {len(edf)} events using method '{method}': {e}")
            return None

    def associate_photons_events(self, time_norm_ns=1.0, spatial_norm_px=1.0, dSpace_px=np.inf, weight_px_in_s=None, max_time_ns=500, verbosity=1, method='auto'):
        """
        Associate photons to events in parallel per file pair using the specified method.

        This method processes each pair of loaded event and photon DataFrames independently,
        associates photons to events based on the chosen method, and concatenates the results.
        For time-window methods ('window', 'simple'), data is assumed to be almost sorted by time for efficiency.
        The 'simple' method uses a forward time window [event_t, event_t + max_time_s] and selects the spatially closest
        photons, assigning only if the center-of-mass (or single photon) distance is within dSpace_px.

        Args:
            time_norm_ns (float): Time normalization factor in nanoseconds for 'kdtree' and 'window' methods.
            spatial_norm_px (float): Spatial normalization factor in pixels for 'kdtree' and 'window' methods.
            dSpace_px (float): Maximum allowed center-of-mass (or single photon) distance in pixels.
            weight_px_in_s (float, optional): Weight for converting pixels to seconds ('lumacam' method).
            max_time_ns (float, optional): Maximum time duration in nanoseconds (default: 500). Converted to seconds internally.
                                          For 'lumacam' and 'window', if None, computed as 3 * std of times.
                                          For 'simple', default 500 ns if not provided.
            verbosity (int): Verbosity level: 0=silent, 1=summary, 2=debug.
            method (str): Association method:
                - 'auto': Uses 'lumacam' if enabled and available, else 'kdtree'.
                - 'lumacam': Uses lumacamTesting library (requires installation).
                - 'kdtree': Full KDTree-based association.
                - 'window': Symmetric time-window KDTree.
                - 'simple': Forward time-window with spatial closest selection and CoG check (fast for small windows).
                - 'mystic': Constrained optimization using mystic framework (requires mystic package).
        """
        if self.pair_dfs is None:
            raise ValueError("Load data first using load().")

        if method == 'auto':
            effective_method = 'lumacam' if self.use_lumacam else 'kdtree'
        else:
            effective_method = method

        if effective_method == 'lumacam' and not LUMACAM_AVAILABLE:
            raise ImportError("lumacamTesting is required for 'lumacam' method.")

        max_time_s = max_time_ns / 1e9 if max_time_ns is not None else None

        self.assoc_method = effective_method
        event_col = 'assoc_cluster_id' if effective_method == 'lumacam' else 'assoc_event_id'

        # Special handling for mystic method - process all data together
        if effective_method == 'mystic':
            # Concatenate all photons and events from pairs
            all_photons = pd.concat([pair[1] for pair in self.pair_dfs], ignore_index=True)
            all_events = pd.concat([pair[0] for pair in self.pair_dfs], ignore_index=True)

            self.associated_df = self._associate_photons_to_events_mystic(
                all_photons, all_events,
                max_dist_px=dSpace_px,
                max_time_ns=max_time_ns,
                verbosity=verbosity
            )
            return self.associated_df

        with ProcessPoolExecutor(max_workers=self.n_threads) as executor:
            futures = [
                executor.submit(
                    self._associate_pair, pair, time_norm_ns, spatial_norm_px, dSpace_px,
                    weight_px_in_s, max_time_s, verbosity, effective_method
                ) for pair in self.pair_dfs
            ]
            associated_list = []
            for future in tqdm(as_completed(futures), total=len(futures), desc="Associating photons to events", disable=(verbosity == 0)):
                result = future.result()
                if result is not None:
                    associated_list.append(result)
        
        if associated_list:
            # Concatenate with ignore_index to avoid index duplication
            self.associated_df = pd.concat(associated_list, ignore_index=True)
            logger.info(f"Before grouping: {self.associated_df['assoc_x'].notna().sum()} photons with non-NaN assoc_x")
            mask = self.associated_df['assoc_x'].notna() & self.associated_df['assoc_y'].notna() & \
                   self.associated_df['assoc_t'].notna() & self.associated_df['assoc_n'].notna() & \
                   self.associated_df['assoc_PSD'].notna()
            if mask.any():
                grouped = self.associated_df.loc[mask].groupby(['assoc_x', 'assoc_y', 'assoc_t', 'assoc_n', 'assoc_PSD'])
                new_ids = grouped.ngroup() + 1
                self.associated_df.loc[mask, event_col] = new_ids
                logger.info(f"After grouping: {self.associated_df[event_col].notna().sum()} photons with non-NaN {event_col}")
            else:
                logger.warning("No photons with all non-NaN assoc columns for grouping")
            # Compute and store statistics for this association
            total_photons = len(self.associated_df)
            matched_photons = self.associated_df[event_col].notna().sum()

            # Count unique events
            if event_col in self.associated_df.columns:
                matched_events = int(self.associated_df[event_col].nunique())
            else:
                matched_events = 0

            # Estimate total events (we don't have unmatched events data in this context)
            total_events = matched_events

            # Compute quality statistics if we have the necessary data
            quality_stats = {'exact_n': 0, 'n_mismatch': 0, 'exact_com': 0, 'good_com': 0,
                           'acceptable_com': 0, 'poor_com': 0}

            # Check if we can compute quality stats
            event_rows = self.associated_df[self.associated_df[event_col].notna()]

            # Check n matching if available
            if 'assoc_n' in event_rows.columns and 'n' in event_rows.columns:
                event_rows_with_n = event_rows[(event_rows['assoc_n'].notna()) & (event_rows['n'].notna())]
                if len(event_rows_with_n) > 0:
                    quality_stats['exact_n'] = int((event_rows_with_n['assoc_n'] == event_rows_with_n['n']).sum())
                    quality_stats['n_mismatch'] = int((event_rows_with_n['assoc_n'] != event_rows_with_n['n']).sum())

            # Compute CoM quality if we have photon and event coordinates
            # Group by event to compute center-of-mass for each event's photons
            if len(event_rows) > 0 and all(col in event_rows.columns for col in ['x', 'y', 'assoc_x', 'assoc_y']):
                # For each event, compute the center of mass of its photons
                for event_id in event_rows[event_col].unique():
                    if pd.isna(event_id):
                        continue

                    event_photons = event_rows[event_rows[event_col] == event_id]
                    if len(event_photons) == 0:
                        continue

                    # Compute center of mass of photons
                    photon_com_x = event_photons['x'].mean()
                    photon_com_y = event_photons['y'].mean()

                    # Get event position (should be same for all rows of this event)
                    event_x = event_photons['assoc_x'].iloc[0]
                    event_y = event_photons['assoc_y'].iloc[0]

                    # Compute CoM distance
                    com_dist = np.sqrt((photon_com_x - event_x)**2 + (photon_com_y - event_y)**2)

                    # Categorize quality (use dSpace_px parameter as search radius)
                    search_radius = dSpace_px if dSpace_px != np.inf else 50.0

                    if com_dist <= 0.1:  # Within 0.1 pixel
                        quality_stats['exact_com'] += 1
                    elif com_dist <= search_radius * 0.3:  # Within 30% of search radius
                        quality_stats['good_com'] += 1
                    elif com_dist <= search_radius * 0.5:  # Within 50% of search radius
                        quality_stats['acceptable_com'] += 1
                    else:  # Within search radius but >50%
                        quality_stats['poor_com'] += 1

            # Store statistics
            self.last_photon_event_stats = {
                'matched_photons': int(matched_photons),
                'total_photons': int(total_photons),
                'matched_events': int(matched_events),
                'total_events': int(total_events),
                'quality': quality_stats
            }

            if verbosity >= 1:
                print(f"✅ Matched {matched_photons} of {total_photons} photons ({100 * matched_photons / total_photons:.1f}%)")
        else:
            self.associated_df = pd.DataFrame()
            logger.warning("No valid association results to concatenate")

    def associate(self, pixel_max_dist_px=None, pixel_max_time_ns=None,
                  photon_time_norm_ns=1.0, photon_spatial_norm_px=1.0, photon_dSpace_px=None,
                  max_time_ns=None, verbosity=None, method='simple', relax=1.0):
        """
        Perform full three-tier association: pixels → photons → events.

        This method automatically handles both single folders and groupby structures:
        - For single folders: Performs standard association and returns a DataFrame
        - For groupby folders: Processes all groups in parallel and returns a dict of DataFrames

        For single folders, the result is stored in self.associated_df as a pixel-centric,
        photon-centric, or event-centric dataframe depending on what data was loaded.

        Args:
            pixel_max_dist_px (float, optional): Maximum spatial distance in pixels for pixel-photon association.
                                                 If None, uses value from settings or defaults to 5.0.
            pixel_max_time_ns (float, optional): Maximum time difference in nanoseconds for pixel-photon association.
                                                 If None, uses value from settings or defaults to 500.
            photon_time_norm_ns (float): Time normalization for photon-event association.
            photon_spatial_norm_px (float): Spatial normalization for photon-event association.
            photon_dSpace_px (float, optional): Maximum center-of-mass distance for photon-event association.
                                                If None, uses value from settings or defaults to 50.0.
            max_time_ns (float, optional): Maximum time window in nanoseconds (used for both associations if method supports it).
                                           If None, uses value from settings or defaults to 500.
            verbosity (int, optional): Verbosity level (0=silent, 1=progress bars only, 2=detailed output).
                                      If None, uses instance verbosity level set in __init__.
            method (str): Association method for both pixel-photon and photon-event stages.
                         Options: 'simple', 'kdtree', 'mystic'.
                         - 'simple': Fast forward time-window with spatial closest selection (default).
                         - 'kdtree': KDTree-based association with iterative CoM refinement.
                         - 'mystic': Constrained optimization using mystic framework (requires mystic package).
                                    Formulates association as optimization problem minimizing CoG distance
                                    subject to time constraints.
            relax (float): Scaling factor for association parameters. Default is 1.0 (no scaling).
                          Values > 1.0 relax the parameters (e.g., 1.5 = 50% more relaxed),
                          values < 1.0 make them more restrictive (e.g., 0.8 = 20% more restrictive).
                          Applied to both pixel-photon and photon-event association parameters.

        Returns:
            pd.DataFrame or dict: For single folders, returns the associated DataFrame.
                                 For groupby folders, returns dict mapping group names to DataFrames.
        """
        # If this is a groupby folder, delegate to associate_groupby
        if self.is_groupby:
            return self.associate_groupby(
                pixel_max_dist_px=pixel_max_dist_px,
                pixel_max_time_ns=pixel_max_time_ns,
                photon_time_norm_ns=photon_time_norm_ns,
                photon_spatial_norm_px=photon_spatial_norm_px,
                photon_dSpace_px=photon_dSpace_px,
                max_time_ns=max_time_ns,
                verbosity=verbosity,
                method=method,
                relax=relax
            )

        # Use instance verbosity if not specified
        if verbosity is None:
            verbosity = self.verbosity

        # Get defaults from settings
        defaults = self._get_association_defaults()

        # Apply defaults if parameters are None
        if pixel_max_dist_px is None:
            pixel_max_dist_px = defaults.get('pixel_max_dist_px', 5.0)
        if pixel_max_time_ns is None:
            pixel_max_time_ns = defaults.get('pixel_max_time_ns', 500)
        if photon_dSpace_px is None:
            photon_dSpace_px = defaults.get('photon_dSpace_px', 50.0)
        if max_time_ns is None:
            max_time_ns = defaults.get('max_time_ns', 500)

        # Apply relax scaling factor to all parameters
        pixel_max_dist_px *= relax
        pixel_max_time_ns *= relax
        photon_dSpace_px *= relax
        max_time_ns *= relax

        if verbosity >= 2:
            print("\n" + "="*70)
            print("Starting Full Multi-Tier Association")
            print("="*70)
            if self.settings:
                print("Using parameters from settings file" + (" (overridden where specified)" if any([
                    pixel_max_dist_px != defaults.get('pixel_max_dist_px') * relax,
                    pixel_max_time_ns != defaults.get('pixel_max_time_ns') * relax,
                    photon_dSpace_px != defaults.get('photon_dSpace_px') * relax,
                    max_time_ns != defaults.get('max_time_ns') * relax
                ]) else ""))
            if relax != 1.0:
                print(f"Relaxation factor applied: {relax}x")
                print(f"  Pixel-photon: max_dist={pixel_max_dist_px:.2f}px, max_time={pixel_max_time_ns:.1f}ns")
                print(f"  Photon-event: dSpace={photon_dSpace_px:.2f}px, max_time={max_time_ns:.1f}ns")

        # Determine what data we have
        has_pixels = self.pixels_df is not None and len(self.pixels_df) > 0
        has_photons = self.photons_df is not None and len(self.photons_df) > 0
        has_events = self.events_df is not None and len(self.events_df) > 0

        if verbosity >= 2:
            print(f"Data available: Pixels={has_pixels}, Photons={has_photons}, Events={has_events}")

        # Case 1: Pixels → Photons → Events (full 3-tier)
        if has_pixels and has_photons and has_events:
            if verbosity >= 2:
                print("\nPerforming 3-tier association: Pixels → Photons → Events")

            # Step 1: Associate pixels to photons
            if verbosity >= 2:
                print(f"\nStep 1/2: Associating pixels to photons (method={method})...")

            # Choose pixel-photon association method
            if method == 'kdtree':
                pixels_associated = self._associate_pixels_to_photons_kdtree(
                    self.pixels_df, self.photons_df,
                    max_dist_px=pixel_max_dist_px,
                    max_time_ns=pixel_max_time_ns,
                    verbosity=verbosity
                )
            elif method == 'mystic':
                pixels_associated = self._associate_pixels_to_photons_mystic(
                    self.pixels_df, self.photons_df,
                    max_dist_px=pixel_max_dist_px,
                    max_time_ns=pixel_max_time_ns,
                    verbosity=verbosity
                )
            else:  # default to 'simple'
                pixels_associated = self._associate_pixels_to_photons_simple(
                    self.pixels_df, self.photons_df,
                    max_dist_px=pixel_max_dist_px,
                    max_time_ns=pixel_max_time_ns,
                    verbosity=verbosity
                )

            # Step 2: Associate photons to events
            if verbosity >= 2:
                print("\nStep 2/2: Associating photons to events...")
            self.associate_photons_events(
                time_norm_ns=photon_time_norm_ns,
                spatial_norm_px=photon_spatial_norm_px,
                dSpace_px=photon_dSpace_px,
                max_time_ns=max_time_ns,
                verbosity=verbosity,
                method=method
            )

            # Step 3: Merge event information into pixel dataframe
            if verbosity >= 2:
                print("\nStep 3/3: Merging pixel-photon-event associations...")

            # Merge: Add event association info to pixels
            photons_with_events = self.associated_df.copy()
            event_col = 'assoc_cluster_id' if self.assoc_method == 'lumacam' else 'assoc_event_id'

            # Prepare photons dataframe with event info for merging
            # Round coordinates to avoid floating point precision issues
            photons_with_events['_merge_x'] = photons_with_events['x'].round(6)
            photons_with_events['_merge_y'] = photons_with_events['y'].round(6)
            photons_with_events['_merge_t'] = photons_with_events['t'].round(12)

            # Prepare pixels dataframe
            pixels_full = pixels_associated.copy()
            pixels_full['_merge_x'] = pixels_full['assoc_phot_x'].round(6)
            pixels_full['_merge_y'] = pixels_full['assoc_phot_y'].round(6)
            pixels_full['_merge_t'] = pixels_full['assoc_phot_t'].round(12)

            # Select only needed columns from photons to merge
            # Include assoc_com_dist for ev/cog (photon-to-event CoG distance)
            merge_cols = ['_merge_x', '_merge_y', '_merge_t',
                         event_col, 'assoc_x', 'assoc_y', 'assoc_t',
                         'assoc_n', 'assoc_PSD']
            # Add assoc_com_dist if present
            if 'assoc_com_dist' in photons_with_events.columns:
                merge_cols.append('assoc_com_dist')
            photon_event_cols = photons_with_events[merge_cols].copy()

            # Rename event columns to avoid conflicts
            photon_event_cols = photon_event_cols.rename(columns={event_col: 'assoc_event_id'})

            # Merge pixels with event info via photon coordinates
            pixels_full = pixels_full.merge(
                photon_event_cols,
                on=['_merge_x', '_merge_y', '_merge_t'],
                how='left',
                suffixes=('', '_event')
            )

            # Clean up temporary merge columns
            pixels_full = pixels_full.drop(columns=['_merge_x', '_merge_y', '_merge_t'])

            # Standardize column names with prefixes
            pixels_full = self._standardize_column_names(pixels_full, verbosity=verbosity)

            self.associated_df = pixels_full

            if verbosity >= 2:
                n_pixels_with_events = pixels_full['assoc_event_id'].notna().sum() if 'assoc_event_id' in pixels_full.columns else 0
                if n_pixels_with_events > 0:
                    print(f"✅ {n_pixels_with_events} pixels associated through full chain to events")

        # Case 2: Pixels → Photons only
        elif has_pixels and has_photons:
            if verbosity >= 2:
                print(f"\nPerforming 2-tier association: Pixels → Photons (method={method})")

            # Choose pixel-photon association method
            if method == 'kdtree':
                pixels_associated = self._associate_pixels_to_photons_kdtree(
                    self.pixels_df, self.photons_df,
                    max_dist_px=pixel_max_dist_px,
                    max_time_ns=pixel_max_time_ns,
                    verbosity=verbosity
                )
            elif method == 'mystic':
                pixels_associated = self._associate_pixels_to_photons_mystic(
                    self.pixels_df, self.photons_df,
                    max_dist_px=pixel_max_dist_px,
                    max_time_ns=pixel_max_time_ns,
                    verbosity=verbosity
                )
            else:  # default to 'simple'
                pixels_associated = self._associate_pixels_to_photons_simple(
                    self.pixels_df, self.photons_df,
                    max_dist_px=pixel_max_dist_px,
                    max_time_ns=pixel_max_time_ns,
                    verbosity=verbosity
                )
            # Standardize column names
            self.associated_df = self._standardize_column_names(pixels_associated, verbosity=verbosity)

        # Case 3: Photons → Events only (standard association)
        elif has_photons and has_events:
            if verbosity >= 2:
                print("\nPerforming standard Photons → Events association")
            self.associate_photons_events(
                time_norm_ns=photon_time_norm_ns,
                spatial_norm_px=photon_spatial_norm_px,
                dSpace_px=photon_dSpace_px,
                max_time_ns=max_time_ns,
                verbosity=verbosity,
                method=method
            )

        else:
            if verbosity >= 2:
                print("\nInsufficient data for association. Need at least two data types loaded.")
            self.associated_df = pd.DataFrame()

        if verbosity >= 2:
            print("\n" + "="*70)
            print("Full Association Complete")
            print("="*70)
            if len(self.associated_df) > 0:
                print(f"Final combined dataframe has {len(self.associated_df)} rows")
                print(f"Columns: {self.associated_df.columns.tolist()}")

        # Auto-save results
        if len(self.associated_df) > 0:
            try:
                output_path = self.save_associations(verbosity=verbosity)
                if verbosity >= 1:
                    print(f"💾 Auto-saved results to: {output_path}")
            except Exception as e:
                if verbosity >= 2:
                    print(f"⚠️  Warning: Could not auto-save results: {e}")

        # Return HTML stats table for display
        from IPython.display import HTML
        return HTML(self._create_stats_html_table())

    def get_association_stats(self):
        """
        Get association statistics as a dictionary.

        Returns:
            dict: Association statistics including match rates and quality metrics.
        """
        if self.is_groupby and hasattr(self, 'groupby_stats') and self.groupby_stats:
            # Return stats for all groups (stored during association)
            return self.groupby_stats
        elif self.last_assoc_stats or self.last_photon_event_stats:
            # Return stats from last association
            stats = {}
            if self.last_assoc_stats:
                stats.update(self.last_assoc_stats)
            if self.last_photon_event_stats:
                stats['photon_event'] = self.last_photon_event_stats
            return stats
        else:
            return {}

    def _compute_stats_for_df(self, df):
        """Compute statistics for a single dataframe."""
        stats = {}

        # Pixel-photon stats
        if 'assoc_photon_id' in df.columns:
            total_pixels = len(df)
            matched_pixels = df['assoc_photon_id'].notna().sum()
            stats['pixels_total'] = total_pixels
            stats['pixels_matched'] = matched_pixels
            stats['pixel_photon_rate'] = matched_pixels / total_pixels if total_pixels > 0 else 0

        # Photon-event stats
        if 'assoc_event_id' in df.columns:
            matched_to_events = df['assoc_event_id'].notna().sum()
            stats['photons_to_events'] = matched_to_events
            stats['photon_event_rate'] = matched_to_events / len(df) if len(df) > 0 else 0
        elif 'assoc_cluster_id' in df.columns:
            matched_to_events = df['assoc_cluster_id'].notna().sum()
            stats['photons_to_events'] = matched_to_events
            stats['photon_event_rate'] = matched_to_events / len(df) if len(df) > 0 else 0

        return stats

    def _repr_html_(self):
        """
        Generate HTML representation for Jupyter notebooks as a metrics table.
        Metrics are columns, groups/datasets are rows.

        Returns:
            str: HTML string with association statistics table.
        """
        return self._create_stats_html_table()

    def _compute_stats_from_dataframe(self, df):
        """Compute association statistics from an already-associated DataFrame.

        This is used when loading pre-existing association results from CSV files.

        Args:
            df: Associated DataFrame with columns like 'ev/n', 'ph/n', etc. (forward slash notation)

        Returns:
            dict: Statistics dictionary with 'pixel_photon' and 'photon_event' keys
        """
        stats = {}

        # Check what columns are available (handle forward slash notation from CSV)
        # Pixels can be represented as individual pixel columns (px/x, px/y, etc.)
        has_pixels = any(col.startswith('px/') for col in df.columns)
        has_photons = 'ph/n' in df.columns
        has_events = 'ev/n' in df.columns

        # Note: Pixel-photon stats cannot be accurately reconstructed from association CSV
        # because the CSV doesn't contain info about unmatched pixels or the original pixel count.
        # These stats should be loaded from a separate stats file or computed during association.

        # Compute photon-event stats if event data is present
        if has_photons and has_events:
            # Count photons matched to events (rows with ev/id not null)
            event_rows = df[df['ev/id'].notna()]
            matched_photons = len(event_rows)
            total_photons = len(df)

            # Count events (unique event IDs)
            if 'ev/id' in df.columns:
                matched_events = int(df['ev/id'].nunique())
                # Total events = matched events (we don't know unmatched events from CSV)
                total_events = matched_events
            else:
                matched_events = len(event_rows)
                total_events = matched_events

            # Compute quality stats if available
            quality_stats = {'exact_n': 0, 'n_mismatch': 0, 'exact_com': 0, 'good_com': 0,
                           'acceptable_com': 0, 'poor_com': 0}

            # Check n matching (compare ev/n with ph/n)
            if 'ev/n' in df.columns and 'ph/n' in df.columns:
                event_rows_with_n = event_rows[(event_rows['ev/n'].notna()) & (event_rows['ph/n'].notna())]
                if len(event_rows_with_n) > 0:
                    quality_stats['exact_n'] = int((event_rows_with_n['ev/n'] == event_rows_with_n['ph/n']).sum())
                    quality_stats['n_mismatch'] = int((event_rows_with_n['ev/n'] != event_rows_with_n['ph/n']).sum())

            # Check CoM quality - look for CoM distance columns
            com_col = None
            for col in ['ev/cog', 'com_dist_ph2ev', 'com_dist']:
                if col in df.columns:
                    com_col = col
                    break

            if com_col:
                com_data = event_rows[com_col].dropna()
                if len(com_data) > 0:
                    search_radius = 50.0  # Default assumption for ph2ev

                    quality_stats['exact_com'] = int((com_data <= 0.1).sum())
                    quality_stats['good_com'] = int(((com_data > 0.1) & (com_data <= search_radius * 0.3)).sum())
                    quality_stats['acceptable_com'] = int(((com_data > search_radius * 0.3) & (com_data <= search_radius * 0.5)).sum())
                    quality_stats['poor_com'] = int((com_data > search_radius * 0.5).sum())

            stats['photon_event'] = {
                'matched_photons': int(matched_photons),
                'total_photons': int(total_photons),
                'matched_events': int(matched_events),
                'total_events': int(total_events),
                'quality': quality_stats
            }

        return stats

    def compute_stats_from_csv(self, verbosity=1):
        """
        Compute association statistics from existing CSV files in AssociatedResults folders.

        This is useful for archive folders that only have AssociatedResults/associated_data.csv
        without the original data files. The method:
        1. Loads CSV files from AssociatedResults folders
        2. Computes statistics from the loaded data
        3. Saves statistics as JSON files
        4. Returns HTML table with the statistics

        Works for both single folders and groupby folder structures.

        Args:
            verbosity (int): Verbosity level (0=silent, 1=info, 2=debug)

        Returns:
            IPython.display.HTML: HTML table with association statistics

        Raises:
            ValueError: If no AssociatedResults CSV files are found

        Example:
            # For single folder
            assoc = nea.Analyse("path/to/archive_folder", events=False, photons=False, pixels=False)
            assoc.compute_stats_from_csv()

            # For groupby folder
            assoc = nea.Analyse("path/to/archive_groupby", events=False, photons=False, pixels=False)
            assoc.compute_stats_from_csv()
        """
        import json
        from IPython.display import HTML

        if self.is_groupby:
            # Process grouped folders
            if not self.groupby_results:
                # Try to load CSVs if not already loaded
                loaded_groups = []
                self.groupby_stats = {}

                for subdir in self.groupby_subdirs:
                    group_path = os.path.join(self.data_folder, subdir)
                    assoc_file = os.path.join(group_path, "AssociatedResults", "associated_data.csv")
                    stats_file = os.path.join(group_path, "AssociatedResults", "association_stats.json")

                    if os.path.exists(assoc_file):
                        try:
                            if verbosity >= 2:
                                print(f"Loading {subdir}...")

                            # Load CSV
                            group_df = pd.read_csv(assoc_file)
                            self.groupby_results[subdir] = group_df

                            # Compute stats from CSV
                            computed_stats = self._compute_stats_from_dataframe(group_df)

                            # Save stats as JSON
                            if computed_stats:
                                # Convert numpy types for JSON
                                def convert_numpy(obj):
                                    import numpy as np
                                    if isinstance(obj, np.integer):
                                        return int(obj)
                                    elif isinstance(obj, np.floating):
                                        return float(obj)
                                    elif isinstance(obj, np.ndarray):
                                        return obj.tolist()
                                    elif isinstance(obj, dict):
                                        return {key: convert_numpy(value) for key, value in obj.items()}
                                    elif isinstance(obj, list):
                                        return [convert_numpy(item) for item in obj]
                                    return obj

                                stats_dict_clean = convert_numpy(computed_stats)

                                # Ensure AssociatedResults directory exists
                                os.makedirs(os.path.dirname(stats_file), exist_ok=True)

                                with open(stats_file, 'w') as f:
                                    json.dump(stats_dict_clean, f, indent=2)

                                if verbosity >= 2:
                                    print(f"  ✅ Saved stats to {stats_file}")

                            self.groupby_stats[subdir] = computed_stats
                            loaded_groups.append(subdir)

                        except Exception as e:
                            if verbosity >= 1:
                                print(f"⚠️  Could not process {subdir}: {e}")

                if not loaded_groups:
                    raise ValueError("No AssociatedResults CSV files found in any group folders")

                if verbosity >= 1:
                    print(f"\n✅ Computed stats for {len(loaded_groups)} group(s):")
                    for group in loaded_groups:
                        n_rows = len(self.groupby_results[group])
                        print(f"   - {group}: {n_rows:,} rows")
            else:
                # Groupby results already loaded, just compute stats
                if verbosity >= 1:
                    print(f"Computing stats from {len(self.groupby_results)} loaded group(s)...")

                for group_name, group_df in self.groupby_results.items():
                    computed_stats = self._compute_stats_from_dataframe(group_df)
                    self.groupby_stats[group_name] = computed_stats

                    # Save stats JSON
                    group_path = os.path.join(self.data_folder, group_name)
                    stats_file = os.path.join(group_path, "AssociatedResults", "association_stats.json")

                    if computed_stats:
                        def convert_numpy(obj):
                            import numpy as np
                            if isinstance(obj, np.integer):
                                return int(obj)
                            elif isinstance(obj, np.floating):
                                return float(obj)
                            elif isinstance(obj, np.ndarray):
                                return obj.tolist()
                            elif isinstance(obj, dict):
                                return {key: convert_numpy(value) for key, value in obj.items()}
                            elif isinstance(obj, list):
                                return [convert_numpy(item) for item in obj]
                            return obj

                        stats_dict_clean = convert_numpy(computed_stats)
                        os.makedirs(os.path.dirname(stats_file), exist_ok=True)

                        with open(stats_file, 'w') as f:
                            json.dump(stats_dict_clean, f, indent=2)

                if verbosity >= 1:
                    print(f"✅ Computed and saved stats for all groups")

        else:
            # Process single folder
            assoc_file = os.path.join(self.data_folder, "AssociatedResults", "associated_data.csv")
            stats_file = os.path.join(self.data_folder, "AssociatedResults", "association_stats.json")

            if not os.path.exists(assoc_file):
                raise ValueError(f"No AssociatedResults CSV found at {assoc_file}")

            if verbosity >= 1:
                print(f"Loading association data from CSV...")

            # Load CSV if not already loaded
            if self.associated_df is None:
                self.associated_df = pd.read_csv(assoc_file)
                if verbosity >= 1:
                    print(f"  Loaded {len(self.associated_df):,} rows")

            # Compute stats
            computed_stats = self._compute_stats_from_dataframe(self.associated_df)

            if verbosity >= 2:
                print(f"  Computed stats keys: {list(computed_stats.keys())}")
                if 'photon_event' in computed_stats:
                    print(f"  Photon-event stats: {computed_stats['photon_event']}")

            if 'pixel_photon' in computed_stats:
                self.last_assoc_stats = computed_stats['pixel_photon']
            if 'photon_event' in computed_stats:
                self.last_photon_event_stats = computed_stats['photon_event']

            if verbosity >= 2:
                print(f"  self.last_assoc_stats: {self.last_assoc_stats}")
                print(f"  self.last_photon_event_stats: {self.last_photon_event_stats}")

            # Save stats as JSON
            if computed_stats:
                def convert_numpy(obj):
                    import numpy as np
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, dict):
                        return {key: convert_numpy(value) for key, value in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_numpy(item) for item in obj]
                    return obj

                stats_dict_clean = convert_numpy(computed_stats)
                os.makedirs(os.path.dirname(stats_file), exist_ok=True)

                with open(stats_file, 'w') as f:
                    json.dump(stats_dict_clean, f, indent=2)

                if verbosity >= 1:
                    print(f"✅ Computed and saved stats to {stats_file}")

        # Return HTML table
        return HTML(self._create_stats_html_table())

    def _create_stats_html_table(self):
        """Create HTML table with metrics as columns and groups as rows."""

        def get_color(value):
            """Color code percentages: red for low, green for good."""
            if value < 50:
                return 'indianred'
            elif value < 90:
                return '#FFA500'  # orange
            else:
                return '#90EE90'  # light green

        # Collect data for table rows
        rows_data = []

        if self.is_groupby and self.groupby_results:
            # Multiple rows for grouped data - use group names
            for group_name in self.groupby_results.keys():
                row = {'Group': group_name}

                # Get stored stats for this group (if available)
                if hasattr(self, 'groupby_stats') and group_name in self.groupby_stats:
                    stats = self.groupby_stats[group_name]
                    row.update(self._extract_row_metrics(stats))

                rows_data.append(row)
        else:
            # Single row for single dataset - use folder name
            import os
            folder_name = os.path.basename(self.data_folder)
            row = {'Group': folder_name}
            # Always try to extract metrics even if stats might be partial
            combined_stats = {
                'pixel_photon': self.last_assoc_stats if self.last_assoc_stats else {},
                'photon_event': self.last_photon_event_stats if self.last_photon_event_stats else {}
            }
            row.update(self._extract_row_metrics(combined_stats))
            rows_data.append(row)

        if not rows_data or all(len(row) == 1 for row in rows_data):
            return "<p><em>No association data available. Run associate() first, or use compute_stats_from_csv() if you have existing CSV files.</em></p>"

        # Build HTML table with multiindex-style headers
        html = """
        <div style="font-family: Arial, sans-serif; font-size: 0.85em;">
            <table style="border-collapse: collapse; border: 1px solid #ddd; table-layout: auto;">
                <thead>
                    <!-- Top level headers (category grouping) -->
                    <tr style="background-color: white; border-bottom: 2px solid #ddd;">
                        <th rowspan="2" style="padding: 8px; border: 1px solid #ddd; text-align: left; font-weight: bold; white-space: nowrap;">Group</th>
                        <th colspan="4" style="padding: 8px; border: 1px solid #ddd; text-align: center; font-weight: bold; border-bottom: 1px solid #999;">Pixel → Photon</th>
                        <th colspan="4" style="padding: 8px; border: 1px solid #ddd; text-align: center; font-weight: bold; border-bottom: 1px solid #999;">Photon → Event</th>
                        <th colspan="2" style="padding: 8px; border: 1px solid #ddd; text-align: center; font-weight: bold; border-bottom: 1px solid #999;">CoM Quality (px2ph)</th>
                        <th colspan="2" style="padding: 8px; border: 1px solid #ddd; text-align: center; font-weight: bold; border-bottom: 1px solid #999;">CoM Quality (ph2ev)</th>
                    </tr>
                    <!-- Second level headers (specific metrics) -->
                    <tr style="background-color: white; border-bottom: 2px solid #444;">
                        <th style="padding: 6px; border: 1px solid #ddd; text-align: center; font-weight: bold; font-size: 0.9em;">Pixels</th>
                        <th style="padding: 6px; border: 1px solid #ddd; text-align: center; font-weight: bold; font-size: 0.9em;">Pix %</th>
                        <th style="padding: 6px; border: 1px solid #ddd; text-align: center; font-weight: bold; font-size: 0.9em;">Photons</th>
                        <th style="padding: 6px; border: 1px solid #ddd; text-align: center; font-weight: bold; font-size: 0.9em;">Phot %</th>
                        <th style="padding: 6px; border: 1px solid #ddd; text-align: center; font-weight: bold; font-size: 0.9em;">Photons</th>
                        <th style="padding: 6px; border: 1px solid #ddd; text-align: center; font-weight: bold; font-size: 0.9em;">Phot %</th>
                        <th style="padding: 6px; border: 1px solid #ddd; text-align: center; font-weight: bold; font-size: 0.9em;">Events</th>
                        <th style="padding: 6px; border: 1px solid #ddd; text-align: center; font-weight: bold; font-size: 0.9em;">Evt %</th>
                        <th style="padding: 6px; border: 1px solid #ddd; text-align: center; font-weight: bold; font-size: 0.9em;">Exact %</th>
                        <th style="padding: 6px; border: 1px solid #ddd; text-align: center; font-weight: bold; font-size: 0.9em;">Good %</th>
                        <th style="padding: 6px; border: 1px solid #ddd; text-align: center; font-weight: bold; font-size: 0.9em;">Exact %</th>
                        <th style="padding: 6px; border: 1px solid #ddd; text-align: center; font-weight: bold; font-size: 0.9em;">Good %</th>
                    </tr>
                </thead>
                <tbody>
        """

        for row in rows_data:
            is_comparison = row['Group'] in getattr(self, 'comparison_groups', [])
            row_bg = '#FFF3E0' if is_comparison else 'white'
            comp_badge = ' <span style="background-color: #FF9800; color: white; padding: 2px 4px; border-radius: 2px; font-size: 0.75em;">cmp</span>' if is_comparison else ''

            html += f"""
                    <tr style="background-color: {row_bg};">
                        <td style="padding: 8px; border: 1px solid #ddd; white-space: nowrap;"><strong>{row['Group']}</strong>{comp_badge}</td>
            """

            # Pixel → Photon: Pixels count, Pix %, Photons count, Phot %
            for key in ['pix_count', 'pix_pct', 'phot_px2ph_count', 'phot_px2ph_pct']:
                if key in row:
                    if '_pct' in key:
                        value = row[key]
                        bg_color = get_color(value)
                        html += f'<td style="padding: 6px; border: 1px solid #ddd; text-align: center; background-color: {bg_color}; font-size: 0.9em;">{value:.1f}%</td>'
                    else:
                        html += f'<td style="padding: 6px; border: 1px solid #ddd; text-align: center; font-size: 0.9em;">{row[key]}</td>'
                else:
                    html += '<td style="padding: 6px; border: 1px solid #ddd; text-align: center; font-size: 0.9em;">—</td>'

            # Photon → Event: Photons count, Phot %, Events count, Evt %
            for key in ['phot_ph2ev_count', 'phot_ph2ev_pct', 'evt_count', 'evt_pct']:
                if key in row:
                    if '_pct' in key:
                        value = row[key]
                        bg_color = get_color(value)
                        html += f'<td style="padding: 6px; border: 1px solid #ddd; text-align: center; background-color: {bg_color}; font-size: 0.9em;">{value:.1f}%</td>'
                    else:
                        html += f'<td style="padding: 6px; border: 1px solid #ddd; text-align: center; font-size: 0.9em;">{row[key]}</td>'
                else:
                    html += '<td style="padding: 6px; border: 1px solid #ddd; text-align: center; font-size: 0.9em;">—</td>'

            # CoM Quality px2ph: Exact %, Good %
            for key in ['com_exact_px2ph', 'com_good_px2ph']:
                if key in row:
                    value = row[key]
                    bg_color = get_color(value)
                    html += f'<td style="padding: 6px; border: 1px solid #ddd; text-align: center; background-color: {bg_color}; font-size: 0.9em;">{value:.1f}%</td>'
                else:
                    html += '<td style="padding: 6px; border: 1px solid #ddd; text-align: center; font-size: 0.9em;">—</td>'

            # CoM Quality ph2ev: Exact %, Good %
            for key in ['com_exact_ph2ev', 'com_good_ph2ev']:
                if key in row:
                    value = row[key]
                    bg_color = get_color(value)
                    html += f'<td style="padding: 6px; border: 1px solid #ddd; text-align: center; background-color: {bg_color}; font-size: 0.9em;">{value:.1f}%</td>'
                else:
                    html += '<td style="padding: 6px; border: 1px solid #ddd; text-align: center; font-size: 0.9em;">—</td>'

            html += """
                    </tr>
            """

        html += """
                </tbody>
            </table>
        </div>
        """
        return html

    def _extract_row_metrics(self, stats):
        """Extract metrics from stats dict for a table row.

        Returns dict with:
        - pix_count: "matched / total" formatted string
        - pix_pct: percentage value
        - phot_px2ph_count: "matched / total" formatted string
        - phot_px2ph_pct: percentage value
        - phot_ph2ev_count: "matched / total" formatted string
        - phot_ph2ev_pct: percentage value
        - evt_count: "matched / total" formatted string
        - evt_pct: percentage value
        - com_exact_px2ph: percentage value
        - com_good_px2ph: percentage value
        - com_exact_ph2ev: percentage value
        - com_good_ph2ev: percentage value
        """
        metrics = {}

        def format_count(matched, total):
            """Format count as 'matched / total' with thousands separators."""
            return f"{matched:,} / {total:,}"

        if isinstance(stats, dict):
            # Handle combined stats (single dataset)
            if 'pixel_photon' in stats and stats['pixel_photon']:
                pxph = stats['pixel_photon']

                # Pixel → Photon: Pixels count and %
                if 'matched_pixels' in pxph and 'total_pixels' in pxph:
                    matched_pix = pxph['matched_pixels']
                    total_pix = pxph['total_pixels']
                    metrics['pix_count'] = format_count(matched_pix, total_pix)
                    metrics['pix_pct'] = 100 * matched_pix / total_pix if total_pix > 0 else 0

                # Pixel → Photon: Photons count and %
                if 'matched_photons' in pxph and 'total_photons' in pxph:
                    matched_phot = pxph['matched_photons']
                    total_phot = pxph['total_photons']
                    metrics['phot_px2ph_count'] = format_count(matched_phot, total_phot)
                    metrics['phot_px2ph_pct'] = 100 * matched_phot / total_phot if total_phot > 0 else 0

                # CoM Quality px2ph
                if 'com_quality' in pxph:
                    total_com = sum(pxph['com_quality'].values())
                    if total_com > 0:
                        metrics['com_exact_px2ph'] = 100 * pxph['com_quality'].get('exact', 0) / total_com
                        metrics['com_good_px2ph'] = 100 * pxph['com_quality'].get('good', 0) / total_com

            if 'photon_event' in stats and stats['photon_event']:
                phev = stats['photon_event']

                # Photon → Event: Photons count and %
                if 'matched_photons' in phev and 'total_photons' in phev:
                    matched_phot = phev['matched_photons']
                    total_phot = phev['total_photons']
                    metrics['phot_ph2ev_count'] = format_count(matched_phot, total_phot)
                    metrics['phot_ph2ev_pct'] = 100 * matched_phot / total_phot if total_phot > 0 else 0

                # Photon → Event: Events count and %
                if 'matched_events' in phev and 'total_events' in phev:
                    matched_evt = phev['matched_events']
                    total_evt = phev['total_events']
                    metrics['evt_count'] = format_count(matched_evt, total_evt)
                    metrics['evt_pct'] = 100 * matched_evt / total_evt if total_evt > 0 else 0

                # CoM Quality ph2ev
                if 'quality' in phev:
                    qual = phev['quality']
                    # Use total CoM quality counts as denominator
                    total_com = (qual.get('exact_com', 0) + qual.get('good_com', 0) +
                                qual.get('acceptable_com', 0) + qual.get('poor_com', 0))
                    if total_com > 0:
                        metrics['com_exact_ph2ev'] = 100 * qual.get('exact_com', 0) / total_com
                        metrics['com_good_ph2ev'] = 100 * qual.get('good_com', 0) / total_com

            # Handle direct stats (from groupby) - same structure
            if 'matched_pixels' in stats and 'total_pixels' in stats:
                matched_pix = stats['matched_pixels']
                total_pix = stats['total_pixels']
                metrics['pix_count'] = format_count(matched_pix, total_pix)
                metrics['pix_pct'] = 100 * matched_pix / total_pix if total_pix > 0 else 0

            if 'matched_photons' in stats and 'total_photons' in stats:
                matched_phot = stats['matched_photons']
                total_phot = stats['total_photons']
                # This is for px2ph stage
                metrics['phot_px2ph_count'] = format_count(matched_phot, total_phot)
                metrics['phot_px2ph_pct'] = 100 * matched_phot / total_phot if total_phot > 0 else 0

            if 'matched_events' in stats and 'total_events' in stats:
                matched_evt = stats['matched_events']
                total_evt = stats['total_events']
                metrics['evt_count'] = format_count(matched_evt, total_evt)
                metrics['evt_pct'] = 100 * matched_evt / total_evt if total_evt > 0 else 0

                # For ph2ev, photons count comes from the events data
                # We need to infer photons from the same stats
                if 'phot_ph2ev_count' not in metrics:
                    # Use matched_photons if available from ph2ev stage
                    metrics['phot_ph2ev_count'] = format_count(matched_phot, total_phot) if 'matched_photons' in stats else "—"
                    metrics['phot_ph2ev_pct'] = 100 * matched_phot / total_phot if ('matched_photons' in stats and total_phot > 0) else 0

            if 'com_quality' in stats:
                total_com = sum(stats['com_quality'].values())
                if total_com > 0:
                    metrics['com_exact_px2ph'] = 100 * stats['com_quality'].get('exact', 0) / total_com
                    metrics['com_good_px2ph'] = 100 * stats['com_quality'].get('good', 0) / total_com

            if 'quality' in stats:
                qual = stats['quality']
                # Use total CoM quality counts as denominator
                total_com = (qual.get('exact_com', 0) + qual.get('good_com', 0) +
                            qual.get('acceptable_com', 0) + qual.get('poor_com', 0))
                if total_com > 0:
                    metrics['com_exact_ph2ev'] = 100 * qual.get('exact_com', 0) / total_com
                    metrics['com_good_ph2ev'] = 100 * qual.get('good_com', 0) / total_com

        return metrics

    def _repr_html_single(self):
        """Generate HTML for single folder results with comprehensive statistics."""
        html = """
        <div style="font-family: Arial, sans-serif; font-size: 0.9em; border: 2px solid #4CAF50; border-radius: 8px; padding: 12px; margin: 10px 0; background-color: #f9f9f9;">
            <h3 style="margin-top: 0; margin-bottom: 10px; color: #4CAF50; font-size: 1.1em;">✅ Association Results</h3>
        """

        # Pixel-Photon Association Stats
        if self.last_assoc_stats:
            stats = self.last_assoc_stats
            html += """
            <table style="width: 100%; border-collapse: collapse; background-color: white; margin-bottom: 10px; font-size: 0.85em;">
                <thead>
                    <tr style="background-color: #4CAF50; color: white;">
                        <th colspan="3" style="padding: 6px; text-align: left; font-size: 0.95em;">Pixel → Photon Association</th>
                    </tr>
                </thead>
                <tbody>
            """

            # Pixel and photon matching rates
            pix_rate = 100 * stats['matched_pixels'] / stats['total_pixels']
            phot_rate = 100 * stats['matched_photons'] / stats['total_photons']
            pix_color = '#4CAF50' if pix_rate > 70 else '#FF9800' if pix_rate > 50 else '#F44336'
            phot_color = '#4CAF50' if phot_rate > 70 else '#FF9800' if phot_rate > 50 else '#F44336'

            html += f"""
                    <tr style="border-bottom: 1px solid #ddd;">
                        <td style="padding: 5px; width: 40%;">Pixels Matched</td>
                        <td style="padding: 5px; width: 35%;">{stats['matched_pixels']:,} / {stats['total_pixels']:,}</td>
                        <td style="padding: 5px; text-align: right; width: 25%;">
                            <span style="background-color: {pix_color}; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.9em;">
                                {pix_rate:.1f}%
                            </span>
                        </td>
                    </tr>
                    <tr style="border-bottom: 1px solid #ddd;">
                        <td style="padding: 5px;">Photons Matched</td>
                        <td style="padding: 5px;">{stats['matched_photons']:,} / {stats['total_photons']:,}</td>
                        <td style="padding: 5px; text-align: right;">
                            <span style="background-color: {phot_color}; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.9em;">
                                {phot_rate:.1f}%
                            </span>
                        </td>
                    </tr>
            """

            # CoM Quality
            if 'com_quality' in stats:
                com = stats['com_quality']
                total = sum(com.values())
                if total > 0:
                    html += """
                    <tr style="background-color: #f0f0f0;">
                        <td colspan="3" style="padding: 5px; font-weight: bold; font-size: 0.9em;">Center-of-Mass Match Quality</td>
                    </tr>
                    """
                    for quality, label in [('exact', 'Exact (≤0.1px)'), ('good', 'Good (≤30% radius)'),
                                          ('acceptable', 'Acceptable (≤50%)'), ('poor', 'Poor (≤100%)'), ('failed', 'Failed (>100%)')]:
                        count = com.get(quality, 0)
                        if count > 0 or quality in ['exact', 'good']:  # Always show exact and good
                            pct = 100 * count / total
                            html += f"""
                    <tr style="border-bottom: 1px solid #ddd;">
                        <td style="padding: 5px; padding-left: 15px;">{label}</td>
                        <td style="padding: 5px;">{count:,}</td>
                        <td style="padding: 5px; text-align: right;">{pct:.1f}%</td>
                    </tr>
                            """

            html += """
                </tbody>
            </table>
            """

        # Photon-Event Association Stats
        if self.last_photon_event_stats:
            stats = self.last_photon_event_stats
            html += """
            <table style="width: 100%; border-collapse: collapse; background-color: white; font-size: 0.85em;">
                <thead>
                    <tr style="background-color: #2196F3; color: white;">
                        <th colspan="3" style="padding: 6px; text-align: left; font-size: 0.95em;">Photon → Event Association</th>
                    </tr>
                </thead>
                <tbody>
            """

            # Photon and event matching rates
            phot_rate = 100 * stats['matched_photons'] / stats['total_photons']
            evt_rate = 100 * stats['matched_events'] / stats['total_events']
            phot_color = '#4CAF50' if phot_rate > 70 else '#FF9800' if phot_rate > 50 else '#F44336'
            evt_color = '#4CAF50' if evt_rate > 70 else '#FF9800' if evt_rate > 50 else '#F44336'

            html += f"""
                    <tr style="border-bottom: 1px solid #ddd;">
                        <td style="padding: 5px; width: 40%;">Photons Matched</td>
                        <td style="padding: 5px; width: 35%;">{stats['matched_photons']:,} / {stats['total_photons']:,}</td>
                        <td style="padding: 5px; text-align: right; width: 25%;">
                            <span style="background-color: {phot_color}; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.9em;">
                                {phot_rate:.1f}%
                            </span>
                        </td>
                    </tr>
                    <tr style="border-bottom: 1px solid #ddd;">
                        <td style="padding: 5px;">Events Matched</td>
                        <td style="padding: 5px;">{stats['matched_events']:,} / {stats['total_events']:,}</td>
                        <td style="padding: 5px; text-align: right;">
                            <span style="background-color: {evt_color}; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.9em;">
                                {evt_rate:.1f}%
                            </span>
                        </td>
                    </tr>
            """

            # Association Quality
            if 'quality' in stats:
                qual = stats['quality']
                matched_total = qual.get('exact_n', 0) + qual.get('n_mismatch', 0)
                if matched_total > 0:
                    html += """
                    <tr style="background-color: #f0f0f0;">
                        <td colspan="3" style="padding: 5px; font-weight: bold; font-size: 0.9em;">Association Quality</td>
                    </tr>
                    """

                    # Photon count match
                    exact_n = qual.get('exact_n', 0)
                    n_mismatch = qual.get('n_mismatch', 0)
                    exact_n_pct = 100 * exact_n / matched_total
                    html += f"""
                    <tr style="border-bottom: 1px solid #ddd;">
                        <td style="padding: 5px; padding-left: 15px;">Photon count matches ev/n</td>
                        <td style="padding: 5px;">{exact_n:,}</td>
                        <td style="padding: 5px; text-align: right;">{exact_n_pct:.1f}%</td>
                    </tr>
                    """
                    if n_mismatch > 0:
                        n_mismatch_pct = 100 * n_mismatch / matched_total
                        html += f"""
                    <tr style="border-bottom: 1px solid #ddd;">
                        <td style="padding: 5px; padding-left: 15px;">Photon count mismatch</td>
                        <td style="padding: 5px;">{n_mismatch:,}</td>
                        <td style="padding: 5px; text-align: right;">{n_mismatch_pct:.1f}%</td>
                    </tr>
                        """

                    # CoM Quality
                    html += """
                    <tr style="background-color: #f0f0f0;">
                        <td colspan="3" style="padding: 5px; font-weight: bold; font-size: 0.9em;">Center-of-Mass Match Quality</td>
                    </tr>
                    """
                    for quality, label in [('exact_com', 'Exact (≤0.1px)'), ('good_com', 'Good (≤30% radius)'),
                                          ('acceptable_com', 'Acceptable (≤50%)'), ('poor_com', 'Poor (>50%)')]:
                        count = qual.get(quality, 0)
                        if count > 0 or quality in ['exact_com', 'good_com']:
                            pct = 100 * count / matched_total
                            html += f"""
                    <tr style="border-bottom: 1px solid #ddd;">
                        <td style="padding: 5px; padding-left: 15px;">{label.split('(')[1].split(')')[0]}</td>
                        <td style="padding: 5px;">{count:,}</td>
                        <td style="padding: 5px; text-align: right;">{pct:.1f}%</td>
                    </tr>
                            """

            html += """
                </tbody>
            </table>
            """

        html += """
        </div>
        """
        return html


    def _repr_html_groupby(self):
        """Generate HTML for grouped results."""
        html = """
        <div style="font-family: Arial, sans-serif; border: 2px solid #2196F3; border-radius: 8px; padding: 15px; margin: 10px 0; background-color: #f9f9f9;">
            <h3 style="margin-top: 0; color: #2196F3;">📊 Groupby Association Results</h3>
            <table style="width: 100%; border-collapse: collapse; background-color: white;">
                <thead>
                    <tr style="background-color: #2196F3; color: white;">
                        <th style="padding: 10px; text-align: left;">Group</th>
                        <th style="padding: 10px; text-align: right;">Pixels</th>
                        <th style="padding: 10px; text-align: right;">Pix→Phot</th>
                        <th style="padding: 10px; text-align: right;">Phot→Evt</th>
                    </tr>
                </thead>
                <tbody>
        """

        for group_name, group_df in self.groupby_results.items():
            stats = self._compute_stats_for_df(group_df)

            pix_rate = stats.get('pixel_photon_rate', 0) * 100
            pix_color = '#4CAF50' if pix_rate > 70 else '#FF9800' if pix_rate > 50 else '#F44336'

            phot_rate = stats.get('photon_event_rate', 0) * 100
            phot_color = '#4CAF50' if phot_rate > 70 else '#FF9800' if phot_rate > 50 else '#F44336'

            html += f"""
                <tr style="border-bottom: 1px solid #ddd;">
                    <td style="padding: 8px;"><strong>{group_name}</strong></td>
                    <td style="padding: 8px; text-align: right;">{len(group_df):,}</td>
                    <td style="padding: 8px; text-align: right;">
                        <span style="background-color: {pix_color}; color: white; padding: 3px 6px; border-radius: 3px; font-size: 0.9em;">
                            {pix_rate:.1f}%
                        </span>
                    </td>
                    <td style="padding: 8px; text-align: right;">
                        <span style="background-color: {phot_color}; color: white; padding: 3px 6px; border-radius: 3px; font-size: 0.9em;">
                            {phot_rate:.1f}%
                        </span>
                    </td>
                </tr>
            """

        html += """
                </tbody>
            </table>
            <p style="margin: 10px 0 0 0; font-size: 0.9em; color: #666;">
                💡 <strong>Tip:</strong> Use <code>.plot_stats()</code> to visualize, <code>.plot_stats(group='name')</code> for specific group
            </p>
        </div>
        """
        return html

    def associate_groupby(self, **kwargs):
        """
        Run association on all groups in a groupby folder structure sequentially with progress bar.

        This method processes each subdirectory as a separate Analyse instance and runs
        association on all groups sequentially (with a progress bar to show progress).

        Args:
            **kwargs: Arguments to pass to the associate() method for each group.
                     Common arguments: relax, method, verbosity, pixel_max_dist_px, etc.

        Returns:
            dict: Dictionary mapping group names to their associated dataframes.

        Raises:
            ValueError: If this is not a groupby folder structure.

        Example:
            assoc = nea.Analyse("archive/pencilbeam/detector_model")
            results = assoc.associate_groupby(relax=1.5, method='simple', verbosity=1)
            # Access results: results['intensifier_gain_50']
        """
        if not self.is_groupby:
            raise ValueError("This is not a groupby folder structure. Use .associate() instead.")

        # Handle verbosity - default to instance verbosity if not specified
        verbosity = kwargs.get('verbosity')
        if verbosity is None:
            verbosity = self.verbosity if self.verbosity is not None else 1

        if verbosity >= 1:
            print(f"\n{'='*70}")
            print(f"Running Association on {len(self.groupby_subdirs)} Groups")
            print(f"{'='*70}")

        # Process groups sequentially with progress bar
        results = {}
        group_stats = []  # Collect pixel-photon statistics from each group
        group_pe_stats = []  # Collect photon-event statistics from each group
        self.groupby_stats = {}  # Store stats per group for HTML display

        for group_name in tqdm(self.groupby_subdirs, desc="Processing groups", disable=(verbosity == 0)):
            group_path = os.path.join(self.data_folder, group_name)
            try:
                # Create Analyse instance for this group
                # Pass parent settings to child groups (child can override with its own settings file)
                group_assoc = Analyse(
                    group_path,
                    export_dir=self.export_dir,
                    n_threads=self.n_threads,  # Use all threads for each group
                    use_lumacam=self.use_lumacam,
                    settings=None,  # Let child auto-detect its own settings first
                    parent_settings=self.settings,  # Provide parent settings as fallback
                    verbosity=0,  # Suppress individual group output
                    limit=self.limit  # Pass limit to child groups
                )

                # Run association with provided kwargs (but override verbosity to 0)
                kwargs_copy = kwargs.copy()
                kwargs_copy['verbosity'] = 0
                group_assoc.associate(**kwargs_copy)

                results[group_name] = group_assoc.associated_df

                # Save association results for this group (includes stats JSON)
                group_assoc.save_associations(verbosity=0)

                # Collect and store statistics for this group
                group_combined_stats = {}
                if hasattr(group_assoc, 'last_assoc_stats') and group_assoc.last_assoc_stats:
                    group_stats.append(group_assoc.last_assoc_stats)
                    # Store px2ph stats under 'pixel_photon' key for consistency
                    group_combined_stats['pixel_photon'] = group_assoc.last_assoc_stats.copy()

                if hasattr(group_assoc, 'last_photon_event_stats') and group_assoc.last_photon_event_stats:
                    group_pe_stats.append(group_assoc.last_photon_event_stats)
                    # Store ph2ev stats under 'photon_event' key for consistency
                    group_combined_stats['photon_event'] = group_assoc.last_photon_event_stats.copy()

                self.groupby_stats[group_name] = group_combined_stats

                if verbosity >= 2:
                    print(f"✅ {group_name}: {len(group_assoc.associated_df)} rows")
            except Exception as e:
                logger.error(f"Error processing group {group_name}: {e}")
                if verbosity >= 1:
                    print(f"❌ {group_name}: Failed - {e}")

        self.groupby_results = results

        if verbosity >= 1:
            print(f"\n{'='*70}")
            print(f"Groupby Association Complete")
            print(f"{'='*70}")
            print(f"Processed {len(results)}/{len(self.groupby_subdirs)} groups successfully")

            # Aggregate and display statistics from all groups
            if group_stats:
                total_matched_pixels = sum(s['matched_pixels'] for s in group_stats)
                total_pixels = sum(s['total_pixels'] for s in group_stats)
                total_matched_photons = sum(s['matched_photons'] for s in group_stats)
                total_photons = sum(s['total_photons'] for s in group_stats)

                # Aggregate CoM quality stats
                aggregated_com = {
                    'exact': sum(s['com_quality']['exact'] for s in group_stats),
                    'good': sum(s['com_quality']['good'] for s in group_stats),
                    'acceptable': sum(s['com_quality']['acceptable'] for s in group_stats),
                    'poor': sum(s['com_quality']['poor'] for s in group_stats),
                    'failed': sum(s['com_quality']['failed'] for s in group_stats)
                }

                print(f"\n✅ Pixel-Photon Association Results (All Groups):")
                if total_pixels > 0:
                    print(f"   Pixels:  {total_matched_pixels:,} / {total_pixels:,} matched ({100 * total_matched_pixels / total_pixels:.1f}%)")
                if total_photons > 0:
                    print(f"   Photons: {total_matched_photons:,} / {total_photons:,} matched ({100 * total_matched_photons / total_photons:.1f}%)")

                # Show CoM quality statistics
                total_processed = sum(aggregated_com.values())
                if total_processed > 0:
                    print(f"   Center-of-Mass Match Quality:")
                    print(f"      Exact (≤0.1px):     {aggregated_com['exact']:,} ({100 * aggregated_com['exact'] / total_processed:.1f}%)")
                    print(f"      Good (≤30% radius): {aggregated_com['good']:,} ({100 * aggregated_com['good'] / total_processed:.1f}%)")
                    print(f"      Acceptable (≤50%):  {aggregated_com['acceptable']:,} ({100 * aggregated_com['acceptable'] / total_processed:.1f}%)")
                    print(f"      Poor (≤100%):       {aggregated_com['poor']:,} ({100 * aggregated_com['poor'] / total_processed:.1f}%)")
                    if aggregated_com['failed'] > 0:
                        print(f"      Failed (>100%):     {aggregated_com['failed']:,} ({100 * aggregated_com['failed'] / total_processed:.1f}%)")

            # Aggregate and display photon-event statistics
            if group_pe_stats:
                total_matched_photons_pe = sum(s['matched_photons'] for s in group_pe_stats)
                total_photons_pe = sum(s['total_photons'] for s in group_pe_stats)
                total_matched_events = sum(s['matched_events'] for s in group_pe_stats)
                total_events = sum(s['total_events'] for s in group_pe_stats)

                # Aggregate quality stats
                aggregated_quality = {
                    'exact_n': sum(s['quality']['exact_n'] for s in group_pe_stats),
                    'n_mismatch': sum(s['quality']['n_mismatch'] for s in group_pe_stats),
                    'exact_com': sum(s['quality']['exact_com'] for s in group_pe_stats),
                    'good_com': sum(s['quality']['good_com'] for s in group_pe_stats),
                    'acceptable_com': sum(s['quality']['acceptable_com'] for s in group_pe_stats),
                    'poor_com': sum(s['quality']['poor_com'] for s in group_pe_stats)
                }

                print(f"\n✅ Photon-Event Association Results (All Groups):")
                if total_photons_pe > 0:
                    print(f"   Photons: {total_matched_photons_pe:,} / {total_photons_pe:,} matched ({100 * total_matched_photons_pe / total_photons_pe:.1f}%)")
                if total_events > 0:
                    print(f"   Events:  {total_matched_events:,} / {total_events:,} matched ({100 * total_matched_events / total_events:.1f}%)")

                # Show quality statistics
                matched_events_total = aggregated_quality['exact_n'] + aggregated_quality['n_mismatch']
                if matched_events_total > 0:
                    print(f"   Association Quality:")
                    print(f"      Photon count matches ev/n: {aggregated_quality['exact_n']:,} ({100 * aggregated_quality['exact_n'] / matched_events_total:.1f}%)")
                    if aggregated_quality['n_mismatch'] > 0:
                        print(f"      Photon count mismatch:     {aggregated_quality['n_mismatch']:,} ({100 * aggregated_quality['n_mismatch'] / matched_events_total:.1f}%)")
                    print(f"   Center-of-Mass Match Quality:")
                    print(f"      Exact (≤0.1px):     {aggregated_quality['exact_com']:,} ({100 * aggregated_quality['exact_com'] / matched_events_total:.1f}%)")
                    print(f"      Good (≤30% radius): {aggregated_quality['good_com']:,} ({100 * aggregated_quality['good_com'] / matched_events_total:.1f}%)")
                    print(f"      Acceptable (≤50%):  {aggregated_quality['acceptable_com']:,} ({100 * aggregated_quality['acceptable_com'] / matched_events_total:.1f}%)")
                    if aggregated_quality['poor_com'] > 0:
                        print(f"      Poor (>50%):        {aggregated_quality['poor_com']:,} ({100 * aggregated_quality['poor_com'] / matched_events_total:.1f}%)")

        # Auto-save results for all groups
        if results:
            if verbosity >= 1:
                print("\n💾 Auto-saving results for all groups...")
            saved_count = 0
            for group_name, group_df in results.items():
                try:
                    # Temporarily set associated_df to save this group
                    original_df = self.associated_df
                    original_folder = self.data_folder
                    self.associated_df = group_df
                    self.data_folder = os.path.join(original_folder, group_name)

                    output_path = self.save_associations(verbosity=0)
                    saved_count += 1
                    if verbosity >= 2:
                        print(f"   ✅ {group_name}: {output_path}")

                    # Restore
                    self.associated_df = original_df
                    self.data_folder = original_folder
                except Exception as e:
                    if verbosity >= 1:
                        print(f"   ⚠️  {group_name}: Could not save - {e}")

            if verbosity >= 1:
                print(f"💾 Saved results for {saved_count}/{len(results)} groups")

        # Return HTML stats table for display
        from IPython.display import HTML
        return HTML(self._create_stats_html_table())

    def _associate_photons_to_events(self, photons_df, events_df, weight_px_in_s, max_time_s, verbosity):
        """
        Associate photons to events using the lumacamTesting library.

        This method requires the lumacamTesting package. It fixes time monotonicity,
        computes scales if parameters are not provided, and performs association.

        Args:
            photons_df (pd.DataFrame): Photon DataFrame with 'x', 'y', 't' columns.
            events_df (pd.DataFrame): Event DataFrame with 'x', 'y', 't', 'n', 'PSD' columns.
            weight_px_in_s (float, optional): Weight for pixel-to-second conversion.
            max_time_s (float, optional): Maximum distance in seconds.
            verbosity (int): Verbosity level.

        Returns:
            pd.DataFrame: Associated photon DataFrame with added columns.
        """
        if not LUMACAM_AVAILABLE:
            raise ImportError("lumacamTesting is required for this method.")
        
        photons = photons_df.rename(columns={"x": "x_px", "y": "y_px", "t": "t_s"}).copy()
        events = events_df.rename(columns={"x": "x_px", "y": "y_px", "t": "t_s"}).copy()

        def fix_time_with_progress(df, label="DataFrame"):
            df = df.sort_values("t_s").reset_index(drop=True)
            t_s = df["t_s"].to_numpy()
            for i in tqdm(range(1, len(t_s)), desc=f"Fixing time for {label}", disable=(verbosity == 0)):
                if t_s[i] <= t_s[i - 1]:
                    t_s[i] = t_s[i - 1] + 1e-12
            df["t_s"] = t_s
            return df

        photons = fix_time_with_progress(photons, label="photons")
        events = fix_time_with_progress(events, label="events")

        if weight_px_in_s is None or max_time_s is None:
            all_x = pd.concat([photons['x_px'], events['x_px']])
            all_y = pd.concat([photons['y_px'], events['y_px']])
            all_t = pd.concat([photons['t_s'], events['t_s']])
            spatial_scale = np.sqrt(np.var(all_x) + np.var(all_y))
            temporal_scale = np.std(all_t)
            weight_px_in_s = temporal_scale / spatial_scale if spatial_scale > 0 else 1.0
            max_time_s = 3 * temporal_scale
            if verbosity >= 2:
                print(f"📏 Spatial scale: {spatial_scale:.2f}")
                print(f"⏱  Temporal scale: {temporal_scale:.2e} s")
                print(f"⚖️  Weight px→s: {weight_px_in_s:.2e}")
                print(f"📐 Max dist: {max_time_s:.2e} s")
        else:
            if verbosity >= 2:
                print(f"⚖️  Using provided weight_px_in_s: {weight_px_in_s}")
                print(f"📐 Using provided max_time_s: {max_time_s}")

        with yaspin.yaspin(text="Associating photons to events...", color="cyan") as spinner:
            assoc = lct.EventAssociation.make_individualShortestConnection(
                weight_px_in_s, max_time_s,
                photons[['t_s', 'x_px', 'y_px']],
                events[['t_s', 'x_px', 'y_px']]
            )
            spinner.ok("✅")

        cluster_associations = assoc.clusterAssociation_groundTruth
        cluster_event_indices = assoc.clusterAssociation_toTest
        result_df = photons_df.copy()
        result_df["assoc_cluster_id"] = cluster_associations
        result_df["assoc_t"] = np.nan
        result_df["assoc_x"] = np.nan
        result_df["assoc_y"] = np.nan
        result_df["assoc_n"] = 0  # Default to 0 if missing
        result_df["assoc_PSD"] = 0  # Default to 0 if missing

        for cluster_id in tqdm(assoc.clusters.index, desc="Assigning event data", disable=(verbosity == 0)):
            photon_indices = np.where(cluster_associations == cluster_id)[0]
            event_indices = np.where(cluster_event_indices == cluster_id)[0]
            if len(event_indices) > 0:
                event_idx = event_indices[0]
                event = events.iloc[event_idx]
                result_df.loc[photon_indices, "assoc_t"] = event["t_s"]
                result_df.loc[photon_indices, "assoc_x"] = event["x_px"]
                result_df.loc[photon_indices, "assoc_y"] = event["y_px"]
                result_df.loc[photon_indices, "assoc_n"] = event.get("n", 0)
                result_df.loc[photon_indices, "assoc_PSD"] = event.get("PSD", 0)

        return result_df

    def _associate_photons_to_events_kdtree(
        self, photons_df, events_df, time_norm_ns, spatial_norm_px, dSpace_px, verbosity
    ):
        """
        Associate photons to events using a full KDTree on normalized coordinates.

        This method builds a KDTree on all photons and queries for each event.
        It handles single and multi-photon events, with center-of-mass check for multi-photons.

        Args:
            photons_df (pd.DataFrame): Photon DataFrame with 'x', 'y', 't' columns.
            events_df (pd.DataFrame): Event DataFrame with 'x', 'y', 't', 'n', 'PSD' columns.
            time_norm_ns (float): Time normalization in nanoseconds.
            spatial_norm_px (float): Spatial normalization in pixels.
            dSpace_px (float): Maximum center-of-mass distance for multi-photons.
            verbosity (int): Verbosity level (0=silent, 1=summary, 2=debug).

        Returns:
            pd.DataFrame: Associated photon DataFrame with added columns.
        """
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
        photons['assoc_status'] = np.nan
        events = events.sort_values('t').reset_index(drop=True)
        events['event_id'] = events.index + 1
        
        # Create a KDTree for photons (x, y, t scaled by time_norm_ns)
        photon_coords = np.vstack([
            photons['x'].to_numpy(),
            photons['y'].to_numpy(),
            photons['t'].to_numpy() * 1e9 / time_norm_ns
        ]).T
        tree = cKDTree(photon_coords)

        for event in tqdm(events.iterrows(), total=len(events), desc="Associating events", disable=(verbosity == 0)):
            i, event = event
            n_photons = int(event['n'])
            ex, ey, et = event['x'], event['y'], event['t']
            eid = event['event_id']
            query_point = np.array([ex, ey, et * 1e9 / time_norm_ns])
            max_dist = 5.0  # Increased from 3.0 to allow more matches
            indices = tree.query_ball_point(query_point, max_dist)
            
            if not indices:
                continue
            
            candidate_photons = photons.iloc[indices]
            time_diff = np.abs(candidate_photons['t'] - et) * 1e9
            spatial_diff = np.sqrt((candidate_photons['x'] - ex)**2 + (candidate_photons['y'] - ey)**2)
            combined_diff = (time_diff / time_norm_ns) + (spatial_diff / spatial_norm_px)
            
            if n_photons == 1:
                best_idx = indices[np.argmin(combined_diff)]
                if np.isnan(photons.loc[best_idx, 'assoc_event_id']):
                    com_dist = spatial_diff.iloc[np.argmin(combined_diff)]
                    if com_dist <= dSpace_px:
                        photons.loc[best_idx, 'assoc_event_id'] = eid
                        photons.loc[best_idx, 'assoc_x'] = ex
                        photons.loc[best_idx, 'assoc_y'] = ey
                        photons.loc[best_idx, 'assoc_t'] = et
                        photons.loc[best_idx, 'assoc_n'] = n_photons
                        photons.loc[best_idx, 'assoc_PSD'] = event.get('PSD', 0)
                        photons.loc[best_idx, 'time_diff_ns'] = time_diff.iloc[np.argmin(combined_diff)]
                        photons.loc[best_idx, 'spatial_diff_px'] = com_dist
                        photons.loc[best_idx, 'assoc_com_dist'] = com_dist
                        photons.loc[best_idx, 'assoc_status'] = 'cog_match'
            else:
                candidate_indices = np.argsort(combined_diff)[:n_photons]
                selected_indices = [indices[i] for i in candidate_indices]
                selected_x = candidate_photons.iloc[candidate_indices]['x']
                selected_y = candidate_photons.iloc[candidate_indices]['y']
                if not np.any(np.isnan(selected_x)) and not np.any(np.isnan(selected_y)):
                    com_x = np.mean(selected_x)
                    com_y = np.mean(selected_y)
                    com_dist = np.sqrt((com_x - ex)**2 + (com_y - ey)**2)
                    if com_dist <= dSpace_px:
                        for idx, diff_idx in zip(selected_indices, candidate_indices):
                            if np.isnan(photons.loc[idx, 'assoc_event_id']):
                                photons.loc[idx, 'assoc_event_id'] = eid
                                photons.loc[idx, 'assoc_x'] = ex
                                photons.loc[idx, 'assoc_y'] = ey
                                photons.loc[idx, 'assoc_t'] = et
                                photons.loc[idx, 'assoc_n'] = n_photons
                                photons.loc[idx, 'assoc_PSD'] = event.get('PSD', 0)
                                photons.loc[idx, 'time_diff_ns'] = time_diff.iloc[diff_idx]
                                photons.loc[idx, 'spatial_diff_px'] = spatial_diff.iloc[diff_idx]
                                photons.loc[idx, 'assoc_com_dist'] = com_dist
                                photons.loc[idx, 'assoc_status'] = 'cog_match'

        photons['assoc_status'] = photons['assoc_status'].astype('category')

        # Compute and store statistics
        matched_photons = photons['assoc_event_id'].notna().sum()
        total_photons = len(photons)

        # Count how many events were matched
        matched_event_ids = photons[photons['assoc_event_id'].notna()]['assoc_event_id'].unique()
        matched_events = len(matched_event_ids)
        total_events = len(events)

        # Quality metrics
        quality_stats = {'exact_n': 0, 'n_mismatch': 0, 'exact_com': 0, 'good_com': 0,
                        'acceptable_com': 0, 'poor_com': 0}

        # For each matched event, check quality
        for event_id in matched_event_ids:
            event_photons = photons[photons['assoc_event_id'] == event_id]
            actual_n = len(event_photons)

            # Get predicted n from first photon's assoc_n
            predicted_n = int(event_photons.iloc[0]['assoc_n'])

            # Check if photon count matches prediction
            if actual_n == predicted_n:
                quality_stats['exact_n'] += 1
            else:
                quality_stats['n_mismatch'] += 1

            # Check CoM distance quality (from first photon)
            com_dist = event_photons.iloc[0]['assoc_com_dist']
            if com_dist <= 0.1:  # Within 0.1 pixel
                quality_stats['exact_com'] += 1
            elif com_dist <= dSpace_px * 0.3:  # Within 30% of search radius
                quality_stats['good_com'] += 1
            elif com_dist <= dSpace_px * 0.5:  # Within 50% of search radius
                quality_stats['acceptable_com'] += 1
            else:  # Within search radius but >50%
                quality_stats['poor_com'] += 1

        # Store statistics as instance variable
        self.last_photon_event_stats = {
            'matched_photons': matched_photons,
            'total_photons': total_photons,
            'matched_events': matched_events,
            'total_events': total_events,
            'quality': quality_stats
        }

        # Print statistics if requested
        if verbosity >= 2:  # Only print detailed stats at verbosity 2
            print(f"✅ Photon-Event Association Results:")
            print(f"   Photons: {matched_photons:,} / {total_photons:,} matched ({100 * matched_photons / total_photons:.1f}%)")
            print(f"   Events:  {matched_events:,} / {total_events:,} matched ({100 * matched_events / total_events:.1f}%)")

            # Show quality statistics
            if matched_events > 0:
                print(f"   Association Quality:")
                print(f"      Photon count matches ev/n: {quality_stats['exact_n']:,} ({100 * quality_stats['exact_n'] / matched_events:.1f}%)")
                if quality_stats['n_mismatch'] > 0:
                    print(f"      Photon count mismatch:     {quality_stats['n_mismatch']:,} ({100 * quality_stats['n_mismatch'] / matched_events:.1f}%)")
                print(f"   Center-of-Mass Match Quality:")
                print(f"      Exact (≤0.1px):     {quality_stats['exact_com']:,} ({100 * quality_stats['exact_com'] / matched_events:.1f}%)")
                print(f"      Good (≤30% radius): {quality_stats['good_com']:,} ({100 * quality_stats['good_com'] / matched_events:.1f}%)")
                print(f"      Acceptable (≤50%):  {quality_stats['acceptable_com']:,} ({100 * quality_stats['acceptable_com'] / matched_events:.1f}%)")
                if quality_stats['poor_com'] > 0:
                    print(f"      Poor (>50%):        {quality_stats['poor_com']:,} ({100 * quality_stats['poor_com'] / matched_events:.1f}%)")

        return photons

    def _associate_photons_to_events_window(
        self, photons_df, events_df, time_norm_ns, spatial_norm_px, dSpace_px, max_time_s, verbosity
    ):
        """
        Associate photons to events using a time-window KDTree.

        This is more efficient when photons/events are almost sorted by time.
        It uses a symmetric sliding window [-max_time_s, +max_time_s] to build local KDTrees.
        If max_time_s is None, it is computed as 3 * std of all times.

        Args:
            photons_df (pd.DataFrame): Photon DataFrame with 'x', 'y', 't' columns.
            events_df (pd.DataFrame): Event DataFrame with 'x', 'y', 't', 'n', 'PSD' columns.
            time_norm_ns (float): Time normalization in nanoseconds.
            spatial_norm_px (float): Spatial normalization in pixels.
            dSpace_px (float): Maximum center-of-mass distance for multi-photons.
            max_time_s (float, optional): Maximum time window in seconds.
            verbosity (int): Verbosity level (0=silent, 1=summary, 2=debug).

        Returns:
            pd.DataFrame: Associated photon DataFrame with added columns.
        """
        if max_time_s is None:
            all_t = np.concatenate((photons_df['t'].to_numpy(), events_df['t'].to_numpy()))
            temporal_scale = np.std(all_t) if len(all_t) > 1 else 1e-6
            max_time_s = 3 * temporal_scale
            if verbosity >= 2:
                print(f"Computed max_time_s: {max_time_s:.2e} s")

        photons = photons_df.copy()
        events = events_df.copy()

        # Init association columns
        photons['assoc_event_id'] = np.nan
        photons['assoc_x'] = np.nan
        photons['assoc_y'] = np.nan
        photons['assoc_t'] = np.nan
        photons['assoc_n'] = 0
        photons['assoc_PSD'] = 0
        photons['time_diff_ns'] = np.nan
        photons['spatial_diff_px'] = np.nan
        photons['assoc_com_dist'] = np.nan
        photons['assoc_status'] = np.nan

        # Ensure sorted by time
        photons = photons.sort_values('t').reset_index(drop=True)
        events = events.sort_values('t').reset_index(drop=True)
        events['event_id'] = events.index + 1

        # Convert to numpy for fast iteration
        p_t = photons['t'].to_numpy()
        p_x = photons['x'].to_numpy()
        p_y = photons['y'].to_numpy()

        left = 0
        n_photons_total = len(photons)

        for _, ev in tqdm(events.iterrows(), total=len(events), desc="Associating events", disable=(verbosity == 0)):
            et, ex, ey, eid, n_photons = ev['t'], ev['x'], ev['y'], ev['event_id'], int(ev['n'])

            # Slide left pointer: drop photons earlier than (et - max_time_s)
            while left < n_photons_total and p_t[left] < et - max_time_s:
                left += 1

            # Right bound: include photons until (et + max_time_s)
            right = left
            while right < n_photons_total and p_t[right] <= et + max_time_s:
                right += 1

            if right == left:
                continue  # no photons in window

            # Build KDTree for this local window
            sub_idx = np.arange(left, right)
            sub_coords = np.vstack([
                p_x[sub_idx],
                p_y[sub_idx],
                p_t[sub_idx] * 1e9 / time_norm_ns
            ]).T
            tree = cKDTree(sub_coords)

            query_point = np.array([ex, ey, et * 1e9 / time_norm_ns])
            indices = tree.query_ball_point(query_point, r=5.0)
            if not indices:
                continue

            # Map back to global photon indices
            indices = sub_idx[indices]
            candidate_photons = photons.iloc[indices]
            time_diff = np.abs(candidate_photons['t'] - et) * 1e9
            spatial_diff = np.sqrt((candidate_photons['x'] - ex) ** 2 +
                                   (candidate_photons['y'] - ey) ** 2)
            combined_diff = (time_diff / time_norm_ns) + (spatial_diff / spatial_norm_px)

            if n_photons == 1:
                min_idx = np.argmin(combined_diff)
                best_idx = indices[min_idx]
                if np.isnan(photons.loc[best_idx, 'assoc_event_id']):
                    com_dist = spatial_diff.iloc[min_idx]
                    if com_dist <= dSpace_px:
                        photons.loc[best_idx, 'assoc_event_id'] = eid
                        photons.loc[best_idx, 'assoc_x'] = ex
                        photons.loc[best_idx, 'assoc_y'] = ey
                        photons.loc[best_idx, 'assoc_t'] = et
                        photons.loc[best_idx, 'assoc_n'] = n_photons
                        photons.loc[best_idx, 'assoc_PSD'] = ev.get('PSD', 0)
                        photons.loc[best_idx, 'time_diff_ns'] = time_diff.iloc[min_idx]
                        photons.loc[best_idx, 'spatial_diff_px'] = com_dist
                        photons.loc[best_idx, 'assoc_com_dist'] = com_dist
                        photons.loc[best_idx, 'assoc_status'] = 'cog_match'
            else:
                candidate_indices = np.argsort(combined_diff)[:n_photons]
                selected_indices = indices[candidate_indices]
                selected_x = candidate_photons.iloc[candidate_indices]['x']
                selected_y = candidate_photons.iloc[candidate_indices]['y']

                if not np.any(np.isnan(selected_x)) and not np.any(np.isnan(selected_y)):
                    com_x = np.mean(selected_x)
                    com_y = np.mean(selected_y)
                    com_dist = np.sqrt((com_x - ex) ** 2 + (com_y - ey) ** 2)
                    if com_dist <= dSpace_px:
                        for i, diff_idx in enumerate(candidate_indices):
                            idx = selected_indices[i]
                            if np.isnan(photons.loc[idx, 'assoc_event_id']):
                                photons.loc[idx, 'assoc_event_id'] = eid
                                photons.loc[idx, 'assoc_x'] = ex
                                photons.loc[idx, 'assoc_y'] = ey
                                photons.loc[idx, 'assoc_t'] = et
                                photons.loc[idx, 'assoc_n'] = n_photons
                                photons.loc[idx, 'assoc_PSD'] = ev.get('PSD', 0)
                                photons.loc[idx, 'time_diff_ns'] = time_diff.iloc[diff_idx]
                                photons.loc[idx, 'spatial_diff_px'] = spatial_diff.iloc[diff_idx]
                                photons.loc[idx, 'assoc_com_dist'] = com_dist
                                photons.loc[idx, 'assoc_status'] = 'cog_match'

        photons['assoc_status'] = photons['assoc_status'].astype('category')

        # Compute and store statistics
        matched_photons = photons['assoc_event_id'].notna().sum()
        total_photons = len(photons)

        # Count how many events were matched
        matched_event_ids = photons[photons['assoc_event_id'].notna()]['assoc_event_id'].unique()
        matched_events = len(matched_event_ids)
        total_events = len(events)

        # Quality metrics
        quality_stats = {'exact_n': 0, 'n_mismatch': 0, 'exact_com': 0, 'good_com': 0,
                        'acceptable_com': 0, 'poor_com': 0}

        # For each matched event, check quality
        for event_id in matched_event_ids:
            event_photons = photons[photons['assoc_event_id'] == event_id]
            actual_n = len(event_photons)

            # Get predicted n from first photon's assoc_n
            predicted_n = int(event_photons.iloc[0]['assoc_n'])

            # Check if photon count matches prediction
            if actual_n == predicted_n:
                quality_stats['exact_n'] += 1
            else:
                quality_stats['n_mismatch'] += 1

            # Check CoM distance quality (from first photon)
            com_dist = event_photons.iloc[0]['assoc_com_dist']
            if com_dist <= 0.1:  # Within 0.1 pixel
                quality_stats['exact_com'] += 1
            elif com_dist <= dSpace_px * 0.3:  # Within 30% of search radius
                quality_stats['good_com'] += 1
            elif com_dist <= dSpace_px * 0.5:  # Within 50% of search radius
                quality_stats['acceptable_com'] += 1
            else:  # Within search radius but >50%
                quality_stats['poor_com'] += 1

        # Store statistics as instance variable
        self.last_photon_event_stats = {
            'matched_photons': matched_photons,
            'total_photons': total_photons,
            'matched_events': matched_events,
            'total_events': total_events,
            'quality': quality_stats
        }

        # Print statistics if requested
        if verbosity >= 2:  # Only print detailed stats at verbosity 2
            print(f"✅ Photon-Event Association Results:")
            print(f"   Photons: {matched_photons:,} / {total_photons:,} matched ({100 * matched_photons / total_photons:.1f}%)")
            print(f"   Events:  {matched_events:,} / {total_events:,} matched ({100 * matched_events / total_events:.1f}%)")

            # Show quality statistics
            if matched_events > 0:
                print(f"   Association Quality:")
                print(f"      Photon count matches ev/n: {quality_stats['exact_n']:,} ({100 * quality_stats['exact_n'] / matched_events:.1f}%)")
                if quality_stats['n_mismatch'] > 0:
                    print(f"      Photon count mismatch:     {quality_stats['n_mismatch']:,} ({100 * quality_stats['n_mismatch'] / matched_events:.1f}%)")
                print(f"   Center-of-Mass Match Quality:")
                print(f"      Exact (≤0.1px):     {quality_stats['exact_com']:,} ({100 * quality_stats['exact_com'] / matched_events:.1f}%)")
                print(f"      Good (≤30% radius): {quality_stats['good_com']:,} ({100 * quality_stats['good_com'] / matched_events:.1f}%)")
                print(f"      Acceptable (≤50%):  {quality_stats['acceptable_com']:,} ({100 * quality_stats['acceptable_com'] / matched_events:.1f}%)")
                if quality_stats['poor_com'] > 0:
                    print(f"      Poor (>50%):        {quality_stats['poor_com']:,} ({100 * quality_stats['poor_com'] / matched_events:.1f}%)")

        return photons

    def _associate_photons_to_events_simple_window(
        self, photons_df, events_df, dSpace_px, max_time_s, verbosity
    ):
        """
        Associate photons to events using a simple forward time-window approach.

        This method is optimized for speed when photons and events are almost sorted by time and windows are small.
        For each event, it considers photons in [event_t, event_t + max_time_s], selects the n spatially closest photons,
        computes the center-of-mass (CoG) distance (or single distance for n=1), and assigns only if <= dSpace_px.
        Adds 'assoc_com_dist' for the CoG distance and 'assoc_status' as categorical ('cog_match' if assigned).

        Args:
            photons_df (pd.DataFrame): Photon DataFrame with 'x', 'y', 't' columns.
            events_df (pd.DataFrame): Event DataFrame with 'x', 'y', 't', 'n', 'PSD' columns.
            dSpace_px (float): Maximum allowed center-of-mass (or single photon) distance in pixels.
            max_time_s (float): Maximum time duration in seconds for the forward window.
            verbosity (int): Verbosity level (0=silent, 1=summary, 2=debug).

        Returns:
            pd.DataFrame: Associated photon DataFrame with added columns including 'assoc_com_dist' and 'assoc_status'.
        """
        if max_time_s is None:
            max_time_s = 500e-9  # Default to 500 ns if not provided
            if verbosity >= 2:
                print(f"Using default max_time_s: {max_time_s:.2e} s")

        photons = photons_df.copy()
        events = events_df.copy()

        # Init association columns
        photons['assoc_event_id'] = np.nan
        photons['assoc_x'] = np.nan
        photons['assoc_y'] = np.nan
        photons['assoc_t'] = np.nan
        photons['assoc_n'] = 0
        photons['assoc_PSD'] = 0
        photons['time_diff_ns'] = np.nan
        photons['spatial_diff_px'] = np.nan
        photons['assoc_com_dist'] = np.nan
        photons['assoc_status'] = np.nan

        # Ensure sorted by time
        photons = photons.sort_values('t').reset_index(drop=True)
        events = events.sort_values('t').reset_index(drop=True)
        events['event_id'] = events.index + 1

        # Convert to numpy for fast iteration
        p_t = photons['t'].to_numpy()
        p_x = photons['x'].to_numpy()
        p_y = photons['y'].to_numpy()

        left = 0
        n_total = len(photons)

        for _, ev in tqdm(events.iterrows(), total=len(events), desc="Associating events", disable=(verbosity == 0)):
            et, ex, ey, eid, n = ev['t'], ev['x'], ev['y'], ev['event_id'], int(ev['n'])
            psd = ev.get('PSD', 0)

            # Slide left to photons t >= et
            while left < n_total and p_t[left] < et:
                left += 1

            # Right to t <= et + max_time_s
            right = left
            while right < n_total and p_t[right] <= et + max_time_s:
                right += 1

            if right - left < n:
                continue  # Not enough photons in window

            sub_idx = np.arange(left, right)
            sub_x = p_x[sub_idx]
            sub_y = p_y[sub_idx]
            sub_t = p_t[sub_idx]

            spatial_diffs = np.sqrt((sub_x - ex)**2 + (sub_y - ey)**2)
            sort_indices = np.argsort(spatial_diffs)
            top_n_indices = sort_indices[:n]

            selected_x = sub_x[top_n_indices]
            selected_y = sub_y[top_n_indices]

            if n == 1:
                com_dist = spatial_diffs[top_n_indices[0]]
            else:
                com_x = np.mean(selected_x)
                com_y = np.mean(selected_y)
                com_dist = np.sqrt((com_x - ex)**2 + (com_y - ey)**2)

            if com_dist > dSpace_px:
                continue

            # Assign if CoG matches
            selected_global_idx = sub_idx[top_n_indices]
            for i, loc_idx in enumerate(selected_global_idx):
                # Skip if already assigned
                if np.isnan(photons.loc[loc_idx, 'assoc_event_id']):
                    photons.loc[loc_idx, 'assoc_event_id'] = eid
                    photons.loc[loc_idx, 'assoc_x'] = ex
                    photons.loc[loc_idx, 'assoc_y'] = ey
                    photons.loc[loc_idx, 'assoc_t'] = et
                    photons.loc[loc_idx, 'assoc_n'] = n
                    photons.loc[loc_idx, 'assoc_PSD'] = psd
                    photons.loc[loc_idx, 'time_diff_ns'] = (sub_t[top_n_indices[i]] - et) * 1e9
                    photons.loc[loc_idx, 'spatial_diff_px'] = spatial_diffs[top_n_indices[i]]
                    photons.loc[loc_idx, 'assoc_com_dist'] = com_dist
                    photons.loc[loc_idx, 'assoc_status'] = 'cog_match'

        photons['assoc_status'] = photons['assoc_status'].astype('category')

        # Compute and store statistics
        matched_photons = photons['assoc_event_id'].notna().sum()
        total_photons = len(photons)

        # Count how many events were matched
        matched_event_ids = photons[photons['assoc_event_id'].notna()]['assoc_event_id'].unique()
        matched_events = len(matched_event_ids)
        total_events = len(events)

        # Quality metrics
        quality_stats = {'exact_n': 0, 'n_mismatch': 0, 'exact_com': 0, 'good_com': 0,
                        'acceptable_com': 0, 'poor_com': 0}

        # For each matched event, check quality
        for event_id in matched_event_ids:
            event_photons = photons[photons['assoc_event_id'] == event_id]
            actual_n = len(event_photons)

            # Get predicted n from first photon's assoc_n
            predicted_n = int(event_photons.iloc[0]['assoc_n'])

            # Check if photon count matches prediction
            if actual_n == predicted_n:
                quality_stats['exact_n'] += 1
            else:
                quality_stats['n_mismatch'] += 1

            # Check CoM distance quality (from first photon)
            com_dist = event_photons.iloc[0]['assoc_com_dist']
            if com_dist <= 0.1:  # Within 0.1 pixel
                quality_stats['exact_com'] += 1
            elif com_dist <= dSpace_px * 0.3:  # Within 30% of search radius
                quality_stats['good_com'] += 1
            elif com_dist <= dSpace_px * 0.5:  # Within 50% of search radius
                quality_stats['acceptable_com'] += 1
            else:  # Within search radius but >50%
                quality_stats['poor_com'] += 1

        # Store statistics as instance variable
        self.last_photon_event_stats = {
            'matched_photons': matched_photons,
            'total_photons': total_photons,
            'matched_events': matched_events,
            'total_events': total_events,
            'quality': quality_stats
        }

        # Print statistics if requested
        if verbosity >= 2:  # Only print detailed stats at verbosity 2
            print(f"✅ Photon-Event Association Results:")
            print(f"   Photons: {matched_photons:,} / {total_photons:,} matched ({100 * matched_photons / total_photons:.1f}%)")
            print(f"   Events:  {matched_events:,} / {total_events:,} matched ({100 * matched_events / total_events:.1f}%)")

            # Show quality statistics
            if matched_events > 0:
                print(f"   Association Quality:")
                print(f"      Photon count matches ev/n: {quality_stats['exact_n']:,} ({100 * quality_stats['exact_n'] / matched_events:.1f}%)")
                if quality_stats['n_mismatch'] > 0:
                    print(f"      Photon count mismatch:     {quality_stats['n_mismatch']:,} ({100 * quality_stats['n_mismatch'] / matched_events:.1f}%)")
                print(f"   Center-of-Mass Match Quality:")
                print(f"      Exact (≤0.1px):     {quality_stats['exact_com']:,} ({100 * quality_stats['exact_com'] / matched_events:.1f}%)")
                print(f"      Good (≤30% radius): {quality_stats['good_com']:,} ({100 * quality_stats['good_com'] / matched_events:.1f}%)")
                print(f"      Acceptable (≤50%):  {quality_stats['acceptable_com']:,} ({100 * quality_stats['acceptable_com'] / matched_events:.1f}%)")
                if quality_stats['poor_com'] > 0:
                    print(f"      Poor (>50%):        {quality_stats['poor_com']:,} ({100 * quality_stats['poor_com'] / matched_events:.1f}%)")

        return photons

    def _associate_pixels_to_photons_simple(self, pixels_df, photons_df, max_dist_px=5.0, max_time_ns=500, verbosity=0):
        """
        Associate pixels to photons using a simple spatial-temporal proximity method.

        For each photon, finds pixels within a time window and spatial radius, then associates
        the closest pixels (by spatial distance) to that photon.

        Args:
            pixels_df (pd.DataFrame): Pixel DataFrame with 'x', 'y', 't', 'tot', 'tof' columns.
            photons_df (pd.DataFrame): Photon DataFrame with 'x', 'y', 't', 'tof' columns.
            max_dist_px (float): Maximum spatial distance in pixels for association.
            max_time_ns (float): Maximum time difference in nanoseconds for association.
            verbosity (int): Verbosity level (0=silent, 1=summary, 2=debug).

        Returns:
            pd.DataFrame: Pixel DataFrame with added association columns.
        """
        if pixels_df is None or photons_df is None or len(pixels_df) == 0 or len(photons_df) == 0:
            if verbosity >= 1:
                print("Warning: Empty pixels or photons dataframe, skipping pixel-photon association")
            return pixels_df

        pixels = pixels_df.copy()
        photons = photons_df.copy()

        # Initialize association columns
        pixels['assoc_photon_id'] = np.nan
        pixels['assoc_phot_x'] = np.nan
        pixels['assoc_phot_y'] = np.nan
        pixels['assoc_phot_t'] = np.nan
        pixels['pixel_com_dist'] = np.nan  # CoM distance from pixel cluster to photon

        # Ensure sorted by time
        pixels = pixels.sort_values('t').reset_index(drop=True)
        photons = photons.sort_values('t').reset_index(drop=True)
        photons['photon_id'] = photons.index + 1

        # Convert to numpy for fast iteration
        pix_t = pixels['t'].to_numpy()
        pix_x = pixels['x'].to_numpy()
        pix_y = pixels['y'].to_numpy()

        max_time_s = max_time_ns / 1e9
        left = 0
        n_pixels_total = len(pixels)

        # Track CoM quality statistics
        com_quality_stats = {'exact': 0, 'good': 0, 'acceptable': 0, 'poor': 0, 'failed': 0}

        for _, phot in tqdm(photons.iterrows(), total=len(photons), desc="Associating pixels to photons", disable=(verbosity == 0)):
            phot_t, phot_x, phot_y, phot_id = phot['t'], phot['x'], phot['y'], phot['photon_id']

            # Slide left to pixels with t >= phot_t (photon time is the FIRST pixel time)
            # Pixels can only come AT or AFTER the first pixel
            while left < n_pixels_total and pix_t[left] < phot_t:
                left += 1

            # Right to t <= phot_t + max_time_s
            right = left
            while right < n_pixels_total and pix_t[right] <= phot_t + max_time_s:
                right += 1

            if right == left:
                continue  # No pixels in time window

            # Get pixels in time window
            sub_idx = np.arange(left, right)
            sub_x = pix_x[sub_idx]
            sub_y = pix_y[sub_idx]
            sub_t = pix_t[sub_idx]

            # Calculate spatial distances
            spatial_diffs = np.sqrt((sub_x - phot_x)**2 + (sub_y - phot_y)**2)

            # Filter by max distance
            valid_mask = spatial_diffs <= max_dist_px
            if not valid_mask.any():
                continue

            # Get valid pixel indices and data
            valid_sub_idx = sub_idx[valid_mask]
            valid_spatial_diffs = spatial_diffs[valid_mask]
            valid_time_diffs = (sub_t[valid_mask] - phot_t) * 1e9
            valid_x = sub_x[valid_mask]
            valid_y = sub_y[valid_mask]

            # Filter out already assigned pixels
            unassigned_mask = np.array([np.isnan(pixels.loc[idx, 'assoc_photon_id']) for idx in valid_sub_idx])
            if not unassigned_mask.any():
                continue

            # Get unassigned pixels
            unassigned_idx = valid_sub_idx[unassigned_mask]
            unassigned_x = valid_x[unassigned_mask]
            unassigned_y = valid_y[unassigned_mask]
            unassigned_spatial_diffs = valid_spatial_diffs[unassigned_mask]
            unassigned_time_diffs = valid_time_diffs[unassigned_mask]

            # Find the best subset of pixels whose center of mass matches the photon
            # Use iterative refinement: start with all, remove outliers, converge
            best_subset_mask = np.ones(len(unassigned_idx), dtype=bool)

            for iteration in range(10):  # Max 10 iterations
                # Compute center of mass of current subset
                com_x = unassigned_x[best_subset_mask].mean()
                com_y = unassigned_y[best_subset_mask].mean()

                # Distance from photon to center of mass
                com_dist = np.sqrt((com_x - phot_x)**2 + (com_y - phot_y)**2)

                # If CoM is close enough, we're done
                if com_dist <= max_dist_px * 0.5:  # CoM should be within half the pixel search radius
                    break

                # Otherwise, remove pixels that are pulling CoM away from photon
                # Calculate how much each pixel pulls the CoM
                subset_x = unassigned_x[best_subset_mask]
                subset_y = unassigned_y[best_subset_mask]

                # Distance of each pixel from photon
                pixel_dists_from_photon = np.sqrt((subset_x - phot_x)**2 + (subset_y - phot_y)**2)

                # Remove the pixel farthest from the photon
                if len(pixel_dists_from_photon) <= 1:
                    break

                worst_pixel_local_idx = np.argmax(pixel_dists_from_photon)
                # Convert local index to mask index
                mask_indices = np.where(best_subset_mask)[0]
                best_subset_mask[mask_indices[worst_pixel_local_idx]] = False

            # Compute final CoM and track quality
            if best_subset_mask.any():
                final_com_x = unassigned_x[best_subset_mask].mean()
                final_com_y = unassigned_y[best_subset_mask].mean()
                final_com_dist = np.sqrt((final_com_x - phot_x)**2 + (final_com_y - phot_y)**2)

                # Categorize CoM quality
                if final_com_dist <= 0.1:  # Within 0.1 pixel
                    com_quality_stats['exact'] += 1
                elif final_com_dist <= max_dist_px * 0.3:  # Within 30% of search radius
                    com_quality_stats['good'] += 1
                elif final_com_dist <= max_dist_px * 0.5:  # Within 50% of search radius
                    com_quality_stats['acceptable'] += 1
                elif final_com_dist <= max_dist_px:  # Within search radius
                    com_quality_stats['poor'] += 1
                else:
                    com_quality_stats['failed'] += 1

            # Assign the best subset to this photon
            final_idx = unassigned_idx[best_subset_mask]
            final_spatial_diffs = unassigned_spatial_diffs[best_subset_mask]
            final_time_diffs = unassigned_time_diffs[best_subset_mask]

            if len(final_idx) > 0:  # Only assign if we found at least one pixel
                for i, pix_idx in enumerate(final_idx):
                    pixels.loc[pix_idx, 'assoc_photon_id'] = phot_id
                    pixels.loc[pix_idx, 'assoc_phot_x'] = phot_x
                    pixels.loc[pix_idx, 'assoc_phot_y'] = phot_y
                    pixels.loc[pix_idx, 'assoc_phot_t'] = phot_t
                    pixels.loc[pix_idx, 'pixel_com_dist'] = final_com_dist  # Same CoM dist for all pixels in cluster

        # Always compute statistics (store for later use)
        matched_pixels = pixels['assoc_photon_id'].notna().sum()
        total_pixels = len(pixels)

        # Count how many photons were matched
        matched_photon_ids = pixels[pixels['assoc_photon_id'].notna()]['assoc_photon_id'].unique()
        matched_photons = len(matched_photon_ids)
        total_photons = len(photons)

        # Store statistics as instance variables
        self.last_assoc_stats = {
            'matched_pixels': matched_pixels,
            'total_pixels': total_pixels,
            'matched_photons': matched_photons,
            'total_photons': total_photons,
            'com_quality': com_quality_stats.copy()
        }

        # Print statistics if requested
        if verbosity >= 2:  # Only print detailed stats at verbosity 2
            print(f"✅ Pixel-Photon Association Results:")
            print(f"   Pixels:  {matched_pixels:,} / {total_pixels:,} matched ({100 * matched_pixels / total_pixels:.1f}%)")
            print(f"   Photons: {matched_photons:,} / {total_photons:,} matched ({100 * matched_photons / total_photons:.1f}%)")

            # Show CoM quality statistics
            total_processed = sum(com_quality_stats.values())
            if total_processed > 0:
                print(f"   Center-of-Mass Match Quality:")
                print(f"      Exact (≤0.1px):     {com_quality_stats['exact']:,} ({100 * com_quality_stats['exact'] / total_processed:.1f}%)")
                print(f"      Good (≤30% radius): {com_quality_stats['good']:,} ({100 * com_quality_stats['good'] / total_processed:.1f}%)")
                print(f"      Acceptable (≤50%):  {com_quality_stats['acceptable']:,} ({100 * com_quality_stats['acceptable'] / total_processed:.1f}%)")
                print(f"      Poor (≤100%):       {com_quality_stats['poor']:,} ({100 * com_quality_stats['poor'] / total_processed:.1f}%)")
                if com_quality_stats['failed'] > 0:
                    print(f"      Failed (>100%):     {com_quality_stats['failed']:,} ({100 * com_quality_stats['failed'] / total_processed:.1f}%)")

            if verbosity >= 2 and matched_photons > 0:
                # Show distribution of pixels per photon
                pixels_per_photon = pixels[pixels['assoc_photon_id'].notna()].groupby('assoc_photon_id').size()
                print(f"   Pixels per photon: min={pixels_per_photon.min()}, "
                      f"mean={pixels_per_photon.mean():.1f}, "
                      f"median={pixels_per_photon.median():.0f}, "
                      f"max={pixels_per_photon.max()}")

        return pixels

    def _associate_pixels_to_photons_kdtree(self, pixels_df, photons_df, max_dist_px=5.0, max_time_ns=500, verbosity=0):
        """
        Associate pixels to photons using kdtree for spatial queries with center-of-mass matching.

        This method uses the same logic as the simple method but with a kdtree for faster
        candidate pixel finding. For each photon:
        1. Use kdtree to find nearby pixels within max_dist_px
        2. Filter by time window [phot_t, phot_t + max_time_ns]
        3. Apply iterative center-of-mass refinement
        4. Assign only pixels whose CoM matches the photon position

        Args:
            pixels_df (pd.DataFrame): Pixel DataFrame with 'x', 'y', 't', 'tot', 'tof' columns.
            photons_df (pd.DataFrame): Photon DataFrame with 'x', 'y', 't', 'tof' columns.
            max_dist_px (float): Maximum spatial distance in pixels for association.
            max_time_ns (float): Maximum time difference in nanoseconds for association.
            verbosity (int): Verbosity level (0=silent, 1=summary, 2=debug).

        Returns:
            pd.DataFrame: Pixel DataFrame with added association columns.
        """
        if pixels_df is None or photons_df is None or len(pixels_df) == 0 or len(photons_df) == 0:
            if verbosity >= 1:
                print("Warning: Empty pixels or photons dataframe, skipping pixel-photon association")
            return pixels_df

        pixels = pixels_df.copy()
        photons = photons_df.copy()

        # Initialize association columns
        pixels['assoc_photon_id'] = np.nan
        pixels['assoc_phot_x'] = np.nan
        pixels['assoc_phot_y'] = np.nan
        pixels['assoc_phot_t'] = np.nan
        pixels['pixel_com_dist'] = np.nan  # CoM distance from pixel cluster to photon

        # Sort by time
        pixels = pixels.sort_values('t').reset_index(drop=True)
        photons = photons.sort_values('t').reset_index(drop=True)
        photons['photon_id'] = photons.index + 1

        # Build kdtree for pixel spatial coordinates
        pixel_coords = np.column_stack([pixels['x'].to_numpy(), pixels['y'].to_numpy()])
        pixel_tree = cKDTree(pixel_coords)

        max_time_s = max_time_ns / 1e9

        # Track CoM quality statistics
        com_quality_stats = {'exact': 0, 'good': 0, 'acceptable': 0, 'poor': 0, 'failed': 0}

        for _, phot in tqdm(photons.iterrows(), total=len(photons), desc="Associating pixels to photons (kdtree)", disable=(verbosity == 0)):
            phot_t, phot_x, phot_y, phot_id = phot['t'], phot['x'], phot['y'], phot['photon_id']

            # Use kdtree to find all pixels within spatial radius
            candidate_indices = pixel_tree.query_ball_point([phot_x, phot_y], max_dist_px)

            if len(candidate_indices) == 0:
                continue

            # Filter by time window: [phot_t, phot_t + max_time_s]
            # Photon time is the FIRST pixel time, so pixels can only come AT or AFTER
            candidate_pixels = pixels.iloc[candidate_indices]
            time_mask = (candidate_pixels['t'] >= phot_t) & (candidate_pixels['t'] <= phot_t + max_time_s)

            if not time_mask.any():
                continue

            # Get pixels in time and spatial window
            valid_pixels = candidate_pixels[time_mask]
            valid_indices = np.array(candidate_indices)[time_mask.to_numpy()]

            # Filter out already assigned pixels
            unassigned_mask = valid_pixels['assoc_photon_id'].isna().to_numpy()
            if not unassigned_mask.any():
                continue

            # Get unassigned pixels
            unassigned_idx = valid_indices[unassigned_mask]
            unassigned_pixels = valid_pixels[unassigned_mask]
            unassigned_x = unassigned_pixels['x'].to_numpy()
            unassigned_y = unassigned_pixels['y'].to_numpy()
            unassigned_t = unassigned_pixels['t'].to_numpy()

            # Calculate spatial and time differences
            unassigned_spatial_diffs = np.sqrt((unassigned_x - phot_x)**2 + (unassigned_y - phot_y)**2)
            unassigned_time_diffs = (unassigned_t - phot_t) * 1e9

            # Find the best subset of pixels whose center of mass matches the photon
            # Use iterative refinement: start with all, remove outliers, converge
            best_subset_mask = np.ones(len(unassigned_idx), dtype=bool)

            for iteration in range(10):  # Max 10 iterations
                # Compute center of mass of current subset
                com_x = unassigned_x[best_subset_mask].mean()
                com_y = unassigned_y[best_subset_mask].mean()

                # Distance from photon to center of mass
                com_dist = np.sqrt((com_x - phot_x)**2 + (com_y - phot_y)**2)

                # If CoM is close enough, we're done
                if com_dist <= max_dist_px * 0.5:  # CoM should be within half the pixel search radius
                    break

                # Otherwise, remove pixels that are pulling CoM away from photon
                subset_x = unassigned_x[best_subset_mask]
                subset_y = unassigned_y[best_subset_mask]

                # Distance of each pixel from photon
                pixel_dists_from_photon = np.sqrt((subset_x - phot_x)**2 + (subset_y - phot_y)**2)

                # Remove the pixel farthest from the photon
                if len(pixel_dists_from_photon) <= 1:
                    break

                worst_pixel_local_idx = np.argmax(pixel_dists_from_photon)
                # Convert local index to mask index
                mask_indices = np.where(best_subset_mask)[0]
                best_subset_mask[mask_indices[worst_pixel_local_idx]] = False

            # Compute final CoM and track quality
            if best_subset_mask.any():
                final_com_x = unassigned_x[best_subset_mask].mean()
                final_com_y = unassigned_y[best_subset_mask].mean()
                final_com_dist = np.sqrt((final_com_x - phot_x)**2 + (final_com_y - phot_y)**2)

                # Categorize CoM quality
                if final_com_dist <= 0.1:  # Within 0.1 pixel
                    com_quality_stats['exact'] += 1
                elif final_com_dist <= max_dist_px * 0.3:  # Within 30% of search radius
                    com_quality_stats['good'] += 1
                elif final_com_dist <= max_dist_px * 0.5:  # Within 50% of search radius
                    com_quality_stats['acceptable'] += 1
                elif final_com_dist <= max_dist_px:  # Within search radius
                    com_quality_stats['poor'] += 1
                else:
                    com_quality_stats['failed'] += 1

            # Assign the best subset to this photon
            final_idx = unassigned_idx[best_subset_mask]

            if len(final_idx) > 0:  # Only assign if we found at least one pixel
                for pix_idx in final_idx:
                    pixels.loc[pix_idx, 'assoc_photon_id'] = phot_id
                    pixels.loc[pix_idx, 'assoc_phot_x'] = phot_x
                    pixels.loc[pix_idx, 'assoc_phot_y'] = phot_y
                    pixels.loc[pix_idx, 'assoc_phot_t'] = phot_t
                    pixels.loc[pix_idx, 'pixel_com_dist'] = final_com_dist  # Same CoM dist for all pixels in cluster

        # Always compute statistics (store for later use)
        matched_pixels = pixels['assoc_photon_id'].notna().sum()
        total_pixels = len(pixels)

        # Count how many photons were matched
        matched_photon_ids = pixels[pixels['assoc_photon_id'].notna()]['assoc_photon_id'].unique()
        matched_photons = len(matched_photon_ids)
        total_photons = len(photons)

        # Store statistics as instance variables
        self.last_assoc_stats = {
            'matched_pixels': matched_pixels,
            'total_pixels': total_pixels,
            'matched_photons': matched_photons,
            'total_photons': total_photons,
            'com_quality': com_quality_stats.copy()
        }

        # Print statistics if requested
        if verbosity >= 2:  # Only print detailed stats at verbosity 2
            print(f"✅ Pixel-Photon Association Results:")
            print(f"   Pixels:  {matched_pixels:,} / {total_pixels:,} matched ({100 * matched_pixels / total_pixels:.1f}%)")
            print(f"   Photons: {matched_photons:,} / {total_photons:,} matched ({100 * matched_photons / total_photons:.1f}%)")

            # Show CoM quality statistics
            total_processed = sum(com_quality_stats.values())
            if total_processed > 0:
                print(f"   Center-of-Mass Match Quality:")
                print(f"      Exact (≤0.1px):     {com_quality_stats['exact']:,} ({100 * com_quality_stats['exact'] / total_processed:.1f}%)")
                print(f"      Good (≤30% radius): {com_quality_stats['good']:,} ({100 * com_quality_stats['good'] / total_processed:.1f}%)")
                print(f"      Acceptable (≤50%):  {com_quality_stats['acceptable']:,} ({100 * com_quality_stats['acceptable'] / total_processed:.1f}%)")
                print(f"      Poor (≤100%):       {com_quality_stats['poor']:,} ({100 * com_quality_stats['poor'] / total_processed:.1f}%)")
                if com_quality_stats['failed'] > 0:
                    print(f"      Failed (>100%):     {com_quality_stats['failed']:,} ({100 * com_quality_stats['failed'] / total_processed:.1f}%)")

            if verbosity >= 2 and matched_photons > 0:
                # Show distribution of pixels per photon
                pixels_per_photon = pixels[pixels['assoc_photon_id'].notna()].groupby('assoc_photon_id').size()
                print(f"   Pixels per photon: min={pixels_per_photon.min()}, "
                      f"mean={pixels_per_photon.mean():.1f}, "
                      f"median={pixels_per_photon.median():.0f}, "
                      f"max={pixels_per_photon.max()}")

        return pixels

    def _associate_pixels_to_photons_mystic(self, pixels_df, photons_df, max_dist_px=5.0, max_time_ns=500,
                                            time_weight=1.0, cog_weight=1.0, min_pixels=1, verbosity=0):
        """
        Associate pixels to photons using mystic optimization framework.

        This method formulates the pixel-to-photon association as a constrained optimization problem.
        For each photon, it finds the optimal subset of nearby pixels by minimizing:
          - The distance between pixel cluster center-of-gravity and photon position
          - A penalty for time mismatch (first pixel time should match photon time)

        Args:
            pixels_df (pd.DataFrame): Pixel DataFrame with 'x', 'y', 't', 'tot' columns.
            photons_df (pd.DataFrame): Photon DataFrame with 'x', 'y', 't' columns.
            max_dist_px (float): Maximum spatial distance in pixels for candidate selection.
            max_time_ns (float): Maximum time difference in nanoseconds for candidate selection.
            time_weight (float): Weight for time mismatch penalty in objective function.
            cog_weight (float): Weight for center-of-gravity mismatch in objective function.
            min_pixels (int): Minimum number of pixels required to form a valid cluster.
            verbosity (int): Verbosity level (0=silent, 1=summary, 2=debug).

        Returns:
            pd.DataFrame: Pixel DataFrame with added association columns.
        """
        try:
            from mystic.solvers import fmin_powell, diffev2
            from mystic.monitors import VerboseMonitor
        except ImportError:
            raise ImportError("mystic package is required for this method. Install with: pip install mystic")

        if pixels_df is None or photons_df is None or len(pixels_df) == 0 or len(photons_df) == 0:
            if verbosity >= 1:
                print("Warning: Empty pixels or photons dataframe, skipping pixel-photon association")
            return pixels_df

        pixels = pixels_df.copy()
        photons = photons_df.copy()

        # Initialize association columns
        pixels['assoc_photon_id'] = np.nan
        pixels['assoc_phot_x'] = np.nan
        pixels['assoc_phot_y'] = np.nan
        pixels['assoc_phot_t'] = np.nan
        pixels['pixel_com_dist'] = np.nan

        # Sort by time
        pixels = pixels.sort_values('t').reset_index(drop=True)
        photons = photons.sort_values('t').reset_index(drop=True)
        photons['photon_id'] = photons.index + 1

        # Build kdtree for pixel spatial coordinates
        pixel_coords = np.column_stack([pixels['x'].to_numpy(), pixels['y'].to_numpy()])
        pixel_tree = cKDTree(pixel_coords)

        max_time_s = max_time_ns / 1e9

        # Track optimization statistics
        opt_stats = {'success': 0, 'fallback': 0, 'failed': 0}
        com_quality_stats = {'exact': 0, 'good': 0, 'acceptable': 0, 'poor': 0, 'failed': 0}

        for _, phot in tqdm(photons.iterrows(), total=len(photons),
                           desc="Associating pixels to photons (mystic)", disable=(verbosity == 0)):
            phot_t, phot_x, phot_y, phot_id = phot['t'], phot['x'], phot['y'], phot['photon_id']

            # Use kdtree to find all pixels within spatial radius
            candidate_indices = pixel_tree.query_ball_point([phot_x, phot_y], max_dist_px)

            if len(candidate_indices) == 0:
                continue

            # Filter by time window: [phot_t, phot_t + max_time_s]
            candidate_pixels = pixels.iloc[candidate_indices]
            time_mask = (candidate_pixels['t'] >= phot_t) & (candidate_pixels['t'] <= phot_t + max_time_s)

            if not time_mask.any():
                continue

            # Get pixels in time and spatial window
            valid_pixels = candidate_pixels[time_mask]
            valid_indices = np.array(candidate_indices)[time_mask.to_numpy()]

            # Filter out already assigned pixels
            unassigned_mask = valid_pixels['assoc_photon_id'].isna().to_numpy()
            if not unassigned_mask.any():
                continue

            # Get unassigned pixels
            unassigned_idx = valid_indices[unassigned_mask]
            unassigned_pixels = valid_pixels[unassigned_mask]
            unassigned_x = unassigned_pixels['x'].to_numpy()
            unassigned_y = unassigned_pixels['y'].to_numpy()
            unassigned_t = unassigned_pixels['t'].to_numpy()

            n_candidates = len(unassigned_idx)

            # For very small candidate sets, use simple assignment
            if n_candidates <= min_pixels:
                # Assign all candidates
                for i, pix_idx in enumerate(unassigned_idx):
                    pixels.loc[pix_idx, 'assoc_photon_id'] = phot_id
                    pixels.loc[pix_idx, 'assoc_phot_x'] = phot_x
                    pixels.loc[pix_idx, 'assoc_phot_y'] = phot_y
                    pixels.loc[pix_idx, 'assoc_phot_t'] = phot_t
                com_x, com_y = unassigned_x.mean(), unassigned_y.mean()
                com_dist = np.sqrt((com_x - phot_x)**2 + (com_y - phot_y)**2)
                for pix_idx in unassigned_idx:
                    pixels.loc[pix_idx, 'pixel_com_dist'] = com_dist
                opt_stats['fallback'] += 1
                continue

            # Define objective function for mystic
            # Decision variables: weights w[i] in [0, 1] for each candidate pixel
            # Objective: minimize CoG distance + time penalty
            def objective(weights):
                weights = np.array(weights)
                w_sum = weights.sum()

                if w_sum < 0.1:  # Avoid division by zero
                    return 1e10

                # Weighted center of gravity
                cog_x = (weights * unassigned_x).sum() / w_sum
                cog_y = (weights * unassigned_y).sum() / w_sum

                # CoG distance from photon position
                cog_dist = np.sqrt((cog_x - phot_x)**2 + (cog_y - phot_y)**2)

                # Time penalty: first pixel (by weight) should have time close to photon time
                # Approximate "first pixel" as weighted minimum time
                # Use softmin: weighted average with exponential emphasis on early times
                time_diffs = (unassigned_t - phot_t) * 1e9  # Convert to ns
                # Penalize if weighted average time is far from photon time
                weighted_time = (weights * time_diffs).sum() / w_sum
                time_penalty = abs(weighted_time)  # Should be close to 0

                # Total objective
                obj = cog_weight * cog_dist**2 + time_weight * (time_penalty / max_time_ns)**2

                return obj

            # Constraint: enforce minimum total weight (at least min_pixels worth)
            def constraint(weights):
                weights = np.array(weights)
                # Ensure sum of weights is at least min_pixels
                w_sum = weights.sum()
                if w_sum < min_pixels:
                    # Scale up weights proportionally
                    scale = min_pixels / max(w_sum, 0.01)
                    weights = np.clip(weights * scale, 0, 1)
                return weights

            # Initial guess: all weights = 0.5
            x0 = np.full(n_candidates, 0.5)

            # Bounds: weights in [0, 1]
            lb = np.zeros(n_candidates)
            ub = np.ones(n_candidates)

            try:
                # Use Powell's method for optimization (fast for small problems)
                if n_candidates <= 20:
                    solution = fmin_powell(objective, x0, bounds=list(zip(lb, ub)),
                                          constraints=constraint, disp=False, gtol=1e-4)
                else:
                    # For larger problems, use differential evolution
                    solution = diffev2(objective, list(zip(lb, ub)), constraints=constraint,
                                       npop=min(20, n_candidates * 2), disp=False, gtol=50)

                optimal_weights = np.array(solution)

                # Threshold weights to get binary assignment
                threshold = 0.3
                assigned_mask = optimal_weights >= threshold

                if assigned_mask.sum() < min_pixels:
                    # Fall back to top-k by weight
                    top_k = min(min_pixels, n_candidates)
                    top_indices = np.argsort(optimal_weights)[-top_k:]
                    assigned_mask = np.zeros(n_candidates, dtype=bool)
                    assigned_mask[top_indices] = True

                opt_stats['success'] += 1

            except Exception as e:
                if verbosity >= 2:
                    print(f"Optimization failed for photon {phot_id}: {e}, using fallback")
                # Fallback: assign all candidates
                assigned_mask = np.ones(n_candidates, dtype=bool)
                opt_stats['failed'] += 1

            # Compute final CoM
            final_idx = unassigned_idx[assigned_mask]
            final_x = unassigned_x[assigned_mask]
            final_y = unassigned_y[assigned_mask]

            if len(final_idx) > 0:
                com_x = final_x.mean()
                com_y = final_y.mean()
                com_dist = np.sqrt((com_x - phot_x)**2 + (com_y - phot_y)**2)

                # Track quality
                if com_dist <= 0.1:
                    com_quality_stats['exact'] += 1
                elif com_dist <= max_dist_px * 0.3:
                    com_quality_stats['good'] += 1
                elif com_dist <= max_dist_px * 0.5:
                    com_quality_stats['acceptable'] += 1
                elif com_dist <= max_dist_px:
                    com_quality_stats['poor'] += 1
                else:
                    com_quality_stats['failed'] += 1

                # Assign pixels
                for pix_idx in final_idx:
                    pixels.loc[pix_idx, 'assoc_photon_id'] = phot_id
                    pixels.loc[pix_idx, 'assoc_phot_x'] = phot_x
                    pixels.loc[pix_idx, 'assoc_phot_y'] = phot_y
                    pixels.loc[pix_idx, 'assoc_phot_t'] = phot_t
                    pixels.loc[pix_idx, 'pixel_com_dist'] = com_dist

        # Store statistics
        matched_pixels = pixels['assoc_photon_id'].notna().sum()
        total_pixels = len(pixels)
        matched_photon_ids = pixels[pixels['assoc_photon_id'].notna()]['assoc_photon_id'].unique()
        matched_photons = len(matched_photon_ids)
        total_photons = len(photons)

        self.last_assoc_stats = {
            'matched_pixels': matched_pixels,
            'total_pixels': total_pixels,
            'matched_photons': matched_photons,
            'total_photons': total_photons,
            'com_quality': com_quality_stats.copy(),
            'optimization': opt_stats.copy()
        }

        if verbosity >= 1:
            print(f"✅ Pixel-Photon Association (mystic optimization):")
            print(f"   Pixels:  {matched_pixels:,} / {total_pixels:,} matched ({100 * matched_pixels / total_pixels:.1f}%)")
            print(f"   Photons: {matched_photons:,} / {total_photons:,} matched ({100 * matched_photons / total_photons:.1f}%)")
            print(f"   Optimization: {opt_stats['success']} success, {opt_stats['fallback']} fallback, {opt_stats['failed']} failed")

        if verbosity >= 2:
            total_processed = sum(com_quality_stats.values())
            if total_processed > 0:
                print(f"   Center-of-Mass Match Quality:")
                print(f"      Exact (≤0.1px):     {com_quality_stats['exact']:,} ({100 * com_quality_stats['exact'] / total_processed:.1f}%)")
                print(f"      Good (≤30% radius): {com_quality_stats['good']:,} ({100 * com_quality_stats['good'] / total_processed:.1f}%)")
                print(f"      Acceptable (≤50%):  {com_quality_stats['acceptable']:,} ({100 * com_quality_stats['acceptable'] / total_processed:.1f}%)")
                print(f"      Poor (≤100%):       {com_quality_stats['poor']:,} ({100 * com_quality_stats['poor'] / total_processed:.1f}%)")

        return pixels

    def _associate_photons_to_events_mystic(self, photons_df, events_df, max_dist_px=10.0, max_time_ns=500,
                                            time_weight=1.0, cog_weight=1.0, min_photons=1, verbosity=0):
        """
        Associate photons to events using mystic optimization framework.

        This method formulates the photon-to-event association as a constrained optimization problem.
        For each event, it finds the optimal subset of nearby photons by minimizing:
          - The distance between photon cluster center-of-gravity and event position
          - A penalty for time mismatch (first photon time should match event time)

        Args:
            photons_df (pd.DataFrame): Photon DataFrame with 'x', 'y', 't' columns.
            events_df (pd.DataFrame): Event DataFrame with 'x', 'y', 't', 'n', 'PSD' columns.
            max_dist_px (float): Maximum spatial distance in pixels for candidate selection.
            max_time_ns (float): Maximum time difference in nanoseconds for candidate selection.
            time_weight (float): Weight for time mismatch penalty in objective function.
            cog_weight (float): Weight for center-of-gravity mismatch in objective function.
            min_photons (int): Minimum number of photons required to form a valid cluster.
            verbosity (int): Verbosity level (0=silent, 1=summary, 2=debug).

        Returns:
            pd.DataFrame: Photon DataFrame with added association columns.
        """
        try:
            from mystic.solvers import fmin_powell, diffev2
        except ImportError:
            raise ImportError("mystic package is required for this method. Install with: pip install mystic")

        if photons_df is None or events_df is None or len(photons_df) == 0 or len(events_df) == 0:
            if verbosity >= 1:
                print("Warning: Empty photons or events dataframe, skipping photon-event association")
            return photons_df

        photons = photons_df.copy()
        events = events_df.copy()

        # Initialize association columns
        photons['assoc_event_id'] = np.nan
        photons['assoc_x'] = np.nan
        photons['assoc_y'] = np.nan
        photons['assoc_t'] = np.nan
        photons['assoc_n'] = np.nan
        photons['assoc_PSD'] = np.nan
        photons['assoc_com_dist'] = np.nan

        # Sort by time
        photons = photons.sort_values('t').reset_index(drop=True)
        events = events.sort_values('t').reset_index(drop=True)
        events['event_id'] = events.index + 1

        # Build kdtree for photon spatial coordinates
        photon_coords = np.column_stack([photons['x'].to_numpy(), photons['y'].to_numpy()])
        photon_tree = cKDTree(photon_coords)

        max_time_s = max_time_ns / 1e9

        # Track optimization statistics
        opt_stats = {'success': 0, 'fallback': 0, 'failed': 0}
        com_quality_stats = {'exact': 0, 'good': 0, 'acceptable': 0, 'poor': 0, 'failed': 0}

        for _, ev in tqdm(events.iterrows(), total=len(events),
                         desc="Associating photons to events (mystic)", disable=(verbosity == 0)):
            ev_t, ev_x, ev_y = ev['t'], ev['x'], ev['y']
            ev_n, ev_psd, ev_id = ev['n'], ev['PSD'], ev['event_id']

            # Use kdtree to find all photons within spatial radius
            candidate_indices = photon_tree.query_ball_point([ev_x, ev_y], max_dist_px)

            if len(candidate_indices) == 0:
                continue

            # Filter by time window
            candidate_photons = photons.iloc[candidate_indices]
            time_mask = (candidate_photons['t'] >= ev_t) & (candidate_photons['t'] <= ev_t + max_time_s)

            if not time_mask.any():
                continue

            valid_photons = candidate_photons[time_mask]
            valid_indices = np.array(candidate_indices)[time_mask.to_numpy()]

            # Filter out already assigned photons
            unassigned_mask = valid_photons['assoc_event_id'].isna().to_numpy()
            if not unassigned_mask.any():
                continue

            unassigned_idx = valid_indices[unassigned_mask]
            unassigned_photons = valid_photons[unassigned_mask]
            unassigned_x = unassigned_photons['x'].to_numpy()
            unassigned_y = unassigned_photons['y'].to_numpy()
            unassigned_t = unassigned_photons['t'].to_numpy()

            n_candidates = len(unassigned_idx)

            if n_candidates <= min_photons:
                # Assign all candidates
                for phot_idx in unassigned_idx:
                    photons.loc[phot_idx, 'assoc_event_id'] = ev_id
                    photons.loc[phot_idx, 'assoc_x'] = ev_x
                    photons.loc[phot_idx, 'assoc_y'] = ev_y
                    photons.loc[phot_idx, 'assoc_t'] = ev_t
                    photons.loc[phot_idx, 'assoc_n'] = ev_n
                    photons.loc[phot_idx, 'assoc_PSD'] = ev_psd
                com_x, com_y = unassigned_x.mean(), unassigned_y.mean()
                com_dist = np.sqrt((com_x - ev_x)**2 + (com_y - ev_y)**2)
                for phot_idx in unassigned_idx:
                    photons.loc[phot_idx, 'assoc_com_dist'] = com_dist
                opt_stats['fallback'] += 1
                continue

            # Define objective function
            def objective(weights):
                weights = np.array(weights)
                w_sum = weights.sum()

                if w_sum < 0.1:
                    return 1e10

                cog_x = (weights * unassigned_x).sum() / w_sum
                cog_y = (weights * unassigned_y).sum() / w_sum
                cog_dist = np.sqrt((cog_x - ev_x)**2 + (cog_y - ev_y)**2)

                time_diffs = (unassigned_t - ev_t) * 1e9
                weighted_time = (weights * time_diffs).sum() / w_sum
                time_penalty = abs(weighted_time)

                return cog_weight * cog_dist**2 + time_weight * (time_penalty / max_time_ns)**2

            def constraint(weights):
                weights = np.array(weights)
                w_sum = weights.sum()
                if w_sum < min_photons:
                    scale = min_photons / max(w_sum, 0.01)
                    weights = np.clip(weights * scale, 0, 1)
                return weights

            x0 = np.full(n_candidates, 0.5)
            lb = np.zeros(n_candidates)
            ub = np.ones(n_candidates)

            try:
                if n_candidates <= 20:
                    solution = fmin_powell(objective, x0, bounds=list(zip(lb, ub)),
                                          constraints=constraint, disp=False, gtol=1e-4)
                else:
                    solution = diffev2(objective, list(zip(lb, ub)), constraints=constraint,
                                       npop=min(20, n_candidates * 2), disp=False, gtol=50)

                optimal_weights = np.array(solution)
                threshold = 0.3
                assigned_mask = optimal_weights >= threshold

                if assigned_mask.sum() < min_photons:
                    top_k = min(min_photons, n_candidates)
                    top_indices = np.argsort(optimal_weights)[-top_k:]
                    assigned_mask = np.zeros(n_candidates, dtype=bool)
                    assigned_mask[top_indices] = True

                opt_stats['success'] += 1

            except Exception as e:
                if verbosity >= 2:
                    print(f"Optimization failed for event {ev_id}: {e}, using fallback")
                assigned_mask = np.ones(n_candidates, dtype=bool)
                opt_stats['failed'] += 1

            final_idx = unassigned_idx[assigned_mask]
            final_x = unassigned_x[assigned_mask]
            final_y = unassigned_y[assigned_mask]

            if len(final_idx) > 0:
                com_x = final_x.mean()
                com_y = final_y.mean()
                com_dist = np.sqrt((com_x - ev_x)**2 + (com_y - ev_y)**2)

                if com_dist <= 0.1:
                    com_quality_stats['exact'] += 1
                elif com_dist <= max_dist_px * 0.3:
                    com_quality_stats['good'] += 1
                elif com_dist <= max_dist_px * 0.5:
                    com_quality_stats['acceptable'] += 1
                elif com_dist <= max_dist_px:
                    com_quality_stats['poor'] += 1
                else:
                    com_quality_stats['failed'] += 1

                for phot_idx in final_idx:
                    photons.loc[phot_idx, 'assoc_event_id'] = ev_id
                    photons.loc[phot_idx, 'assoc_x'] = ev_x
                    photons.loc[phot_idx, 'assoc_y'] = ev_y
                    photons.loc[phot_idx, 'assoc_t'] = ev_t
                    photons.loc[phot_idx, 'assoc_n'] = ev_n
                    photons.loc[phot_idx, 'assoc_PSD'] = ev_psd
                    photons.loc[phot_idx, 'assoc_com_dist'] = com_dist

        matched_photons = photons['assoc_event_id'].notna().sum()
        total_photons = len(photons)
        matched_event_ids = photons[photons['assoc_event_id'].notna()]['assoc_event_id'].unique()
        matched_events = len(matched_event_ids)
        total_events = len(events)

        if verbosity >= 1:
            print(f"✅ Photon-Event Association (mystic optimization):")
            print(f"   Photons: {matched_photons:,} / {total_photons:,} matched ({100 * matched_photons / total_photons:.1f}%)")
            print(f"   Events:  {matched_events:,} / {total_events:,} matched ({100 * matched_events / total_events:.1f}%)")
            print(f"   Optimization: {opt_stats['success']} success, {opt_stats['fallback']} fallback, {opt_stats['failed']} failed")

        return photons

    def _standardize_column_names(self, df, verbosity=0):
        """
        Standardize column names with prefixes: px/ for pixels, ph/ for photons, ev/ for events.
        Also add ph/n column counting pixels per photon.

        Args:
            df (pd.DataFrame): DataFrame to standardize
            verbosity (int): Verbosity level

        Returns:
            pd.DataFrame: DataFrame with standardized column names
        """
        rename_map = {}

        # Pixel columns (original data)
        if 'x' in df.columns:
            rename_map['x'] = 'px/x'
        if 'y' in df.columns:
            rename_map['y'] = 'px/y'
        if 't' in df.columns:
            rename_map['t'] = 'px/toa'
        if 'tot' in df.columns:
            rename_map['tot'] = 'px/tot'
        if 'tof' in df.columns:
            rename_map['tof'] = 'px/tof'

        # Photon association columns
        if 'assoc_photon_id' in df.columns:
            rename_map['assoc_photon_id'] = 'ph/id'
        if 'assoc_phot_x' in df.columns:
            rename_map['assoc_phot_x'] = 'ph/x'
        if 'assoc_phot_y' in df.columns:
            rename_map['assoc_phot_y'] = 'ph/y'
        if 'assoc_phot_t' in df.columns:
            rename_map['assoc_phot_t'] = 'ph/toa'

        # Photon CoG distance (quality of pixel-to-photon association)
        if 'pixel_com_dist' in df.columns:
            rename_map['pixel_com_dist'] = 'ph/cog'

        # Event association columns
        if 'assoc_event_id' in df.columns:
            rename_map['assoc_event_id'] = 'ev/id'
        if 'assoc_x' in df.columns:
            rename_map['assoc_x'] = 'ev/x'
        if 'assoc_y' in df.columns:
            rename_map['assoc_y'] = 'ev/y'
        if 'assoc_t' in df.columns:
            rename_map['assoc_t'] = 'ev/toa'
        if 'assoc_n' in df.columns:
            rename_map['assoc_n'] = 'ev/n'
        if 'assoc_PSD' in df.columns:
            rename_map['assoc_PSD'] = 'ev/psd'

        # Event CoG distance (quality of photon-to-event association)
        if 'assoc_com_dist' in df.columns:
            rename_map['assoc_com_dist'] = 'ev/cog'

        # Apply renaming
        df = df.rename(columns=rename_map)

        # Drop unnecessary columns (time_diff_ns and spatial_diff_px)
        cols_to_drop = ['pixel_time_diff_ns', 'pixel_spatial_diff_px', 'time_diff_ns', 'spatial_diff_px']
        existing_drop_cols = [c for c in cols_to_drop if c in df.columns]
        if existing_drop_cols:
            df = df.drop(columns=existing_drop_cols)

        # Add ph/n column: count of pixels per photon
        if 'ph/id' in df.columns:
            photon_pixel_counts = df.groupby('ph/id').size()
            df['ph/n'] = df['ph/id'].map(photon_pixel_counts).fillna(0).astype(int)

            if verbosity >= 2:
                print(f"Added ph/n column: {df['ph/n'].describe()}")

        return df

    def compute_ellipticity(self, x_col='x', y_col='y', event_col=None, verbosity=1):
        """
        Compute ellipticity for associated events using principal component analysis.

        For each event group, computes the covariance of spatial coordinates,
        finds eigenvalues/vectors for major/minor axes, and derives ellipticity.

        Args:
            x_col (str): Column name for x-coordinate (default: 'x').
            y_col (str): Column name for y-coordinate (default: 'y').
            event_col (str, optional): Column name for event ID. Defaults to 'assoc_cluster_id' for lumacam, else 'assoc_event_id'.
            verbosity (int): Verbosity level (0=silent, 1=summary, 2=debug).

        Returns:
            None: Updates self.associated_df with ellipticity columns.
        """
        if self.associated_df is None:
            raise ValueError("Associate photons and events first.")
        if event_col is None:
            event_col = 'assoc_cluster_id' if self.assoc_method == 'lumacam' else 'assoc_event_id'
        self.associated_df = self._compute_event_ellipticity(
            self.associated_df, x_col, y_col, event_col, verbosity
        )

    def _compute_event_ellipticity(self, df, x_col, y_col, event_col, verbosity):
        """
        Internal method to compute ellipticity for grouped events.

        Args:
            df (pd.DataFrame): Associated photon DataFrame.
            x_col (str): Column name for x-coordinate.
            y_col (str): Column name for y-coordinate.
            event_col (str): Column name for event grouping.
            verbosity (int): Verbosity level (0=silent, 1=summary, 2=debug).

        Returns:
            pd.DataFrame: DataFrame with added ellipticity columns ('major_x', 'major_y', 'angle_deg', 'ellipticity').
        """
        df = df.copy()
        df['major_x'] = np.nan
        df['major_y'] = np.nan
        df['angle_deg'] = np.nan
        df['ellipticity'] = np.nan
        event_ids = df[event_col].dropna().unique()
        iterator = tqdm(event_ids, desc="🧮 Computing ellipticity")
        for eid in iterator:
            group = df[df[event_col] == eid]
            if len(group) < 2:
                continue
            coords = group[[x_col, y_col]].values
            coords -= coords.mean(axis=0)
            cov = np.cov(coords, rowvar=False)
            eigvals, eigvecs = np.linalg.eigh(cov)
            major_idx = np.argmax(eigvals)
            major_axis = eigvecs[:, major_idx]
            major_len = np.sqrt(eigvals[major_idx])
            minor_len = np.sqrt(eigvals[1 - major_idx])
            angle = np.degrees(np.arctan2(major_axis[1], major_axis[0]))
            ellipticity = minor_len / major_len if major_len > 0 else 0.0
            df.loc[df[event_col] == eid, 'major_x'] = major_axis[0]
            df.loc[df[event_col] == eid, 'major_y'] = major_axis[1]
            df.loc[df[event_col] == eid, 'angle_deg'] = angle
            df.loc[df[event_col] == eid, 'ellipticity'] = ellipticity
        if verbosity >= 1:
            print(f"✅ Computed shape for {len(event_ids)} events.")
        return df

    def get_combined_dataframe(self):
        """
        Retrieve the associated DataFrame.

        Returns:
            pd.DataFrame: The concatenated associated photon DataFrame with all association columns.

        Raises:
            ValueError: If association has not been performed.
        """
        if self.associated_df is None:
            raise ValueError("Associate photons and events first.")
        return self.associated_df

    def plot_event(self, id_selector, id_type='ev', x_col='x', y_col='y', title=None, max_plots=20):
        """
        Plot events or photons with their associated data.

        This method can plot either by event IDs or photon IDs, and accepts:
        - Single integer: plots one event/photon
        - Slice: plots a range (e.g., slice(0, 10) or 0:10)
        - List of integers: plots specific IDs

        Args:
            id_selector (int, slice, or list): ID(s) to plot. Can be:
                - int: Single ID (e.g., 1130)
                - slice: Range of IDs (e.g., slice(0, 10) or use notation id_selector=slice(0, 10))
                - list: Specific IDs (e.g., [1130, 1131, 1135])
            id_type (str): Type of ID - 'ev' for event IDs or 'ph' for photon IDs (default: 'ev')
            x_col (str): Column name for x-coordinate (default: 'x')
            y_col (str): Column name for y-coordinate (default: 'y')
            title (str, optional): Custom title for the plot (only used for single plots)
            max_plots (int): Maximum number of plots to generate (default: 20)

        Raises:
            ValueError: If association has not been performed or invalid parameters.

        Examples:
            # Plot single event
            assoc.plot_event(1130)

            # Plot range of events
            assoc.plot_event(slice(1130, 1140))

            # Plot specific events
            assoc.plot_event([1130, 1135, 1140])

            # Plot by photon ID
            assoc.plot_event(100, id_type='ph')

            # Plot range of photons
            assoc.plot_event(slice(0, 10), id_type='ph')
        """
        if self.associated_df is None:
            raise ValueError("Associate photons and events first.")

        # Import Plotter here to avoid circular import
        from .plotter import Plotter
        import matplotlib.pyplot as plt

        # Determine column names based on id_type
        if id_type == 'ev':
            # Use exported column names (ev/id) if available, otherwise internal names
            if 'ev/id' in self.associated_df.columns:
                id_col = 'ev/id'
            else:
                id_col = 'assoc_cluster_id' if self.assoc_method == 'lumacam' else 'assoc_event_id'
        elif id_type == 'ph':
            # Use exported column names (ph/id) if available, otherwise internal names
            if 'ph/id' in self.associated_df.columns:
                id_col = 'ph/id'
            else:
                id_col = 'assoc_photon_id' if 'assoc_photon_id' in self.associated_df.columns else 'id'
        else:
            raise ValueError(f"Invalid id_type: {id_type}. Must be 'ev' or 'ph'.")

        # Get unique IDs from the DataFrame
        unique_ids = self.associated_df[id_col].dropna().unique()

        # Process id_selector to get list of IDs to plot
        if isinstance(id_selector, int):
            # Single ID
            ids_to_plot = [id_selector]
        elif isinstance(id_selector, slice):
            # Slice - apply to sorted unique IDs
            sorted_ids = sorted(unique_ids)
            ids_to_plot = sorted_ids[id_selector]
        elif isinstance(id_selector, (list, tuple)):
            # List of IDs
            ids_to_plot = list(id_selector)
        else:
            raise ValueError(f"Invalid id_selector type: {type(id_selector)}. Must be int, slice, or list.")

        # Limit number of plots
        if len(ids_to_plot) > max_plots:
            print(f"⚠️  Requested {len(ids_to_plot)} plots, but max_plots={max_plots}. Plotting first {max_plots} only.")
            ids_to_plot = ids_to_plot[:max_plots]

        # Plot each ID
        for idx, selected_id in enumerate(ids_to_plot):
            # Filter data for this ID
            data = self.associated_df[self.associated_df[id_col] == selected_id]

            if data.empty:
                print(f"⚠️  No data found for {id_type}/id = {selected_id}")
                continue

            # Create plot
            fig, ax = plt.subplots(figsize=(8, 8))

            # Determine what to plot based on id_type
            if id_type == 'ev':
                # Plotting by event - show associated photons
                # Use ph/x, ph/y if available (exported format), otherwise x, y
                if 'ph/x' in data.columns and 'ph/y' in data.columns:
                    ax.scatter(data['ph/x'], data['ph/y'], c='blue', label='Photons', alpha=0.6, s=50)
                else:
                    ax.scatter(data[x_col], data[y_col], c='blue', label='Photons', alpha=0.6, s=50)

                # Plot event center if available
                if 'ev/x' in data.columns and 'ev/y' in data.columns:
                    ev_x = data['ev/x'].iloc[0]
                    ev_y = data['ev/y'].iloc[0]
                    ax.scatter(ev_x, ev_y, c='red', marker='x', s=200, linewidths=3, label='Event Center')
                elif 'assoc_x' in data.columns and 'assoc_y' in data.columns:
                    ev_x = data['assoc_x'].iloc[0]
                    ev_y = data['assoc_y'].iloc[0]
                    ax.scatter(ev_x, ev_y, c='red', marker='x', s=200, linewidths=3, label='Event Center')

                plot_title = title if title and len(ids_to_plot) == 1 else f'Event ID: {selected_id}'

                # Add n count if available
                if 'ev/n' in data.columns:
                    n_value = data['ev/n'].iloc[0]
                    plot_title += f' (n={int(n_value)})'
                elif 'ph/n' in data.columns:
                    n_value = data['ph/n'].iloc[0]
                    plot_title += f' (n={int(n_value)})'

            else:  # id_type == 'ph'
                # Plotting by photon - show the photon and any associated pixels/events
                # Plot photon position
                if 'ph/x' in data.columns and 'ph/y' in data.columns:
                    ph_x = data['ph/x'].iloc[0]
                    ph_y = data['ph/y'].iloc[0]
                    ax.scatter(ph_x, ph_y, c='green', marker='o', s=200, label='Photon Center')
                else:
                    ph_x = data[x_col].iloc[0]
                    ph_y = data[y_col].iloc[0]
                    ax.scatter(ph_x, ph_y, c='green', marker='o', s=200, label='Photon Center')

                # Plot associated pixels if available
                if 'px/x' in data.columns and 'px/y' in data.columns:
                    pixel_data = data[data['px/x'].notna()]
                    if not pixel_data.empty:
                        ax.scatter(pixel_data['px/x'], pixel_data['px/y'], c='blue', marker='s',
                                 s=100, alpha=0.5, label='Pixels')

                # Plot associated event if available
                if 'ev/x' in data.columns and 'ev/y' in data.columns:
                    ev_x = data['ev/x'].iloc[0]
                    ev_y = data['ev/y'].iloc[0]
                    if not pd.isna(ev_x):
                        ax.scatter(ev_x, ev_y, c='red', marker='x', s=200, linewidths=3, label='Event Center')

                plot_title = title if title and len(ids_to_plot) == 1 else f'Photon ID: {selected_id}'

            ax.set_xlabel('X (pixels)', fontsize=12)
            ax.set_ylabel('Y (pixels)', fontsize=12)
            ax.legend(fontsize=10)
            ax.set_title(plot_title, fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')

            plt.tight_layout()
            plt.show()

        if len(ids_to_plot) > 1:
            print(f"✅ Plotted {len(ids_to_plot)} {id_type}/id(s)")

    def _rename_columns_for_export(self, df):
        """
        Rename columns to a clean, user-friendly naming scheme for export.

        Uses prefixes to identify data source:
        - px/* for pixel columns
        - ph/* for photon columns
        - ev/* for event columns

        Args:
            df (pd.DataFrame): Dataframe with association results.

        Returns:
            pd.DataFrame: Copy of dataframe with renamed columns.
        """
        # Make a copy to avoid modifying the original
        df_export = df.copy()

        # Determine if this is pixel-centric, photon-centric, or event-centric data
        # by checking for pixel-specific columns
        has_pixel_data = 'tot' in df.columns or 'assoc_photon_id' in df.columns

        # Create base mapping for association columns (same for all cases)
        rename_map = {
            # Pixel-photon association columns
            'assoc_photon_id': 'ph/id',
            'assoc_phot_x': 'ph/x',
            'assoc_phot_y': 'ph/y',
            'assoc_phot_t': 'ph/toa',
            'pixel_com_dist': 'ph/cog',  # Pixel-photon CoG distance

            # Photon-event association columns
            'assoc_event_id': 'ev/id',
            'assoc_cluster_id': 'ev/id',  # For lumacam method
            'assoc_x': 'ev/x',
            'assoc_y': 'ev/y',
            'assoc_t': 'ev/toa',
            'assoc_n': 'ev/n',
            'assoc_PSD': 'ev/psd',
            'assoc_com_dist': 'ev/cog',  # Photon-event CoG distance
        }

        if has_pixel_data:
            # Pixel-centric data: x,y,t,tot,tof are pixel columns
            rename_map.update({
                'x': 'px/x',
                'y': 'px/y',
                't': 'px/toa',
                'tot': 'px/tot',
                'tof': 'px/tof',
            })
        else:
            # Photon-centric data: x,y,t,tof are photon columns
            rename_map.update({
                'x': 'ph/x',
                'y': 'ph/y',
                't': 'ph/toa',
                'tof': 'ph/tof',
            })

        # Only rename columns that exist in the dataframe
        cols_to_rename = {old: new for old, new in rename_map.items() if old in df_export.columns}
        df_export = df_export.rename(columns=cols_to_rename)

        # Drop unnecessary columns (time_diff_ns and spatial_diff_px)
        cols_to_drop = ['pixel_time_diff_ns', 'pixel_spatial_diff_px', 'time_diff_ns', 'spatial_diff_px']
        existing_drop_cols = [c for c in cols_to_drop if c in df_export.columns]
        if existing_drop_cols:
            df_export = df_export.drop(columns=existing_drop_cols)

        return df_export

    def _create_readme_if_needed(self, output_dir):
        """
        Create a README.md file in the output directory explaining the data structure.

        Args:
            output_dir (str): Directory where the README should be created.
        """
        readme_path = os.path.join(output_dir, "README.md")

        # Only create if it doesn't exist
        if os.path.exists(readme_path):
            return

        readme_content = """# Associated Results Data Structure

This directory contains neutron event analysis results with pixel-photon-event associations.

## Column Naming Convention

Columns use prefixes to identify the data source:
- `px/*` - Pixel data (from Timepix3 detector)
- `ph/*` - Photon data (from scintillator)
- `ev/*` - Event data (neutron events)

## Column Definitions

### Pixel Columns (px/*)
| Column | Description | Units |
|--------|-------------|-------|
| `px/x` | Pixel x-coordinate | pixels |
| `px/y` | Pixel y-coordinate | pixels |
| `px/toa` | Pixel time of arrival | seconds |
| `px/tot` | Pixel time over threshold | seconds |
| `px/tof` | Pixel time of flight | seconds |

### Photon Columns (ph/*)
| Column | Description | Units |
|--------|-------------|-------|
| `ph/x` | Photon x-coordinate | pixels |
| `ph/y` | Photon y-coordinate | pixels |
| `ph/toa` | Photon time of arrival | seconds |
| `ph/tof` | Photon time of flight | seconds |
| `ph/id` | Associated photon ID | - |
| `ph/n` | Number of pixels in photon | - |
| `ph/cog` | Distance from photon CoG to pixel CoM | pixels |

### Event Columns (ev/*)
| Column | Description | Units |
|--------|-------------|-------|
| `ev/x` | Event center-of-mass x-coordinate | pixels |
| `ev/y` | Event center-of-mass y-coordinate | pixels |
| `ev/toa` | Event time of arrival | seconds |
| `ev/n` | Number of photons in event | - |
| `ev/psd` | Pulse shape discrimination value | - |
| `ev/id` | Associated event ID | - |
| `ev/cog` | Distance from event CoG to photon CoM | pixels |

## Data Structure

Depending on which data types were loaded and associated, the CSV files contain:

### Full 3-Tier Association (Pixels → Photons → Events)
- One row per pixel
- Each pixel may be associated with a photon (`ph/id`)
- Each photon may be associated with an event (`ev/id`)
- Contains all `px/*`, `ph/*`, and `ev/*` columns

### Photon-Event Association Only
- One row per photon
- Each photon may be associated with an event (`ev/id`)
- Contains `ph/*` and `ev/*` columns only

### Pixel-Photon Association Only
- One row per pixel
- Each pixel may be associated with a photon (`ph/id`)
- Contains `px/*` and `ph/*` columns only

## Missing Values

- Unassociated entries have `NaN` (Not a Number) values in association columns
- For example, pixels without a matched photon will have `NaN` in `ph/id`, `ph/x`, `ph/y`, etc.

## Units Summary

- **Position**: pixels (coordinate system depends on detector configuration)
- **Time**: seconds (for toa/tof)
- **IDs**: Integer identifiers (0-indexed)
- **Counts**: Integer values (for ev/n, ph/n)
- **PSD**: Dimensionless discrimination value (typically 0-1)
- **CoG**: Center-of-gravity/mass distance in pixels

## Analysis Workflow

The data in this directory was generated using the neutron_event_analyzer package:
1. Raw detector data (pixels, photons, events) was loaded
2. Association algorithms matched pixels to photons and photons to events
3. Results were exported with standardized column names

For more information, see: https://github.com/nuclear/neutron_event_analyzer

---
*Generated automatically by neutron_event_analyzer*
"""

        with open(readme_path, 'w') as f:
            f.write(readme_content)

    def save_associations(self, output_dir=None, filename="associated_data.csv", format='csv', verbosity=1):
        """
        Save associated results to a file.

        Args:
            output_dir (str, optional): Output directory path. If None, creates 'AssociatedResults' folder in data_folder.
            filename (str, optional): Output filename (default: 'associated_data.csv').
            format (str, optional): Output format - 'csv' or 'parquet' (default: 'csv').
            verbosity (int, optional): Verbosity level (0=silent, 1=info).

        Returns:
            str: Path to the saved file.

        Raises:
            ValueError: If no association has been performed yet.
        """
        if self.associated_df is None or len(self.associated_df) == 0:
            raise ValueError("No association data to save. Run associate() or associate_full() first.")

        # Rename columns to user-friendly export format
        df_to_save = self._rename_columns_for_export(self.associated_df)

        # Determine output directory
        if output_dir is None:
            output_dir = os.path.join(self.data_folder, "AssociatedResults")

        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Create README explaining data structure
        self._create_readme_if_needed(output_dir)

        # Construct full output path
        output_path = os.path.join(output_dir, filename)

        # Save based on format
        if format.lower() == 'csv':
            df_to_save.to_csv(output_path, index=False)
        elif format.lower() == 'parquet':
            try:
                df_to_save.to_parquet(output_path, index=False)
            except ImportError:
                raise ImportError("Parquet format requires pyarrow or fastparquet. Install with: pip install pyarrow")
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'csv' or 'parquet'.")

        # Also save statistics as JSON for later retrieval
        stats_dict = {}
        if self.last_assoc_stats:
            stats_dict['pixel_photon'] = self.last_assoc_stats
        if self.last_photon_event_stats:
            stats_dict['photon_event'] = self.last_photon_event_stats

        if stats_dict:
            import json
            stats_path = os.path.join(output_dir, "association_stats.json")
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy(obj):
                import numpy as np
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                return obj

            stats_dict_clean = convert_numpy(stats_dict)
            with open(stats_path, 'w') as f:
                json.dump(stats_dict_clean, f, indent=2)

        if verbosity >= 1:
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # Size in MB
            print(f"✅ Saved {len(df_to_save)} rows to {output_path}")
            print(f"   File size: {file_size:.2f} MB")
            print(f"   Columns: {len(df_to_save.columns)}")

        return output_path

    def plot_stats(self, output_dir=None, verbosity=None, group=None, inline=False):
        """
        Generate comprehensive association quality plots.

        Automatically handles both single folders and groupby structures:
        - For single folders: Generates plots for the single dataset
        - For groupby folders: Generates plots for each group (or specific group if specified)

        Creates plots showing:
        - Pixel-photon association statistics (if pixels were loaded)
        - Photon-event association statistics
        - Correlation plots
        - Distribution comparisons

        Args:
            output_dir (str, optional): Output directory for plots. If None, uses 'AssociatedResults' folder.
            verbosity (int, optional): Verbosity level (0=silent, 1=summary, 2=debug). If None, uses instance verbosity.
            group (str, optional): For groupby structures, specify which group to plot.
                                  If None, plots all groups. Ignored for single folders.
            inline (bool): If True, returns a matplotlib figure for inline display (e.g., Jupyter).
                          If False, saves plots to files and returns file paths. Default: False.

        Returns:
            If inline=False:
                dict or list: For groupby, dict mapping group names to plot file lists.
                             For single folder, list of plot file paths.
            If inline=True:
                matplotlib.figure.Figure: Combined figure with all plots for inline display.

        Raises:
            ValueError: If no association has been performed yet.

        Example:
            # Single folder - save to files
            assoc = nea.Analyse("data/single/")
            assoc.associate(relax=1.5)
            plots = assoc.plot_stats()

            # Single folder - inline display
            fig = assoc.plot_stats(inline=True)
            plt.show()

            # Grouped folders - all groups
            assoc = nea.Analyse("data/grouped/")
            assoc.associate(relax=1.5)
            all_plots = assoc.plot_stats()  # Returns dict with all groups

            # Grouped folders - specific group
            plots = assoc.plot_stats(group='intensifier_gain_50')
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        # Use instance verbosity if not specified
        if verbosity is None:
            verbosity = self.verbosity
            if verbosity is None:
                verbosity = 1

        # Handle groupby structures
        if self.is_groupby:
            if not self.groupby_results:
                raise ValueError("No association data for groups. Run associate() first.")

            # If specific group requested
            if group is not None:
                if group not in self.groupby_results:
                    raise ValueError(f"Group '{group}' not found. Available groups: {list(self.groupby_results.keys())}")

                # Temporarily set associated_df to this group's data
                original_df = self.associated_df
                original_folder = self.data_folder
                self.associated_df = self.groupby_results[group]
                self.data_folder = os.path.join(original_folder, group)

                try:
                    result = self._plot_stats_single(output_dir, verbosity, inline=inline)
                finally:
                    self.associated_df = original_df
                    self.data_folder = original_folder

                return result

            # Plot all groups (inline not supported for multiple groups)
            if inline:
                raise ValueError("inline=True not supported when plotting all groups. Specify a single group with the 'group' parameter.")

            all_results = {}
            if verbosity >= 1:
                print(f"\n📊 Generating plots for {len(self.groupby_results)} groups...")

            for group_name, group_df in tqdm(self.groupby_results.items(),
                                            desc="Plotting groups",
                                            disable=(verbosity == 0)):
                # Temporarily set associated_df
                original_df = self.associated_df
                original_folder = self.data_folder
                self.associated_df = group_df
                self.data_folder = os.path.join(original_folder, group_name)

                try:
                    if output_dir is None:
                        group_output_dir = os.path.join(original_folder, group_name, "AssociatedResults")
                    else:
                        group_output_dir = os.path.join(output_dir, group_name)

                    result = self._plot_stats_single(group_output_dir, 0, inline=False)  # Suppress individual messages
                    all_results[group_name] = result

                    if verbosity >= 2:
                        print(f"✅ {group_name}: {len(result)} plots generated")
                finally:
                    self.associated_df = original_df
                    self.data_folder = original_folder

            if verbosity >= 1:
                print(f"\n✅ Generated plots for {len(all_results)} groups")

            return all_results

        # Single folder - check if data exists
        if self.associated_df is None or len(self.associated_df) == 0:
            raise ValueError("No association data to plot. Run associate() first.")

        return self._plot_stats_single(output_dir, verbosity, inline=inline)

    def _plot_stats_single(self, output_dir=None, verbosity=1, inline=False):
        """
        Internal method to generate plots for a single dataset.

        Args:
            output_dir (str, optional): Output directory for plots (ignored if inline=True).
            verbosity (int): Verbosity level (0=silent, 1=summary, 2=debug).
            inline (bool): If True, returns a combined figure for inline display.

        Returns:
            If inline=False: list of paths to generated plot files.
            If inline=True: matplotlib.figure.Figure with combined plots.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        # Set seaborn style
        sns.set_theme(style="whitegrid")

        df = self.associated_df

        # Check for event ID column (new format: ev/id, old format: assoc_event_id or assoc_cluster_id)
        if 'ev/id' in df.columns:
            associated_mask = df['ev/id'].notna()
            event_col = 'ev/id'
        elif 'assoc_event_id' in df.columns:
            associated_mask = df['assoc_event_id'].notna()
            event_col = 'assoc_event_id'
        elif 'assoc_cluster_id' in df.columns:
            associated_mask = df['assoc_cluster_id'].notna()
            event_col = 'assoc_cluster_id'
        else:
            raise ValueError("No event ID column found in association data (expected 'ev/id', 'assoc_event_id', or 'assoc_cluster_id')")

        total_photons = len(df)
        associated_count = associated_mask.sum()
        unassociated_count = total_photons - associated_count

        # Determine x/y column names (new format: px/x, px/y or ph/x, ph/y; old format: x, y)
        x_col = y_col = None
        for candidate_x, candidate_y in [('px/x', 'px/y'), ('ph/x', 'ph/y'), ('x', 'y')]:
            if candidate_x in df.columns and candidate_y in df.columns:
                x_col, y_col = candidate_x, candidate_y
                break

        # Inline mode: create combined figure
        if inline:
            if verbosity >= 2:
                print("📊 Generating inline summary plot...")

            # Create a 2x2 grid for the main plots
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))

            # Plot 1: Association rate overview (top-left)
            ax = axes[0, 0]
            data = pd.DataFrame({
                'Status': ['Associated', 'Unassociated'],
                'Count': [associated_count, unassociated_count],
                'Percentage': [100 * associated_count / total_photons, 100 * unassociated_count / total_photons]
            })
            sns.barplot(data=data, x='Status', y='Percentage', ax=ax, palette='Set2')
            ax.set_ylabel('Percentage (%)')
            ax.set_title(f'Association Rate: {100 * associated_count / total_photons:.1f}%', fontweight='bold')
            for i, (count, pct) in enumerate(zip(data['Count'], data['Percentage'])):
                ax.text(i, pct + 1, f'{pct:.1f}%\n({count})', ha='center', va='bottom', fontsize=9)

            # Plot 2: Event size distribution (top-right)
            ax = axes[0, 1]
            if associated_count > 0:
                event_sizes = df[associated_mask].groupby(event_col).size()
                sns.histplot(event_sizes, bins=range(1, min(event_sizes.max() + 2, 50)), ax=ax,
                            kde=False, edgecolor='black', color='mediumseagreen')
                ax.axvline(event_sizes.median(), color='red', linestyle='--', linewidth=2,
                          label=f'Median: {event_sizes.median():.1f}')
                ax.axvline(event_sizes.mean(), color='blue', linestyle='--', linewidth=2,
                          label=f'Mean: {event_sizes.mean():.1f}')
                ax.set_xlabel('Photons per Event')
                ax.set_ylabel('Number of Events')
                ax.set_title('Event Size Distribution', fontweight='bold')
                ax.legend(fontsize=8)
            else:
                ax.text(0.5, 0.5, 'No associated data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Event Size Distribution', fontweight='bold')

            # Plot 3: Position scatter (bottom-left)
            ax = axes[1, 0]
            if x_col is not None and associated_count > 0:
                unassoc_df = df[~associated_mask]
                if len(unassoc_df) > 0:
                    ax.scatter(unassoc_df[x_col], unassoc_df[y_col], alpha=0.3, s=5, c='gray', label='Unassociated')
                assoc_df = df[associated_mask]
                ax.scatter(assoc_df[x_col], assoc_df[y_col], alpha=0.5, s=5, c='blue', label='Associated')
                ax.set_xlabel('X Position (pixels)')
                ax.set_ylabel('Y Position (pixels)')
                ax.set_title('Photon Positions', fontweight='bold')
                ax.legend(fontsize=8)
                ax.set_aspect('equal')
            else:
                ax.text(0.5, 0.5, 'No position data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Photon Positions', fontweight='bold')

            # Plot 4: Density heatmap (bottom-right)
            ax = axes[1, 1]
            if x_col is not None and associated_count > 0:
                assoc_df = df[associated_mask]
                h = ax.hist2d(assoc_df[x_col], assoc_df[y_col], bins=50, cmap='hot')
                plt.colorbar(h[3], ax=ax, label='Count')
                ax.set_xlabel('X Position (pixels)')
                ax.set_ylabel('Y Position (pixels)')
                ax.set_title('Associated Photon Density', fontweight='bold')
                ax.set_aspect('equal')
            else:
                ax.text(0.5, 0.5, 'No position data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Associated Photon Density', fontweight='bold')

            plt.tight_layout()

            if verbosity >= 1:
                print(f"✅ Generated inline summary plot ({associated_count}/{total_photons} associated)")

            return fig

        # File mode: save individual plots
        if output_dir is None:
            output_dir = os.path.join(self.data_folder, "AssociatedResults")
        os.makedirs(output_dir, exist_ok=True)
        plot_files = []

        # Plot 1: Association rate overview
        if verbosity >= 1:
            print("📊 Generating association rate plot...")

        fig, ax = plt.subplots(figsize=(10, 6))

        data = pd.DataFrame({
            'Status': ['Associated', 'Unassociated'],
            'Count': [associated_count, unassociated_count],
            'Percentage': [100 * associated_count / total_photons, 100 * unassociated_count / total_photons]
        })

        sns.barplot(data=data, x='Status', y='Percentage', ax=ax, palette='Set2')
        ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.set_title(f'Association Rate: {100 * associated_count / total_photons:.1f}%', fontsize=14, fontweight='bold')

        # Add percentage labels on bars
        for i, (count, pct) in enumerate(zip(data['Count'], data['Percentage'])):
            ax.text(i, pct + 1, f'{pct:.1f}%\n({count} photons)', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'association_rate.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        plot_files.append(plot_path)

        # Plot 2: Spatial differences (photon-event)
        if 'spatial_diff_px' in df.columns and associated_count > 0:
            if verbosity >= 1:
                print("📊 Generating spatial difference distribution...")

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Histogram
            spatial_diffs = df[associated_mask]['spatial_diff_px'].dropna()
            axes[0].hist(spatial_diffs, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
            axes[0].axvline(spatial_diffs.median(), color='red', linestyle='--', linewidth=2, label=f'Median: {spatial_diffs.median():.2f} px')
            axes[0].set_xlabel('Spatial Difference (pixels)', fontsize=12)
            axes[0].set_ylabel('Frequency', fontsize=12)
            axes[0].set_title('Photon-Event Spatial Difference Distribution', fontsize=13, fontweight='bold')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Box plot
            sns.boxplot(y=spatial_diffs, ax=axes[1], color='steelblue')
            axes[1].set_ylabel('Spatial Difference (pixels)', fontsize=12)
            axes[1].set_title('Spatial Difference Box Plot', fontsize=13, fontweight='bold')
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plot_path = os.path.join(output_dir, 'spatial_difference_distribution.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            plot_files.append(plot_path)

        # Plot 3: Time differences (photon-event)
        if 'time_diff_ns' in df.columns and associated_count > 0:
            if verbosity >= 1:
                print("📊 Generating temporal difference distribution...")

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            time_diffs = df[associated_mask]['time_diff_ns'].dropna()

            # Histogram
            axes[0].hist(time_diffs, bins=50, edgecolor='black', alpha=0.7, color='coral')
            axes[0].axvline(time_diffs.median(), color='red', linestyle='--', linewidth=2, label=f'Median: {time_diffs.median():.2f} ns')
            axes[0].set_xlabel('Time Difference (nanoseconds)', fontsize=12)
            axes[0].set_ylabel('Frequency', fontsize=12)
            axes[0].set_title('Photon-Event Time Difference Distribution', fontsize=13, fontweight='bold')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Box plot
            sns.boxplot(y=time_diffs, ax=axes[1], color='coral')
            axes[1].set_ylabel('Time Difference (nanoseconds)', fontsize=12)
            axes[1].set_title('Time Difference Box Plot', fontsize=13, fontweight='bold')
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plot_path = os.path.join(output_dir, 'time_difference_distribution.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            plot_files.append(plot_path)

        # Plot 4: Correlation plot (spatial vs temporal differences)
        if 'spatial_diff_px' in df.columns and 'time_diff_ns' in df.columns and associated_count > 0:
            if verbosity >= 1:
                print("📊 Generating correlation plot...")

            fig, ax = plt.subplots(figsize=(10, 8))

            plot_df = df[associated_mask][['spatial_diff_px', 'time_diff_ns']].dropna()

            # Hexbin for better visualization with many points
            hexbin = ax.hexbin(plot_df['time_diff_ns'], plot_df['spatial_diff_px'],
                              gridsize=50, cmap='YlOrRd', mincnt=1)
            plt.colorbar(hexbin, ax=ax, label='Count')

            ax.set_xlabel('Time Difference (nanoseconds)', fontsize=12)
            ax.set_ylabel('Spatial Difference (pixels)', fontsize=12)
            ax.set_title('Spatial vs Temporal Difference Correlation', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Add correlation coefficient
            corr = plot_df.corr().iloc[0, 1]
            ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes,
                   fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            plt.tight_layout()
            plot_path = os.path.join(output_dir, 'spatial_temporal_correlation.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            plot_files.append(plot_path)

        # Plot 5: Event size distribution
        if associated_count > 0:
            if verbosity >= 1:
                print("📊 Generating event size distribution...")

            event_sizes = df[associated_mask].groupby(event_col).size()

            fig, ax = plt.subplots(figsize=(10, 6))

            sns.histplot(event_sizes, bins=range(1, event_sizes.max() + 2), ax=ax,
                        kde=False, edgecolor='black', color='mediumseagreen')

            ax.axvline(event_sizes.median(), color='red', linestyle='--', linewidth=2,
                      label=f'Median: {event_sizes.median():.1f} photons/event')
            ax.axvline(event_sizes.mean(), color='blue', linestyle='--', linewidth=2,
                      label=f'Mean: {event_sizes.mean():.1f} photons/event')

            ax.set_xlabel('Photons per Event', fontsize=12)
            ax.set_ylabel('Number of Events', fontsize=12)
            ax.set_title('Event Size Distribution', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            plot_path = os.path.join(output_dir, 'event_size_distribution.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            plot_files.append(plot_path)

        # Plot 6: X-Y position scatter (showing association quality)
        if x_col is not None and associated_count > 0:
            if verbosity >= 1:
                print("📊 Generating position scatter plot...")

            fig, axes = plt.subplots(1, 2, figsize=(16, 7))

            # Unassociated photons
            unassoc_df = df[~associated_mask]
            if len(unassoc_df) > 0:
                axes[0].scatter(unassoc_df[x_col], unassoc_df[y_col], alpha=0.3, s=10, c='gray', label='Unassociated')

            # Associated photons
            assoc_df = df[associated_mask]
            axes[0].scatter(assoc_df[x_col], assoc_df[y_col], alpha=0.5, s=10, c='blue', label='Associated')

            axes[0].set_xlabel('X Position (pixels)', fontsize=12)
            axes[0].set_ylabel('Y Position (pixels)', fontsize=12)
            axes[0].set_title('Photon Positions: Associated vs Unassociated', fontsize=13, fontweight='bold')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            axes[0].set_aspect('equal')

            # Heatmap of associated photons
            if len(assoc_df) > 0:
                h = axes[1].hist2d(assoc_df[x_col], assoc_df[y_col], bins=50, cmap='hot')
                plt.colorbar(h[3], ax=axes[1], label='Photon Count')
                axes[1].set_xlabel('X Position (pixels)', fontsize=12)
                axes[1].set_ylabel('Y Position (pixels)', fontsize=12)
                axes[1].set_title('Associated Photon Density Heatmap', fontsize=13, fontweight='bold')
                axes[1].set_aspect('equal')

            plt.tight_layout()
            plot_path = os.path.join(output_dir, 'position_scatter.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            plot_files.append(plot_path)

        # Summary
        if verbosity >= 1:
            print(f"\n✅ Generated {len(plot_files)} plots:")
            for i, path in enumerate(plot_files, 1):
                print(f"   {i}. {os.path.basename(path)}")
            print(f"\n📁 Plots saved to: {output_dir}")

        return plot_files

    def plot_violin(self, columns=None, split_pairs=True, hue='associated',
                    output_dir=None, figsize=None, show=True, inline=False,
                    sample_size=10000, verbosity=None):
        """
        Generate violin plots showing distributions of association data columns.

        Creates a grid of despined subplots, each showing the distribution of a column.
        Similar columns (like x/y pairs) can be combined into split violin plots.

        Args:
            columns (list, optional): Columns to plot. If None, auto-detects available columns.
                                     Use column names like 'px/x', 'ph/cog', 'ev/psd', etc.
            split_pairs (bool): If True, combines x/y and similar column pairs into split violins.
                               Default: True.
            hue (str): Variable to use for split violins. Options:
                      - 'associated': Split by association status (associated vs unassociated)
                      - 'coordinate': Split by coordinate type (x vs y) for paired columns
                      - None: No splitting, single violin per column
                      Default: 'associated'.
            output_dir (str, optional): Output directory for saved plot. If None, uses 'AssociatedResults'.
            figsize (tuple, optional): Figure size. If None, auto-calculated based on number of plots.
            show (bool): Whether to display plot inline. Default: True.
            inline (bool): If True, returns the figure without saving. Default: False.
            sample_size (int): Max samples per violin to avoid slow rendering. Default: 10000.
            verbosity (int, optional): Verbosity level (0=silent, 1=summary, 2=debug).

        Returns:
            If inline=True: matplotlib.figure.Figure
            If inline=False: str path to saved plot file

        Example:
            # Auto-detect and plot all available columns
            assoc.plot_violin()

            # Plot specific columns
            assoc.plot_violin(columns=['px/x', 'px/y', 'ph/cog', 'ev/psd'])

            # Inline display without saving
            fig = assoc.plot_violin(inline=True)

            # Split by coordinate (x vs y) instead of association status
            assoc.plot_violin(hue='coordinate')
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        # Use instance verbosity if not specified
        if verbosity is None:
            verbosity = self.verbosity if self.verbosity is not None else 1

        # Check data availability
        if self.associated_df is None or len(self.associated_df) == 0:
            raise ValueError("No association data available. Run associate() first.")

        df = self.associated_df

        # Column definitions: column_name -> (description, unit, pair_group)
        # pair_group is used to combine similar columns (e.g., 'px_pos' for px/x and px/y)
        column_info = {
            # Pixel columns
            'px/x': ('Pixel X position', 'pixels', 'px_pos'),
            'px/y': ('Pixel Y position', 'pixels', 'px_pos'),
            'px/toa': ('Pixel time-of-arrival', 's', None),
            'px/tot': ('Pixel time-over-threshold', 'a.u.', None),
            # Photon columns
            'ph/x': ('Photon X position', 'pixels', 'ph_pos'),
            'ph/y': ('Photon Y position', 'pixels', 'ph_pos'),
            'ph/toa': ('Photon time-of-arrival', 's', None),
            'ph/cog': ('Pixel-to-photon CoG distance', 'pixels', None),
            # Event columns
            'ev/x': ('Event X position', 'pixels', 'ev_pos'),
            'ev/y': ('Event Y position', 'pixels', 'ev_pos'),
            'ev/t': ('Event time', 's', None),
            'ev/n': ('Event multiplicity', 'count', None),
            'ev/psd': ('Event PSD', 'a.u.', None),
            'ev/cog': ('Photon-to-event CoG distance', 'pixels', None),
        }

        # Determine which columns to plot
        if columns is None:
            # Auto-detect available columns
            available = [col for col in column_info.keys() if col in df.columns]
        else:
            available = [col for col in columns if col in df.columns]
            missing = [col for col in columns if col not in df.columns]
            if missing and verbosity >= 1:
                print(f"⚠️  Columns not found: {missing}")

        if not available:
            raise ValueError("No valid columns found to plot.")

        # Determine event ID column for association status
        if 'ev/id' in df.columns:
            event_col = 'ev/id'
        elif 'assoc_event_id' in df.columns:
            event_col = 'assoc_event_id'
        elif 'assoc_cluster_id' in df.columns:
            event_col = 'assoc_cluster_id'
        else:
            event_col = None
            if hue == 'associated':
                hue = None
                if verbosity >= 1:
                    print("⚠️  No event ID column found, disabling association split")

        # Group columns for split violins if enabled
        if split_pairs and hue == 'coordinate':
            # Group by pair_group
            plot_groups = {}
            standalone = []
            for col in available:
                info = column_info.get(col, (col, '', None))
                pair_group = info[2]
                if pair_group:
                    if pair_group not in plot_groups:
                        plot_groups[pair_group] = []
                    plot_groups[pair_group].append(col)
                else:
                    standalone.append(col)
            # Convert to list of tuples (cols_to_plot, is_paired)
            plots = [(cols, True) for cols in plot_groups.values() if len(cols) == 2]
            plots += [([col], False) for col in standalone]
            # Add unpaired from groups
            for cols in plot_groups.values():
                if len(cols) == 1:
                    plots.append((cols, False))
        else:
            # Each column gets its own subplot
            plots = [([col], False) for col in available]

        n_plots = len(plots)
        if n_plots == 0:
            raise ValueError("No columns available to plot.")

        # Calculate figure size
        if figsize is None:
            n_cols = min(3, n_plots)
            n_rows = (n_plots + n_cols - 1) // n_cols
            figsize = (4.5 * n_cols, 3.5 * n_rows)

        n_cols_grid = min(3, n_plots)
        n_rows_grid = (n_plots + n_cols_grid - 1) // n_cols_grid

        if verbosity >= 1:
            print(f"📊 Generating violin plots for {len(available)} columns...")

        # Create figure and axes
        fig, axes = plt.subplots(n_rows_grid, n_cols_grid, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        # Hide unused axes
        for idx in range(n_plots, len(axes)):
            axes[idx].set_visible(False)

        # Set seaborn style
        sns.set_theme(style="white")

        for idx, (cols, is_paired) in enumerate(plots):
            ax = axes[idx]

            if is_paired and hue == 'coordinate':
                # Split violin for x/y pair - both halves in one violin
                col_x, col_y = cols[0], cols[1]
                info_x = column_info.get(col_x, (col_x, '', None))

                # Sample data
                data_x = df[col_x].dropna()
                data_y = df[col_y].dropna()
                if len(data_x) > sample_size:
                    data_x = data_x.sample(sample_size, random_state=42)
                if len(data_y) > sample_size:
                    data_y = data_y.sample(sample_size, random_state=42)

                # Combine for split violin - single x position, split by coordinate
                base_name = col_x.rsplit('/', 1)[0]
                plot_data = pd.DataFrame({
                    'value': pd.concat([data_x, data_y], ignore_index=True),
                    'coord': ['X'] * len(data_x) + ['Y'] * len(data_y),
                    'group': [base_name] * (len(data_x) + len(data_y))
                })

                sns.violinplot(data=plot_data, x='group', y='value', hue='coord',
                              split=True, inner='quart', ax=ax, palette=['#4C72B0', '#DD8452'],
                              density_norm='width', cut=0, legend=True)

                # Labels
                ax.set_xlabel(f'{base_name}/x,y')
                ax.set_ylabel(info_x[1])
                ax.set_title(info_x[0].replace(' X ', ' '), fontsize=10, fontweight='bold')
                ax.legend(title='', loc='upper right', fontsize=8)

            else:
                # Single column violin
                col = cols[0]
                info = column_info.get(col, (col, '', None))

                data = df[col].dropna()
                if len(data) > sample_size:
                    data = data.sample(sample_size, random_state=42)

                if hue == 'associated' and event_col is not None:
                    # Get association status for sampled indices
                    sampled_idx = data.index
                    assoc_status = df.loc[sampled_idx, event_col].notna()
                    plot_data = pd.DataFrame({
                        'value': data.values,
                        'status': ['Associated' if s else 'Unassociated' for s in assoc_status]
                    })

                    # Check if we have both categories
                    n_assoc = (plot_data['status'] == 'Associated').sum()
                    n_unassoc = (plot_data['status'] == 'Unassociated').sum()

                    if n_assoc > 0 and n_unassoc > 0:
                        sns.violinplot(data=plot_data, x='status', y='value', hue='status',
                                      split=False, inner='quart', ax=ax,
                                      palette={'Associated': '#55A868', 'Unassociated': '#C44E52'},
                                      density_norm='width', cut=0, legend=False)
                        ax.set_xlabel('')
                    else:
                        # Only one category, plot without split
                        sns.violinplot(y=data, ax=ax, inner='quart',
                                      color='#55A868' if n_assoc > 0 else '#C44E52',
                                      density_norm='width', cut=0)
                        ax.set_xlabel('Associated' if n_assoc > 0 else 'Unassociated')
                else:
                    # No hue splitting
                    sns.violinplot(y=data, ax=ax, inner='quart', color='#4C72B0',
                                  density_norm='width', cut=0)
                    ax.set_xlabel('')

                ax.set_ylabel(info[1])
                ax.set_title(f'{col}\n({info[0]})', fontsize=10, fontweight='bold')

            # Despine
            sns.despine(ax=ax, left=False, bottom=True)
            ax.tick_params(bottom=False)

        plt.tight_layout()

        if verbosity >= 1:
            print(f"✅ Generated violin plot with {n_plots} subplots")

        if inline:
            return fig

        # Save plot
        if output_dir is None:
            output_dir = os.path.join(self.data_folder, "AssociatedResults")
        os.makedirs(output_dir, exist_ok=True)

        plot_path = os.path.join(output_dir, 'violin_distributions.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()

        plt.close()

        if verbosity >= 1:
            print(f"📁 Plot saved to: {plot_path}")

        return plot_path

    def get_violin_stats(self, groupby='photons', time_scale=1e7):
        """
        Compute per-photon or per-event statistics for all groups (helper method for violin plots).

        This method computes the same statistics as plot_violin() but returns them as
        a dictionary of DataFrames for inspection or custom plotting.

        Args:
            groupby (str): What to group by: 'photons' (default) or 'events'.
            time_scale (float): Scale factor for time columns (default: 1e7 for ~100ns units).

        Returns:
            dict: Dictionary mapping group names to DataFrames with statistics.
                  For groupby='photons': columns are px/x, px/y, px/tot, px/n, px/toa
                  For groupby='events': columns are ph/x, ph/y, ph/n, ph/toa, ev/psd

        Raises:
            ValueError: If no grouped association data is available.

        Example:
            assoc = nea.Analyse("archive/pencilbeam1/detector_model/")
            assoc.associate(relax=1.5)

            # Get per-photon statistics
            photon_stats = assoc.get_violin_stats(groupby='photons')
            print(photon_stats['full_physics']['px/x'].describe())

            # Get per-event statistics
            event_stats = assoc.get_violin_stats(groupby='events')
            print(event_stats['full_physics']['ph/n'].describe())
        """
        import pandas as pd

        if not self.is_groupby or not self.groupby_results:
            raise ValueError("This method requires grouped analysis data. Use associate() on a groupby folder first.")

        if groupby not in ['photons', 'events']:
            raise ValueError(f"groupby must be 'photons' or 'events', got '{groupby}'")

        all_stats = {}
        group_col = 'ph/id' if groupby == 'photons' else 'ev/id'

        for group_name, group_df in self.groupby_results.items():
            if group_col not in group_df.columns:
                continue

            stats_list = []
            col_names = []

            if groupby == 'photons':
                # Compute per-photon statistics
                # px/x std
                if 'px/x' in group_df.columns:
                    stats_list.append(group_df.groupby(group_col)['px/x'].std())
                    col_names.append('px/x')

                # px/y std
                if 'px/y' in group_df.columns:
                    stats_list.append(group_df.groupby(group_col)['px/y'].std())
                    col_names.append('px/y')

                # px/tot std
                if 'px/tot' in group_df.columns:
                    stats_list.append(group_df.groupby(group_col)['px/tot'].std())
                    col_names.append('px/tot')

                # px/n count
                stats_list.append(group_df.groupby(group_col)[group_col].count())
                col_names.append('px/n')

                # px/toa std (scaled)
                if 'px/toa' in group_df.columns:
                    stats_list.append(group_df.groupby(group_col)['px/toa'].std() * time_scale)
                    col_names.append(f'px/toa [{1e9/time_scale:.0f}ns]')

            else:  # groupby == 'events'
                # Compute per-event statistics
                # ph/x std
                if 'ph/x' in group_df.columns:
                    stats_list.append(group_df.groupby(group_col)['ph/x'].std())
                    col_names.append('ph/x')

                # ph/y std
                if 'ph/y' in group_df.columns:
                    stats_list.append(group_df.groupby(group_col)['ph/y'].std())
                    col_names.append('ph/y')

                # ph/n count
                stats_list.append(group_df.groupby(group_col)[group_col].count())
                col_names.append('ph/n')

                # ph/toa std (scaled)
                if 'ph/toa' in group_df.columns:
                    stats_list.append(group_df.groupby(group_col)['ph/toa'].std() * time_scale)
                    col_names.append(f'ph/toa [{1e9/time_scale:.0f}ns]')

                # ev/psd mean
                if 'ev/psd' in group_df.columns:
                    stats_list.append(group_df.groupby(group_col)['ev/psd'].mean())
                    col_names.append('ev/psd')

            # Combine into DataFrame
            if stats_list:
                stats_df = pd.concat(stats_list, axis=1)
                stats_df.columns = pd.Index(col_names)
                all_stats[group_name] = stats_df

        return all_stats
