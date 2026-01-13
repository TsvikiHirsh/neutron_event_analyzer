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
        self.last_assoc_stats = None  # Statistics from last association

        # Check if this is a groupby folder structure
        is_groupby, subdirs = self._is_groupby_folder(data_folder)
        self.is_groupby = is_groupby
        self.groupby_subdirs = subdirs if is_groupby else []
        self.groupby_results = {}  # Store results from grouped analyses

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
            for subdir in subdirs:
                group_path = os.path.join(data_folder, subdir)
                assoc_file = os.path.join(group_path, "AssociatedResults", "associated_data.csv")
                if os.path.exists(assoc_file):
                    try:
                        group_df = pd.read_csv(assoc_file)
                        self.groupby_results[subdir] = group_df
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
        if os.path.exists(assoc_file):
            try:
                self.associated_df = pd.read_csv(assoc_file)
                if verbosity >= 1:
                    print(f"\n📂 Auto-loaded association results: {len(self.associated_df):,} rows")
                    print("ℹ️  You can now use plot_stats(), save_associations(), etc. without running associate()")
                    print("    To reload raw data, use .load() method")
                # Skip loading raw data if association results exist (user can call .load() manually if needed)
                return
            except Exception as e:
                if verbosity >= 2:
                    print(f"⚠️  Could not load association results: {e}")
                    print("    Will load raw data instead...")

        # Auto-load raw data if no association results found
        self.load(events=events, photons=photons, pixels=pixels, limit=limit, query=query, verbosity=verbosity)

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
             limit=None, query=None, verbosity=0,
             # Backward compatibility - deprecated
             load_events=None, load_photons=None, load_pixels=None):
        """
        Load paired event, photon, and optionally pixel files.

        This method identifies paired files based on matching base filenames (excluding extensions).
        For each file, it first checks for pre-exported CSV files in ExportedEvents/ExportedPhotons/ExportedPixels folders.
        If CSV files exist, they are used directly. Otherwise, it falls back to converting the original
        files using empir binaries.

        Args:
            event_glob (str, optional): Glob pattern relative to data_folder for event files.
            photon_glob (str, optional): Glob pattern relative to data_folder for photon files.
            pixel_glob (str, optional): Glob pattern relative to data_folder for pixel (TPX3) files.
            events (bool, optional): Whether to load events (default: True).
            photons (bool, optional): Whether to load photons (default: True).
            pixels (bool, optional): Whether to load pixels (default: False).
            limit (int, optional): If provided, limit the number of rows loaded for all data types.
            query (str, optional): If provided, apply a pandas query string to filter the events dataframe (e.g., "n>2").
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

            # Apply limit if provided
            if limit is not None:
                if events and len(self.events_df) > 0:
                    self.events_df = self.events_df.head(limit)
                if photons and len(self.photons_df) > 0:
                    self.photons_df = self.photons_df.head(limit)
                if verbosity >= 2:
                    print(f"Applied limit of {limit} rows.")

            # Update pair_dfs to reflect the filtered data for association
            if query is not None or limit is not None:
                if events and photons:
                    self.pair_dfs = [(self.events_df, self.photons_df)]
                    if verbosity >= 2:
                        print(f"Updated pair_dfs with filtered data for association.")

            if events or photons:
                status_parts = []
                if load_events:
                    status_parts.append(f"{len(self.events_df)} events")
                if load_photons:
                    status_parts.append(f"{len(self.photons_df)} photons")
                if verbosity >= 2:
                    print(f"Loaded {' and '.join(status_parts)} in total.")

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

                # Apply limit if provided
                if limit is not None:
                    self.pixels_df = self.pixels_df.head(limit)

                if verbosity >= 1:
                    print(f"Loaded {len(self.pixels_df)} pixels in total.")
            else:
                logger.error("No pixel data could be loaded. Check that ExportedPixels folder exists or empir binaries are available.")
                self.pixels_df = pd.DataFrame()

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
            logger.info(f"Using existing CSV: {exported_csv}")
            csv_file = exported_csv
        elif os.path.exists(exported_csv_with_prefix):
            # Use CSV with exported_ prefix
            logger.info(f"Using existing CSV: {exported_csv_with_prefix}")
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
            logger.info(f"Using existing CSV: {exported_csv}")
            csv_file = exported_csv
        elif os.path.exists(exported_csv_with_prefix):
            # Use CSV with exported_ prefix
            logger.info(f"Using existing CSV: {exported_csv_with_prefix}")
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
            logger.info(f"Using existing CSV: {exported_csv}")
            csv_file = exported_csv
        elif os.path.exists(exported_csv_with_prefix):
            # Use CSV with exported_ prefix
            logger.info(f"Using existing CSV: {exported_csv_with_prefix}")
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
        
        with ProcessPoolExecutor(max_workers=self.n_threads) as executor:
            futures = [
                executor.submit(
                    self._associate_pair, pair, time_norm_ns, spatial_norm_px, dSpace_px,
                    weight_px_in_s, max_time_s, verbosity, effective_method
                ) for pair in self.pair_dfs
            ]
            associated_list = []
            for future in tqdm(as_completed(futures), total=len(futures), desc="Associating pairs", disable=(verbosity == 0)):
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
            if verbosity >= 1:
                total = len(self.associated_df)
                matched = self.associated_df[event_col].notna().sum()
                print(f"✅ Matched {matched} of {total} photons ({100 * matched / total:.1f}%)")
        else:
            self.associated_df = pd.DataFrame()
            logger.warning("No valid association results to concatenate")

    def associate(self, pixel_max_dist_px=None, pixel_max_time_ns=None,
                  photon_time_norm_ns=1.0, photon_spatial_norm_px=1.0, photon_dSpace_px=None,
                  max_time_ns=None, verbosity=None, method='simple', pixel_method='simple', relax=1.0):
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
            method (str): Association method for photon-event association ('simple', 'kdtree', 'window', 'lumacam').
            pixel_method (str): Association method for pixel-photon association ('simple', 'kdtree'). Default is 'simple'.
                               Both methods use the same center-of-mass matching logic.
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
                pixel_method=pixel_method,
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
                print(f"\nStep 1/2: Associating pixels to photons (method={pixel_method})...")

            # Choose pixel-photon association method
            if pixel_method == 'kdtree':
                pixels_associated = self._associate_pixels_to_photons_kdtree(
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
            photon_event_cols = photons_with_events[['_merge_x', '_merge_y', '_merge_t',
                                                     event_col, 'assoc_x', 'assoc_y', 'assoc_t',
                                                     'assoc_n', 'assoc_PSD']].copy()

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
                print(f"\nPerforming 2-tier association: Pixels → Photons (method={pixel_method})")

            # Choose pixel-photon association method
            if pixel_method == 'kdtree':
                pixels_associated = self._associate_pixels_to_photons_kdtree(
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
                if verbosity >= 1:
                    print(f"⚠️  Warning: Could not auto-save results: {e}")

        return self.associated_df

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
        group_stats = []  # Collect statistics from each group

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
                    verbosity=0  # Suppress individual group output
                )

                # Run association with provided kwargs (but override verbosity to 0)
                kwargs_copy = kwargs.copy()
                kwargs_copy['verbosity'] = 0
                group_assoc.associate(**kwargs_copy)

                results[group_name] = group_assoc.associated_df

                # Collect statistics if available
                if hasattr(group_assoc, 'last_assoc_stats') and group_assoc.last_assoc_stats:
                    group_stats.append(group_assoc.last_assoc_stats)

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

        return results

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
        pixels['pixel_time_diff_ns'] = np.nan
        pixels['pixel_spatial_diff_px'] = np.nan

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
                    pixels.loc[pix_idx, 'pixel_time_diff_ns'] = final_time_diffs[i]
                    pixels.loc[pix_idx, 'pixel_spatial_diff_px'] = final_spatial_diffs[i]

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
        if verbosity >= 1:
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
        pixels['pixel_time_diff_ns'] = np.nan
        pixels['pixel_spatial_diff_px'] = np.nan

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
            final_spatial_diffs = unassigned_spatial_diffs[best_subset_mask]
            final_time_diffs = unassigned_time_diffs[best_subset_mask]

            if len(final_idx) > 0:  # Only assign if we found at least one pixel
                for i, pix_idx in enumerate(final_idx):
                    pixels.loc[pix_idx, 'assoc_photon_id'] = phot_id
                    pixels.loc[pix_idx, 'assoc_phot_x'] = phot_x
                    pixels.loc[pix_idx, 'assoc_phot_y'] = phot_y
                    pixels.loc[pix_idx, 'assoc_phot_t'] = phot_t
                    pixels.loc[pix_idx, 'pixel_time_diff_ns'] = final_time_diffs[i]
                    pixels.loc[pix_idx, 'pixel_spatial_diff_px'] = final_spatial_diffs[i]

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
        if verbosity >= 1:
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

    def _standardize_column_names(self, df, verbosity=0):
        r"""
        Standardize column names with prefixes: px\ for pixels, ph\ for photons, ev\ for events.
        Also add ph\n column counting pixels per photon.

        Args:
            df (pd.DataFrame): DataFrame to standardize
            verbosity (int): Verbosity level

        Returns:
            pd.DataFrame: DataFrame with standardized column names
        """
        rename_map = {}

        # Pixel columns (original data)
        if 'x' in df.columns:
            rename_map['x'] = 'px\\x'
        if 'y' in df.columns:
            rename_map['y'] = 'px\\y'
        if 't' in df.columns:
            rename_map['t'] = 'px\\toa'
        if 'tot' in df.columns:
            rename_map['tot'] = 'px\\tot'
        if 'tof' in df.columns:
            rename_map['tof'] = 'px\\tof'

        # Photon association columns
        if 'assoc_photon_id' in df.columns:
            rename_map['assoc_photon_id'] = 'ph\\id'
        if 'assoc_phot_x' in df.columns:
            rename_map['assoc_phot_x'] = 'ph\\x'
        if 'assoc_phot_y' in df.columns:
            rename_map['assoc_phot_y'] = 'ph\\y'
        if 'assoc_phot_t' in df.columns:
            rename_map['assoc_phot_t'] = 'ph\\toa'

        # Pixel-photon difference columns
        if 'pixel_time_diff_ns' in df.columns:
            rename_map['pixel_time_diff_ns'] = 'px\\dt'
        if 'pixel_spatial_diff_px' in df.columns:
            rename_map['pixel_spatial_diff_px'] = 'px\\dr'

        # Event association columns
        if 'assoc_event_id' in df.columns:
            rename_map['assoc_event_id'] = 'ev\\id'
        if 'assoc_x' in df.columns:
            rename_map['assoc_x'] = 'ev\\x'
        if 'assoc_y' in df.columns:
            rename_map['assoc_y'] = 'ev\\y'
        if 'assoc_t' in df.columns:
            rename_map['assoc_t'] = 'ev\\toa'
        if 'assoc_n' in df.columns:
            rename_map['assoc_n'] = 'ev\\n'
        if 'assoc_PSD' in df.columns:
            rename_map['assoc_PSD'] = 'ev\\psd'

        # Apply renaming
        df = df.rename(columns=rename_map)

        # Add ph\n column: count of pixels per photon
        if 'ph\\id' in df.columns:
            photon_pixel_counts = df.groupby('ph\\id').size()
            df['ph\\n'] = df['ph\\id'].map(photon_pixel_counts).fillna(0).astype(int)

            if verbosity >= 2:
                print(f"Added ph\\n column: {df['ph\\n'].describe()}")

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

    def plot_event(self, event_id, x_col='x', y_col='y', title=None):
        """
        Plot a specific event with its associated photons.

        This is a convenience method that wraps Plotter.plot_event().

        Args:
            event_id: ID of the event to plot.
            x_col (str): Column name for x-coordinate (default: 'x').
            y_col (str): Column name for y-coordinate (default: 'y').
            title (str, optional): Custom title for the plot.

        Raises:
            ValueError: If association has not been performed.
        """
        if self.associated_df is None:
            raise ValueError("Associate photons and events first.")

        # Import Plotter here to avoid circular import
        from .plotter import Plotter

        # Determine event column based on association method
        event_col = 'assoc_cluster_id' if self.assoc_method == 'lumacam' else 'assoc_event_id'

        Plotter.plot_event(
            event_id=event_id,
            df=self.associated_df,
            event_col=event_col,
            x_col=x_col,
            y_col=y_col,
            title=title
        )

    def _rename_columns_for_export(self, df):
        """
        Rename columns to a clean, user-friendly naming scheme for export.

        Uses prefixes to identify data source:
        - px\\* for pixel columns
        - ph\\* for photon columns
        - ev\\* for event columns

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
            'assoc_photon_id': 'ph\\id',
            'assoc_phot_x': 'ph\\x',
            'assoc_phot_y': 'ph\\y',
            'assoc_phot_t': 'ph\\toa',
            'pixel_time_diff_ns': 'px\\dt',
            'pixel_spatial_diff_px': 'px\\dr',

            # Photon-event association columns
            'assoc_event_id': 'ev\\id',
            'assoc_cluster_id': 'ev\\id',  # For lumacam method
            'assoc_x': 'ev\\x',
            'assoc_y': 'ev\\y',
            'assoc_t': 'ev\\toa',
            'assoc_n': 'ev\\n',
            'assoc_PSD': 'ev\\psd',
        }

        if has_pixel_data:
            # Pixel-centric data: x,y,t,tot,tof are pixel columns
            rename_map.update({
                'x': 'px\\x',
                'y': 'px\\y',
                't': 'px\\toa',
                'tot': 'px\\tot',
                'tof': 'px\\tof',
            })
        else:
            # Photon-centric data: x,y,t,tof are photon columns
            rename_map.update({
                'x': 'ph\\x',
                'y': 'ph\\y',
                't': 'ph\\toa',
                'tof': 'ph\\tof',
            })

        # Only rename columns that exist in the dataframe
        cols_to_rename = {old: new for old, new in rename_map.items() if old in df_export.columns}
        df_export = df_export.rename(columns=cols_to_rename)

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
- `px\\*` - Pixel data (from Timepix3 detector)
- `ph\\*` - Photon data (from scintillator)
- `ev\\*` - Event data (neutron events)

## Column Definitions

### Pixel Columns (px\\*)
| Column | Description | Units |
|--------|-------------|-------|
| `px\\x` | Pixel x-coordinate | pixels |
| `px\\y` | Pixel y-coordinate | pixels |
| `px\\toa` | Pixel time of arrival | seconds |
| `px\\tot` | Pixel time over threshold | seconds |
| `px\\tof` | Pixel time of flight | seconds |
| `px\\dt` | Time difference to associated photon | nanoseconds |
| `px\\dr` | Spatial distance to associated photon | pixels |

### Photon Columns (ph\\*)
| Column | Description | Units |
|--------|-------------|-------|
| `ph\\x` | Photon x-coordinate | pixels |
| `ph\\y` | Photon y-coordinate | pixels |
| `ph\\toa` | Photon time of arrival | seconds |
| `ph\\tof` | Photon time of flight | seconds |
| `ph\\id` | Associated photon ID | - |

### Event Columns (ev\\*)
| Column | Description | Units |
|--------|-------------|-------|
| `ev\\x` | Event center-of-mass x-coordinate | pixels |
| `ev\\y` | Event center-of-mass y-coordinate | pixels |
| `ev\\toa` | Event time of arrival | seconds |
| `ev\\n` | Number of photons in event | - |
| `ev\\psd` | Pulse shape discrimination value | - |
| `ev\\id` | Associated event ID | - |

## Data Structure

Depending on which data types were loaded and associated, the CSV files contain:

### Full 3-Tier Association (Pixels → Photons → Events)
- One row per pixel
- Each pixel may be associated with a photon (`ph\\id`)
- Each photon may be associated with an event (`ev\\id`)
- Contains all `px\\*`, `ph\\*`, and `ev\\*` columns

### Photon-Event Association Only
- One row per photon
- Each photon may be associated with an event (`ev\\id`)
- Contains `ph\\*` and `ev\\*` columns only

### Pixel-Photon Association Only
- One row per pixel
- Each pixel may be associated with a photon (`ph\\id`)
- Contains `px\\*` and `ph\\*` columns only

## Missing Values

- Unassociated entries have `NaN` (Not a Number) values in association columns
- For example, pixels without a matched photon will have `NaN` in `ph\\id`, `ph\\x`, `ph\\y`, etc.

## Units Summary

- **Position**: pixels (coordinate system depends on detector configuration)
- **Time**: seconds (for toa/tof), nanoseconds (for dt)
- **IDs**: Integer identifiers (0-indexed)
- **Counts**: Integer values (for ev\\n)
- **PSD**: Dimensionless discrimination value (typically 0-1)

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

        if verbosity >= 1:
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # Size in MB
            print(f"✅ Saved {len(df_to_save)} rows to {output_path}")
            print(f"   File size: {file_size:.2f} MB")
            print(f"   Columns: {len(df_to_save.columns)}")

        return output_path

    def plot_stats(self, output_dir=None, verbosity=None, group=None):
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
            verbosity (int, optional): Verbosity level. If None, uses instance verbosity.
            group (str, optional): For groupby structures, specify which group to plot.
                                  If None, plots all groups. Ignored for single folders.

        Returns:
            dict or list: For groupby, dict mapping group names to plot file lists.
                         For single folder, list of plot file paths.

        Raises:
            ValueError: If no association has been performed yet.

        Example:
            # Single folder
            assoc = nea.Analyse("data/single/")
            assoc.associate(relax=1.5)
            plots = assoc.plot_stats()

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
                    result = self._plot_stats_single(output_dir, verbosity)
                finally:
                    self.associated_df = original_df
                    self.data_folder = original_folder

                return result

            # Plot all groups
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

                    result = self._plot_stats_single(group_output_dir, 0)  # Suppress individual messages
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

        return self._plot_stats_single(output_dir, verbosity)

    def _plot_stats_single(self, output_dir=None, verbosity=1):
        """
        Internal method to generate plots for a single dataset.

        Args:
            output_dir (str, optional): Output directory for plots.
            verbosity (int): Verbosity level.

        Returns:
            list: Paths to generated plot files.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        
        # Determine output directory
        if output_dir is None:
            output_dir = os.path.join(self.data_folder, "AssociatedResults")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Set seaborn style
        sns.set_theme(style="whitegrid")
        plot_files = []
        
        df = self.associated_df
        
        # Plot 1: Association rate overview
        if verbosity >= 1:
            print("📊 Generating association rate plot...")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        total_photons = len(df)
        associated_mask = df['assoc_event_id'].notna() if 'assoc_event_id' in df.columns else df['assoc_cluster_id'].notna()
        associated_count = associated_mask.sum()
        unassociated_count = total_photons - associated_count
        
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
            
            event_col = 'assoc_event_id' if 'assoc_event_id' in df.columns else 'assoc_cluster_id'
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
        if 'x' in df.columns and 'y' in df.columns and associated_count > 0:
            if verbosity >= 1:
                print("📊 Generating position scatter plot...")
            
            fig, axes = plt.subplots(1, 2, figsize=(16, 7))
            
            # Unassociated photons
            unassoc_df = df[~associated_mask]
            if len(unassoc_df) > 0:
                axes[0].scatter(unassoc_df['x'], unassoc_df['y'], alpha=0.3, s=10, c='gray', label='Unassociated')
            
            # Associated photons
            assoc_df = df[associated_mask]
            axes[0].scatter(assoc_df['x'], assoc_df['y'], alpha=0.5, s=10, c='blue', label='Associated')
            
            axes[0].set_xlabel('X Position (pixels)', fontsize=12)
            axes[0].set_ylabel('Y Position (pixels)', fontsize=12)
            axes[0].set_title('Photon Positions: Associated vs Unassociated', fontsize=13, fontweight='bold')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            axes[0].set_aspect('equal')
            
            # Heatmap of associated photons
            if len(assoc_df) > 0:
                h = axes[1].hist2d(assoc_df['x'], assoc_df['y'], bins=50, cmap='hot')
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

    def plot_violin(self, y=None, groupby='photons', hue=None, title=None, time_scale=1e7,
                    output_dir=None, figsize=(10, 6), ylim=None, show=True, verbosity=None):
        """
        Generate violin plots comparing distributions across grouped analyses.

        This method computes per-photon statistics (e.g., std of pixel properties) and creates
        violin plots to compare distributions across different groups.

        Args:
            y (str or list): Column(s) to plot. Options include:
                           - 'px\\x': std of pixel x-coordinates per photon
                           - 'px\\y': std of pixel y-coordinates per photon
                           - 'px\\tot': std of pixel time-over-threshold per photon
                           - 'px\\toa': std of pixel time-of-arrival per photon (scaled by time_scale)
                           - 'px\\n': number of pixels per photon (count)
                           If None, plots all available metrics.
                           Can be a string for single metric or list for multiple.
            groupby (str): What to group by. Options:
                          - 'photons': Compute statistics per photon (default)
                          Currently only 'photons' is supported.
            hue (str, optional): For groupby folders, the hue variable for the violin plot.
                               Typically the group name (e.g., 'detector_model').
                               If None, uses 'group' as the hue variable.
            title (str, optional): Plot title. If None, generates automatic title.
            time_scale (float): Scale factor for time columns (default: 1e7 for ~100ns units).
            output_dir (str, optional): Output directory for plots. If None, uses 'AssociatedResults' folder.
            figsize (tuple): Figure size as (width, height). Default: (10, 6).
            ylim (tuple, optional): Y-axis limits as (ymin, ymax). If None, auto-scales.
            show (bool): Whether to display plots inline. Default: True.
            verbosity (int, optional): Verbosity level. If None, uses instance verbosity.

        Returns:
            str or dict: For single folder, returns the plot file path.
                        For groupby folders, returns dict with metric names as keys and plot paths as values.

        Raises:
            ValueError: If no association data is available or if unsupported groupby option.

        Example:
            # Grouped analysis with violin plot
            assoc = nea.Analyse("archive/pencilbeam1/detector_model/")
            assoc.associate(relax=1.5)
            assoc.plot_violin(y="px\\toa", title="Pixel time spread per photon")

            # Plot multiple metrics
            assoc.plot_violin(y=["px\\x", "px\\y", "px\\tot"], ylim=(-1, 5))
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd

        # Use instance verbosity if not specified
        if verbosity is None:
            verbosity = self.verbosity if self.verbosity is not None else 1

        if groupby != 'photons':
            raise ValueError(f"Currently only groupby='photons' is supported, got '{groupby}'")

        # Determine if we have grouped data
        if not self.is_groupby or not self.groupby_results:
            raise ValueError("Violin plots require grouped analysis data. Use associate() on a groupby folder first.")

        # Define available metrics
        available_metrics = {
            'px\\x': ('px\\x', 'std', 'Pixel X std (px)'),
            'px\\y': ('px\\y', 'std', 'Pixel Y std (px)'),
            'px\\tot': ('px\\tot', 'std', 'Pixel ToT std (s)'),
            'px\\n': ('ph\\id', 'count', 'Pixels per photon'),
            'px\\toa': ('px\\toa', 'std', f'Pixel ToA std ({1e9/time_scale:.0f}ns)')
        }

        # Handle y parameter
        if y is None:
            metrics_to_plot = list(available_metrics.keys())
        elif isinstance(y, str):
            metrics_to_plot = [y]
        else:
            metrics_to_plot = list(y)

        # Validate metrics
        for metric in metrics_to_plot:
            if metric not in available_metrics:
                raise ValueError(f"Unknown metric '{metric}'. Available: {list(available_metrics.keys())}")

        # Set output directory
        if output_dir is None:
            output_dir = os.path.join(self.data_folder, "AssociatedResults")
        os.makedirs(output_dir, exist_ok=True)

        if verbosity >= 1:
            print(f"\n📊 Generating violin plots for {len(metrics_to_plot)} metric(s)...")

        plot_files = {}

        # Process each metric
        for metric in metrics_to_plot:
            col, agg_func, ylabel = available_metrics[metric]

            # Collect data from all groups
            all_data = []
            for group_name, group_df in self.groupby_results.items():
                if 'ph\\id' not in group_df.columns:
                    if verbosity >= 1:
                        print(f"⚠️  Skipping {group_name}: no ph\\id column found")
                    continue

                if col not in group_df.columns and metric != 'px\\n':
                    if verbosity >= 1:
                        print(f"⚠️  Skipping {group_name}: column {col} not found")
                    continue

                # Compute per-photon statistics
                if agg_func == 'std':
                    if metric == 'px\\toa':
                        stats = group_df.groupby('ph\\id')[col].std() * time_scale
                    else:
                        stats = group_df.groupby('ph\\id')[col].std()
                elif agg_func == 'count':
                    stats = group_df.groupby('ph\\id')[col].count()

                # Create dataframe with group label
                df_metric = pd.DataFrame({
                    'value': stats.values,
                    hue if hue else 'group': group_name
                })
                all_data.append(df_metric)

            if not all_data:
                if verbosity >= 1:
                    print(f"⚠️  No data available for metric {metric}")
                continue

            # Combine all groups
            combined_df = pd.concat(all_data, ignore_index=True)

            # Create violin plot
            plt.figure(figsize=figsize)
            sns.violinplot(data=combined_df, y='value', x=hue if hue else 'group',
                          split=False, inner="quart")

            # Set labels and title
            plt.ylabel(ylabel, fontsize=12)
            plt.xlabel(hue if hue else 'Group', fontsize=12)
            if title:
                plt.title(title, fontsize=13, fontweight='bold')
            else:
                plt.title(f'{ylabel} - Comparison across groups', fontsize=13, fontweight='bold')

            # Set y-axis limits if provided
            if ylim is not None:
                plt.ylim(ylim)

            # Rotate x-axis labels if needed
            plt.xticks(rotation=45, ha='right')

            plt.tight_layout()

            # Save plot
            plot_filename = f'violin_{metric.replace(chr(92), "_")}.png'
            plot_path = os.path.join(output_dir, plot_filename)
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')

            # Show plot inline if requested
            if show:
                plt.show()

            plt.close()

            plot_files[metric] = plot_path

            if verbosity >= 2:
                print(f"✅ Generated {plot_filename}")

        if verbosity >= 1:
            print(f"\n✅ Generated {len(plot_files)} violin plot(s)")
            print(f"📁 Plots saved to: {output_dir}")

        return plot_files if len(plot_files) > 1 else (list(plot_files.values())[0] if plot_files else None)

    def get_violin_stats(self, time_scale=1e7):
        r"""
        Compute per-photon statistics for all groups (helper method for violin plots).

        This method computes the same statistics as plot_violin() but returns them as
        a dictionary of DataFrames for inspection or custom plotting.

        Args:
            time_scale (float): Scale factor for time columns (default: 1e7 for ~100ns units).

        Returns:
            dict: Dictionary mapping group names to DataFrames with per-photon statistics.
                  Each DataFrame has columns: px\x, px\y, px\tot, px\n, px\toa

        Raises:
            ValueError: If no grouped association data is available.

        Example:
            assoc = nea.Analyse("archive/pencilbeam1/detector_model/")
            assoc.associate(relax=1.5)
            stats = assoc.get_violin_stats()

            # Access specific group
            full_physics_stats = stats['full_physics']
            print(full_physics_stats['px\\x'].describe())  # Stats for pixel x std per photon
        """
        import pandas as pd

        if not self.is_groupby or not self.groupby_results:
            raise ValueError("This method requires grouped analysis data. Use associate() on a groupby folder first.")

        all_stats = {}

        for group_name, group_df in self.groupby_results.items():
            if 'ph\\id' not in group_df.columns:
                continue

            # Compute per-photon statistics (same as violin_analysis)
            stats_list = []
            col_names = []

            # px\x std
            if 'px\\x' in group_df.columns:
                stats_list.append(group_df.groupby('ph\\id')['px\\x'].std())
                col_names.append('px\\x')

            # px\y std
            if 'px\\y' in group_df.columns:
                stats_list.append(group_df.groupby('ph\\id')['px\\y'].std())
                col_names.append('px\\y')

            # px\tot std
            if 'px\\tot' in group_df.columns:
                stats_list.append(group_df.groupby('ph\\id')['px\\tot'].std())
                col_names.append('px\\tot')

            # px\n count
            stats_list.append(group_df.groupby('ph\\id')['ph\\id'].count())
            col_names.append('px\\n')

            # px\toa std (scaled)
            if 'px\\toa' in group_df.columns:
                stats_list.append(group_df.groupby('ph\\id')['px\\toa'].std() * time_scale)
                col_names.append(f'px\\toa [{1e9/time_scale:.0f}ns]')

            # Combine into DataFrame
            if stats_list:
                stats_df = pd.concat(stats_list, axis=1)
                stats_df.columns = pd.Index(col_names)
                all_stats[group_name] = stats_df

        return all_stats
