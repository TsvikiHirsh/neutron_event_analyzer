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

# Check for lumacamTesting availability
try:
    import lumacamTesting as lct
    import yaspin
    LUMACAM_AVAILABLE = True
except ImportError:
    LUMACAM_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class Analyse:
    def __init__(self, data_folder, export_dir="./export", n_threads=10, use_lumacam=False):
        """
        Initialize the Analyse object.

        This class provides tools for loading, associating, and analyzing neutron event and photon data
        from paired files. It supports multiple association methods:
        - 'lumacam': Uses lumacamTesting library (requires installation).
        - 'kdtree': Full KDTree-based association on normalized space-time coordinates.
        - 'window': Time-window KDTree for nearly time-sorted data, using symmetric window.
        - 'simple': Simple forward time-window association, selecting closest photons in space,
          with center-of-mass check. Optimized for speed with small windows.

        The class can work with either:
        1. Pre-exported CSV files in 'ExportedEvents' and 'ExportedPhotons' subdirectories (preferred), or
        2. Original .empirevent and .empirphot files (requires empir binaries in export_dir)

        Args:
            data_folder (str): Path to the data folder containing 'photonFiles'/'eventFiles' subdirectories
                              and optionally 'ExportedPhotons'/'ExportedEvents' subdirectories with CSV files.
            export_dir (str): Path to the directory containing export binaries (empir_export_events, empir_export_photons).
                             Only required if pre-exported CSV files are not available.
            n_threads (int): Number of threads for parallel processing (default: 10).
            use_lumacam (bool): If True, prefer 'lumacam' for association when method='auto' (if available).
        """
        self.data_folder = data_folder
        self.export_dir = export_dir
        self.n_threads = n_threads
        self.use_lumacam = use_lumacam and LUMACAM_AVAILABLE
        if use_lumacam and not LUMACAM_AVAILABLE:
            print("Warning: lumacamTesting not installed. Cannot use lumacam association.")
            self.use_lumacam = False
        self.pair_files = None
        self.pair_dfs = None
        self.events_df = None
        self.photons_df = None
        self.associated_df = None
        self.assoc_method = None

    def _process_pair(self, pair, tmp_dir, verbosity=0):
        """
        Process a pair of event and photon files by converting them to CSV and loading into DataFrames.

        Args:
            pair (tuple): Tuple of (event_file, photon_file) paths.
            tmp_dir (str): Path to temporary directory for CSV output.
            verbosity (int): Verbosity level (0=silent, 1=warnings).

        Returns:
            tuple: (event_df, photon_df) if successful, None otherwise.
        """
        event_file, photon_file = pair
        event_df = self._convert_event_file(event_file, tmp_dir, verbosity)
        photon_df = self._convert_photon_file(photon_file, tmp_dir, verbosity)
        if event_df is not None and photon_df is not None:
            return event_df, photon_df
        return None

    def load(self, event_glob="[Ee]ventFiles/*.empirevent", photon_glob="[Pp]hotonFiles/*.empirphot", limit=None, query=None, verbosity=0):
        """
        Load paired event and photon files independently without concatenating into single DataFrames initially.

        This method identifies paired files based on matching base filenames (excluding extensions).
        For each file, it first checks for pre-exported CSV files in ExportedEvents/ExportedPhotons folders.
        If CSV files exist, they are used directly. Otherwise, it falls back to converting the original
        files using empir binaries.

        Args:
            event_glob (str, optional): Glob pattern relative to data_folder for event files.
            photon_glob (str, optional): Glob pattern relative to data_folder for photon files.
            limit (int, optional): If provided, limit the number of rows loaded for both events and photons.
            query (str, optional): If provided, apply a pandas query string to filter the events dataframe (e.g., "n>2").
            verbosity (int, optional): Verbosity level (0=silent, 1=warnings). Default is 0.
        """
        event_files = glob.glob(os.path.join(self.data_folder, event_glob))
        photon_files = glob.glob(os.path.join(self.data_folder, photon_glob))

        def get_key(f):
            return os.path.basename(f).rsplit('.', 1)[0]

        event_dict = {get_key(f): f for f in event_files}
        photon_dict = {get_key(f): f for f in photon_files}

        common_keys = sorted(set(event_dict) & set(photon_dict))
        self.pair_files = [(event_dict[k], photon_dict[k]) for k in common_keys]
        print(f"Found {len(self.pair_files)} paired files.")

        with tempfile.TemporaryDirectory(dir="/tmp") as tmp_dir:
            with ProcessPoolExecutor(max_workers=self.n_threads) as executor:
                futures = [executor.submit(self._process_pair, pair, tmp_dir, verbosity) for pair in self.pair_files]
                self.pair_dfs = []
                for future in tqdm(as_completed(futures), total=len(futures), desc="Loading pairs"):
                    result = future.result()
                    if result is not None:
                        self.pair_dfs.append(result)

        # Optionally concatenate for full DataFrames
        if self.pair_dfs:
            self.events_df = pd.concat([edf for edf, pdf in self.pair_dfs], ignore_index=True).replace(" nan", float("nan"))
            self.photons_df = pd.concat([pdf for edf, pdf in self.pair_dfs], ignore_index=True).replace(" nan", float("nan"))

            # Apply query filter to events if provided
            if query is not None:
                original_events_len = len(self.events_df)
                self.events_df = self.events_df.query(query)
                print(f"Applied query '{query}': {original_events_len} -> {len(self.events_df)} events")

            # Apply limit to both dataframes if provided
            if limit is not None:
                self.events_df = self.events_df.head(limit)
                self.photons_df = self.photons_df.head(limit)
                print(f"Applied limit of {limit} rows to events and photons.")

            # Update pair_dfs to reflect the filtered data for association
            if query is not None or limit is not None:
                self.pair_dfs = [(self.events_df, self.photons_df)]
                print(f"Updated pair_dfs with filtered data for association.")

            print(f"Loaded {len(self.events_df)} events and {len(self.photons_df)} photons in total.")
        else:
            self.events_df = pd.DataFrame()
            self.photons_df = pd.DataFrame()

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
        exported_csv = os.path.join(self.data_folder, "ExportedEvents", f"{basename}.csv")

        if os.path.exists(exported_csv):
            # Use already exported CSV
            logging.info(f"Using existing CSV: {exported_csv}")
            csv_file = exported_csv
        else:
            # Fall back to empir binary conversion
            export_bin = os.path.join(self.export_dir, "empir_export_events")
            if not os.path.exists(export_bin):
                if verbosity >= 1:
                    print(f"Warning: empir_export_events binary not found at {export_bin} and no exported CSV found at {exported_csv}")
                return None

            csv_file = os.path.join(tmp_dir, f"{uuid.uuid4().hex}.csv")
            logging.info(f"Converting {eventfile} using empir_export_events")
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
            if verbosity >= 1:
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
        exported_csv = os.path.join(self.data_folder, "ExportedPhotons", f"{basename}.csv")

        if os.path.exists(exported_csv):
            # Use already exported CSV
            logging.info(f"Using existing CSV: {exported_csv}")
            csv_file = exported_csv
        else:
            # Fall back to empir binary conversion
            export_bin = os.path.join(self.export_dir, "empir_export_photons")
            if not os.path.exists(export_bin):
                if verbosity >= 1:
                    print(f"Warning: empir_export_photons binary not found at {export_bin} and no exported CSV found at {exported_csv}")
                return None

            csv_file = os.path.join(tmp_dir, f"{uuid.uuid4().hex}.csv")
            logging.info(f"Converting {photonfile} using empir_export_photons")
            os.system(f"{export_bin} {photonfile} {csv_file} csv")

        try:
            df = pd.read_csv(csv_file)

            if verbosity >= 2:
                print(f"Read photon CSV with shape {df.shape}, columns: {df.columns.tolist()}")

            df.columns = ["x", "y", "t", "tof"]
            df["x"] = df["x"].astype(float)
            df["y"] = df["y"].astype(float)
            df["t"] = df["t"].astype(float)
            df["tof"] = pd.to_numeric(df["tof"], errors="coerce")

            if len(df) == 0:
                if verbosity >= 1:
                    print(f"Warning: Photon DataFrame is empty after processing {os.path.basename(csv_file)}")

            return df
        except Exception as e:
            if verbosity >= 1:
                print(f"Error processing photon file {csv_file}: {e}")
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
            logging.error(f"Event DataFrame missing required columns: {required_event_cols}")
            return None
        if not all(col in pdf.columns for col in required_photon_cols):
            logging.error(f"Photon DataFrame missing required columns: {required_photon_cols}")
            return None
        # Check for excessive NaNs
        if edf[required_event_cols].isna().any().any() or pdf[required_photon_cols].isna().any().any():
            logging.warning(f"NaN values detected in event or photon DataFrame for pair")
        if verbosity >= 2:
            logging.info(f"Starting association for pair with {len(edf)} events and {len(pdf)} photons using method '{method}'")
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
                logging.info(f"Finished association for pair with {len(edf)} events, {matched} photons matched")
            return result
        except Exception as e:
            logging.error(f"Error associating pair with {len(edf)} events using method '{method}': {e}")
            return None

    def associate(self, time_norm_ns=1.0, spatial_norm_px=1.0, dSpace_px=np.inf, weight_px_in_s=None, max_time_ns=500, verbosity=1, method='auto'):
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
            for future in tqdm(as_completed(futures), total=len(futures), desc="Associating pairs"):
                result = future.result()
                if result is not None:
                    associated_list.append(result)
        
        if associated_list:
            # Concatenate with ignore_index to avoid index duplication
            self.associated_df = pd.concat(associated_list, ignore_index=True)
            logging.info(f"Before grouping: {self.associated_df['assoc_x'].notna().sum()} photons with non-NaN assoc_x")
            mask = self.associated_df['assoc_x'].notna() & self.associated_df['assoc_y'].notna() & \
                   self.associated_df['assoc_t'].notna() & self.associated_df['assoc_n'].notna() & \
                   self.associated_df['assoc_PSD'].notna()
            if mask.any():
                grouped = self.associated_df.loc[mask].groupby(['assoc_x', 'assoc_y', 'assoc_t', 'assoc_n', 'assoc_PSD'])
                new_ids = grouped.ngroup() + 1
                self.associated_df.loc[mask, event_col] = new_ids
                logging.info(f"After grouping: {self.associated_df[event_col].notna().sum()} photons with non-NaN {event_col}")
            else:
                logging.warning("No photons with all non-NaN assoc columns for grouping")
            if verbosity >= 1:
                total = len(self.associated_df)
                matched = self.associated_df[event_col].notna().sum()
                print(f"✅ Matched {matched} of {total} photons ({100 * matched / total:.1f}%)")
        else:
            self.associated_df = pd.DataFrame()
            logging.warning("No valid association results to concatenate")

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
            for i in tqdm(range(1, len(t_s)), desc=f"Fixing time for {label}"):
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

        for cluster_id in tqdm(assoc.clusters.index, desc="Assigning event data"):
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

        for event in tqdm(events.iterrows(), total=len(events), desc="Associating events"):
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

        for _, ev in tqdm(events.iterrows(), total=len(events), desc="Associating events"):
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

        for _, ev in tqdm(events.iterrows(), total=len(events), desc="Associating events"):
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
