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

        Args:
            data_folder (str): Path to the data folder containing 'photonFiles' and 'eventFiles'.
            export_dir (str): Path to the directory containing export binaries (empir_export_events, empir_export_photons).
            n_threads (int): Number of threads for parallel processing (default: 10).
            use_lumacam (bool): If True, use lumacamTesting for association (if available).
        """
        self.data_folder = data_folder
        self.export_dir = export_dir
        self.n_threads = n_threads
        self.use_lumacam = use_lumacam and LUMACAM_AVAILABLE
        if use_lumacam and not LUMACAM_AVAILABLE:
            print("Warning: lumacamTesting not installed. Falling back to simple association.")
            self.use_lumacam = False
        self.pair_files = None
        self.pair_dfs = None
        self.events_df = None
        self.photons_df = None
        self.associated_df = None

    def _process_pair(self, pair, tmp_dir):
        """
        Process a pair of event and photon files.

        Args:
            pair (tuple): Tuple of (event_file, photon_file) paths.
            tmp_dir (str): Path to temporary directory for CSV output.

        Returns:
            tuple: (event_df, photon_df) if successful, None otherwise.
        """
        event_file, photon_file = pair
        event_df = self._convert_event_file(event_file, tmp_dir)
        photon_df = self._convert_photon_file(photon_file, tmp_dir)
        if event_df is not None and photon_df is not None:
            return event_df, photon_df
        return None

    def load(self, event_glob="eventFiles/*.empirevent", photon_glob="photonFiles/*.empirphot"):
        """
        Load paired event and photon files independently without concatenating into single DataFrames initially.

        Args:
            event_glob (str, optional): Glob pattern relative to data_folder for event files.
            photon_glob (str, optional): Glob pattern relative to data_folder for photon files.
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
                futures = [executor.submit(self._process_pair, pair, tmp_dir) for pair in self.pair_files]
                self.pair_dfs = []
                for future in tqdm(as_completed(futures), total=len(futures), desc="Loading pairs"):
                    result = future.result()
                    if result is not None:
                        self.pair_dfs.append(result)

        # Optionally concatenate for full DataFrames
        if self.pair_dfs:
            self.events_df = pd.concat([edf for edf, pdf in self.pair_dfs], ignore_index=True).replace(" nan", float("nan"))
            self.photons_df = pd.concat([pdf for edf, pdf in self.pair_dfs], ignore_index=True).replace(" nan", float("nan"))
            print(f"Loaded {len(self.events_df)} events and {len(self.photons_df)} photons in total.")
        else:
            self.events_df = pd.DataFrame()
            self.photons_df = pd.DataFrame()

    def _convert_event_file(self, eventfile, tmp_dir):
        out_file = os.path.join(tmp_dir, f"{uuid.uuid4().hex}.csv")
        export_bin = os.path.join(self.export_dir, "empir_export_events")
        os.system(f"{export_bin} {eventfile} {out_file} csv")
        try:
            df = pd.read_csv(out_file).query("` PSD value` > 0")
            df.columns = ["x", "y", "t", "n", "PSD", "tof"]
            df["tof"] = df["tof"].astype(float)
            df["PSD"] = df["PSD"].astype(float)
            return df
        except Exception as e:
            print(f"Error processing {out_file}: {e}")
            return None

    def _convert_photon_file(self, photonfile, tmp_dir):
        out_file = os.path.join(tmp_dir, f"{uuid.uuid4().hex}.csv")
        export_bin = os.path.join(self.export_dir, "empir_export_photons")
        os.system(f"{export_bin} {photonfile} {out_file} csv")
        try:
            df = pd.read_csv(out_file)
            df.columns = ["x", "y", "t", "tof"]
            df["x"] = df["x"].astype(float)
            df["y"] = df["y"].astype(float)
            df["t"] = df["t"].astype(float)
            df["tof"] = pd.to_numeric(df["tof"], errors="coerce")
            return df
        except Exception as e:
            print(f"Error processing {out_file}: {e}")
            return None

    def _associate_pair(self, pair, time_norm_ns, spatial_norm_px, dSpace_px, weight_px_in_s, max_dist_s, verbosity):
        """
        Associate photons to events for a single pair of event and photon DataFrames.

        Args:
            pair (tuple): Tuple of (event_df, photon_df).
            time_norm_ns (float): Time normalization factor (ns) for simple association.
            spatial_norm_px (float): Spatial normalization factor (px) for simple association.
            dSpace_px (float): Max allowed center-of-mass distance for multiphoton matches.
            weight_px_in_s (float, optional): Weight for pixel-to-second conversion (lumacam).
            max_dist_s (float, optional): Max distance in seconds (lumacam).
            verbosity (int): 0=silent, 1=summary, 2=debug.

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
            logging.info(f"Starting association for pair with {len(edf)} events and {len(pdf)} photons")
        try:
            if self.use_lumacam:
                result = self._associate_photons_to_events(pdf, edf, weight_px_in_s, max_dist_s, verbosity=0)
            else:
                result = self._associate_photons_to_events_simple(pdf, edf, time_norm_ns, spatial_norm_px, dSpace_px, verbosity=0)
            # Log number of matched photons
            event_col = 'assoc_cluster_id' if self.use_lumacam else 'assoc_event_id'
            matched = result[event_col].notna().sum()
            logging.info(f"Finished association for pair with {len(edf)} events, {matched} photons matched")
            return result
        except Exception as e:
            logging.error(f"Error associating pair with {len(edf)} events: {e}")
            return None

    def associate(self, time_norm_ns=1.0, spatial_norm_px=1.0, dSpace_px=np.inf, weight_px_in_s=None, max_dist_s=None, verbosity=1):
        """
        Associate photons to events in parallel per file pair.

        Args:
            time_norm_ns (float): Time normalization factor (ns) for simple association.
            spatial_norm_px (float): Spatial normalization factor (px) for simple association.
            dSpace_px (float): Max allowed center-of-mass distance for multiphoton matches (simple association).
            weight_px_in_s (float, optional): Weight for pixel-to-second conversion (lumacam association).
            max_dist_s (float, optional): Max distance in seconds (lumacam association).
            verbosity (int): 0=silent, 1=summary, 2=debug.
        """
        if self.pair_dfs is None:
            raise ValueError("Load data first using load().")
        
        with ProcessPoolExecutor(max_workers=self.n_threads) as executor:
            futures = [
                executor.submit(
                    self._associate_pair, pair, time_norm_ns, spatial_norm_px, dSpace_px,
                    weight_px_in_s, max_dist_s, verbosity
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
            event_col = 'assoc_cluster_id' if self.use_lumacam else 'assoc_event_id'
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

    def _associate_photons_to_events(self, photons_df, events_df, weight_px_in_s, max_dist_s, verbosity):
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

        if weight_px_in_s is None or max_dist_s is None:
            all_x = pd.concat([photons['x_px'], events['x_px']])
            all_y = pd.concat([photons['y_px'], events['y_px']])
            all_t = pd.concat([photons['t_s'], events['t_s']])
            spatial_scale = np.sqrt(np.var(all_x) + np.var(all_y))
            temporal_scale = np.std(all_t)
            weight_px_in_s = temporal_scale / spatial_scale if spatial_scale > 0 else 1.0
            max_dist_s = 3 * temporal_scale
            if verbosity >= 2:
                print(f"📏 Spatial scale: {spatial_scale:.2f}")
                print(f"⏱  Temporal scale: {temporal_scale:.2e} s")
                print(f"⚖️  Weight px→s: {weight_px_in_s:.2e}")
                print(f"📐 Max dist: {max_dist_s:.2e} s")
        else:
            if verbosity >= 2:
                print(f"⚖️  Using provided weight_px_in_s: {weight_px_in_s}")
                print(f"📐 Using provided max_dist_s: {max_dist_s}")

        with yaspin.yaspin(text="Associating photons to events...", color="cyan") as spinner:
            assoc = lct.EventAssociation.make_individualShortestConnection(
                weight_px_in_s, max_dist_s,
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

    def _associate_photons_to_events_simple(
        self, photons_df, events_df, time_norm_ns, spatial_norm_px, dSpace_px, verbosity
    ):
        photons = photons_df.copy()
        events = events_df.copy()
        photons['assoc_event_id'] = np.nan
        photons['assoc_x'] = np.nan
        photons['assoc_y'] = np.nan
        photons['assoc_t'] = np.nan
        photons['assoc_n'] = 0  # Default to 0
        photons['assoc_PSD'] = 0  # Default to 0
        photons['time_diff_ns'] = np.nan
        photons['spatial_diff_px'] = np.nan
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
                photons.loc[best_idx, 'assoc_event_id'] = eid
                photons.loc[best_idx, 'assoc_x'] = ex
                photons.loc[best_idx, 'assoc_y'] = ey
                photons.loc[best_idx, 'assoc_t'] = et
                photons.loc[best_idx, 'assoc_n'] = n_photons
                photons.loc[best_idx, 'assoc_PSD'] = event.get('PSD', 0)
                photons.loc[best_idx, 'time_diff_ns'] = time_diff.iloc[np.argmin(combined_diff)]
                photons.loc[best_idx, 'spatial_diff_px'] = spatial_diff.iloc[np.argmin(combined_diff)]
            else:
                candidate_indices = np.argsort(combined_diff)[:n_photons]
                selected_indices = [indices[i] for i in candidate_indices]
                selected_x = candidate_photons.iloc[candidate_indices]['x']
                selected_y = candidate_photons.iloc[candidate_indices]['y']
                if not np.any(np.isnan(selected_x)) and not np.any(np.isnan(selected_y)):
                    com_x = np.mean(selected_x)
                    com_y = np.mean(selected_y)
                    com_dist = np.sqrt((com_x - ex)**2 + (com_y - ey)**2)
                    if com_dist > dSpace_px:
                        continue
                for idx, diff_idx in zip(selected_indices, candidate_indices):
                    photons.loc[idx, 'assoc_event_id'] = eid
                    photons.loc[idx, 'assoc_x'] = ex
                    photons.loc[idx, 'assoc_y'] = ey
                    photons.loc[idx, 'assoc_t'] = et
                    photons.loc[idx, 'assoc_n'] = n_photons
                    photons.loc[idx, 'assoc_PSD'] = event.get('PSD', 0)
                    photons.loc[idx, 'time_diff_ns'] = time_diff.iloc[diff_idx]
                    photons.loc[idx, 'spatial_diff_px'] = spatial_diff.iloc[diff_idx]
        return photons

    def compute_ellipticity(self, x_col='x', y_col='y', event_col=None, verbosity=1):
        """
        Compute ellipticity for associated events.

        Args:
            x_col, y_col (str): Column names for spatial coordinates.
            event_col (str, optional): Column name for event ID. Defaults to 'assoc_cluster_id' if use_lumacam, else 'assoc_event_id'.
            verbosity (int): 0 = silent, 1 = print summary.
        """
        if event_col is None:
            event_col = 'assoc_cluster_id' if self.use_lumacam else 'assoc_event_id'
        if self.associated_df is None:
            raise ValueError("Associate photons and events first.")
        self.associated_df = self._compute_event_ellipticity(
            self.associated_df, x_col, y_col, event_col, verbosity
        )

    def _compute_event_ellipticity(self, df, x_col, y_col, event_col, verbosity):
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
        if self.associated_df is None:
            raise ValueError("Associate photons and events first.")
        return self.associated_df