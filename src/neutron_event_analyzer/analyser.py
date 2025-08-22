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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class Analyse:
    def __init__(self, data_folder, export_dir="./export", n_threads=10):
        """
        Initialize the Analyse object.

        Args:
            data_folder (str): Path to the data folder containing 'photonFiles' and 'eventFiles'.
            export_dir (str): Path to the directory containing export binaries (empir_export_events, empir_export_photons).
            n_threads (int): Number of threads for parallel processing (default: 10).
        """
        self.data_folder = data_folder
        self.export_dir = export_dir
        self.n_threads = n_threads
        self.pair_files = None
        self.pair_dfs = None
        self.events_df = None
        self.photons_df = None
        self.associated_df = None
        self.association_methods = {
            "simple": self._associate_simple,
            "kmeans": self._associate_kmeans,
        }

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

    def _associate_pair(self, pair, time_norm_ns, spatial_norm_px, dSpace_px, verbosity, method):
        """
        Associate photons to events for a single pair of event and photon DataFrames.

        Args:
            pair (tuple): Tuple of (event_df, photon_df).
            time_norm_ns (float): Time normalization factor (ns) for simple association.
            spatial_norm_px (float): Spatial normalization factor (px) for simple association.
            dSpace_px (float): Max allowed center-of-mass distance for multiphoton matches.
            verbosity (int): 0=silent, 1=summary, 2=debug.
            method (str): Association method ('simple' or 'kmeans').

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
            func = self.association_methods[method]
            result = func(pdf, edf, time_norm_ns, spatial_norm_px, dSpace_px, verbosity=0)
            # Log number of matched photons
            event_col = 'assoc_event_id'
            matched = result[event_col].notna().sum()
            if verbosity >= 1:
                logging.info(f"Finished association for pair with {len(edf)} events, {matched} photons matched")
            return result
        except Exception as e:
            logging.error(f"Error associating pair with {len(edf)} events: {e}")
            return None

    def associate(self, time_norm_ns=1.0, spatial_norm_px=1.0, dSpace_px=np.inf, verbosity=1, method="simple"):
        """
        Associate photons to events in parallel per file pair.

        Args:
            time_norm_ns (float): Time normalization factor (ns) for association.
            spatial_norm_px (float): Spatial normalization factor (px) for association.
            dSpace_px (float): Max allowed center-of-mass distance for multiphoton matches.
            verbosity (int): 0=silent, 1=summary, 2=debug.
            method (str): Association method ('simple' or 'kmeans').
        """
        if method not in self.association_methods:
            raise ValueError(f"Unknown association method '{method}'. Available: {list(self.association_methods.keys())}")
        if self.pair_dfs is None:
            raise ValueError("Load data first using load().")
        
        with ProcessPoolExecutor(max_workers=self.n_threads) as executor:
            futures = [
                executor.submit(
                    self._associate_pair, pair, time_norm_ns, spatial_norm_px, dSpace_px, verbosity, method
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
            event_col = 'assoc_event_id'
            if verbosity >= 2:
                logging.info(f"Before grouping: {self.associated_df['assoc_x'].notna().sum()} photons with non-NaN assoc_x")
            mask = self.associated_df['assoc_x'].notna() & self.associated_df['assoc_y'].notna() & \
                   self.associated_df['assoc_t'].notna() & self.associated_df['assoc_n'].notna() & \
                   self.associated_df['assoc_PSD'].notna()
            if mask.any():
                grouped = self.associated_df.loc[mask].groupby(['assoc_x', 'assoc_y', 'assoc_t', 'assoc_n', 'assoc_PSD'])
                new_ids = grouped.ngroup() + 1
                self.associated_df.loc[mask, event_col] = new_ids
                if verbosity >= 2:
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

    def _associate_simple(self, photons_df, events_df, time_norm_ns, spatial_norm_px, dSpace_px, verbosity):
        photons = photons_df.copy()
        events = events_df.copy()
        photons['assoc_event_id'] = np.nan
        photons['assoc_x'] = np.nan
        photons['assoc_y'] = np.nan
        photons['assoc_t'] = np.nan
        photons['assoc_n'] = np.nan
        photons['assoc_PSD'] = np.nan
        photons['time_diff_ns'] = np.nan
        photons['spatial_diff_px'] = np.nan
        events = events.sort_values('t').reset_index(drop=True)
        events['event_id'] = events.index + 1
        photon_times = photons['t'].to_numpy()
        photon_x = photons['x'].to_numpy()
        photon_y = photons['y'].to_numpy()
        
        for i, event in tqdm(events.iterrows(), total=len(events), desc="Associating events (simple)"):
            n_photons = int(event['n'])
            ex, ey, et = event['x'], event['y'], event['t']
            eid = event['event_id']
            time_diff = np.abs(photon_times - et) * 1e9
            spatial_diff = np.sqrt((photon_x - ex)**2 + (photon_y - ey)**2)
            combined_diff = (time_diff / time_norm_ns) + (spatial_diff / spatial_norm_px)
            if n_photons == 1:
                best_idx = np.argmin(combined_diff)
                min_t = photon_times[best_idx]
                com_x = photon_x[best_idx]
                com_y = photon_y[best_idx]
                photons.loc[best_idx, 'assoc_event_id'] = eid
                photons.loc[best_idx, 'assoc_x'] = com_x
                photons.loc[best_idx, 'assoc_y'] = com_y
                photons.loc[best_idx, 'assoc_t'] = min_t
                photons.loc[best_idx, 'assoc_n'] = n_photons
                photons.loc[best_idx, 'assoc_PSD'] = event['PSD']
                photons.loc[best_idx, 'time_diff_ns'] = time_diff[best_idx]
                photons.loc[best_idx, 'spatial_diff_px'] = spatial_diff[best_idx]
            else:
                candidate_indices = np.argsort(combined_diff)[:n_photons]
                selected_t = photon_times[candidate_indices]
                selected_x = photon_x[candidate_indices]
                selected_y = photon_y[candidate_indices]
                if not np.any(np.isnan(selected_x)) and not np.any(np.isnan(selected_y)):
                    min_t = np.min(selected_t)
                    com_x = np.mean(selected_x)
                    com_y = np.mean(selected_y)
                    com_dist = np.sqrt((com_x - ex)**2 + (com_y - ey)**2)
                    if com_dist > dSpace_px:
                        continue
                    for idx in candidate_indices:
                        photons.loc[idx, 'assoc_event_id'] = eid
                        photons.loc[idx, 'assoc_x'] = com_x
                        photons.loc[idx, 'assoc_y'] = com_y
                        photons.loc[idx, 'assoc_t'] = min_t
                        photons.loc[idx, 'assoc_n'] = n_photons
                        photons.loc[idx, 'assoc_PSD'] = event['PSD']
                        photons.loc[idx, 'time_diff_ns'] = (photon_times[idx] - min_t) * 1e9
                        photons.loc[idx, 'spatial_diff_px'] = np.sqrt((photon_x[idx] - com_x)**2 + (photon_y[idx] - com_y)**2)
        return photons

    def _associate_kmeans(self, photons_df, events_df, time_norm_ns, spatial_norm_px, dSpace_px, verbosity):
        from sklearn.cluster import KMeans
        from scipy.spatial.distance import cdist
        
        photons = photons_df.copy()
        events = events_df.copy()
        photons['assoc_event_id'] = np.nan
        photons['assoc_x'] = np.nan
        photons['assoc_y'] = np.nan
        photons['assoc_t'] = np.nan
        photons['assoc_n'] = np.nan
        photons['assoc_PSD'] = np.nan
        photons['time_diff_ns'] = np.nan
        photons['spatial_diff_px'] = np.nan
        events = events.sort_values('t').reset_index(drop=True)
        events['event_id'] = events.index + 1
        
        # Normalize coordinates for clustering
        scale_space = 1 / spatial_norm_px
        scale_time = 1e9 / time_norm_ns
        photon_coords = np.column_stack([
            photons['x'] * scale_space,
            photons['y'] * scale_space,
            photons['t'] * scale_time
        ])
        event_coords = np.column_stack([
            events['x'] * scale_space,
            events['y'] * scale_space,
            events['t'] * scale_time
        ])
        
        # Perform KMeans clustering initialized with event coordinates
        kmeans = KMeans(n_clusters=len(events), init=event_coords, n_init=1, random_state=0)
        kmeans.fit(photon_coords)
        
        labels = kmeans.labels_
        
        # Map cluster centers to events by finding closest initial event coordinates
        distances = cdist(kmeans.cluster_centers_, event_coords)
        cluster_to_event = np.argmin(distances, axis=1)
        
        # Process each cluster
        for cluster_id in tqdm(range(len(events)), desc="Assigning clusters (kmeans)"):
            mask = labels == cluster_id
            if np.sum(mask) == 0:
                continue
            cluster_photons = photons[mask]
            min_t = cluster_photons['t'].min()
            mean_x = cluster_photons['x'].mean()
            mean_y = cluster_photons['y'].mean()
            event_idx = cluster_to_event[cluster_id]
            event = events.iloc[event_idx]
            # Optional: Check CoM distance (converted back to original scale)
            center_x = kmeans.cluster_centers_[cluster_id, 0] / scale_space
            center_y = kmeans.cluster_centers_[cluster_id, 1] / scale_space
            com_dist = np.sqrt((mean_x - event['x'])**2 + (mean_y - event['y'])**2)
            if com_dist > dSpace_px:
                continue
            photons.loc[mask, 'assoc_event_id'] = event['event_id']
            photons.loc[mask, 'assoc_x'] = mean_x
            photons.loc[mask, 'assoc_y'] = mean_y
            photons.loc[mask, 'assoc_t'] = min_t
            photons.loc[mask, 'assoc_n'] = np.sum(mask)
            photons.loc[mask, 'assoc_PSD'] = event['PSD']
            photons.loc[mask, 'time_diff_ns'] = (photons.loc[mask, 't'] - min_t) * 1e9
            photons.loc[mask, 'spatial_diff_px'] = np.sqrt((photons.loc[mask, 'x'] - mean_x)**2 + (photons.loc[mask, 'y'] - mean_y)**2)
        
        return photons

    def compute_ellipticity(self, x_col='x', y_col='y', event_col=None, verbosity=1):
        """
        Compute ellipticity for associated events.

        Args:
            x_col, y_col (str): Column names for spatial coordinates.
            event_col (str, optional): Column name for event ID. Defaults to 'assoc_event_id'.
            verbosity (int): 0 = silent, 1 = print summary.
        """
        if event_col is None:
            event_col = 'assoc_event_id'
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