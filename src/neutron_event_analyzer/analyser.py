# src/neutron_event_analyzer/analyser.py

import os
import glob
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
import uuid
import matplotlib.pyplot as plt

# Check for lumacamTesting availability
try:
    import lumacamTesting as lct
    import yaspin
    LUMACAM_AVAILABLE = True
except ImportError:
    LUMACAM_AVAILABLE = False

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

        def process_pair(pair, tmp_dir):
            event_file, photon_file = pair
            event_df = self._convert_event_file(event_file, tmp_dir)
            photon_df = self._convert_photon_file(photon_file, tmp_dir)
            if event_df is not None and photon_df is not None:
                return event_df, photon_df
            return None

        with tempfile.TemporaryDirectory(dir="/tmp") as tmp_dir:
            with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
                futures = [executor.submit(process_pair, pair, tmp_dir) for pair in self.pair_files]
                self.pair_dfs = []
                for future in tqdm(as_completed(futures), total=len(futures), desc="Loading pairs"):
                    result = future.result()
                    if result is not None:
                        self.pair_dfs.append(result)

        # Optionally concatenate for full DataFrames
        if self.pair_dfs:
            self.events_df = pd.concat([edf for edf, pdf in self.pair_dfs]).replace(" nan", float("nan"))
            self.photons_df = pd.concat([pdf for edf, pdf in self.pair_dfs]).replace(" nan", float("nan"))
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
        
        def associate_pair(pair):
            edf, pdf = pair
            if self.use_lumacam:
                return self._associate_photons_to_events(pdf, edf, weight_px_in_s, max_dist_s, verbosity=0)
            else:
                return self._associate_photons_to_events_simple(pdf, edf, time_norm_ns, spatial_norm_px, dSpace_px, verbosity=0)
        
        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            futures = [executor.submit(associate_pair, pair) for pair in self.pair_dfs]
            associated_list = []
            for future in tqdm(as_completed(futures), total=len(futures), desc="Associating pairs"):
                result = future.result()
                if result is not None:
                    associated_list.append(result)
        
        if associated_list:
            self.associated_df = pd.concat(associated_list)
            # Make event IDs unique across all pairs
            event_col = 'assoc_cluster_id' if self.use_lumacam else 'assoc_event_id'
            mask = self.associated_df['assoc_x'].notna()
            if mask.any():
                grouped = self.associated_df.loc[mask].groupby(['assoc_x', 'assoc_y', 'assoc_t', 'assoc_n', 'assoc_PSD'])
                new_ids = grouped.ngroup() + 1
                self.associated_df.loc[mask, event_col] = new_ids
            if verbosity >= 1:
                total = len(self.associated_df)
                matched = self.associated_df[event_col].notna().sum()
                print(f"✅ Matched {matched} of {total} photons ({100 * matched / total:.1f}%)")
        else:
            self.associated_df = pd.DataFrame()

    # Rest of the class remains the same: _associate_photons_to_events, _associate_photons_to_events_simple, compute_ellipticity, _compute_event_ellipticity, plot_four, plot_event, get_combined_dataframe
    def _associate_photons_to_events(self, photons_df, events_df, weight_px_in_s, max_dist_s, verbosity):
        if not LUMACAM_AVAILABLE:
            raise ImportError("lumacamTesting is required for this method.")
        
        photons = photons_df.rename(columns={"x": "x_px", "y": "y_px", "t": "t_s"}).copy()
        events = events_df.rename(columns={"x": "x_px", "y": "y_px", "t": "t_s"}).copy()

        def fix_time_with_progress(df, label="DataFrame"):
            df = df.sort_values("t_s").reset_index(drop=True)
            t_s = df["t_s"].to_numpy()
            for i in range(1, len(t_s)):
                if t_s[i] <= t_s[i - 1]:
                    t_s[i] = t_s[i - 1] + 1e-12
            df["t_s"] = t_s
            return df

        photons = fix_time_with_progress(photons)
        events = fix_time_with_progress(events)

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
        result_df["assoc_n"] = np.nan
        result_df["assoc_PSD"] = np.nan

        for cluster_id in assoc.clusters.index:
            photon_indices = np.where(cluster_associations == cluster_id)[0]
            event_indices = np.where(cluster_event_indices == cluster_id)[0]
            if len(event_indices) > 0:
                event_idx = event_indices[0]
                event = events.iloc[event_idx]
                result_df.loc[photon_indices, "assoc_t"] = event["t_s"]
                result_df.loc[photon_indices, "assoc_x"] = event["x_px"]
                result_df.loc[photon_indices, "assoc_y"] = event["y_px"]
                result_df.loc[photon_indices, "assoc_n"] = event.get("n", np.nan)
                result_df.loc[photon_indices, "assoc_PSD"] = event.get("PSD", np.nan)

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
        photons['assoc_n'] = np.nan
        photons['assoc_PSD'] = np.nan
        photons['time_diff_ns'] = np.nan
        photons['spatial_diff_px'] = np.nan
        events = events.sort_values('t').reset_index(drop=True)
        events['event_id'] = events.index + 1
        photon_times = photons['t'].to_numpy()
        photon_x = photons['x'].to_numpy()
        photon_y = photons['y'].to_numpy()
        for i, event in events.iterrows():
            n_photons = int(event['n'])
            ex, ey, et = event['x'], event['y'], event['t']
            eid = event['event_id']
            time_diff = np.abs(photon_times - et) * 1e9
            spatial_diff = np.sqrt((photon_x - ex)**2 + (photon_y - ey)**2)
            combined_diff = (time_diff / time_norm_ns) + (spatial_diff / spatial_norm_px)
            if n_photons == 1:
                best_idx = np.argmin(combined_diff)
                photons.loc[best_idx, 'assoc_event_id'] = eid
                photons.loc[best_idx, 'assoc_x'] = ex
                photons.loc[best_idx, 'assoc_y'] = ey
                photons.loc[best_idx, 'assoc_t'] = et
                photons.loc[best_idx, 'assoc_n'] = n_photons
                photons.loc[best_idx, 'assoc_PSD'] = event['PSD']
                photons.loc[best_idx, 'time_diff_ns'] = time_diff[best_idx]
                photons.loc[best_idx, 'spatial_diff_px'] = spatial_diff[best_idx]
            else:
                candidate_indices = np.argsort(combined_diff)[:n_photons]
                selected_x = photon_x[candidate_indices]
                selected_y = photon_y[candidate_indices]
                if not np.any(np.isnan(selected_x)) and not np.any(np.isnan(selected_y)):
                    com_x = np.mean(selected_x)
                    com_y = np.mean(selected_y)
                    com_dist = np.sqrt((com_x - ex)**2 + (com_y - ey)**2)
                    if com_dist > dSpace_px:
                        continue
                for idx in candidate_indices:
                    photons.loc[idx, 'assoc_event_id'] = eid
                    photons.loc[idx, 'assoc_x'] = ex
                    photons.loc[idx, 'assoc_y'] = ey
                    photons.loc[idx, 'assoc_t'] = et
                    photons.loc[idx, 'assoc_n'] = n_photons
                    photons.loc[idx, 'assoc_PSD'] = event['PSD']
                    photons.loc[idx, 'time_diff_ns'] = time_diff[idx]
                    photons.loc[idx, 'spatial_diff_px'] = spatial_diff[idx]
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

    def plot_four(self, name="run", min_n=1, max_n=1000, min_psd=1e-10, max_psd=1, df=None):
        if df is None:
            if self.events_df is None:
                raise ValueError("Load data first.")
            df = self.events_df
        fig, ax = plt.subplots(2, 2, figsize=(8, 6), sharex=False, sharey=False)
        df.query("0<tof<1e-6").plot.hexbin(x="tof", y="PSD", bins="log", yscale="log", ax=ax[0][0], cmap="turbo", gridsize=100)
        df.query("0<tof<1e-6").plot.hexbin(x="n", y="PSD", bins="log", yscale="log", ax=ax[0][1], cmap="turbo")
        df.query("0<tof<1e-6").plot.hexbin(x="tof", y="n", bins="log", ax=ax[1][0], cmap="turbo")
        df.query("0<tof<1e-6 and @min_n<n<@max_n and @min_psd<PSD<@max_psd").tof.plot.hist(
            bins=np.arange(0, 1e-6, 1.5625e-9), histtype="step", ax=ax[1][1],
            label=f"n:({min_n},{max_n}) PSD:({min_psd},{max_psd})", legend=True
        )
        fig.suptitle(name, y=0.94)
        plt.subplots_adjust(hspace=0.32, wspace=0.32)
        plt.show()

    def plot_event(self, event_id, df=None, event_col=None, x_col='x', y_col='y', title=None):
        if df is None:
            if self.associated_df is None:
                raise ValueError("Associate photons and events first.")
            df = self.associated_df
        if event_col is None:
            event_col = 'assoc_cluster_id' if self.use_lumacam else 'assoc_event_id'
        event_data = df[df[event_col] == event_id]
        if event_data.empty:
            print(f"No data found for event ID {event_id}")
            return
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(event_data[x_col], event_data[y_col], c='blue', label='Photons', alpha=0.6)
        if 'assoc_x' in event_data.columns and 'assoc_y' in event_data.columns:
            ax.scatter(event_data['assoc_x'].iloc[0], event_data['assoc_y'].iloc[0], c='red', marker='x', s=100, label='Event Center')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.legend()
        ax.set_title(title if title else f'Event {event_id}')
        plt.show()

    def get_combined_dataframe(self):
        if self.associated_df is None:
            raise ValueError("Associate photons and events first.")
        return self.associated_df