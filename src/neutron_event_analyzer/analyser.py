import os
import glob
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
import uuid
import matplotlib.pyplot as plt

class Analyse:
    def __init__(self, data_folder, export_dir="./export", n_threads=10):
        """
        Initialize the Analyse object.

        Args:
            data_folder (str): Path to the data folder containing 'photonFiles' and 'eventFiles'.
            export_dir (str): Path to the directory containing export binaries (empir_export_events, empir_export_photons).
            n_threads (int): Number of threads for parallel processing.
        """
        self.data_folder = data_folder
        self.export_dir = export_dir
        self.n_threads = n_threads
        self.events_df = None
        self.photons_df = None
        self.associated_df = None

    def load_events(self, glob_string=None):
        """
        Load and convert event files into a DataFrame.

        Args:
            glob_string (str, optional): Glob pattern for event files. Defaults to all .empirevent in eventFiles.
        """
        if glob_string is None:
            glob_string = os.path.join(self.data_folder, "eventFiles", "*.empirevent")
        self.events_df = self._convert_events(glob_string)
        print(f"Loaded {len(self.events_df)} events.")

    def load_photons(self, glob_string=None):
        """
        Load and convert photon files into a DataFrame.

        Args:
            glob_string (str, optional): Glob pattern for photon files. Defaults to all .empirphot in photonFiles.
        """
        if glob_string is None:
            glob_string = os.path.join(self.data_folder, "photonFiles", "*.empirphot")
        self.photons_df = self._convert_photons(glob_string)
        print(f"Loaded {len(self.photons_df)} photons.")

    def _convert_event_file(self, eventfile, tmp_dir):
        out_file = os.path.join(tmp_dir, f"{uuid.uuid4().hex}.csv")
        export_bin = os.path.join(self.export_dir, "empir_export_events")
        os.system(f"{export_bin} {eventfile} {out_file} csv")
        try:
            df = pd.read_csv(out_file).query("` PSD value` > 0")
            df.columns = ["x", "y", "t", "n", "PSD", "tof"]
            df["tof"] = df["tof"].astype(float)
            df["PSD"] = df["PSD"].astype(float)
            # Uncomment and adjust if needed:
            # y = df.tof.where((df.tof > 0) & (df.tof < 1800e-9))
            # df["tof"] = (y - 820e-9) % (1144 * 1.5625e-9)
            return df
        except Exception as e:
            print(f"Error processing {out_file}: {e}")
            return None

    def _convert_events(self, glob_string):
        eventfiles = glob.glob(glob_string)
        dfs = []
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp_dir:
            with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
                futures = [executor.submit(self._convert_event_file, f, tmp_dir) for f in eventfiles]
                for future in tqdm(as_completed(futures), total=len(futures)):
                    result = future.result()
                    if result is not None:
                        dfs.append(result)
        if dfs:
            return pd.concat(dfs).replace(" nan", float("nan"))
        return pd.DataFrame()

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

    def _convert_photons(self, glob_string):
        photonfiles = glob.glob(glob_string)
        dfs = []
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp_dir:
            with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
                futures = [executor.submit(self._convert_photon_file, f, tmp_dir) for f in photonfiles]
                for future in tqdm(as_completed(futures), total=len(futures)):
                    result = future.result()
                    if result is not None:
                        dfs.append(result)
        if dfs:
            return pd.concat(dfs).replace(" nan", float("nan"))
        return pd.DataFrame()

    def associate(self, time_norm_ns=1.0, spatial_norm_px=1.0, dSpace_px=np.inf, verbosity=1):
        """
        Associate photons to events and store the result.

        Args:
            time_norm_ns (float): Time normalization factor (ns).
            spatial_norm_px (float): Spatial normalization factor (px).
            dSpace_px (float): Max allowed center-of-mass distance for multiphoton matches.
            verbosity (int): 0=silent, 1=summary, 2=debug.
        """
        if self.photons_df is None or self.events_df is None:
            raise ValueError("Load photons and events first.")
        self.associated_df = self._associate_photons_to_events_simple(
            self.photons_df, self.events_df, time_norm_ns, spatial_norm_px, dSpace_px, verbosity
        )

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
        iterator = tqdm(events.iterrows(), total=len(events), desc="🔗 Associating events")
        for i, event in iterator:
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
        if verbosity >= 1:
            total = len(photons)
            matched = photons['assoc_event_id'].notna().sum()
            print(f"✅ Matched {matched} of {total} photons ({100 * matched / total:.1f}%)")
        return photons

    def compute_ellipticity(self, x_col='x', y_col='y', event_col='assoc_event_id', verbosity=1):
        """
        Compute ellipticity for associated events.

        Args:
            x_col, y_col (str): Column names for spatial coordinates.
            event_col (str): Column name for event ID.
            verbosity (int): 0 = silent, 1 = print summary.
        """
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
        """
        Plot four diagnostic plots for events.

        Args:
            name (str): Title prefix.
            min_n, max_n (int): Range for n.
            min_psd, max_psd (float): Range for PSD.
            df (pd.DataFrame, optional): DataFrame to plot (defaults to self.events_df).
        """
        if df is None:
            if self.events_df is None:
                raise ValueError("Load events first.")
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

    def get_combined_dataframe(self):
        """
        Get the combined DataFrame with associated photon and event information.

        Returns:
            pd.DataFrame: The associated DataFrame.
        """
        if self.associated_df is None:
            raise ValueError("Associate photons and events first.")
        return self.associated_df