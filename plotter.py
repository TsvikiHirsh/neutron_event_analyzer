import pandas as pd
import matplotlib.pyplot as plt

class Plotter:
    @staticmethod
    def plot_four(name="run", min_n=1, max_n=1000, min_psd=1e-10, max_psd=1, df=None):
        """
        Plot four diagnostic plots for event data.

        Args:
            name (str): Title for the plot.
            min_n (int): Minimum number of photons for filtering.
            max_n (int): Maximum number of photons for filtering.
            min_psd (float): Minimum PSD value for filtering.
            max_psd (float): Maximum PSD value for filtering.
            df (pandas.DataFrame): DataFrame containing event data with columns 'tof', 'PSD', 'n'.
        """
        if df is None:
            raise ValueError("A DataFrame must be provided.")
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

    @staticmethod
    def plot_event(event_id, df, event_col=None, x_col='x', y_col='y', title=None):
        """
        Plot a specific event with its associated photons.

        Args:
            event_id: ID of the event to plot.
            df (pandas.DataFrame): DataFrame containing associated photon data.
            event_col (str, optional): Column name for event ID. Defaults to None.
            x_col, y_col (str): Column names for spatial coordinates.
            title (str, optional): Custom title for the plot.
        """
        if df is None:
            raise ValueError("A DataFrame must be provided.")
        if event_col is None:
            event_col = 'assoc_cluster_id'  # Default, can be overridden
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