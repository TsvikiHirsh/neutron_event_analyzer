"""Per-photon and per-event topology feature computation."""

import numpy as np
import pandas as pd
from itertools import combinations


def photon_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-photon topology features from pixel hits.

    Parameters
    ----------
    df : DataFrame
        Association output with columns:
        ev/id, ph/id, px/x, px/y, px/toa, px/tot

    Returns
    -------
    DataFrame indexed by (ev/id, ph/id) with feature columns.
    All time quantities are in the same units as px/toa (seconds for CSV output).
    """
    def _features(g):
        x   = g["px/x"].values.astype(float)
        y   = g["px/y"].values.astype(float)
        toa = g["px/toa"].values.astype(float)
        tot = g["px/tot"].values.astype(float)
        npix = len(x)

        # ── spatial spread ───────────────────────────────────────────────────
        dx = x.max() - x.min()
        dy = y.max() - y.min()
        bbox_area     = (dx + 1) * (dy + 1)
        fill_fraction = npix / bbox_area if bbox_area > 0 else 1.0
        aspect        = (dx + 1) / (dy + 1) if dy > 0 else float(dx + 1)

        # ── center of gravity vs highest-ToT pixel ───────────────────────────
        total_tot = tot.sum()
        if total_tot > 0:
            cog_x = (x * tot).sum() / total_tot
            cog_y = (y * tot).sum() / total_tot
        else:
            cog_x, cog_y = x.mean(), y.mean()
        imax       = tot.argmax()
        cog_offset = np.sqrt((cog_x - x[imax])**2 + (cog_y - y[imax])**2)

        # ── ToA spread ───────────────────────────────────────────────────────
        toa_range = toa.max() - toa.min()   # seconds (same units as px/toa)

        if npix >= 3:
            sorted_toa  = np.sort(toa)
            gaps        = np.diff(sorted_toa)
            median_gap  = np.median(gaps)
            toa_bimodal_ratio = (gaps.max() / median_gap) if median_gap > 0 else 0.0
        else:
            toa_bimodal_ratio = 0.0

        # ── pairwise pixel distances ─────────────────────────────────────────
        if npix >= 2:
            dists    = [np.sqrt((x[a]-x[b])**2 + (y[a]-y[b])**2)
                        for a, b in combinations(range(npix), 2)]
            max_dist = max(dists)
        else:
            max_dist = 0.0

        # ── intensity ────────────────────────────────────────────────────────
        hot_ratio = float(tot.max() / tot.mean()) if tot.mean() > 0 else 1.0

        # ── track: correlation of position along major axis with ToA ─────────
        # A relativistic particle crossing the sensor leaves a trail where
        # arrival time progresses monotonically along the track direction.
        track_corr = 0.0
        if npix >= 4:
            coords = np.column_stack([x - x.mean(), y - y.mean()])
            _, _, vt = np.linalg.svd(coords, full_matrices=False)
            proj   = coords @ vt[0]          # projection along major axis
            toa_c  = toa - toa.mean()
            if np.std(proj) > 0 and np.std(toa_c) > 0:
                track_corr = abs(float(np.corrcoef(proj, toa_c)[0, 1]))

        return pd.Series({
            "ph/npix":            npix,
            "ph/bbox_area":       bbox_area,
            "ph/fill_frac":       fill_fraction,
            "ph/aspect":          aspect,
            "ph/max_dist":        max_dist,
            "ph/toa_range":       toa_range,       # seconds
            "ph/toa_bimodal":     toa_bimodal_ratio,
            "ph/cog_offset":      cog_offset,
            "ph/cog_on_max_tot":  bool(cog_offset < 0.6),
            "ph/total_tot":       float(total_tot),
            "ph/max_tot":         float(tot.max()),
            "ph/mean_tot":        float(tot.mean()),
            "ph/hot_ratio":       hot_ratio,
            "ph/track_corr":      track_corr,
        })

    return df.groupby(["ev/id", "ph/id"]).apply(_features, include_groups=False)


def event_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-event topology features from photon positions.

    Parameters
    ----------
    df : DataFrame
        Association output with columns:
        ev/id, ph/id, ph/x, ph/y, px/toa, ev/x, ev/y

    Returns
    -------
    DataFrame indexed by ev/id with feature columns.
    """
    def _features(g):
        ph  = g.groupby("ph/id").first()
        nph = len(ph)

        px = ph["ph/x"].values.astype(float)
        py = ph["ph/y"].values.astype(float)

        if nph >= 2:
            dists        = [np.sqrt((px[a]-px[b])**2 + (py[a]-py[b])**2)
                            for a, b in combinations(range(nph), 2)]
            max_ph_dist  = max(dists)
            mean_ph_dist = float(np.mean(dists))
            min_ph_dist  = min(dists)
        else:
            max_ph_dist  = 0.0
            mean_ph_dist = 0.0
            min_ph_dist  = 0.0

        # ToA spread between photons (earliest pixel per photon)
        ph_toas   = g.groupby("ph/id")["px/toa"].min()
        toa_spread = float(ph_toas.max() - ph_toas.min()) if nph >= 2 else 0.0

        # distance of photons from reconstructed event position
        ev_x    = float(g["ev/x"].iloc[0])
        ev_y    = float(g["ev/y"].iloc[0])
        radii   = np.sqrt((px - ev_x)**2 + (py - ev_y)**2)
        mean_radius = float(radii.mean())

        # linearity of photon arrangement (SVD ratio of singular values)
        if nph >= 3:
            centered = np.column_stack([px - px.mean(), py - py.mean()])
            _, s, _  = np.linalg.svd(centered, full_matrices=False)
            linearity = float(s[0] / (s[1] + 1e-9))
        else:
            linearity = 0.0

        return pd.Series({
            "ev/nph":           nph,
            "ev/max_ph_dist":   max_ph_dist,
            "ev/mean_ph_dist":  mean_ph_dist,
            "ev/min_ph_dist":   min_ph_dist,
            "ev/ph_toa_spread": toa_spread,
            "ev/mean_radius":   mean_radius,
            "ev/linearity":     linearity,
        })

    return df.groupby("ev/id").apply(_features, include_groups=False)
