"""Main tagging pipeline entry point."""

import pandas as pd
from tqdm import tqdm

from .dtypes import frozenset_to_str
from .features import photon_features, event_features
from .tags import assign_ph_tags, assign_ev_tags

_PX_REQUIRED = {"ev/id", "ph/id", "px/x", "px/y", "px/toa", "px/tot"}
_EV_REQUIRED = {"ev/id", "ph/id", "ph/x", "ph/y", "px/toa", "ev/x", "ev/y"}


def add_topology_tags(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Compute features and assign topology tags.

    Adds (or replaces) columns:
      - ``ph/tags`` — Unicode symbol string of photon topology tags
      - ``ev/tags`` — Unicode symbol string of event topology tags

    Parameters
    ----------
    df : DataFrame
        Association output produced by ``Analyse.associate()`` or loaded from
        the saved CSV.  Must contain pixel-level rows with columns:
        ev/id, ph/id, px/x, px/y, px/toa, px/tot, ph/x, ph/y, ev/x, ev/y.
    verbose : bool
        Show tqdm progress bars (default True).

    Returns
    -------
    DataFrame with ``ph/tags`` and ``ev/tags`` columns appended (or replaced).
    """
    missing_ph = _PX_REQUIRED - set(df.columns)
    missing_ev = _EV_REQUIRED - set(df.columns)
    if missing_ph or missing_ev:
        raise ValueError(
            f"Missing columns for tagging — "
            f"photon features need: {missing_ph or 'OK'}, "
            f"event features need: {missing_ev or 'OK'}"
        )

    # Drop existing tag columns so we can cleanly replace them.
    df = df.drop(columns=[c for c in ("ph/tags", "ev/tags") if c in df.columns])

    # Separate tagged vs unassociated rows.
    mask     = df["ph/id"].notna() & df["ev/id"].notna()
    tagged   = df[mask].copy()
    untagged = df[~mask].copy()

    n_ph = tagged["ph/id"].nunique()
    n_ev = tagged["ev/id"].nunique()

    steps = [
        ("photon features", None),
        ("photon tags",     None),
        ("event features",  None),
        ("event tags",      None),
    ]

    bar = tqdm(
        total=4,
        desc="Tagging",
        unit="step",
        disable=not verbose,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
    )

    # ── 1. Photon features ────────────────────────────────────────────────────
    bar.set_postfix_str(f"photon features  ({n_ph:,} photons)")
    if verbose:
        tqdm.pandas(desc="  ph features", leave=False, unit="ph")
        pf = tagged.groupby(["ev/id", "ph/id"]).progress_apply(
            _ph_feature_row, include_groups=False
        )
    else:
        pf = photon_features(tagged)
    bar.update(1)

    # ── 2. Photon tags ────────────────────────────────────────────────────────
    bar.set_postfix_str(f"photon tags  ({n_ph:,} photons)")
    pf["_bright"] = pf["ph/total_tot"] > pf["ph/total_tot"].quantile(0.9)
    pf["_dim"]    = pf["ph/max_tot"]   < pf["ph/max_tot"].quantile(0.1)
    pf["ph/tags"] = pf.apply(assign_ph_tags, axis=1).apply(frozenset_to_str)
    tagged = tagged.join(pf["ph/tags"], on=["ev/id", "ph/id"])
    bar.update(1)

    # ── 3. Event features ─────────────────────────────────────────────────────
    bar.set_postfix_str(f"event features  ({n_ev:,} events)")
    if verbose:
        tqdm.pandas(desc="  ev features", leave=False, unit="ev")
        ef = tagged.groupby("ev/id").progress_apply(
            _ev_feature_row, include_groups=False
        )
    else:
        ef = event_features(tagged)
    bar.update(1)

    # ── 4. Event tags ─────────────────────────────────────────────────────────
    bar.set_postfix_str(f"event tags  ({n_ev:,} events)")
    ef["ev/tags"] = ef.apply(assign_ev_tags, axis=1).apply(frozenset_to_str)
    tagged = tagged.join(ef["ev/tags"], on="ev/id")
    bar.update(1)

    bar.set_postfix_str("done")
    bar.close()

    # Re-attach unassociated rows with empty tag strings.
    result = pd.concat([tagged, untagged], sort=False)
    result["ph/tags"] = result["ph/tags"].fillna("")
    result["ev/tags"] = result["ev/tags"].fillna("")

    return result.sort_index()


# ── Feature row helpers (same logic as features.py, kept here so progress_apply
#    can wrap them directly without re-importing the module functions) ──────────

import numpy as np
from itertools import combinations


def _ph_feature_row(g):
    """Per-photon feature computation (mirrors photon_features._features)."""
    x   = g["px/x"].values.astype(float)
    y   = g["px/y"].values.astype(float)
    toa = g["px/toa"].values.astype(float)
    tot = g["px/tot"].values.astype(float)
    npix = len(x)

    dx = x.max() - x.min()
    dy = y.max() - y.min()
    bbox_area     = (dx + 1) * (dy + 1)
    fill_fraction = npix / bbox_area if bbox_area > 0 else 1.0
    aspect        = (dx + 1) / (dy + 1) if dy > 0 else float(dx + 1)

    total_tot = tot.sum()
    if total_tot > 0:
        cog_x = (x * tot).sum() / total_tot
        cog_y = (y * tot).sum() / total_tot
    else:
        cog_x, cog_y = x.mean(), y.mean()
    imax       = tot.argmax()
    cog_offset = np.sqrt((cog_x - x[imax])**2 + (cog_y - y[imax])**2)

    toa_range = toa.max() - toa.min()
    if npix >= 3:
        gaps   = np.diff(np.sort(toa))
        med_g  = np.median(gaps)
        toa_bimodal_ratio = (gaps.max() / med_g) if med_g > 0 else 0.0
    else:
        toa_bimodal_ratio = 0.0

    if npix >= 2:
        dists    = [np.sqrt((x[a]-x[b])**2 + (y[a]-y[b])**2)
                    for a, b in combinations(range(npix), 2)]
        max_dist = max(dists)
    else:
        max_dist = 0.0

    hot_ratio  = float(tot.max() / tot.mean()) if tot.mean() > 0 else 1.0
    track_corr = 0.0
    if npix >= 4:
        coords = np.column_stack([x - x.mean(), y - y.mean()])
        _, _, vt = np.linalg.svd(coords, full_matrices=False)
        proj  = coords @ vt[0]
        toa_c = toa - toa.mean()
        if np.std(proj) > 0 and np.std(toa_c) > 0:
            track_corr = abs(float(np.corrcoef(proj, toa_c)[0, 1]))

    return pd.Series({
        "ph/npix":        npix,
        "ph/bbox_area":   bbox_area,
        "ph/fill_frac":   fill_fraction,
        "ph/aspect":      aspect,
        "ph/max_dist":    max_dist,
        "ph/toa_range":   toa_range,
        "ph/toa_bimodal": toa_bimodal_ratio,
        "ph/cog_offset":  cog_offset,
        "ph/cog_on_max_tot": bool(cog_offset < 0.6),
        "ph/total_tot":   float(total_tot),
        "ph/max_tot":     float(tot.max()),
        "ph/mean_tot":    float(tot.mean()),
        "ph/hot_ratio":   hot_ratio,
        "ph/track_corr":  track_corr,
    })


def _ev_feature_row(g):
    """Per-event feature computation (mirrors event_features._features)."""
    ph  = g.groupby("ph/id").first()
    nph = len(ph)
    px  = ph["ph/x"].values.astype(float)
    py  = ph["ph/y"].values.astype(float)

    if nph >= 2:
        dists        = [np.sqrt((px[a]-px[b])**2 + (py[a]-py[b])**2)
                        for a, b in combinations(range(nph), 2)]
        max_ph_dist  = max(dists)
        mean_ph_dist = float(np.mean(dists))
        min_ph_dist  = min(dists)
    else:
        max_ph_dist = mean_ph_dist = min_ph_dist = 0.0

    ph_toas    = g.groupby("ph/id")["px/toa"].min()
    toa_spread = float(ph_toas.max() - ph_toas.min()) if nph >= 2 else 0.0

    ev_x = float(g["ev/x"].iloc[0])
    ev_y = float(g["ev/y"].iloc[0])
    radii       = np.sqrt((px - ev_x)**2 + (py - ev_y)**2)
    mean_radius = float(radii.mean())

    if nph >= 3:
        centered  = np.column_stack([px - px.mean(), py - py.mean()])
        _, s, _   = np.linalg.svd(centered, full_matrices=False)
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
