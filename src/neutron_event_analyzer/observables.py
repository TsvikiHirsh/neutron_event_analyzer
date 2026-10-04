"""Per-event observables of event-mode data and their comparison.

Five shape-normalised distributions are built from an associated pixel table
(``Analyse.associate()``: one row per pixel, columns ``px/*``, ``ph/*``, ``ev/*``):

==========  ==================================================================
``ph_n``    pixels per photon cluster (integer bins 0-20)
``ev_n``    photon clusters per event (integer bins 0-10)
``ev_dx``   x residual between a pixel and its event position (40 bins, -10..10 px)
``ev_dtoa`` arrival time of a pixel after its event (40 bins, 0..500 ns)
``ev_sep``  distance of each extra cluster from the largest cluster of its event
            (0.5 px bins up to 12 px, 4 px bins up to 60 px)
==========  ==================================================================

:func:`compare` returns the weighted symmetric chi-squared distance of each
observable and their sum, the objective used to calibrate a detector model
against measured data.

Two selections are available when building the distributions: an event-size
window (``ev_n_min``, ``ev_n_max``, e.g. single-cluster or multi-cluster events)
and a neutron-energy window from the event time of flight (``energy_window``,
with an optional time-of-flight calibration). Per-event weights (column
``ev/w``, see :func:`spectrum_weights`) let one data set be compared at the
neutron-energy mix of another.

Example
-------
>>> from neutron_event_analyzer import Analyse, observables as obs
>>> exp = Analyse('measured_run').associate(); sim = Analyse('simulated_run').associate()
>>> d_exp = obs.extract_distributions(exp_df, ev_n_min=2)
>>> d_sim = obs.extract_distributions(sim_df, ev_n_min=2)
>>> obs.compare(d_exp, d_sim)['total']
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

PHN_BINS = np.arange(21)
EVN_BINS = np.arange(11)
DX_BINS = np.linspace(-10, 10, 41)
DTOA_BINS = np.linspace(0, 5e-7, 41)
SEP_BINS = np.r_[np.arange(0, 12, 0.5), np.arange(12, 64, 4)]
DEFAULT_WEIGHTS = {'ph_n': 3.0, 'ev_n': 3.0, 'ev_dx': 5.0, 'ev_dtoa': 3.0, 'ev_sep': 5.0}
METRICS = tuple(DEFAULT_WEIGHTS)

NEUTRON_MASS_MEV = 939.56542
C_M_PER_S = 299_792_458.0


# ---------------------------------------------------------------- time of flight

def energy_from_tof(tof_s, flight_path_m: float, L0: float = 1.0, t0_s: float = 0.0):
    """Neutron kinetic energy (MeV) from the time of flight (s), relativistic.

    ``L0`` and ``t0_s`` apply the time-of-flight calibration in the convention of
    nres (``tof_true = tof + (1 - L0) tof + t0``); the defaults mean no calibration.
    Non-physical times give NaN.
    """
    tof = np.asarray(tof_s, dtype=float)
    tof = tof + (1.0 - L0) * tof + t0_s
    with np.errstate(divide='ignore', invalid='ignore'):
        beta = flight_path_m / (tof * C_M_PER_S)
        gamma = 1.0 / np.sqrt(1.0 - beta ** 2)
    e = NEUTRON_MASS_MEV * (gamma - 1.0)
    return np.where((tof > 0) & (beta < 1), e, np.nan)


def event_energies(df: pd.DataFrame, flight_path_m: float, L0: float = 1.0, t0_s: float = 0.0) -> pd.Series:
    """Energy of every event (index ``ev/id``) from the ``ev/tof`` column."""
    ev = df.dropna(subset=['ev/id']).drop_duplicates('ev/id').set_index('ev/id')['ev/tof']
    return pd.Series(energy_from_tof(ev.to_numpy(), flight_path_m, L0, t0_s), index=ev.index)


# ---------------------------------------------------------------- selection

def _select(df, ev_n_min=None, ev_n_max=None, energy_window=None, flight_path_m=None,
            L0=1.0, t0_s=0.0):
    """Rows matched to both a photon cluster and an event, inside the selections.

    The cluster size (``ph/n``) is EMPIR's pixel count of the cluster when the
    export carries it (column ``ph/npx``), else the number of associated pixels;
    the event size (``ev/n``) is the number of clusters associated to the event."""
    d = df.dropna(subset=['ph/id', 'ev/id'])
    # cluster size: EMPIR's own pixel count when exported (EMPIR >= 1.0.1), else the
    # number of pixels associated to the cluster
    nph = d['ph/npx'] if 'ph/npx' in d.columns else d.groupby('ph/id')['ph/id'].transform('size')
    d = d.assign(**{'ph/n': nph, 'ev/n': d.groupby('ev/id')['ph/id'].transform('nunique')})
    keep = pd.Series(True, index=d.index)
    if ev_n_min is not None:
        keep &= d['ev/n'] >= ev_n_min
    if ev_n_max is not None:
        keep &= d['ev/n'] <= ev_n_max
    if energy_window is not None:
        if flight_path_m is None:
            raise ValueError('energy_window needs flight_path_m')
        e = energy_from_tof(d['ev/tof'].to_numpy(dtype=float), flight_path_m, L0, t0_s)
        keep &= (e >= energy_window[0]) & (e < energy_window[1])
    return d[keep.to_numpy()]


# ---------------------------------------------------------------- distributions

def _hist(values, bins, weights, normalize):
    h, _ = np.histogram(values, bins=bins, weights=weights)
    h = h.astype(float)
    s = h.sum()
    return h / s if (normalize and s > 0) else h


def extract_distributions(df: pd.DataFrame, ev_n_min: Optional[int] = None, ev_n_max: Optional[int] = None,
                          energy_window=None, flight_path_m: Optional[float] = None, L0: float = 1.0,
                          t0_s: float = 0.0, weight_col: Optional[str] = None, normalize: bool = True) -> dict:
    """The five distributions (see the module docstring) of an associated table.

    ``weight_col`` names a per-row column holding the weight of the row's event
    (e.g. ``ev/w`` from :func:`spectrum_weights`); rows without it count once.
    """
    d = _select(df, ev_n_min, ev_n_max, energy_window, flight_path_m, L0, t0_s)
    w = d[weight_col].to_numpy(dtype=float) if weight_col else np.ones(len(d))
    d = d.assign(_w=w)
    out = {}
    ph = d.drop_duplicates('ph/id')
    out['ph_n'] = _hist(ph['ph/n'].to_numpy(), np.r_[PHN_BINS, PHN_BINS[-1] + 1] - 0.5, ph['_w'].to_numpy(), normalize)
    ev = d.drop_duplicates('ev/id')
    out['ev_n'] = _hist(ev['ev/n'].to_numpy(), np.r_[EVN_BINS, EVN_BINS[-1] + 1] - 0.5, ev['_w'].to_numpy(), normalize)
    out['ev_dx'] = _hist((d['px/x'] - d['ev/x']).to_numpy(), DX_BINS, d['_w'].to_numpy(), normalize)
    out['ev_dtoa'] = _hist((d['px/toa'] - d['ev/toa']).to_numpy(), DTOA_BINS, d['_w'].to_numpy(), normalize)
    # extra clusters of multi-cluster events against the largest cluster of the event
    phs = ph[ph.groupby('ev/id')['ph/id'].transform('size') >= 2]
    sep, wsep = np.array([]), np.array([])
    if len(phs):
        phs = phs.sort_values(['ev/id', 'ph/n'], ascending=[True, False], kind='stable')
        lead = phs.groupby('ev/id')[['ph/x', 'ph/y']].transform('first')
        rest = (phs.groupby('ev/id').cumcount() > 0).to_numpy()
        sep = np.hypot((phs['ph/x'] - lead['ph/x']).to_numpy()[rest], (phs['ph/y'] - lead['ph/y']).to_numpy()[rest])
        wsep = phs['_w'].to_numpy()[rest]
    out['ev_sep'] = _hist(sep, SEP_BINS, wsep, normalize)
    return out


def select_exact(pixels: pd.DataFrame, photons: pd.DataFrame, ev_n_min: Optional[int] = None,
                 ev_n_max: Optional[int] = None, energy_window=None, flight_path_m: Optional[float] = None,
                 L0: float = 1.0, t0_s: float = 0.0):
    """Events of an exact association (``Analyse.associate_exact(pixels=True)`` and its
    ``exact_photons_df``) inside the selections, by EMPIR's photon count per event (``ev/n``) and
    the event energy. Returns (pixel rows, photon rows, event table with ``ev/E`` if a window is set)."""
    ev = photons.drop_duplicates('ev/id')[['ev/id', 'ev/n', 'ev/tof']].set_index('ev/id')
    keep = pd.Series(True, index=ev.index)
    if ev_n_min is not None:
        keep &= ev['ev/n'] >= ev_n_min
    if ev_n_max is not None:
        keep &= ev['ev/n'] <= ev_n_max
    if energy_window is not None:
        if flight_path_m is None:
            raise ValueError('energy_window needs flight_path_m')
        e = energy_from_tof(ev['ev/tof'].to_numpy(dtype=float), flight_path_m, L0, t0_s)
        ev = ev.assign(**{'ev/E': e})
        keep &= (e >= energy_window[0]) & (e < energy_window[1])
    ids = ev.index[keep.to_numpy()]
    return pixels[pixels['ev/id'].isin(ids)], photons[photons['ev/id'].isin(ids)], ev.loc[ids]


def extract_distributions_exact(pixels: pd.DataFrame, photons: pd.DataFrame, weights=None,
                                normalize: bool = True) -> dict:
    """The five distributions from an exact association (see :func:`select_exact`).

    Cluster size (EMPIR's pixel count), clusters per event (EMPIR's count) and the distance of
    every extra cluster from the largest one come from the photon rows, which hold every cluster
    of every event; the pixel residual and the pixel arrival time come from the pixel rows.
    ``weights``: per-event weights (Series indexed by ``ev/id``), else every event counts once.
    """
    wmap = (lambda ids: ids.map(weights).to_numpy(dtype=float)) if weights is not None else \
        (lambda ids: np.ones(len(ids)))
    out = {}
    out['ph_n'] = _hist(photons['ph/npx'].to_numpy(), np.r_[PHN_BINS, PHN_BINS[-1] + 1] - 0.5,
                        wmap(photons['ev/id']), normalize)
    ev = photons.drop_duplicates('ev/id')
    out['ev_n'] = _hist(ev['ev/n'].to_numpy(), np.r_[EVN_BINS, EVN_BINS[-1] + 1] - 0.5, wmap(ev['ev/id']), normalize)
    d = pixels.dropna(subset=['ph/id', 'ev/id'])
    wd = wmap(d['ev/id'])
    out['ev_dx'] = _hist((d['px/x'] - d['ev/x']).to_numpy(), DX_BINS, wd, normalize)
    out['ev_dtoa'] = _hist((d['px/toa'] - d['ev/toa']).to_numpy(), DTOA_BINS, wd, normalize)
    m = photons[photons['ev/n'] >= 2].sort_values(['ev/id', 'ph/npx', 'ph/toa'], ascending=[True, False, True],
                                                  kind='stable')
    sep, wsep = np.array([]), np.array([])
    if len(m):
        lead = m.groupby('ev/id')[['ph/x', 'ph/y']].transform('first')
        rest = (m.groupby('ev/id').cumcount() > 0).to_numpy()
        sep = np.hypot((m['ph/x'] - lead['ph/x']).to_numpy()[rest], (m['ph/y'] - lead['ph/y']).to_numpy()[rest])
        wsep = wmap(m['ev/id'])[rest]
    out['ev_sep'] = _hist(sep, SEP_BINS, wsep, normalize)
    return out


def chi2_sym(p, q, eps: float = 1e-10) -> float:
    """Symmetric chi-squared distance of two histograms."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    return float(np.sum((p - q) ** 2 / (p + q + eps)))


def compare(exp: dict, sim: dict, weights: Optional[dict] = None) -> dict:
    """Weighted symmetric chi-squared per observable and their sum (``'total'``)."""
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)
    scores = {k: chi2_sym(exp[k], sim[k]) * w[k] for k in METRICS if k in exp and k in sim}
    scores['total'] = sum(scores.values())
    return scores


# ---------------------------------------------------------------- energy weights

def spectrum_weights(target_energies, source_energies, edges=None, max_weight: float = 20.0):
    """Per-event weights that give the source events the energy mix of the target.

    Both inputs are per-event energies (MeV); events outside ``edges`` get weight 0.
    Returns an array aligned with ``source_energies``, normalised to a mean of 1 over
    the weighted events. Bins without source events cannot be reweighted and are
    reported by a warning-free zero weight; ``max_weight`` caps the weight of
    sparsely populated source bins.
    """
    if edges is None:
        edges = np.arange(1.0, 10.0001, 0.5)
    t, _ = np.histogram(np.asarray(target_energies, float), bins=edges)
    s, _ = np.histogram(np.asarray(source_energies, float), bins=edges)
    t = t / max(t.sum(), 1); s = s / max(s.sum(), 1)
    ratio = np.divide(t, s, out=np.zeros_like(t, dtype=float), where=s > 0)
    ratio = np.minimum(ratio, max_weight)
    idx = np.digitize(np.asarray(source_energies, float), edges) - 1
    inside = (idx >= 0) & (idx < len(ratio))
    w = np.zeros(len(idx)); w[inside] = ratio[idx[inside]]
    m = w[w > 0].mean() if (w > 0).any() else 1.0
    return w / m
