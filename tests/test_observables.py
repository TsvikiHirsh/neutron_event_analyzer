"""Tests of the per-event observables (neutron_event_analyzer.observables)."""
import numpy as np
import pandas as pd
import pytest

from neutron_event_analyzer import observables as obs


def _table():
    """Two events: event 1 = one cluster of 3 pixels, event 2 = two clusters (4 and 2 pixels, 5 px apart)."""
    rows = []
    def cluster(ev, ph, x0, y0, n, t0, ex, ey, et, tof):
        for i in range(n):
            rows.append({'px/x': x0 + 0.1 * i, 'px/y': y0, 'px/toa': t0 + i * 10e-9, 'ph/id': ph, 'ph/x': x0, 'ph/y': y0,
                         'ph/toa': t0, 'ev/id': ev, 'ev/x': ex, 'ev/y': ey, 'ev/toa': et, 'ev/tof': tof})
    cluster(1, 1, 10.0, 10.0, 3, 1.0, 10.0, 10.0, 1.0, 400e-9)
    cluster(2, 2, 50.0, 50.0, 4, 2.0, 52.5, 50.0, 2.0, 300e-9)
    cluster(2, 3, 55.0, 50.0, 2, 2.0 + 20e-9, 52.5, 50.0, 2.0, 300e-9)
    return pd.DataFrame(rows)


def test_counts_and_separation():
    d = obs.extract_distributions(_table(), normalize=False)
    assert d['ph_n'][3] == 1 and d['ph_n'][4] == 1 and d['ph_n'][2] == 1
    assert d['ev_n'][1] == 1 and d['ev_n'][2] == 1
    assert d['ev_sep'].sum() == 1                      # one extra cluster
    assert d['ev_sep'][np.searchsorted(obs.SEP_BINS, 5.0, side='right') - 1] == 1
    assert d['ev_dx'].sum() == 9 and d['ev_dtoa'].sum() == 9


def test_event_size_selection():
    d = obs.extract_distributions(_table(), ev_n_min=2, normalize=False)
    assert d['ev_n'].sum() == 1 and d['ph_n'].sum() == 2


def test_energy_window_and_tof():
    e = obs.energy_from_tof(np.array([300e-9, 400e-9]), 10.85)
    assert 6.0 < e[0] < 7.0 and 3.5 < e[1] < 4.0     # ~6.9 and ~3.8 MeV over 10.85 m
    d = obs.extract_distributions(_table(), energy_window=(5, 10), flight_path_m=10.85, normalize=False)
    assert d['ev_n'][2] == 1 and d['ev_n'][1] == 0    # only the faster event remains


def test_compare_and_weights():
    d = obs.extract_distributions(_table())
    s = obs.compare(d, d)
    assert s['total'] == pytest.approx(0.0)
    w = obs.spectrum_weights([2.0, 2.0, 7.0], [2.0, 7.0, 7.0, 7.0], edges=np.array([1.0, 5.0, 10.0]))
    assert w[0] > w[1] and np.mean(w) == pytest.approx(1.0)


def test_weighted_histograms():
    t = _table()
    t['ev/w'] = t['ev/id'].map({1: 2.0, 2: 1.0})
    d = obs.extract_distributions(t, weight_col='ev/w', normalize=False)
    assert d['ev_n'][1] == 2.0 and d['ev_n'][2] == 1.0
