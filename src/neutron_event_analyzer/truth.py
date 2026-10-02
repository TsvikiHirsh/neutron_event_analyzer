"""Join the simulation truth of G4LumaCam to an associated table.

G4LumaCam writes, next to every simulated tpx3 file, a TracedPhotons table with one
row per pixel hit (pixel coordinates, the hit time in 1.5625 ns clock ticks, whether
the hit reached the tpx3 file, the id of the scintillation photon and its expected
image position) and a SimPhotons table with the Geant4 truth of every photon
(emission point, parent particle and its creation vertex, neutron energy, last
interaction and entry point of the neutron).

:func:`join_truth` attaches this truth to every pixel row of an associated table
(``Analyse.associate()``): exported pixel hits are matched to traced hits on the
exact key (x, y, clock tick), which is unique because a pixel cannot fire twice
within its dead time, and traced hits to SimPhotons rows on (file, photon id,
pulse id). Truth columns are prefixed ``sim/``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

TICK_S = 1.5625e-9
DEFAULT_TRUTH = ('x', 'y', 'z', 'px', 'py', 'pz', 'parentName', 'parentEnergy', 'neutronEnergy',
                 'nx', 'ny', 'nz', 'nx0', 'ny0', 'nz0')


def _file_index(path: Path) -> int:
    return int(path.stem.rsplit('_', 1)[-1])


def load_truth(archive, trace_dir='TracedPhotons', columns: Iterable[str] = DEFAULT_TRUTH) -> pd.DataFrame:
    """Traced pixel hits that reached the tpx3 files, with their SimPhotons truth."""
    archive = Path(archive)
    tdir = Path(trace_dir) if Path(trace_dir).is_absolute() else archive / trace_dir
    parts = []
    for tf in sorted(tdir.glob('traced_sim_data_*.csv'), key=_file_index):
        k = _file_index(tf)
        t = pd.read_csv(tf)
        if 'in_tpx3' in t.columns:
            t = t[t['in_tpx3'].astype(str).str.lower().isin(('true', '1'))]
        sf = archive / 'SimPhotons' / f'sim_data_{k}.csv'
        head = pd.read_csv(sf, nrows=0).columns
        cols = ['id', 'pulse_id'] + [c for c in columns if c in head and c not in ('id', 'pulse_id')]
        s = pd.read_csv(sf, usecols=cols).drop_duplicates(['id', 'pulse_id'])
        t = t.merge(s, on=['id', 'pulse_id'], how='left', suffixes=('', '_sim'))
        t['file'] = k
        parts.append(t)
    if not parts:
        raise FileNotFoundError(f'no traced_sim_data_*.csv in {tdir}')
    return pd.concat(parts, ignore_index=True)


def join_truth(assoc: pd.DataFrame, archive, trace_dir='TracedPhotons', columns: Iterable[str] = DEFAULT_TRUTH,
               truth: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Associated table with the simulation truth of every pixel row (``sim/*`` columns).

    Rows whose pixel hit is not found in the traced hits keep NaN truth (e.g. hits
    from another tool or noise). The match rate is stored in ``df.attrs['truth_match']``.
    """
    tr = load_truth(archive, trace_dir, columns) if truth is None else truth
    tr = tr.assign(_k=list(zip(tr['pixel_x'].astype(int), tr['pixel_y'].astype(int), tr['toa_tick'].astype(np.int64))))
    tr = tr.drop_duplicates('_k')
    keep = ['_k', 'id', 'sim_id', 'neutron_id', 'pulse_id', 'pulse_time_ns', 'x_opt', 'y_opt', 'file'] + \
           [c for c in columns if c in tr.columns]
    tr = tr[[c for c in keep if c in tr.columns]]
    tr = tr.rename(columns={c: f'sim/{c}' for c in tr.columns if c != '_k'})
    a = assoc.copy()
    tick = np.rint(a['px/toa'].to_numpy(dtype=float) / TICK_S).astype(np.int64)
    a['_k'] = list(zip(a['px/x'].round().astype(int), a['px/y'].round().astype(int), tick))
    out = a.merge(tr, on='_k', how='left').drop(columns='_k')
    out.attrs['truth_match'] = float(out['sim/id'].notna().mean()) if len(out) else float('nan')
    return out
