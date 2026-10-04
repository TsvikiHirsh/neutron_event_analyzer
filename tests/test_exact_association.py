"""Exact photon-event association (neutron_event_analyzer.exact) on a synthetic EMPIR-like list."""
import numpy as np
import pandas as pd
from neutron_event_analyzer.exact import associate_exact, event_positions, TICK_S


def _event(rows, members):
    xs = np.array([rows[k][0] for k in members]); ys = np.array([rows[k][1] for k in members])
    return round(xs.mean(), 2), round(ys.mean(), 2)


def test_exact_membership_and_largest():
    # photons: (x, y, tick, npx); event A = {0, 2, 3}, event B = {1, 4}, single C = {5};
    # photon 6 lies in A's window but is not part of A (a nearer-centroid decoy for a naive match)
    rows = [(10.00, 10.00, 100, 3), (60.00, 60.00, 101, 5), (14.00, 11.00, 103, 9), (12.50, 9.50, 120, 2),
            (62.00, 61.00, 130, 4), (200.0, 5.0, 500, 7), (12.10, 10.20, 104, 6)]
    ph = pd.DataFrame({"x": [r[0] for r in rows], "y": [r[1] for r in rows],
                       "t": [r[2] * TICK_S for r in rows], "npx": [r[3] for r in rows]})
    groups = [[0, 2, 3], [1, 4], [5]]
    ev = pd.DataFrame([dict(zip(("x", "y"), _event(rows, g)), t=rows[g[0]][2] * TICK_S, n=len(g)) for g in groups])
    fe = pd.DataFrame([dict(x=rows[g[0]][0], y=rows[g[0]][1], t=rows[g[0]][2] * TICK_S, n=len(g)) for g in groups])
    members, status = associate_exact(ph, ev, fe)
    assert (status == 0).all()
    got = [sorted(members.loc[members.event == i, "photon"]) for i in range(3)]
    assert got == [sorted(g) for g in groups]
    pos = event_positions(ph, ev, fe, members)
    assert pos["ev/x_largest"].tolist() == [14.0, 60.0, 200.0]     # most pixels in each event
