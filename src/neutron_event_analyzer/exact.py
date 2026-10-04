"""
Exact photon-to-event association for EMPIR photon2event output.

EMPIR's event lists do not record which photons form an event, but three properties of
photon2event fix the membership completely:

* an event starts at its earliest photon (the event time is that photon's time),
* it spans at most the maximum event duration, and
* its position is the mean of the positions of its photons.

Reconstructing the same photons a second time with the earliest-photon position
(algorithm ``noBranchChain_ghostPhoton_firstPhotonPos_direct``, identical event list)
identifies the earliest photon of every event. The remaining n - 1 photons are then the
subset of the photons within the event duration whose mean, together with the earliest
photon, reproduces the event position. The subset is searched depth first in time order,
so the earliest consistent photons are taken. Photons may belong to two events (EMPIR's
ghost photons), so events are matched independently.

Works on the standard CSV exports: times are rounded to the 1.5625 ns clock tick (exact
for t < 10^4 s with the 13 significant digits of the export) and positions are compared in
units of 0.01 pixel, the precision of the export.
"""
import numpy as np
import pandas as pd

TICK_S = 1.5625e-9
STATUS = {0: "exact", 1: "earliest photon not found", 2: "no exact subset",
          3: "search capped", 4: "too many candidate photons"}


def _ticks(t):
    return np.rint(np.asarray(t, dtype=float) / TICK_S).astype(np.int64)


def _units(v, per_px):
    return np.rint(np.asarray(v, dtype=float) * per_px).astype(np.int64)


def _search(pool, px, py, need, tx, ty, tol, max_steps):
    """Depth-first search for `need` photons of `pool` with sums (tx, ty) +- tol.
    Returns (status, chosen) with status 1 found, 0 none, -1 capped."""
    m = len(pool)
    if m < need:
        return 0, None
    x = px[pool]
    y = py[pool]
    # suffix minima / maxima for pruning
    mnx = np.minimum.accumulate(x[::-1])[::-1]
    mxx = np.maximum.accumulate(x[::-1])[::-1]
    mny = np.minimum.accumulate(y[::-1])[::-1]
    mxy = np.maximum.accumulate(y[::-1])[::-1]
    stack = [-1] * need
    sx = [0] * (need + 1)
    sy = [0] * (need + 1)
    depth, steps = 0, 0
    while depth >= 0:
        steps += 1
        if steps > max_steps:
            return -1, None
        stack[depth] += 1
        j = stack[depth]
        if j > m - (need - depth):
            depth -= 1
            continue
        sx[depth + 1] = sx[depth] + x[j]
        sy[depth + 1] = sy[depth] + y[j]
        if depth == need - 1:
            if abs(sx[need] - tx) <= tol and abs(sy[need] - ty) <= tol:
                return 1, [pool[k] for k in stack]
            continue
        k = need - depth - 1
        nxt = j + 1
        rx, ry = tx - sx[depth + 1], ty - sy[depth + 1]
        if (rx < k * mnx[nxt] - tol or rx > k * mxx[nxt] + tol or
                ry < k * mny[nxt] - tol or ry > k * mxy[nxt] + tol):
            continue
        depth += 1
        stack[depth] = stack[depth - 1]
    return 0, None


def associate_exact(photons, events, first_events, duration_s=1e-6, units_per_px=100,
                    max_pool=80, max_steps=200_000):
    """
    Associate EMPIR photons with EMPIR events exactly.

    Args:
        photons (DataFrame): photon export (columns x, y, t; npx optional).
        events (DataFrame): event export of the photon-mean reconstruction (x, y, t, n).
        first_events (DataFrame): event export of the earliest-photon reconstruction of the
            same photons with the same settings (x, y, t, n).
        duration_s (float): photon2event durationMax_s.
        units_per_px (int): position units of the comparison (100 = 0.01 px, CSV exports).
        max_pool, max_steps: limits for burst events; beyond them the n - 1 window photons
            closest to the event position are taken (status 3 or 4).

    Returns:
        members (DataFrame): one row per (event, photon): ``event`` (row index into
            ``events``), ``photon`` (row index into ``photons``), ``rank`` (0 = earliest).
        status (ndarray): per event, see ``STATUS``.
    """
    ev_t = _ticks(events["t"])
    if not (np.array_equal(ev_t, _ticks(first_events["t"])) and
            np.array_equal(np.asarray(events["n"], int), np.asarray(first_events["n"], int))):
        raise ValueError("photon-mean and earliest-photon event lists differ")
    order = np.argsort(_ticks(photons["t"]), kind="stable")
    pt = _ticks(photons["t"])[order]
    px = _units(photons["x"], units_per_px)[order]
    py = _units(photons["y"], units_per_px)[order]
    ex = _units(events["x"], units_per_px)
    ey = _units(events["y"], units_per_px)
    fx = _units(first_events["x"], units_per_px)
    fy = _units(first_events["y"], units_per_px)
    en = np.asarray(events["n"], dtype=np.int64)
    dur = int(round(duration_s / TICK_S))
    lo_all = np.searchsorted(pt, ev_t, "left")
    hi_all = np.searchsorted(pt, ev_t + dur, "right")
    n_ev = len(ev_t)
    status = np.zeros(n_ev, np.int8)
    ev_idx, ph_idx, rank = [], [], []
    for i in range(n_ev):
        lo, hi = lo_all[i], hi_all[i]
        f = -1
        j = lo
        while j < hi and pt[j] == ev_t[i]:
            if px[j] == fx[i] and py[j] == fy[i]:
                f = j
                break
            j += 1
        if f < 0:
            status[i] = 1
            continue
        chosen = [f]
        n = en[i]
        if n > 1:
            pool = [k for k in range(lo, hi) if k != f]
            res = 0
            if len(pool) <= max_pool:
                res, sub = _search(np.asarray(pool, np.int64), px, py, n - 1,
                                   ex[i] * n - px[f], ey[i] * n - py[f], n + 1, max_steps)
            if res == 1:
                chosen += sub
            else:
                status[i] = 4 if len(pool) > max_pool else (3 if res == -1 else 2)
                d = (px[pool] - ex[i]) ** 2 + (py[pool] - ey[i]) ** 2
                chosen += [pool[k] for k in np.argsort(d, kind="stable")[:n - 1]]
        ev_idx += [i] * len(chosen)
        ph_idx += list(order[chosen])
        rank += list(range(len(chosen)))
    members = pd.DataFrame({"event": np.asarray(ev_idx, np.int64), "photon": np.asarray(ph_idx, np.int64),
                            "rank": np.asarray(rank, np.int64)})
    return members, status


def event_positions(photons, events, first_events, members):
    """Photon-mean, earliest-photon and largest-cluster positions per event.

    The largest cluster is the member with the most pixels (photon column ``npx``,
    exported by EMPIR >= 1.0.1); ties go to the earliest photon."""
    if "npx" not in photons.columns:
        raise ValueError("photon export without pixel count (EMPIR >= 1.0.1 needed)")
    m = members.join(photons[["x", "y", "npx"]].reset_index(drop=True), on="photon")
    big = m.sort_values(["event", "npx", "rank"], ascending=[True, False, True]).groupby("event").first()
    out = pd.DataFrame({"ev/x_cog": np.asarray(events["x"], float), "ev/y_cog": np.asarray(events["y"], float),
                        "ev/x_first": np.asarray(first_events["x"], float),
                        "ev/y_first": np.asarray(first_events["y"], float)})
    out["ev/x_largest"] = big["x"].reindex(out.index)
    out["ev/y_largest"] = big["y"].reindex(out.index)
    out["ev/npx_largest"] = big["npx"].reindex(out.index)
    return out
