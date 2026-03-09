"""
benchmark_photon_event.py
=========================
Profile and compare photon-event association implementations.

Run with:
    /root/.local/bin/micromamba run -n base python3 notebooks/benchmark_photon_event.py
"""

import time
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

RNG = np.random.default_rng(42)

# ── Realistic sizes for PTB out-of-focus data ─────────────────────────────────
N_EVENTS  = 30_000
N_PHOTONS = 80_000
MAX_N_PH_PER_EV = 6      # max photons per event (ev['n'])
DSPACE_PX  = 50.0        # photon_dSpace_px
MAX_TIME_S = 500e-9      # max_time_ns / 1e9


def make_data(n_ev, n_ph, seed=42):
    rng = np.random.default_rng(seed)
    ev_t  = np.sort(rng.uniform(0, 1, n_ev))
    ev_x  = rng.uniform(50, 462, n_ev)
    ev_y  = rng.uniform(50, 462, n_ev)
    ev_n  = rng.integers(1, MAX_N_PH_PER_EV + 1, n_ev)
    ev_psd = rng.uniform(0, 1, n_ev)

    events = pd.DataFrame({'t': ev_t, 'x': ev_x, 'y': ev_y,
                           'n': ev_n, 'PSD': ev_psd})
    events = events.sort_values('t').reset_index(drop=True)
    events['event_id'] = events.index + 1

    # Photons: half matched to an event (within window), half noise
    n_matched = n_ph // 2
    ph_ev_idx = rng.integers(0, n_ev, n_matched)
    ph_t_m = ev_t[ph_ev_idx] + rng.uniform(0, MAX_TIME_S, n_matched)
    ph_x_m = ev_x[ph_ev_idx] + rng.normal(0, 3, n_matched)
    ph_y_m = ev_y[ph_ev_idx] + rng.normal(0, 3, n_matched)

    n_noise = n_ph - n_matched
    ph_t_n = rng.uniform(0, 1, n_noise)
    ph_x_n = rng.uniform(0, 512, n_noise)
    ph_y_n = rng.uniform(0, 512, n_noise)

    photons = pd.DataFrame({
        't': np.concatenate([ph_t_m, ph_t_n]),
        'x': np.concatenate([ph_x_m, ph_x_n]),
        'y': np.concatenate([ph_y_m, ph_y_n]),
    }).sort_values('t').reset_index(drop=True)

    return photons, events


# ── Current implementation ────────────────────────────────────────────────────

def assoc_current(photons_df, events_df, dSpace_px=DSPACE_PX, max_time_s=MAX_TIME_S):
    """Verbatim copy of the current _associate_photons_to_events_simple_window."""
    photons = photons_df.copy()
    events  = events_df.copy()

    photons['assoc_event_id'] = np.nan
    photons['assoc_x']   = np.nan; photons['assoc_y']   = np.nan
    photons['assoc_t']   = np.nan; photons['assoc_n']   = 0
    photons['assoc_PSD'] = 0;      photons['assoc_com_dist'] = np.nan

    photons = photons.sort_values('t').reset_index(drop=True)
    events  = events.sort_values('t').reset_index(drop=True)
    events['event_id'] = events.index + 1

    p_t = photons['t'].to_numpy()
    p_x = photons['x'].to_numpy()
    p_y = photons['y'].to_numpy()
    n_total = len(photons)

    photon_candidates = {}
    left = 0
    for _, ev in events.iterrows():
        et, ex, ey, eid, n = ev['t'], ev['x'], ev['y'], ev['event_id'], int(ev['n'])
        psd = ev.get('PSD', 0)

        while left < n_total and p_t[left] < et:
            left += 1
        right = left
        while right < n_total and p_t[right] <= et + max_time_s:
            right += 1

        if right - left < n:
            continue

        sub_idx = np.arange(left, right)
        spatial_diffs = np.sqrt((p_x[sub_idx] - ex)**2 + (p_y[sub_idx] - ey)**2)
        sort_i = np.argsort(spatial_diffs)[:n]
        sel_x  = p_x[sub_idx][sort_i]
        sel_y  = p_y[sub_idx][sort_i]

        com_dist = spatial_diffs[sort_i[0]] if n == 1 else \
            np.sqrt((sel_x.mean() - ex)**2 + (sel_y.mean() - ey)**2)
        if com_dist > dSpace_px:
            continue

        global_idx = sub_idx[sort_i]
        ev_data = {'ex': ex, 'ey': ey, 'et': et, 'n': n, 'psd': psd}
        for i, loc_idx in enumerate(global_idx):
            t_diff  = (p_t[sub_idx[sort_i[i]]] - et) * 1e9
            sp_diff = spatial_diffs[sort_i[i]]
            if loc_idx not in photon_candidates:
                photon_candidates[loc_idx] = []
            photon_candidates[loc_idx].append((eid, com_dist, sp_diff, t_diff, ev_data))

    for loc_idx, candidates in photon_candidates.items():
        best = min(candidates, key=lambda x: x[1])
        eid, com_dist, sp_diff, t_diff, ev_data = best
        photons.loc[loc_idx, 'assoc_event_id'] = eid
        photons.loc[loc_idx, 'assoc_x']   = ev_data['ex']
        photons.loc[loc_idx, 'assoc_y']   = ev_data['ey']
        photons.loc[loc_idx, 'assoc_t']   = ev_data['et']
        photons.loc[loc_idx, 'assoc_n']   = ev_data['n']
        photons.loc[loc_idx, 'assoc_PSD'] = ev_data['psd']
        photons.loc[loc_idx, 'assoc_com_dist'] = com_dist

    return photons


# ── Optimized implementation ──────────────────────────────────────────────────

def assoc_optimized(photons_df, events_df, dSpace_px=DSPACE_PX, max_time_s=MAX_TIME_S):
    """
    Optimized photon-event association with:
    - Pre-computed window boundaries via np.searchsorted (vectorized, no while loops)
    - Spatial bbox pre-filter before computing sqrt
    - Pre-allocated output arrays (no per-photon dict, no Pass 2)
    - Single-pass conflict resolution via lowest-com array comparison
    - No iterrows() — all event data extracted as numpy arrays upfront
    """
    photons = photons_df.sort_values('t').reset_index(drop=True)
    events  = events_df.sort_values('t').reset_index(drop=True)
    events['event_id'] = events.index + 1

    p_t = photons['t'].to_numpy()
    p_x = photons['x'].to_numpy()
    p_y = photons['y'].to_numpy()
    n_ph = len(photons)

    # Extract all event columns as contiguous arrays — no iterrows
    e_t   = events['t'].to_numpy()
    e_x   = events['x'].to_numpy()
    e_y   = events['y'].to_numpy()
    e_n   = events['n'].to_numpy().astype(np.int32)
    e_psd = events['PSD'].to_numpy() if 'PSD' in events.columns else np.zeros(len(events))
    e_id  = events['event_id'].to_numpy()

    # ── Pre-compute ALL window boundaries at once (vectorized, O(log n) each) ─
    left_arr  = np.searchsorted(p_t, e_t,              side='left')
    right_arr = np.searchsorted(p_t, e_t + max_time_s, side='right')

    # ── Pre-allocated output arrays ───────────────────────────────────────────
    out_eid  = np.full(n_ph, np.nan)
    out_ex   = np.full(n_ph, np.nan)
    out_ey   = np.full(n_ph, np.nan)
    out_et   = np.full(n_ph, np.nan)
    out_en   = np.zeros(n_ph)
    out_psd  = np.zeros(n_ph)
    out_com  = np.full(n_ph, np.inf)   # inf = unassigned; conflict → lower com wins

    # ── Single pass over events (no iterrows, no candidate dict) ─────────────
    for i in range(len(events)):
        lo = int(left_arr[i]);  hi = int(right_arr[i])
        en = e_n[i]
        if hi - lo < en:
            continue

        ex = e_x[i];  ey = e_y[i];  et = e_t[i]
        epsd = e_psd[i];  eid = e_id[i]

        sub = np.arange(lo, hi)

        # Spatial bbox pre-filter (cheap abs, avoids sqrt for distant photons)
        dx = p_x[sub] - ex
        dy = p_y[sub] - ey
        bbox = (np.abs(dx) <= dSpace_px) & (np.abs(dy) <= dSpace_px)
        sub = sub[bbox]
        dx  = dx[bbox]
        dy  = dy[bbox]
        if len(sub) < en:
            continue

        # Exact distances only for bbox survivors
        dists = np.sqrt(dx*dx + dy*dy)
        order = np.argsort(dists)[:en]
        sel   = sub[order]
        sel_d = dists[order]

        com_dist = sel_d[0] if en == 1 else \
            np.sqrt((p_x[sel].mean() - ex)**2 + (p_y[sel].mean() - ey)**2)
        if com_dist > dSpace_px:
            continue

        # Conflict resolution inline: overwrite only if com_dist improves
        improve = out_com[sel] > com_dist
        sel2 = sel[improve]
        if len(sel2) == 0:
            continue
        out_eid[sel2] = eid
        out_ex[sel2]  = ex;   out_ey[sel2]  = ey;   out_et[sel2]  = et
        out_en[sel2]  = en;   out_psd[sel2] = epsd;  out_com[sel2] = com_dist

    # Write back to DataFrame
    out_com_clean = np.where(np.isinf(out_com), np.nan, out_com)
    photons = photons.assign(
        assoc_event_id = out_eid,
        assoc_x  = out_ex,   assoc_y  = out_ey,   assoc_t  = out_et,
        assoc_n  = out_en,   assoc_PSD = out_psd,  assoc_com_dist = out_com_clean,
        time_diff_ns     = np.nan,
        spatial_diff_px  = np.nan,
    )
    return photons


# ── Profile each step separately ─────────────────────────────────────────────

def _time_section(label, fn, *args, n_rep=3, **kwargs):
    best = np.inf
    for _ in range(n_rep):
        t0 = time.perf_counter()
        r = fn(*args, **kwargs)
        best = min(best, time.perf_counter() - t0)
    print(f"  {label:<50s}  {best*1000:7.1f} ms")
    return r, best


def profile_current(photons, events):
    """Break current implementation into sub-steps and time each."""
    photons = photons.sort_values('t').reset_index(drop=True)
    events  = events.sort_values('t').reset_index(drop=True)
    events['event_id'] = events.index + 1

    p_t = photons['t'].to_numpy()
    p_x = photons['x'].to_numpy()
    p_y = photons['y'].to_numpy()
    n_total = len(photons)

    # ── Step A: Pass 1 — build candidates dict ────────────────────────────────
    t0 = time.perf_counter()
    photon_candidates = {}
    left = 0
    for _, ev in events.iterrows():
        et, ex, ey, eid, n = ev['t'], ev['x'], ev['y'], ev['event_id'], int(ev['n'])
        psd = ev.get('PSD', 0)
        while left < n_total and p_t[left] < et:
            left += 1
        right = left
        while right < n_total and p_t[right] <= et + MAX_TIME_S:
            right += 1
        if right - left < n:
            continue
        sub_idx = np.arange(left, right)
        spatial_diffs = np.sqrt((p_x[sub_idx] - ex)**2 + (p_y[sub_idx] - ey)**2)
        sort_i = np.argsort(spatial_diffs)[:n]
        com_dist = spatial_diffs[sort_i[0]] if n == 1 else \
            np.sqrt((p_x[sub_idx][sort_i].mean() - ex)**2 +
                    (p_y[sub_idx][sort_i].mean() - ey)**2)
        if com_dist > DSPACE_PX:
            continue
        global_idx = sub_idx[sort_i]
        ev_data = {'ex': ex, 'ey': ey, 'et': et, 'n': n, 'psd': psd}
        for i, loc_idx in enumerate(global_idx):
            t_diff = (p_t[sub_idx[sort_i[i]]] - et) * 1e9
            sp_diff = spatial_diffs[sort_i[i]]
            if loc_idx not in photon_candidates:
                photon_candidates[loc_idx] = []
            photon_candidates[loc_idx].append((eid, com_dist, sp_diff, t_diff, ev_data))
    t_pass1 = time.perf_counter() - t0
    print(f"    Pass 1 (build candidates dict)               {t_pass1*1000:7.1f} ms  "
          f"({len(photon_candidates)} conflicted photons)")

    # ── Step B: Pass 2 — resolve conflicts ───────────────────────────────────
    t0 = time.perf_counter()
    out = {}
    for loc_idx, candidates in photon_candidates.items():
        best = min(candidates, key=lambda x: x[1])
        out[loc_idx] = best
    t_pass2 = time.perf_counter() - t0
    print(f"    Pass 2 (conflict resolution)                 {t_pass2*1000:7.1f} ms")

    # ── Step C: Write back ────────────────────────────────────────────────────
    t0 = time.perf_counter()
    for loc_idx, (eid, com_dist, sp_diff, t_diff, ev_data) in out.items():
        photons.loc[loc_idx, 'assoc_event_id'] = eid
        photons.loc[loc_idx, 'assoc_com_dist'] = com_dist
    t_wb = time.perf_counter() - t0
    print(f"    Write-back (loc[] assignments)               {t_wb*1000:7.1f} ms")

    return t_pass1 + t_pass2 + t_wb


def profile_optimized(photons, events):
    """Break optimized implementation into sub-steps and time each."""
    photons = photons.sort_values('t').reset_index(drop=True)
    events  = events.sort_values('t').reset_index(drop=True)
    events['event_id'] = events.index + 1

    p_t = photons['t'].to_numpy()
    p_x = photons['x'].to_numpy()
    p_y = photons['y'].to_numpy()
    n_ph = len(photons)

    e_t   = events['t'].to_numpy()
    e_x   = events['x'].to_numpy()
    e_y   = events['y'].to_numpy()
    e_n   = events['n'].to_numpy().astype(np.int32)
    e_psd = events['PSD'].to_numpy() if 'PSD' in events.columns else np.zeros(len(events))
    e_id  = events['event_id'].to_numpy()

    # ── Step A: searchsorted (vectorized) ─────────────────────────────────────
    t0 = time.perf_counter()
    left_arr  = np.searchsorted(p_t, e_t,              side='left')
    right_arr = np.searchsorted(p_t, e_t + MAX_TIME_S, side='right')
    t_ss = time.perf_counter() - t0
    print(f"    searchsorted (all boundaries at once)        {t_ss*1000:7.1f} ms")

    # ── Step B: main loop ─────────────────────────────────────────────────────
    t0 = time.perf_counter()
    out_eid = np.full(n_ph, np.nan); out_ex = np.full(n_ph, np.nan)
    out_ey  = np.full(n_ph, np.nan); out_et = np.full(n_ph, np.nan)
    out_en  = np.zeros(n_ph);        out_psd = np.zeros(n_ph)
    out_com = np.full(n_ph, np.inf)

    for i in range(len(events)):
        lo = int(left_arr[i]);  hi = int(right_arr[i])
        en = e_n[i]
        if hi - lo < en:
            continue
        ex = e_x[i];  ey = e_y[i];  et = e_t[i]
        epsd = e_psd[i];  eid = e_id[i]
        sub = np.arange(lo, hi)
        dx = p_x[sub] - ex;  dy = p_y[sub] - ey
        bbox = (np.abs(dx) <= DSPACE_PX) & (np.abs(dy) <= DSPACE_PX)
        sub = sub[bbox];  dx = dx[bbox];  dy = dy[bbox]
        if len(sub) < en:
            continue
        dists = np.sqrt(dx*dx + dy*dy)
        order = np.argsort(dists)[:en]
        sel = sub[order];  sel_d = dists[order]
        com_dist = sel_d[0] if en == 1 else \
            np.sqrt((p_x[sel].mean() - ex)**2 + (p_y[sel].mean() - ey)**2)
        if com_dist > DSPACE_PX:
            continue
        improve = out_com[sel] > com_dist
        sel2 = sel[improve]
        if len(sel2) == 0:
            continue
        out_eid[sel2]=eid; out_ex[sel2]=ex; out_ey[sel2]=ey; out_et[sel2]=et
        out_en[sel2]=en;   out_psd[sel2]=epsd; out_com[sel2]=com_dist
    t_loop = time.perf_counter() - t0
    print(f"    Main loop (bbox + single-pass conflicts)     {t_loop*1000:7.1f} ms")

    # ── Step C: write back ────────────────────────────────────────────────────
    t0 = time.perf_counter()
    _ = photons.assign(
        assoc_event_id=out_eid, assoc_x=out_ex, assoc_y=out_ey,
        assoc_t=out_et, assoc_n=out_en, assoc_PSD=out_psd,
        assoc_com_dist=np.where(np.isinf(out_com), np.nan, out_com),
    )
    t_wb = time.perf_counter() - t0
    print(f"    Write-back (assign)                          {t_wb*1000:7.1f} ms")

    return t_ss + t_loop + t_wb


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*62}")
    print(f"  Photon-event association benchmark")
    print(f"  events={N_EVENTS:,}  photons={N_PHOTONS:,}")
    print(f"  dSpace={DSPACE_PX} px   max_time={MAX_TIME_S*1e9:.0f} ns")
    print(f"{'='*62}\n")

    photons, events = make_data(N_EVENTS, N_PHOTONS)
    print(f"Data: {len(photons):,} photons, {len(events):,} events\n")

    print("Current implementation — step breakdown:")
    t_cur = profile_current(photons.copy(), events.copy())
    print(f"    → total                                      {t_cur*1000:7.1f} ms\n")

    print("Optimized implementation — step breakdown:")
    t_opt = profile_optimized(photons.copy(), events.copy())
    print(f"    → total                                      {t_opt*1000:7.1f} ms\n")

    print("End-to-end timing (best of 3 runs):")
    _, t1 = _time_section("Current  (iterrows + while-loops + dict)", assoc_current,
                           photons.copy(), events.copy())
    _, t2 = _time_section("Optimized (searchsorted + bbox + arrays)",  assoc_optimized,
                           photons.copy(), events.copy())

    print(f"\n  Speedup: ×{t1/t2:.2f}  ({t1*1000:.0f} ms → {t2*1000:.0f} ms)\n")

    # Correctness check
    r1 = assoc_current(photons, events)
    r2 = assoc_optimized(photons, events)
    n1 = r1['assoc_event_id'].notna().sum()
    n2 = r2['assoc_event_id'].notna().sum()
    print(f"  Matched photons: current={n1:,}  optimized={n2:,}")
    if abs(n1 - n2) / max(n1, 1) < 0.01:
        print("  Result agreement: ✓ (within 1%)")
    else:
        print(f"  WARNING: mismatch > 1% — check conflict resolution logic")


if __name__ == '__main__':
    main()
