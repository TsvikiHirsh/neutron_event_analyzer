"""
benchmark_association.py
========================
Compare timing of the current 3-step association pipeline against two
alternatives that avoid the slow float-keyed pandas merge:

  Current   : pixel-photon (empir) + photon-event + merge on rounded (x,y,t)
  Alt-A     : same associations + merge on integer photon_id  (1 key, int)
  Alt-B     : same associations + dict map  (no merge at all)
  Alt-C     : event-first direct join  (traverse ev→ph→px, build rows inline)

Run with:
    /root/.local/bin/micromamba run -n base python3 notebooks/benchmark_association.py
"""

import time
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

# ── Synthetic data sizes (tweak to match real workload) ──────────────────────
N_EVENTS  = 3_000
N_PHOTONS = 12_000   # ~4 photons per event on average
N_PIXELS  = 40_000   # ~3 pixels per photon on average

# ── Helpers ──────────────────────────────────────────────────────────────────

def _timer(label: str, fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    dt = time.perf_counter() - t0
    print(f"  {label:<45s} {dt*1000:7.1f} ms")
    return result, dt


def make_synthetic_data(n_ev, n_ph, n_px, seed=42):
    """
    Generate realistic synthetic pixel / photon / event DataFrames where
    the ground-truth associations are known.

    Events  : random (x, y) in [0, 512)^2, t uniform in [0, 1) s
    Photons : n_ph photons, each offset slightly from a random event
    Pixels  : n_px pixels, each offset slightly from a random photon
    """
    rng = np.random.default_rng(seed)

    # ── Events ────────────────────────────────────────────────────────────────
    ev_x   = rng.uniform(50, 462, n_ev)
    ev_y   = rng.uniform(50, 462, n_ev)
    ev_t   = np.sort(rng.uniform(0, 1, n_ev))
    ev_n   = rng.integers(1, 8, n_ev)
    ev_psd = rng.uniform(0, 1, n_ev)

    events_df = pd.DataFrame({
        'x': ev_x, 'y': ev_y, 't': ev_t,
        'n': ev_n, 'PSD': ev_psd,
    })

    # ── Photons ───────────────────────────────────────────────────────────────
    # Each photon is drawn from a random event, offset by a few pixels / ns
    ph_ev_idx = rng.integers(0, n_ev, n_ph)
    ph_x = ev_x[ph_ev_idx] + rng.normal(0, 2, n_ph)
    ph_y = ev_y[ph_ev_idx] + rng.normal(0, 2, n_ph)
    ph_t = ev_t[ph_ev_idx] + rng.uniform(0, 200e-9, n_ph)   # 0–200 ns after event

    photons_df = pd.DataFrame({
        'x': ph_x, 'y': ph_y, 't': ph_t,
    }).sort_values('t').reset_index(drop=True)
    photons_df['photon_id'] = photons_df.index + 1

    # ── Pixels ────────────────────────────────────────────────────────────────
    px_ph_idx  = rng.integers(0, n_ph, n_px)
    px_x  = photons_df['x'].values[px_ph_idx] + rng.normal(0, 1, n_px)
    px_y  = photons_df['y'].values[px_ph_idx] + rng.normal(0, 1, n_px)
    px_t  = photons_df['t'].values[px_ph_idx] + rng.uniform(0, 100e-9, n_px)
    px_tot = rng.integers(10, 500, n_px)

    pixels_df = pd.DataFrame({
        'x': px_x, 'y': px_y, 't': px_t, 'tot': px_tot,
    }).sort_values('t').reset_index(drop=True)

    return pixels_df, photons_df, events_df


def make_pixels_assoc(pixels_df, photons_df):
    """
    Simplified pixel-photon association (nearest photon within 10 px / 300 ns).
    Returns pixels_df with assoc_photon_id / assoc_phot_x/y/t added.
    """
    max_dist = 10.0
    max_time = 300e-9

    pix = pixels_df.copy()
    pix['assoc_photon_id'] = np.nan
    pix['assoc_phot_x']    = np.nan
    pix['assoc_phot_y']    = np.nan
    pix['assoc_phot_t']    = np.nan
    pix['pixel_com_dist']  = np.nan

    px_t = pix['t'].values
    px_x = pix['x'].values
    px_y = pix['y'].values
    ph_t = photons_df['t'].values
    ph_x = photons_df['x'].values
    ph_y = photons_df['y'].values
    ph_id = photons_df['photon_id'].values

    claimed = np.zeros(len(pix), dtype=bool)
    assoc_id  = np.full(len(pix), np.nan)
    assoc_x   = np.full(len(pix), np.nan)
    assoc_y   = np.full(len(pix), np.nan)
    assoc_t   = np.full(len(pix), np.nan)
    assoc_com = np.full(len(pix), np.nan)

    for j in range(len(photons_df)):
        pht = ph_t[j]; phx = ph_x[j]; phy = ph_y[j]; phid = ph_id[j]
        lo = np.searchsorted(px_t, pht)
        hi = np.searchsorted(px_t, pht + max_time, side='right')
        cands = np.arange(lo, hi)
        cands = cands[~claimed[cands]]
        if len(cands) == 0:
            continue
        dx = px_x[cands] - phx;  dy = px_y[cands] - phy
        mask = dx*dx + dy*dy <= max_dist*max_dist
        cands = cands[mask]
        if len(cands) == 0:
            continue
        com_dist = np.sqrt((px_x[cands] - phx)**2 + (px_y[cands] - phy)**2).mean()
        claimed[cands] = True
        assoc_id[cands]  = phid
        assoc_x[cands]   = phx
        assoc_y[cands]   = phy
        assoc_t[cands]   = pht
        assoc_com[cands] = com_dist

    pix['assoc_photon_id'] = assoc_id
    pix['assoc_phot_x']    = assoc_x
    pix['assoc_phot_y']    = assoc_y
    pix['assoc_phot_t']    = assoc_t
    pix['pixel_com_dist']  = assoc_com
    return pix


def make_photons_assoc(photons_df, events_df):
    """
    Simplified photon-event association (forward time window, nearest CoM).
    Returns photons_df with assoc_event_id / assoc_x/y/t/n/PSD added.
    """
    max_dist = 30.0
    max_time = 500e-9

    ph = photons_df.copy()
    ph['assoc_event_id'] = np.nan
    ph['assoc_x'] = np.nan; ph['assoc_y'] = np.nan; ph['assoc_t'] = np.nan
    ph['assoc_n'] = 0; ph['assoc_PSD'] = 0.0; ph['assoc_com_dist'] = np.nan

    p_t = ph['t'].values; p_x = ph['x'].values; p_y = ph['y'].values
    e_t = events_df['t'].values; e_x = events_df['x'].values
    e_y = events_df['y'].values; e_n = events_df['n'].values
    e_psd = events_df['PSD'].values
    n_ph = len(ph)

    assoc_eid  = np.full(n_ph, np.nan)
    assoc_ex   = np.full(n_ph, np.nan)
    assoc_ey   = np.full(n_ph, np.nan)
    assoc_et   = np.full(n_ph, np.nan)
    assoc_en   = np.zeros(n_ph)
    assoc_epsd = np.zeros(n_ph)
    assoc_com  = np.full(n_ph, np.nan)

    left = 0
    for i in range(len(events_df)):
        et = e_t[i]; ex = e_x[i]; ey = e_y[i]; en = int(e_n[i]); epsd = e_psd[i]
        while left < n_ph and p_t[left] < et:
            left += 1
        right = left
        while right < n_ph and p_t[right] <= et + max_time:
            right += 1
        if right - left == 0:
            continue
        sub = np.arange(left, right)
        dx = p_x[sub] - ex; dy = p_y[sub] - ey
        dists = np.sqrt(dx*dx + dy*dy)
        order = np.argsort(dists)[:en]
        sel = sub[order]
        com_dist = dists[order].mean()
        if com_dist > max_dist:
            continue
        eid = i + 1
        mask_unassigned = np.isnan(assoc_eid[sel])
        sel2 = sel[mask_unassigned]
        if len(sel2) == 0:
            continue
        assoc_eid[sel2]  = eid
        assoc_ex[sel2]   = ex;  assoc_ey[sel2]  = ey;  assoc_et[sel2]  = et
        assoc_en[sel2]   = en;  assoc_epsd[sel2] = epsd
        assoc_com[sel2]  = com_dist

    ph['assoc_event_id'] = assoc_eid
    ph['assoc_x'] = assoc_ex; ph['assoc_y'] = assoc_ey; ph['assoc_t'] = assoc_et
    ph['assoc_n'] = assoc_en; ph['assoc_PSD'] = assoc_epsd
    ph['assoc_com_dist'] = assoc_com
    return ph


# ── Join strategies ───────────────────────────────────────────────────────────

def join_float_merge(pixels_assoc, photons_ev):
    """Current production approach: merge on 3 rounded float columns."""
    pe = photons_ev.copy()
    pe['_mx'] = pe['x'].round(6)
    pe['_my'] = pe['y'].round(6)
    pe['_mt'] = pe['t'].round(12)

    pa = pixels_assoc.copy()
    pa['_mx'] = pa['assoc_phot_x'].round(6)
    pa['_my'] = pa['assoc_phot_y'].round(6)
    pa['_mt'] = pa['assoc_phot_t'].round(12)

    merge_cols = ['_mx', '_my', '_mt', 'assoc_event_id',
                  'assoc_x', 'assoc_y', 'assoc_t', 'assoc_n', 'assoc_PSD', 'assoc_com_dist']
    merge_cols = [c for c in merge_cols if c in pe.columns]

    out = pa.merge(pe[merge_cols], on=['_mx', '_my', '_mt'], how='left')
    out = out.drop(columns=['_mx', '_my', '_mt'])
    return out


def join_int_merge(pixels_assoc, photons_ev):
    """Alt-A: merge on integer photon_id (1 column, integer key)."""
    ev_cols = ['photon_id', 'assoc_event_id',
               'assoc_x', 'assoc_y', 'assoc_t', 'assoc_n', 'assoc_PSD', 'assoc_com_dist']
    ev_cols = [c for c in ev_cols if c in photons_ev.columns]
    out = pixels_assoc.merge(
        photons_ev[ev_cols],
        left_on='assoc_photon_id', right_on='photon_id',
        how='left',
    ).drop(columns=['photon_id'], errors='ignore')
    return out


def join_dict_map(pixels_assoc, photons_ev):
    """
    Alt-B: no merge at all — build a dict from photon_id → event data,
    then map each pixel's assoc_photon_id.
    """
    ev_cols = ['assoc_event_id', 'assoc_x', 'assoc_y',
               'assoc_t', 'assoc_n', 'assoc_PSD', 'assoc_com_dist']
    ev_cols = [c for c in ev_cols if c in photons_ev.columns]

    ph_sub = photons_ev[['photon_id'] + ev_cols].set_index('photon_id')
    ph_records = ph_sub.to_dict(orient='index')   # {ph_id: {col: val, ...}}

    out = pixels_assoc.copy()
    ph_ids = out['assoc_photon_id'].values
    for col in ev_cols:
        out[col] = [ph_records.get(pid, {}).get(col, np.nan) for pid in ph_ids]
    return out


def join_event_first(pixels_assoc, photons_ev, events_df):
    """
    Alt-C: event-first traversal — for each event look up its photons,
    for each photon look up its pixels, emit rows directly.
    No merge, no intermediate DataFrames.
    """
    # Build lookup tables from the already-computed associations
    # photon_id → event row data
    ev_cols = ['assoc_event_id', 'assoc_x', 'assoc_y',
               'assoc_t', 'assoc_n', 'assoc_PSD', 'assoc_com_dist']
    ev_cols = [c for c in ev_cols if c in photons_ev.columns]

    ph_to_ev = {}   # photon_id → dict of event cols
    for _, row in photons_ev[['photon_id'] + ev_cols].iterrows():
        ph_to_ev[row['photon_id']] = {c: row[c] for c in ev_cols}

    # pixel index → photon_id (for pixels that got assigned)
    px_to_ph = pixels_assoc['assoc_photon_id'].values   # NaN if unassigned

    # Group pixel indices by their photon_id
    ph_to_px_rows = {}   # photon_id → list of pixel row-dicts
    px_records = pixels_assoc.to_dict(orient='records')
    for i, ph_id in enumerate(px_to_ph):
        if not np.isnan(ph_id):
            ph_to_px_rows.setdefault(ph_id, []).append(px_records[i])

    # Walk: photon → event lookup → emit rows
    rows = []
    for ph_id, px_list in ph_to_px_rows.items():
        ev_data = ph_to_ev.get(ph_id, {c: np.nan for c in ev_cols})
        for px_row in px_list:
            rows.append({**px_row, **ev_data})

    # Pixels with no photon assignment — keep them with NaN event cols
    unassigned = [px_records[i] for i, pid in enumerate(px_to_ph) if np.isnan(pid)]
    for px_row in unassigned:
        rows.append({**px_row, **{c: np.nan for c in ev_cols}})

    return pd.DataFrame(rows)


# ── Main benchmark ────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print(f"  Association join benchmark")
    print(f"  events={N_EVENTS:,}  photons={N_PHOTONS:,}  pixels={N_PIXELS:,}")
    print(f"{'='*60}\n")

    print("Generating synthetic data …")
    pixels_df, photons_df, events_df = make_synthetic_data(N_EVENTS, N_PHOTONS, N_PIXELS)
    print(f"  pixels={len(pixels_df):,}  photons={len(photons_df):,}  events={len(events_df):,}\n")

    print("Step 1 — pixel-photon association (shared for all join methods):")
    pixels_assoc, t_ppas = _timer("pixel-photon (simplified empir)", make_pixels_assoc,
                                   pixels_df, photons_df)

    print("\nStep 2 — photon-event association (shared for all join methods):")
    photons_ev, t_peas = _timer("photon-event (simplified window)", make_photons_assoc,
                                 photons_df, events_df)

    # Sanity: how many were actually matched?
    n_px_assoc = pixels_assoc['assoc_photon_id'].notna().sum()
    n_ph_assoc = photons_ev['assoc_event_id'].notna().sum()
    print(f"\n  → {n_px_assoc:,}/{len(pixels_assoc):,} pixels matched to photons")
    print(f"  → {n_ph_assoc:,}/{len(photons_ev):,} photons matched to events\n")

    # ── Warm-up run (avoid cold-start bias) ──────────────────────────────────
    _ = join_float_merge(pixels_assoc, photons_ev)

    print("Step 3 — join strategies (repeated 5× each, best time shown):\n")
    N_REP = 5

    times = {}
    for name, fn, extra in [
        ("Current: float merge (3 rounded cols)", join_float_merge,  {}),
        ("Alt-A:  int   merge (photon_id)",       join_int_merge,    {}),
        ("Alt-B:  dict map   (no merge)",          join_dict_map,     {}),
        ("Alt-C:  event-first traversal",          join_event_first,  {'events_df': events_df}),
    ]:
        best = np.inf
        for _ in range(N_REP):
            t0 = time.perf_counter()
            if extra:
                fn(pixels_assoc, photons_ev, **extra)
            else:
                fn(pixels_assoc, photons_ev)
            best = min(best, time.perf_counter() - t0)
        times[name] = best
        print(f"  {name:<45s} {best*1000:7.1f} ms")

    baseline = times["Current: float merge (3 rounded cols)"]
    print(f"\n  {'Speedups vs float-merge':^45s}")
    print(f"  {'-'*52}")
    for name, t in times.items():
        sp = baseline / t
        print(f"  {name:<45s} ×{sp:.2f}")

    total_current  = t_ppas + t_peas + baseline
    best_join_t    = min(times["Alt-A:  int   merge (photon_id)"],
                         times["Alt-B:  dict map   (no merge)"],
                         times["Alt-C:  event-first traversal"])
    total_best     = t_ppas + t_peas + best_join_t

    print(f"\n  {'Total pipeline time':^52}")
    print(f"  {'Current  (step1+step2+float-merge)':<45s} {total_current*1000:7.1f} ms")
    print(f"  {'Best alt (step1+step2+best-join)':<45s} {total_best*1000:7.1f} ms")
    print(f"  {'Saving':<45s} {(total_current-total_best)*1000:7.1f} ms "
          f"({100*(total_current-total_best)/total_current:.1f}%)\n")


if __name__ == '__main__':
    main()
