"""Tag assignment functions for photons and events.

Thresholds are calibrated against PTB2024 Timepix3 data where px/toa is
stored in seconds (1 Timepix3 clock tick ≈ 1.5625 ns ≈ 1.5625e-9 s).
"""

from .dtypes import PH_TAG_DTYPE, EV_TAG_DTYPE, validate_tags

# ToA spread threshold for 'spread' tag (seconds).
# p90 of intra-cluster toa_range ≈ 53 ns → use 75 ns to select the top ~5%.
_TOA_SPREAD_S = 75e-9

# Track correlation threshold: abs(corr) > 0.85 selects the top ~10% of
# multi-pixel clusters with strong position-time correlation.
_TRACK_CORR_THR = 0.85

# Hot pixel: one pixel dominates charge deposit (max_tot / mean_tot).
# p99 ≈ 4.25 in fast_neutrons data; use 4.0 to capture ~1% of photons.
_HOT_RATIO_THR = 4.0


def assign_ph_tags(row, dtype=PH_TAG_DTYPE) -> frozenset:
    """Compute tags for a single photon from its feature row."""
    tags = set()
    n = row["ph/npix"]

    # ── size ─────────────────────────────────────────────────────────────────
    if n == 1:
        tags.add("point")
    elif n <= 4:
        tags.add("small")
    else:
        tags.add("large")

    # ── shape (multi-pixel only) ─────────────────────────────────────────────
    if n >= 2:
        aspect = row["ph/aspect"]
        fill   = row["ph/fill_frac"]
        if 0.4 <= aspect <= 2.5 and fill > 0.6:
            tags.add("round")
        elif aspect > 2.5 or aspect < 0.4:
            tags.add("elongated")

    if n >= 3 and row["ph/fill_frac"] < 0.4:
        tags.add("sparse")

    # ── center-of-gravity ────────────────────────────────────────────────────
    if row["ph/cog_on_max_tot"]:
        tags.add("centered")
    if row["ph/cog_offset"] > 1.5:
        tags.add("offset")

    # ── temporal ─────────────────────────────────────────────────────────────
    # bimodal: two distinct arrival-time populations within the cluster
    if row["ph/toa_bimodal"] > 5:
        tags.add("bimodal")

    # spread: cluster pixels arrive over an unusually long window
    if n >= 3 and row["ph/toa_range"] > _TOA_SPREAD_S:
        tags.add("spread")

    # ── intensity ────────────────────────────────────────────────────────────
    if row.get("_bright", False):
        tags.add("bright")
    if row.get("_dim", False):
        tags.add("dim")

    # ── physics ──────────────────────────────────────────────────────────────
    # track: ToA progresses monotonically along the cluster's major axis
    # (signature of a charged particle crossing the sensor)
    if n >= 4 and row["ph/track_corr"] > _TRACK_CORR_THR:
        tags.add("track")

    # hot: charge dominated by a single pixel (Bragg peak / proton recoil)
    if n >= 2 and row["ph/hot_ratio"] > _HOT_RATIO_THR:
        tags.add("hot")

    return validate_tags(tags, dtype)


def assign_ev_tags(row, dtype=EV_TAG_DTYPE) -> frozenset:
    """Compute tags for a single event from its feature row."""
    tags = set()
    nph = row["ev/nph"]

    # ── multiplicity ─────────────────────────────────────────────────────────
    if nph == 1:
        tags.add("solo")
    elif nph == 2:
        tags.add("pair")
    else:
        tags.add("multi")

    # ── geometry (2+ photons) ─────────────────────────────────────────────────
    if nph >= 2:
        if row["ev/max_ph_dist"] < 3:
            tags.add("tight")
        if row["ev/min_ph_dist"] > 5:
            tags.add("loose")
        # wide: photons far from the reconstructed event position → poor
        # localisation or secondary scatter
        if row["ev/mean_radius"] > 10:
            tags.add("wide")

    # ── geometry (3+ photons) ─────────────────────────────────────────────────
    if nph >= 3:
        if row["ev/linearity"] > 5:
            tags.add("linear")
        elif row["ev/linearity"] < 2 and row["ev/mean_radius"] < 3:
            tags.add("ring")

    return validate_tags(tags, dtype)
