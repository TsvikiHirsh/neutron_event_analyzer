"""Tag assignment functions for photons and events."""

from .dtypes import PH_TAG_DTYPE, EV_TAG_DTYPE, validate_tags


def assign_ph_tags(row, dtype=PH_TAG_DTYPE) -> frozenset:
    """Compute tags for a single photon from its feature row."""
    tags = set()
    n = row["ph/npix"]

    # size
    if n == 1:
        tags.add("single_pixel")
    elif n <= 4:
        tags.add("small_blob")
    else:
        tags.add("large_blob")

    # shape (only for multi-pixel)
    if n >= 2:
        if 0.5 <= row["ph/aspect"] <= 2.0 and row["ph/fill_frac"] > 0.6:
            tags.add("symmetric")
        else:
            tags.add("asymmetric")
        if row["ph/aspect"] > 3 or row["ph/aspect"] < 1 / 3:
            tags.add("line")

    # spatial
    if n >= 3 and row["ph/fill_frac"] < 0.5:
        tags.add("sparse")
    if row["ph/cog_on_max_tot"]:
        tags.add("centered")
    if row["ph/cog_offset"] > 1.5:
        tags.add("satellite")
    if row["ph/max_dist"] > 4:
        tags.add("distant_pixels")

    # temporal
    if row["ph/toa_bimodal_ratio"] > 5:
        tags.add("delayed")
    if row["ph/toa_range_ns"] > 50 and n >= 3:
        tags.add("sparse_time")

    # intensity (population-level flags set by the pipeline)
    if row.get("_high_yield", False):
        tags.add("high_yield")
    if row.get("_shallow_tot", False):
        tags.add("shallow_tot")

    return validate_tags(tags, dtype)


def assign_ev_tags(row, dtype=EV_TAG_DTYPE) -> frozenset:
    """Compute tags for a single event from its feature row."""
    tags = set()
    nph = row["ev/nph"]

    # multiplicity
    if nph == 1:
        tags.add("single_ph")
    elif nph == 2:
        tags.add("double_ph")
    elif nph == 3:
        tags.add("triple_ph")
    else:
        tags.add("multi_ph")

    # spatial (2+ photons)
    if nph >= 2:
        if row["ev/max_ph_dist"] < 3:
            tags.add("close")
        if row["ev/min_ph_dist"] > 5:
            tags.add("far")
        if row["ev/mean_radius"] < 2:
            tags.add("centered")
        if row["ev/max_ph_dist"] > 3 * row["ev/mean_ph_dist"]:
            tags.add("satellite")

    # temporal
    if nph >= 2 and row["ev/ph_toa_spread"] > 50:
        tags.add("delayed")

    # geometry (3+)
    if nph >= 3:
        if row["ev/linearity"] > 5:
            tags.add("in_line")
        if row["ev/linearity"] < 2 and row["ev/mean_radius"] < 3:
            tags.add("symmetric")

    return validate_tags(tags, dtype)
