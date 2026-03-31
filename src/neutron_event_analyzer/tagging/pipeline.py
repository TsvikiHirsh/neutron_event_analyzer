"""Main tagging pipeline entry point."""

import pandas as pd

from .dtypes import frozenset_to_str
from .features import photon_features, event_features
from .tags import assign_ph_tags, assign_ev_tags

# Required pixel-level columns for feature computation.
_PH_REQUIRED = {"ev/id", "ph/id", "px/x", "px/y", "px/toa", "px/tot"}
# Required event-level columns.
_EV_REQUIRED = {"ev/id", "ph/id", "ph/x", "ph/y", "px/toa", "ev/x", "ev/y"}


def add_topology_tags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute features and assign topology tags.

    Adds columns:
      - ``ph/tags`` — pipe-delimited photon topology tag string
      - ``ev/tags`` — pipe-delimited event topology tag string

    Parameters
    ----------
    df : DataFrame
        Association output produced by ``Analyse.associate()``.  Must contain
        pixel-level rows with columns: ev/id, ph/id, px/x, px/y, px/toa,
        px/tot, ph/x, ph/y, ev/x, ev/y.

    Returns
    -------
    DataFrame with ``ph/tags`` and ``ev/tags`` columns appended.
    """
    missing_ph = _PH_REQUIRED - set(df.columns)
    missing_ev = _EV_REQUIRED - set(df.columns)
    if missing_ph or missing_ev:
        raise ValueError(
            f"Missing columns for tagging — photon features need: {missing_ph or 'OK'}, "
            f"event features need: {missing_ev or 'OK'}"
        )

    # Drop rows without a valid photon or event assignment.
    mask = df["ph/id"].notna() & df["ev/id"].notna()
    tagged = df[mask].copy()
    untagged = df[~mask].copy()

    # --- photon features & tags ---
    pf = photon_features(tagged)

    pf["_high_yield"] = pf["ph/total_tot"] > pf["ph/total_tot"].quantile(0.9)
    pf["_shallow_tot"] = pf["ph/max_tot"] < pf["ph/max_tot"].quantile(0.1)

    pf["ph/tags"] = pf.apply(assign_ph_tags, axis=1).apply(frozenset_to_str)

    # Merge photon tags back to pixel rows.
    tagged = tagged.join(pf["ph/tags"], on=["ev/id", "ph/id"])

    # --- event features & tags ---
    ef = event_features(tagged)
    ef["ev/tags"] = ef.apply(assign_ev_tags, axis=1).apply(frozenset_to_str)

    tagged = tagged.join(ef["ev/tags"], on="ev/id")

    # Re-attach unassociated rows (tags will be NaN → fill with empty string).
    result = pd.concat([tagged, untagged], sort=False)
    result["ph/tags"] = result["ph/tags"].fillna("")
    result["ev/tags"] = result["ev/tags"].fillna("")

    return result.sort_index()
