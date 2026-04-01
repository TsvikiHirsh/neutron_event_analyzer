"""Tag vocabulary definitions.

Photon tags — morphology of a single scintillation cluster (pixel group):

  Size      : point, small, large
  Shape     : round, elongated, sparse
  CoG       : centered, offset
  Temporal  : bimodal, spread
  Intensity : bright, dim
  Physics   : track, hot

Event tags — spatial/temporal arrangement of photons in one neutron event:

  Multiplicity : solo, pair, multi
  Geometry     : tight, loose, linear, ring, wide
"""

import pandas as pd

PH_TAG_DTYPE = pd.CategoricalDtype(categories=[
    # size
    "point", "small", "large",
    # shape
    "round", "elongated", "sparse",
    # center-of-gravity
    "centered", "offset",
    # temporal
    "bimodal", "spread",
    # intensity
    "bright", "dim",
    # physics
    "track", "hot",
], ordered=False)

EV_TAG_DTYPE = pd.CategoricalDtype(categories=[
    # multiplicity
    "solo", "pair", "multi",
    # geometry
    "tight", "loose", "linear", "ring", "wide",
], ordered=False)


def validate_tags(tags: set, dtype: pd.CategoricalDtype) -> frozenset:
    """Validate that all tags belong to the vocabulary and return as frozenset."""
    invalid = tags - set(dtype.categories)
    if invalid:
        raise ValueError(f"Unknown tags: {invalid}")
    return frozenset(tags)


def frozenset_to_str(tags: frozenset) -> str:
    """Serialise a frozenset of tags to a pipe-delimited string."""
    return "|".join(sorted(tags)) if tags else ""


def str_to_frozenset(s: str) -> frozenset:
    """Deserialise a pipe-delimited string back to a frozenset of tags."""
    return frozenset(s.split("|")) if s else frozenset()
