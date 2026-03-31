"""Tag vocabulary definitions."""

import pandas as pd

PH_TAG_DTYPE = pd.CategoricalDtype(categories=[
    "single_pixel", "small_blob", "large_blob",
    "symmetric", "asymmetric", "line",
    "sparse", "centered", "satellite", "distant_pixels",
    "delayed", "sparse_time",
    "high_yield", "shallow_tot",
], ordered=False)

EV_TAG_DTYPE = pd.CategoricalDtype(categories=[
    "single_ph", "double_ph", "triple_ph", "multi_ph",
    "close", "far", "centered", "satellite",
    "delayed", "in_line", "symmetric",
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
