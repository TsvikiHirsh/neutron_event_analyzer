"""Convenience helpers for filtering and querying tag columns."""

import pandas as pd


def _to_set(val) -> set:
    """Normalise a tag value (frozenset or pipe-delimited string) to a set."""
    if isinstance(val, (frozenset, set)):
        return val
    if isinstance(val, str):
        return set(val.split("|")) if val else set()
    return set()


def has_tag(series: pd.Series, tag: str) -> pd.Series:
    """Boolean mask: rows where the tag column contains *tag*."""
    return series.apply(lambda t: tag in _to_set(t))


def has_all_tags(series: pd.Series, *tags: str) -> pd.Series:
    """Boolean mask: rows containing ALL of the given tags."""
    required = set(tags)
    return series.apply(lambda t: required <= _to_set(t))


def has_any_tag(series: pd.Series, *tags: str) -> pd.Series:
    """Boolean mask: rows containing ANY of the given tags."""
    query = set(tags)
    return series.apply(lambda t: bool(_to_set(t) & query))
