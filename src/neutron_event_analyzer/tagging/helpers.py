"""Convenience helpers for filtering and querying tag columns.

Tag columns store Unicode symbol strings (e.g. "●⊕★").
All helpers accept either the tag name ("round") or its symbol ("●").
"""

import pandas as pd
from .dtypes import TAG_SYMBOLS, SYMBOL_TAGS


def _normalise(tag: str) -> str:
    """Return the symbol for *tag*, accepting either a name or a symbol."""
    if tag in TAG_SYMBOLS:       # text name → symbol
        return TAG_SYMBOLS[tag]
    if tag in SYMBOL_TAGS:       # already a symbol
        return tag
    raise KeyError(f"Unknown tag: {tag!r}")


def _tag_set(val) -> set:
    """Return the set of symbols present in a tag-column value."""
    if isinstance(val, (frozenset, set)):
        # frozenset of names → convert each to symbol
        return {TAG_SYMBOLS.get(t, t) for t in val}
    if isinstance(val, str):
        return set(val)   # each character is one symbol
    return set()


def has_tag(series: pd.Series, tag: str) -> pd.Series:
    """Boolean mask: rows where the tag column contains *tag*."""
    sym = _normalise(tag)
    return series.apply(lambda t: sym in _tag_set(t))


def has_all_tags(series: pd.Series, *tags: str) -> pd.Series:
    """Boolean mask: rows containing ALL of the given tags."""
    syms = {_normalise(t) for t in tags}
    return series.apply(lambda t: syms <= _tag_set(t))


def has_any_tag(series: pd.Series, *tags: str) -> pd.Series:
    """Boolean mask: rows containing ANY of the given tags."""
    syms = {_normalise(t) for t in tags}
    return series.apply(lambda t: bool(_tag_set(t) & syms))
