"""Tag vocabulary definitions.

Photon tags — morphology of a single scintillation cluster:

  Size      : point ·   small ◾   large ■
  Shape     : round ●   elongated ▬   sparse ∴
  CoG       : centered ⊕   offset ⊗
  Temporal  : bimodal ∿   spread ↔
  Intensity : bright ★   dim ☆
  Physics   : track ↗   hot ⧫

Event tags — arrangement of photons in one neutron event:

  Multiplicity : solo ①   pair ②   multi ③
  Geometry     : tight ⊡   loose ⟺   linear ⋯   ring ◎   wide ⊠

In CSV the tag columns store a compact string of Unicode symbols (no separator
needed — every symbol is a single codepoint).  Example: "●⊕★" means the photon
is round, centered, and bright.  Use `has_tag` with the text name or symbol:

    has_tag(df["ph/tags"], "round")   # or has_tag(df["ph/tags"], "●")
"""

import pandas as pd

# ── Symbol ↔ name mappings ────────────────────────────────────────────────────

TAG_SYMBOLS: dict[str, str] = {
    # photon — size
    "point":     "·",   # U+00B7  MIDDLE DOT
    "small":     "◾",   # U+25FE  BLACK MEDIUM SMALL SQUARE
    "large":     "■",   # U+25A0  BLACK SQUARE
    # photon — shape
    "round":     "●",   # U+25CF  BLACK CIRCLE
    "elongated": "▬",   # U+25AC  BLACK RECTANGLE
    "sparse":    "∴",   # U+2234  THEREFORE  (three scattered dots)
    # photon — centre-of-gravity
    "centered":  "⊕",   # U+2295  CIRCLED PLUS   (crosshair on pixel)
    "offset":    "⊗",   # U+2297  CIRCLED TIMES  (CoG displaced)
    # photon — temporal
    "bimodal":   "∿",   # U+223F  SINE WAVE      (two ToA populations)
    "spread":    "↔",   # U+2194  LEFT RIGHT ARROW (wide ToA window)
    # photon — intensity
    "bright":    "★",   # U+2605  BLACK STAR
    "dim":       "☆",   # U+2606  WHITE STAR
    # photon — physics
    "track":     "↗",   # U+2197  NORTH EAST ARROW (directed particle track)
    "hot":       "⧫",   # U+29EB  BLACK LOZENGE   (single dominant pixel)
    # event — multiplicity
    "solo":      "①",   # U+2460  CIRCLED DIGIT ONE
    "pair":      "②",   # U+2461  CIRCLED DIGIT TWO
    "multi":     "③",   # U+2462  CIRCLED DIGIT THREE (3+)
    # event — geometry
    "tight":     "⊡",   # U+22A1  SQUARED DOT       (compact cluster)
    "loose":     "⟺",   # U+27FA  LONG LR DBL ARROW (well separated)
    "linear":    "⋯",   # U+22EF  MIDLINE ELLIPSIS  (in a line)
    "ring":      "◎",   # U+25CE  BULLSEYE          (symmetric ring)
    "wide":      "⊠",   # U+22A0  SQUARED TIMES     (far from event pos.)
}

# Reverse mapping: symbol → tag name
SYMBOL_TAGS: dict[str, str] = {v: k for k, v in TAG_SYMBOLS.items()}

PH_TAGS = [
    "point", "small", "large",
    "round", "elongated", "sparse",
    "centered", "offset",
    "bimodal", "spread",
    "bright", "dim",
    "track", "hot",
]

EV_TAGS = [
    "solo", "pair", "multi",
    "tight", "loose", "linear", "ring", "wide",
]

PH_TAG_DTYPE = pd.CategoricalDtype(categories=PH_TAGS, ordered=False)
EV_TAG_DTYPE = pd.CategoricalDtype(categories=EV_TAGS, ordered=False)


def validate_tags(tags: set, dtype: pd.CategoricalDtype) -> frozenset:
    """Validate that all tags belong to the vocabulary and return as frozenset."""
    invalid = tags - set(dtype.categories)
    if invalid:
        raise ValueError(f"Unknown tags: {invalid}")
    return frozenset(tags)


# ── Serialisation ─────────────────────────────────────────────────────────────

def frozenset_to_str(tags: frozenset) -> str:
    """Serialise a frozenset of tag names to a compact Unicode symbol string.

    Tags are emitted in a canonical order (matching TAG_SYMBOLS key order) so
    the string is reproducible across runs.
    """
    order = list(TAG_SYMBOLS.keys())
    return "".join(TAG_SYMBOLS[t] for t in order if t in tags)


def str_to_frozenset(s: str) -> frozenset:
    """Deserialise a Unicode symbol string back to a frozenset of tag names."""
    if not s:
        return frozenset()
    return frozenset(SYMBOL_TAGS[c] for c in s if c in SYMBOL_TAGS)


def tags_to_names(s: str) -> str:
    """Convert a symbol string to a human-readable pipe-delimited name string."""
    if not s:
        return ""
    return "|".join(SYMBOL_TAGS[c] for c in s if c in SYMBOL_TAGS)
