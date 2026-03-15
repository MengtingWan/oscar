"""
src/utils.py
============
Shared utility functions used across method scripts.
"""

import ast
import math
import re

from src.config import CAT_NORM


def parse_codes(s):
    """Parse a stringified list of IMDB codes into a Python list."""
    try:
        v = ast.literal_eval(str(s))
        return [x for x in v if x and isinstance(x, str)]
    except Exception:
        return []


def norm_cat(cat_name):
    """Normalize a free-form award category name to a canonical category key."""
    cl = cat_name.lower().strip()

    # Pre-checks: specific patterns that must be caught before generic CAT_NORM scan
    if "directorial achievement" in cl or re.search(r"\bdirecting\b", cl):
        return "director"
    if "cinematography" in cl:
        return "cinematography"
    if "sound mixing" in cl or "sound editing" in cl:
        return "sound"
    if re.search(r"\bbest edited\b|\bedited.*feature film\b", cl):
        return "editing"
    if "documentary" in cl and "short" not in cl:
        return "documentary"
    if "international" in cl and "feature" in cl and "short" not in cl:
        return "international"
    if re.search(r"animated\s+(?:feature|theatrical|motion)", cl):
        return "animated"
    if re.search(r"(?:period|contemporary|fantasy)\s+feature\s+film", cl):
        return "production_design"
    if re.search(r"cast in.*motion picture|cast of.*motion picture", cl):
        return "cast"
    # SAG/GG use "Female Actor" / "Male Actor" — disambiguate before generic patterns
    # GG also uses "Actor/Actress in a Motion Picture" (no "Leading" keyword)
    if "female actor" in cl or ("actress" in cl and "motion picture" in cl):
        if "supporting" in cl:
            return "supp_actress"
        if "television" not in cl and "limited series" not in cl:
            return "actress"
    if "male actor" in cl or ("actor" in cl and "motion picture" in cl and "actress" not in cl):
        if "supporting" in cl:
            return "supp_actor"
        if "television" not in cl and "limited series" not in cl:
            return "actor"

    for key, patterns in CAT_NORM.items():
        for pat in patterns:
            if re.search(pat, cl):
                return key
    return None


def make_feature_name(event_name, cat_key, is_winner):
    """Build a feature string: '<event>::<cat_key>::W|N'."""
    return f"{event_name}::{cat_key}::{'W' if is_winner else 'N'}"


def softmax(scores):
    """Softmax over a list of floats, returning a list of the same length."""
    s = [float(x) for x in scores]
    m = max(s) if s else 0.0
    e = [math.exp(x - m) for x in s]
    t = sum(e)
    return [x / t for x in e] if t > 0 else [1.0 / len(s)] * len(s)


def get_codes_for_nom(nom):
    """Return all IMDB codes (primary + secondary) for a nominee dict."""
    return nom.get("primaryCodes", []) + nom.get("secondaryCodes", [])


def fuzzy_match(name, lookup_dict):
    """
    Case-insensitive substring match of `name` against dict keys.
    Returns (matched_value, matched_key) or (0.0, None).
    """
    nl = name.lower()
    for k, v in lookup_dict.items():
        if nl in k.lower() or k.lower() in nl:
            return v, k
    return 0.0, None


def short_event(feat_str):
    """
    Shorten an 'event::cat::W/N' feature string for human display.
    e.g. 'Screen Actors Guild Awards::actor::W' → 'SAG Win'
    """
    abbr = {
        "Broadcast Film Critics Association Awards": "Critics Choice",
        "Screen Actors Guild Awards":                "SAG",
        "Directors Guild of America, USA":           "DGA",
        "Golden Globes, USA":                        "Golden Globe",
        "BAFTA Awards":                              "BAFTA",
        "PGA Awards":                                "PGA",
        "Writers Guild of America, USA":             "WGA",
        "American Cinema Editors, USA":              "ACE",
        "Art Directors Guild":                       "ADG",
        "Costume Designers Guild Awards":            "CDG",
        "Film Independent Spirit Awards":            "Spirit",
        "Annie Awards":                              "Annie",
        "Grammy Awards":                             "Grammy",
        "American Society of Cinematographers, USA": "ASC",
        "Cinema Audio Society, USA":                 "CAS",
        "Visual Effects Society Awards":             "VES",
        "National Society of Film Critics Awards, USA": "NSFC",
        "Online Film Critics Society Awards":        "OFCS",
        "Satellite Awards":                          "Satellite",
        "Cinema Eye Honors Awards, US":              "Cinema Eye",
        "International Documentary Association, US": "IDA",
        "London Critics Circle Film Awards":         "London Critics",
    }
    parts = feat_str.split("::")
    ev    = abbr.get(parts[0], parts[0]) if parts else feat_str
    if len(parts) >= 3:
        ev += " Win" if parts[2] == "W" else " Nom"
    return ev
