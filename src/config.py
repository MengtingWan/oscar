"""
src/config.py
=============
Shared constants for the Oscar prediction pipeline.
All method scripts import from here.
"""

# ── Ceremony ──────────────────────────────────────────────────────────────────
PRED_YEAR        = 2026
CEREMONY_NAME    = "98th Academy Awards"
CEREMONY_DATE    = "March 15, 2026"
OSCAR_EVENT_ID   = "ev0000003"
SEASON_YEARS     = [2025, 2026]          # award-season years for 2026 nominees
TRAIN_YEARS      = list(range(2000, 2026))
VALIDATION_YEARS = list(range(2020, 2026))

# ── File paths (relative to repo root) ───────────────────────────────────────
NOMINEES_PATH        = "data/processed/oscar2026_nominees.json"
DF_FULL_PATH         = "data/processed/df_data_full.csv"
GD_ODDS_PATH         = "data/processed/gold_derby_odds_2026.json"
GD_HIST_ODDS_PATH    = "data/processed/gd_historical.json"
FEATURES_PATH        = "data/processed/year_features.pkl"
HIST_CAT_PATH        = "data/processed/hist_by_cat.pkl"

STATS_PRED_PATH  = "results/stats/predictions.json"
STATS_VALID_PATH = "results/stats/validation.json"
ML_PRED_PATH     = "results/ml_model/predictions.json"
ML_VALID_PATH    = "results/ml_model/validation.json"
LLM_PRED_PATH    = "results/llm_analysis/predictions.json"
GD_PRED_PATH     = "results/goldderby/predictions.json"
GD_VALID_PATH    = "results/goldderby/validation.json"
GD_HIST_ODDS_FILE = "data/processed/gd_historical_odds.json"
CALIB_WEIGHTS_PATH = "data/processed/calibrated_weights.json"

# ── Category normalization ────────────────────────────────────────────────────
import re

CAT_NORM = {
    "picture":             ["best film","best motion picture","outstanding motion picture",
                            "feature film","theatrical motion picture","best picture",
                            "outstanding producer of theatrical motion pictures"],
    "director":            ["best director","directorial achievement",
                            "directorial achievement in theatrical",
                            "outstanding directorial achievement in theatrical"],
    "actor":               ["leading actor","actor in a leading","male actor in a leading",
                            "best actor","best male lead","performance by a male actor in a leading"],
    "actress":             ["leading actress","actress in a leading","female actor in a leading",
                            "best actress","best female lead","performance by a female actor in a leading"],
    "supp_actor":          ["supporting actor","male actor in a supporting",
                            "actor in a supporting","best male supporting",
                            "performance by a male actor in a supporting"],
    "supp_actress":        ["supporting actress","female actor in a supporting",
                            "actress in a supporting","best female supporting",
                            "performance by a female actor in a supporting"],
    "original_screenplay": ["original screenplay","writing original","screenplay - original",
                            "screenplay (original)","best screenplay.*original"],
    "adapted_screenplay":  ["adapted screenplay","writing adapted","screenplay - adapted",
                            "screenplay (adapted)","best screenplay.*adapted",
                            "best screenplay (adapted)"],
    "screenplay":          ["best screenplay$","best screenplay -"],
    "cinematography":      ["cinematography"],
    "editing":             ["editing","edited feature film - drama","edited feature film - comedy",
                            "best edited feature film","best edited.*feature","edited.*feature film"],
    "production_design":   ["production design","art direction","period feature film",
                            "contemporary feature film","fantasy feature film",
                            r"^period film$",r"^contemporary film$",r"^fantasy film$",
                            "period or fantasy film","best production design"],
    "costume":             ["costume design","excellence in period film","excellence in contemporary",
                            "excellence in period","excellence in fantasy","excellence in sci-fi",
                            "costume illustration"],
    "sound":               ["best sound","sound mixing","sound editing"],
    "makeup":              ["makeup","make up","hair & makeup","hairstyling"],
    "score":               ["original score","best score","best film music"],
    "song":                ["original song","best song","best original song"],
    "visual_effects":      ["visual effects"],
    "documentary":         ["documentary feature","best documentary feature","best documentary$"],
    "animated":            ["animated feature","animated theatrical","animated motion picture$"],
    "animated_short":      ["animated short"],
    "live_action_short":   ["live action short","short film.*live action"],
    "documentary_short":   ["documentary short","edited documentary film"],
    "international":       ["foreign language","non-english language","international feature",
                            "best international","non-english language film"],
    "cast":                ["cast in a motion picture",r"cast in.*motion picture",
                            r"cast of.*motion picture","casting","ensemble cast",
                            "ensemble & casting","best casting"],
}

OSCAR_KEY = {
    "Best Motion Picture of the Year":                                                   "picture",
    "Best Performance by an Actor in a Leading Role":                                    "actor",
    "Best Performance by an Actress in a Leading Role":                                  "actress",
    "Best Performance by an Actor in a Supporting Role":                                 "supp_actor",
    "Best Performance by an Actress in a Supporting Role":                               "supp_actress",
    "Best Achievement in Directing":                                                     "director",
    "Best Original Screenplay":                                                          "original_screenplay",
    "Best Adapted Screenplay":                                                           "adapted_screenplay",
    "Best Achievement in Cinematography":                                                "cinematography",
    "Best Achievement in Film Editing":                                                  "editing",
    "Best Achievement in Production Design":                                             "production_design",
    "Best Achievement in Costume Design":                                                "costume",
    "Best Sound":                                                                        "sound",
    "Best Achievement in Makeup and Hairstyling":                                        "makeup",
    "Best Achievement in Music Written for Motion Pictures (Original Score)":            "score",
    "Best Achievement in Music Written for Motion Pictures (Original Song)":             "song",
    "Best Achievement in Visual Effects":                                                "visual_effects",
    "Best Documentary Feature Film":                                                     "documentary",  # IMDB uses both names
    "Best Documentary Feature":                                                          "documentary",  # for the same category
    "Best Animated Feature Film":                                                        "animated",
    "Best Animated Short Film":                                                          "animated_short",
    "Best Live Action Short Film":                                                       "live_action_short",
    "Best Documentary Short Film":                                                       "documentary_short",
    "Best International Feature Film":                                                   "international",
    "Best Casting":                                                                      "cast",
}
