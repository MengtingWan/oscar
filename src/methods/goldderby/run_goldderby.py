#!/usr/bin/env python3
"""
src/methods/goldderby/run_goldderby.py
=====================================
Gold Derby crowd-odds method.

Uses Gold Derby consensus odds from gold_derby_odds_2026.json for 2026
predictions, gd_historical.json (top-1 picks) for basic validation,
and gd_historical_odds.json (full odds where available) for AUC computation.

Produces:
  results/goldderby/predictions.json  — 2026 predictions with rationale
  results/goldderby/validation.json   — 2020-2025 accuracy + AUC (where available)

Run from repo root:
  python -m src.methods.goldderby.run_goldderby
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from src.config import (
    OSCAR_KEY, OSCAR_EVENT_ID, NOMINEES_PATH, GD_ODDS_PATH, GD_HIST_ODDS_PATH,
    GD_PRED_PATH, GD_VALID_PATH, GD_HIST_ODDS_FILE, VALIDATION_YEARS, DF_FULL_PATH,
)
from src.utils import norm_cat, fuzzy_match

os.makedirs("results/goldderby", exist_ok=True)

KEY_TO_CAT = {v: k for k, v in OSCAR_KEY.items()}

# Category name mappings (GD odds use short names, Oscar data uses long names)
GD_CAT_MAP = {
    "Best Picture": "picture", "Best Director": "director",
    "Best Actor": "actor", "Best Actress": "actress",
    "Best Supporting Actor": "supp_actor", "Best Supporting Actress": "supp_actress",
    "Best Original Screenplay": "original_screenplay",
    "Best Adapted Screenplay": "adapted_screenplay",
    "Best Cinematography": "cinematography", "Best Film Editing": "editing",
    "Best Production Design": "production_design",
    "Best Costume Design": "costume", "Best Sound": "sound",
    "Best Makeup and Hairstyling": "makeup", "Best Score": "score",
    "Best Song": "song", "Best Visual Effects": "visual_effects",
    "Best Animated Feature": "animated",
    "Best Documentary Feature": "documentary",
    "Best International Film": "international",
}

print("=" * 60)
print("Gold Derby Method: Crowd Consensus Odds")
print("=" * 60)

# ── Load data ──────────────────────────────────────────────────────────────────
with open(NOMINEES_PATH) as f:
    NOMINEES = json.load(f)
print(f"\nNominees: {sum(len(c['nominations']) for c in NOMINEES)} across {len(NOMINEES)} categories")

with open(GD_ODDS_PATH) as f:
    gd_odds = json.load(f)

with open(GD_HIST_ODDS_PATH) as f:
    gd_hist = json.load(f)

# Load full historical odds if available
gd_hist_odds = {}
if os.path.isfile(GD_HIST_ODDS_FILE):
    with open(GD_HIST_ODDS_FILE) as f:
        gd_hist_odds = json.load(f)
    print(f"Historical odds: {sorted(gd_hist_odds.keys())} ({sum(len(v) for v in gd_hist_odds.values())} category-years)")
else:
    print("No historical odds file found — AUC will only use top-1 data")

# Load Oscar results for validation
import pandas as pd
df_full = pd.read_csv(DF_FULL_PATH)

# Filter out meta keys
gd_cats_2026 = {k: v for k, v in gd_odds.items() if not k.startswith("_")}
gd_hist_years = {k: v for k, v in gd_hist.items() if k.isdigit()}
print(f"GD 2026 categories: {len(gd_cats_2026)}")
print(f"GD historical top-1: {sorted(gd_hist_years.keys())}")


def match_nom_to_odds(nom, cat_key, gd_cat_odds):
    """Match a nominee to a GD odds entry using fuzzy matching."""
    for name in nom.get("primaryNames", []) + nom.get("secondaryNames", []):
        val, matched_key = fuzzy_match(name, gd_cat_odds)
        if matched_key is not None:
            return matched_key, val
    return None, 0.0


# ── 2026 Predictions ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2026 Predictions")
print("=" * 60)
print(f"\n  {'Category':<52} {'Predicted Winner':<30} {'Prob':>6}")
print("  " + "-" * 95)

predictions = {}

for cat_data in NOMINEES:
    cat_name = cat_data["category"]
    cat_key = OSCAR_KEY.get(cat_name, norm_cat(cat_name))
    if cat_key is None:
        continue

    gd_cat_odds = gd_cats_2026.get(cat_key, {})
    noms = cat_data["nominations"]

    cat_preds = []
    total_matched = 0.0

    for nom in noms:
        matched_key, gd_prob = match_nom_to_odds(nom, cat_key, gd_cat_odds)
        primary_name = ", ".join(nom.get("primaryNames", []))
        cat_preds.append({
            "name": primary_name,
            "_gd_key": matched_key,
            "_raw_prob": gd_prob,
        })
        total_matched += gd_prob

    # Normalize
    denom = total_matched if total_matched > 0 else 1.0
    n_noms = len(noms)

    predictions[cat_name] = []
    for i, (nom, pred) in enumerate(zip(noms, cat_preds)):
        raw_prob = pred["_raw_prob"]
        if raw_prob == 0.0:
            residual = max(0.0, 1.0 - total_matched)
            unmatched_count = sum(1 for p in cat_preds if p["_raw_prob"] == 0.0)
            norm_prob = residual / max(unmatched_count, 1) / n_noms
        else:
            norm_prob = raw_prob / denom if denom > 0 else 1.0 / n_noms

        predictions[cat_name].append({
            "name": pred["name"],
            "probability": round(float(norm_prob), 4),
            "rationale": {
                "raw_odds": round(raw_prob, 4) if raw_prob > 0 else None,
                "gd_key": pred["_gd_key"],
                "odds_rank": None,
                "n_candidates": len(gd_cat_odds),
            }
        })

    sorted_by_prob = sorted(predictions[cat_name], key=lambda x: -x["probability"])
    for rank_i, entry in enumerate(sorted_by_prob, start=1):
        for pred_entry in predictions[cat_name]:
            if pred_entry["name"] == entry["name"]:
                pred_entry["rationale"]["odds_rank"] = rank_i
                break

    top = max(predictions[cat_name], key=lambda x: x["probability"])
    print(f"  {cat_name[:52]:<52} {top['name'][:30]:<30} {top['probability']:>5.0%}")

with open(GD_PRED_PATH, "w") as f:
    json.dump({"method": "goldderby", "year": 2026, "categories": predictions}, f, indent=2)
print(f"\nSaved -> {GD_PRED_PATH}")


# ── 2020-2025 Validation ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"Validation: {VALIDATION_YEARS[0]}-{VALIDATION_YEARS[-1]}")
print("=" * 60)

KEY_CATS = [
    "picture", "director", "actor", "actress", "supp_actor", "supp_actress",
    "original_screenplay", "adapted_screenplay", "cinematography", "editing",
    "production_design", "costume", "sound", "makeup", "score", "song",
    "visual_effects", "documentary", "animated", "international",
]

# Build reverse mapping from GD category names to canonical keys
GD_CAT_REV = {v: k for k, v in GD_CAT_MAP.items()}

by_year = {}
all_correct = []
all_aucs = []

for yr in VALIDATION_YEARS:
    yr_str = str(yr)
    yr_top1 = gd_hist_years.get(yr_str, {})
    yr_odds = gd_hist_odds.get(yr_str, {})

    if not yr_top1 and not yr_odds:
        print(f"\n  {yr}: no GD data")
        continue

    # Get actual Oscar winners for this year
    year_oscar = df_full[
        (df_full["eventId"] == OSCAR_EVENT_ID) &
        (df_full["year"] == yr)
    ]

    yr_results = {}
    n_correct = 0
    yr_aucs = []

    for cat_key in KEY_CATS:
        # Get top-1 pick
        top1_entry = yr_top1.get(cat_key, {})
        gd_pick = top1_entry.get("gd_pick", "?")
        actual = top1_entry.get("actual", "?")
        correct = bool(top1_entry.get("correct", False))

        if not top1_entry:
            continue

        # Try to compute AUC from full odds
        auc = None
        gd_cat_name = GD_CAT_REV.get(cat_key)
        odds_data = yr_odds.get(gd_cat_name, []) if gd_cat_name else []

        if odds_data:
            # Get actual winner for this category from Oscar data
            cat_oscar = year_oscar[
                year_oscar["categoryName"].apply(
                    lambda x: norm_cat(str(x)) == cat_key if pd.notna(x) else False
                )
            ]

            if not cat_oscar.empty:
                winner_names = []
                for _, row in cat_oscar.iterrows():
                    if bool(row.get("isWinner", False)):
                        pn = str(row.get("primaryNomineeName", ""))
                        if pn:
                            winner_names.append(pn)

                if winner_names:
                    # Match odds nominees to Oscar nominees, mark winner
                    scores = []
                    labels = []
                    for od in odds_data:
                        p = od["probability"]
                        name = od["name"]
                        # Check if this nominee is the winner
                        is_win = 0
                        for wn in winner_names:
                            if wn.lower() in name.lower() or name.lower() in wn.lower():
                                is_win = 1
                                break
                        scores.append(p)
                        labels.append(is_win)

                    if sum(labels) == 1 and len(labels) >= 2:
                        try:
                            from sklearn.metrics import roc_auc_score
                            auc = float(roc_auc_score(labels, scores))
                        except Exception:
                            pass

        yr_results[cat_key] = {
            "predicted": gd_pick,
            "actual": actual,
            "correct": correct,
            "auc": round(auc, 4) if auc is not None else None,
        }
        all_correct.append(correct)
        if correct:
            n_correct += 1
        if auc is not None:
            yr_aucs.append(auc)
            all_aucs.append(auc)

    n_total = len(yr_results)
    pct = f"{n_correct/n_total:.0%}" if n_total > 0 else "N/A"
    auc_str = f"AUC {np.mean(yr_aucs):.3f} ({len(yr_aucs)} cats)" if yr_aucs else "no AUC data"
    has_odds = "with full odds" if yr_odds else "top-1 only"
    print(f"\n  {yr}: {n_correct}/{n_total} = {pct}  [{has_odds}, {auc_str}]")
    for ck, res in yr_results.items():
        if not res["correct"]:
            print(f"    x {ck:<25} predicted={res['predicted'][:25]!r:27} actual={res['actual'][:25]!r}")

    by_year[yr_str] = yr_results

# Per-category summary
by_category = {}
for cat_key in KEY_CATS:
    cat_vals = [(yr, d[cat_key]) for yr, d in by_year.items() if cat_key in d]
    if not cat_vals:
        continue
    acc = sum(1 for _, v in cat_vals if v["correct"]) / len(cat_vals)
    cat_aucs = [v["auc"] for _, v in cat_vals if v["auc"] is not None]
    by_category[cat_key] = {
        "accuracy": round(acc, 3),
        "auc": round(float(np.mean(cat_aucs)), 3) if cat_aucs else None,
        "n_years": len(cat_vals),
        "n_auc_years": len(cat_aucs),
    }

overall_acc = sum(all_correct) / max(len(all_correct), 1)
overall_auc = float(np.mean(all_aucs)) if all_aucs else None
print(f"\nOverall: {overall_acc:.1%} accuracy ({len(all_correct)} cat-years)")
if all_aucs:
    print(f"Overall AUC: {overall_auc:.3f} ({len(all_aucs)} cat-years with full odds)")
else:
    print("No AUC data available (need gd_historical_odds.json with full odds)")

print(f"\n{'Category':<25} {'Acc':>5}  {'AUC':>6}  {'N':>3}")
print("-" * 45)
for ck in KEY_CATS:
    if ck in by_category:
        d = by_category[ck]
        auc_s = f"{d['auc']:.3f}" if d['auc'] is not None else "  N/A"
        print(f"{ck:<25} {d['accuracy']:>4.0%}  {auc_s:>6}  {d['n_years']:>3}")

with open(GD_VALID_PATH, "w") as f:
    json.dump({
        "method": "goldderby",
        "validation_years": VALIDATION_YEARS,
        "overall_accuracy": round(overall_acc, 3),
        "overall_auc": round(overall_auc, 3) if overall_auc is not None else None,
        "n_cat_years": len(all_correct),
        "n_auc_cat_years": len(all_aucs),
        "note": "AUC computed where full odds available; top-1 accuracy for all years",
        "by_year": by_year,
        "by_category": by_category,
    }, f, indent=2)
print(f"\nSaved -> {GD_VALID_PATH}")
print("\nGold Derby method complete.")


if __name__ == "__main__":
    pass
