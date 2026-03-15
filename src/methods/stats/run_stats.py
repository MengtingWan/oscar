#!/usr/bin/env python3
"""
src/methods/stats/run_stats.py
=====================================
Stats model: Bayesian-calibrated season award weights.

For each (award, category) pair, estimates P(Oscar win | season award win)
from historical data (2005-2025) using a Beta(1,1) prior (Laplace smoothing).
Win and nomination weights are estimated separately — no arbitrary factors.

Score per nominee = sum of calibrated weights for their season awards.
Probabilities via softmax normalization per category.

Produces:
  results/stats/predictions.json         — 2026 predictions with rationale
  results/stats/validation.json          — 2020-2025 LOO-CV accuracy + AUC
  data/processed/calibrated_weights.json — estimated weights for transparency

Run from repo root:
  python -m src.methods.stats.run_stats
"""

import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from src.config import (
    OSCAR_EVENT_ID, OSCAR_KEY, VALIDATION_YEARS,
    NOMINEES_PATH, FEATURES_PATH, DF_FULL_PATH,
    STATS_PRED_PATH, STATS_VALID_PATH, CALIB_WEIGHTS_PATH,
)
from src.utils import norm_cat, parse_codes, softmax, get_codes_for_nom

os.makedirs("results/stats", exist_ok=True)

# Categories to evaluate
KEY_CATS = [
    "picture", "director", "actor", "actress", "supp_actor", "supp_actress",
    "original_screenplay", "adapted_screenplay", "cinematography", "editing",
    "production_design", "costume", "sound", "makeup", "score", "song",
    "visual_effects", "documentary", "animated", "international",
]

print("=" * 60)
print("Stats Model: Bayesian-Calibrated Season Award Weights")
print("=" * 60)

# ── Load artifacts ────────────────────────────────────────────────────────────
with open(NOMINEES_PATH) as f:
    NOMINEES = json.load(f)
print(f"\nNominees: {sum(len(c['nominations']) for c in NOMINEES)} across {len(NOMINEES)} categories")

with open(FEATURES_PATH, "rb") as f:
    feat_data = pickle.load(f)
year_code_features = feat_data["year_code_features"]
current_code_features = feat_data["current_code_features"]
print(f"Feature lookup: {len(current_code_features)} current-season codes")

df_full = pd.read_csv(DF_FULL_PATH)
print(f"Historical data: {len(df_full):,} rows")


# ── Weight calibration ────────────────────────────────────────────────────────
def calibrate_weights(df, yr_feats_all, years, cat_keys=KEY_CATS):
    """
    Estimate P(Oscar win | season award, category) from historical data.

    For each (award_event, oscar_category, W/N) triple, counts:
      hits  = nominee had this season award AND won the Oscar
      total = nominee had this season award (regardless of Oscar outcome)

    Posterior mean with Beta(1,1) prior: P = (hits + 1) / (total + 2)

    Returns: {event_name: {cat_key: {"W": float, "N": float}}}
    """
    counts = {}  # (event, cat_key, status) -> [hits, total]

    for year in years:
        yr_feats = yr_feats_all.get(year, {})
        if not yr_feats:
            continue

        year_oscar = df[
            (df["eventId"] == OSCAR_EVENT_ID) &
            (df["year"] == year)
        ]

        for cat_key in cat_keys:
            cat_noms = []
            for _, row in year_oscar.iterrows():
                ck = norm_cat(str(row["categoryName"]))
                if ck != cat_key:
                    continue
                codes = parse_codes(row["primaryNomineeCode"]) + \
                        parse_codes(row["secondaryNomineeCode"])
                is_winner = bool(row["isWinner"])
                cat_noms.append((codes, is_winner))

            if not cat_noms:
                continue

            for codes, is_winner in cat_noms:
                seen = set()
                for code in codes:
                    for feat in yr_feats.get(code, set()):
                        if feat in seen:
                            continue
                        seen.add(feat)
                        parts = feat.split("::")
                        if len(parts) != 3:
                            continue
                        event_name, feat_cat, status = parts
                        if feat_cat != cat_key or status not in ("W", "N"):
                            continue

                        key = (event_name, cat_key, status)
                        if key not in counts:
                            counts[key] = [0, 0]
                        counts[key][1] += 1
                        if is_winner:
                            counts[key][0] += 1

    # Compute posterior mean: Beta(1,1) prior
    weights = {}
    for (event, cat_key, status), (hits, total) in counts.items():
        p = (hits + 1) / (total + 2)
        if event not in weights:
            weights[event] = {}
        if cat_key not in weights[event]:
            weights[event][cat_key] = {}
        weights[event][cat_key][status] = round(p, 4)

    return weights, counts


# Calibrate on full training data for 2026 prediction
CALIB_YEARS = list(range(2005, 2026))  # 20 seasons of data
print(f"\nCalibrating weights from {CALIB_YEARS[0]}-{CALIB_YEARS[-1]} ({len(CALIB_YEARS)} years)...")
WEIGHTS, RAW_COUNTS = calibrate_weights(df_full, year_code_features, CALIB_YEARS)

# Print top weights
print(f"\n  {'Award':<45} {'Category':<18} {'Type':>4}  {'P':>6}  {'Hits/Total':>10}")
print("  " + "-" * 90)
top_items = []
for event, cats in WEIGHTS.items():
    for cat_key, statuses in cats.items():
        for status, p in statuses.items():
            key = (event, cat_key, status)
            hits, total = RAW_COUNTS.get(key, [0, 0])
            top_items.append((p, event, cat_key, status, hits, total))
top_items.sort(reverse=True)
for p, event, cat_key, status, hits, total in top_items[:25]:
    print(f"  {event[:45]:<45} {cat_key:<18} {status:>4}  {p:>5.3f}  {hits:>3}/{total:<3}")

# Save calibrated weights
calib_path = CALIB_WEIGHTS_PATH
calib_export = {}
for event, cats in WEIGHTS.items():
    for cat_key, statuses in cats.items():
        for status, p in statuses.items():
            key = (event, cat_key, status)
            hits, total = RAW_COUNTS.get(key, [0, 0])
            calib_export[f"{event}::{cat_key}::{status}"] = {
                "weight": p, "hits": hits, "total": total,
                "note": f"P(Oscar|{status}) = ({hits}+1)/({total}+2)"
            }
with open(calib_path, "w") as f:
    json.dump(calib_export, f, indent=2, ensure_ascii=False)
print(f"\nSaved calibrated weights -> {calib_path} ({len(calib_export)} entries)")


# ── Scoring helper ────────────────────────────────────────────────────────────
def score_nominee(codes, cat_key, code_feat_lookup, weights):
    """
    Score a nominee by summing Bayesian-calibrated season weights.
    Wins and nominations use separately estimated weights.
    """
    score = 0.0
    season_wins = []
    season_noms = []
    seen_feats = set()
    for code in codes:
        for feat in code_feat_lookup.get(code, set()):
            if feat in seen_feats:
                continue
            seen_feats.add(feat)
            parts = feat.split("::")
            if len(parts) != 3:
                continue
            event_name, feat_cat, status = parts
            if feat_cat != cat_key:
                continue
            w = weights.get(event_name, {}).get(cat_key, {}).get(status, 0.0)
            if w <= 0:
                continue
            score += w
            entry = {"award": event_name, "category": feat_cat, "weight": round(w, 4)}
            if status == "W":
                season_wins.append(entry)
            else:
                season_noms.append(entry)
    return score, season_wins, season_noms


# ── 2026 Predictions ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2026 Predictions")
print("=" * 60)
print(f"\n  {'Category':<52} {'Predicted Winner':<28} {'Prob':>6}  Key wins")
print("  " + "-" * 115)

predictions = {}

for cat_data in NOMINEES:
    cat_name = cat_data["category"]
    cat_key = OSCAR_KEY.get(cat_name, norm_cat(cat_name))
    if cat_key is None:
        continue

    noms = cat_data["nominations"]
    raw_scores, all_wins, all_noms = [], [], []

    for nom in noms:
        codes = get_codes_for_nom(nom)
        raw, wins, s_noms = score_nominee(codes, cat_key, current_code_features, WEIGHTS)
        raw_scores.append(raw + 1e-4)
        all_wins.append(wins)
        all_noms.append(s_noms)

    probs = softmax(raw_scores)

    predictions[cat_name] = []
    for nom, prob, raw, wins, s_noms in zip(noms, probs, raw_scores, all_wins, all_noms):
        predictions[cat_name].append({
            "name": ", ".join(nom.get("primaryNames", [])),
            "probability": round(float(prob), 4),
            "rationale": {
                "score_raw": round(raw - 1e-4, 4),
                "season_wins": wins,
                "season_nominations": s_noms,
            }
        })

    top = max(predictions[cat_name], key=lambda x: x["probability"])
    key_wins_str = "; ".join(
        w["award"].split(",")[0] for w in top["rationale"]["season_wins"][:3]
    ) or "no season wins"
    print(f"  {cat_name[:52]:<52} {top['name'][:28]:<28} {top['probability']:>5.0%}  {key_wins_str}")

with open(STATS_PRED_PATH, "w") as f:
    json.dump({"method": "stats", "year": 2026, "categories": predictions}, f, indent=2)
print(f"\nSaved -> {STATS_PRED_PATH}")


# ── 2020-2025 Validation (LOO-calibrated weights) ────────────────────────────
print("\n" + "=" * 60)
print(f"Validation: {VALIDATION_YEARS[0]}-{VALIDATION_YEARS[-1]} (LOO-calibrated)")
print("=" * 60)

by_year = {}

for val_year in VALIDATION_YEARS:
    # LOO: calibrate weights on all years EXCEPT the held-out year
    loo_years = [y for y in CALIB_YEARS if y != val_year]
    loo_weights, _ = calibrate_weights(df_full, year_code_features, loo_years)

    yr_feats = year_code_features.get(val_year, {})
    year_oscar = df_full[
        (df_full["eventId"] == OSCAR_EVENT_ID) &
        (df_full["year"] == val_year)
    ]
    if year_oscar.empty:
        print(f"\n  {val_year}: no Oscar data - skipping")
        continue

    year_results = {}
    for cat_key in KEY_CATS:
        cat_rows = []
        for _, r in year_oscar.iterrows():
            ck = norm_cat(str(r["categoryName"]))
            if ck != cat_key:
                continue
            p_codes = parse_codes(r["primaryNomineeCode"])
            s_codes = parse_codes(r["secondaryNomineeCode"])
            codes = p_codes + s_codes
            label = int(bool(r["isWinner"]))
            nom_name = str(r.get("primaryNomineeName", "?"))
            cat_rows.append((codes, nom_name, label))

        if not cat_rows or sum(lb for _, _, lb in cat_rows) == 0:
            continue

        raw_scores = []
        for (codes, _, _) in cat_rows:
            raw, _, _ = score_nominee(codes, cat_key, yr_feats, loo_weights)
            raw_scores.append(raw + 1e-4)

        probs = softmax(raw_scores)
        labels = [lb for _, _, lb in cat_rows]

        predicted_idx = int(np.argmax(probs))
        actual_idx = next((i for i, lb in enumerate(labels) if lb == 1), -1)
        correct = (predicted_idx == actual_idx)

        rank_of_winner = (
            sorted(range(len(probs)), key=lambda x: -probs[x]).index(actual_idx) + 1
            if actual_idx >= 0 else -1
        )

        auc = None
        if len(labels) >= 2 and sum(labels) == 1:
            try:
                auc = float(roc_auc_score(labels, probs))
            except Exception:
                pass

        year_results[cat_key] = {
            "predicted": cat_rows[predicted_idx][1],
            "actual": cat_rows[actual_idx][1] if actual_idx >= 0 else "?",
            "correct": correct,
            "confidence": round(probs[predicted_idx], 4),
            "winner_prob": round(probs[actual_idx], 4) if actual_idx >= 0 else 0.0,
            "rank_of_winner": rank_of_winner,
            "n_nominees": len(cat_rows),
            "auc": round(auc, 4) if auc is not None else None,
        }

    n_correct = sum(1 for v in year_results.values() if v["correct"])
    n_total = len(year_results)
    pct = f"{n_correct/n_total:.0%}" if n_total > 0 else "N/A"
    print(f"\n  {val_year}: {n_correct}/{n_total} = {pct}")
    for ck, res in year_results.items():
        if not res["correct"]:
            print(f"    x {ck:<25} predicted={res['predicted'][:25]!r:27} actual={res['actual'][:25]!r}")

    by_year[str(val_year)] = year_results

# ── Per-category summary ─────────────────────────────────────────────────────
by_category = {}
for cat_key in KEY_CATS:
    preds = [v[cat_key] for v in by_year.values() if cat_key in v]
    if not preds:
        continue
    acc = sum(1 for p in preds if p["correct"]) / len(preds)
    aucs = [p["auc"] for p in preds if p["auc"] is not None]
    by_category[cat_key] = {
        "accuracy": round(acc, 3),
        "auc": round(float(np.mean(aucs)), 3) if aucs else None,
        "avg_winner_prob": round(float(np.mean([p["winner_prob"] for p in preds])), 3),
        "n_years": len(preds),
    }

all_preds_flat = [v for yr_dict in by_year.values() for v in yr_dict.values()]
overall_acc = sum(1 for p in all_preds_flat if p["correct"]) / max(len(all_preds_flat), 1)
all_aucs = [p["auc"] for p in all_preds_flat if p["auc"] is not None]
overall_auc = float(np.mean(all_aucs)) if all_aucs else 0.0

print(f"\nOverall accuracy: {overall_acc:.1%}   AUC: {overall_auc:.3f}   ({len(all_preds_flat)} cat-years)")
print(f"\n{'Category':<25} {'Acc':>5}  {'AUC':>6}  {'AvgWinP':>8}  {'N':>3}")
print("-" * 52)
for ck in KEY_CATS:
    if ck in by_category:
        d = by_category[ck]
        auc_s = f"{d['auc']:.3f}" if d['auc'] is not None else "  N/A"
        print(f"{ck:<25} {d['accuracy']:>4.0%}  {auc_s:>6}  {d['avg_winner_prob']:>7.3f}  {d['n_years']:>3}")

with open(STATS_VALID_PATH, "w") as f:
    json.dump({
        "method": "stats",
        "validation_years": VALIDATION_YEARS,
        "overall_accuracy": round(overall_acc, 3),
        "overall_auc": round(overall_auc, 3),
        "n_cat_years": len(all_preds_flat),
        "by_year": by_year,
        "by_category": by_category,
    }, f, indent=2)
print(f"\nSaved -> {STATS_VALID_PATH}")
print("\nStats model complete.")


if __name__ == "__main__":
    pass  # script executes on import when run with python -m
