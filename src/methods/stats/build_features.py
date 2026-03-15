#!/usr/bin/env python3
"""
src/methods/stats/build_features.py
=====================================
Build year-indexed feature lookups from df_data_full.csv.

Outputs (gitignored intermediates):
  data/processed/year_features.pkl   — {year: {code: {feature_strings}}}
  data/processed/hist_by_cat.pkl     — {cat_key: [(year, codes, label, name)]}

KEY DESIGN: Oscar event (ev0000003) is excluded so that Oscar wins never
appear as input features — only as labels. This prevents data leakage.
Year-indexed lookup means each prediction for year N can only use features
from season years [N-1, N].

Run from repo root:
  python -m src.methods.stats.build_features
"""

import os
import pickle
import sys
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from src.config import OSCAR_EVENT_ID, TRAIN_YEARS, SEASON_YEARS, DF_FULL_PATH, FEATURES_PATH, HIST_CAT_PATH
from src.utils import parse_codes, norm_cat, make_feature_name

os.makedirs("data/processed", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("=" * 60)
print("Build Year-Indexed Feature Lookup")
print("=" * 60)

# ── Load data ─────────────────────────────────────────────────────────────────
print(f"\nLoading {DF_FULL_PATH} ...")
df_full = pd.read_csv(DF_FULL_PATH)
print(f"  {len(df_full):,} rows  |  years {int(df_full['year'].min())}–{int(df_full['year'].max())}")

# ── Build year-indexed feature lookup ─────────────────────────────────────────
print("\nBuilding year_code_features (Oscar event excluded) ...")
year_code_features = defaultdict(lambda: defaultdict(set))

skipped_oscar = 0
added = 0
for _, row in df_full.iterrows():
    event_id = str(row.get("eventId", ""))
    if event_id == OSCAR_EVENT_ID:
        skipped_oscar += 1
        continue

    event    = str(row["eventName"])
    cat_raw  = str(row["categoryName"])
    cat_key  = norm_cat(cat_raw)
    if cat_key is None:
        continue

    year      = int(row["year"])
    is_winner = bool(row["isWinner"])
    feat      = make_feature_name(event, cat_key, is_winner)

    for code in parse_codes(row["primaryNomineeCode"]):
        year_code_features[year][code].add(feat)
    for code in parse_codes(row["secondaryNomineeCode"]):
        year_code_features[year][code].add(feat)
    added += 1

print(f"  Rows processed: {added:,}  (Oscar rows skipped: {skipped_oscar:,})")
print(f"  Year-code pairs: {sum(len(v) for v in year_code_features.values()):,}")
print(f"  Years covered: {sorted(year_code_features.keys())[:5]} … {sorted(year_code_features.keys())[-3:]}")

# ── Build current-season lookup ───────────────────────────────────────────────
current_code_features = defaultdict(set)
for yr in SEASON_YEARS:
    for code, feats in year_code_features[yr].items():
        current_code_features[code].update(feats)
print(f"\nCurrent season ({SEASON_YEARS}) unique codes: {len(current_code_features):,}")

# ── Build Oscar training rows ─────────────────────────────────────────────────
print(f"\nBuilding hist_by_cat from Oscar nominations (years {TRAIN_YEARS[0]}–{TRAIN_YEARS[-1]}) ...")
oscar_rows = df_full[
    (df_full["eventId"] == OSCAR_EVENT_ID) &
    (df_full["year"].isin(TRAIN_YEARS))
].copy()
print(f"  Oscar training rows: {len(oscar_rows):,}")

hist_by_cat = defaultdict(list)  # cat_key → [(year, codes, label, nom_name)]
for _, r in oscar_rows.iterrows():
    year    = int(r["year"])
    cat_key = norm_cat(str(r["categoryName"]))
    if cat_key is None:
        continue
    p_codes  = parse_codes(r["primaryNomineeCode"])
    s_codes  = parse_codes(r["secondaryNomineeCode"])
    codes    = p_codes + s_codes
    label    = int(bool(r["isWinner"]))
    nom_name = str(r.get("primaryNomineeName", ""))
    hist_by_cat[cat_key].append((year, codes, label, nom_name))

print(f"  Categories with training data: {len(hist_by_cat)}")
for ck, rows in sorted(hist_by_cat.items()):
    n_wins = sum(l for _, _, l, _ in rows)
    print(f"    {ck:<25}  {len(rows):>4} rows  {n_wins} winners")

# ── Save artifacts ────────────────────────────────────────────────────────────
with open(FEATURES_PATH, "wb") as f:
    pickle.dump({
        "year_code_features":    dict(year_code_features),
        "current_code_features": dict(current_code_features),
    }, f, protocol=4)
print(f"\nSaved → {FEATURES_PATH}")

with open(HIST_CAT_PATH, "wb") as f:
    pickle.dump(dict(hist_by_cat), f, protocol=4)
print(f"Saved → {HIST_CAT_PATH}")
print("\nBuild features complete.")


if __name__ == "__main__":
    pass  # script executes on import when run with python -m
