#!/usr/bin/env python3
"""
src/methods/ml_model/run_ml.py
=====================================
ML model: Elastic-Net / Gradient Boosting with per-category model selection.

Improvements over baseline:
  1. Enhanced features: binary season awards + aggregate counts + sweep indicators
  2. Per-category hyperparameter tuning via LOO-CV (alpha, l1_ratio grid)
  3. Model selection: elastic-net grid + gradient boosting candidates
  4. Extended training window (2000-2025, configured via TRAIN_YEARS)

Produces:
  results/ml_model/predictions.json  — 2026 predictions with rationale
  results/ml_model/validation.json   — 2020-2025 LOO-CV accuracy + AUC

Run from repo root:
  python -m src.methods.ml_model.run_ml
"""

import json
import math
import os
import pickle
import sys
import warnings

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from src.config import (
    OSCAR_KEY, SEASON_YEARS, TRAIN_YEARS, VALIDATION_YEARS,
    NOMINEES_PATH, FEATURES_PATH, HIST_CAT_PATH,
    ML_PRED_PATH, ML_VALID_PATH,
)
from src.utils import norm_cat, get_codes_for_nom, short_event

os.makedirs("results/ml_model", exist_ok=True)

print("=" * 60)
print("ML Model: Enhanced Elastic-Net + GBT (per-category tuning)")
print("=" * 60)

# ── Load artifacts ─────────────────────────────────────────────────────────────
with open(NOMINEES_PATH) as f:
    NOMINEES = json.load(f)
print(f"\nNominees: {sum(len(c['nominations']) for c in NOMINEES)} across {len(NOMINEES)} categories")

with open(FEATURES_PATH, "rb") as f:
    feat_data = pickle.load(f)
year_code_features = feat_data["year_code_features"]

with open(HIST_CAT_PATH, "rb") as f:
    hist_by_cat = pickle.load(f)
print(f"Training rows: {sum(len(v) for v in hist_by_cat.values())} across {len(hist_by_cat)} categories")
print(f"Training years: {TRAIN_YEARS[0]}–{TRAIN_YEARS[-1]}")

# ── Key awards for sweep detection ────────────────────────────────────────────
KEY_AWARDS = {
    "SAG":    "Screen Actors Guild Awards",
    "BAFTA":  "BAFTA Awards",
    "GG":     "Golden Globes, USA",
    "CC":     "Broadcast Film Critics Association Awards",
    "DGA":    "Directors Guild of America, USA",
    "WGA":    "Writers Guild of America, USA",
    "PGA":    "PGA Awards",
    "Spirit": "Film Independent Spirit Awards",
}
KEY_AWARD_NAMES = set(KEY_AWARDS.values())

SWEEP_COMBOS = [
    ("SAG", "BAFTA"),
    ("SAG", "GG"),
    ("BAFTA", "GG"),
    ("SAG", "BAFTA", "GG"),
    ("BAFTA", "GG", "CC"),
]

ENG_NAMES = [
    "__n_wins_in_cat__",
    "__n_noms_in_cat__",
    "__n_wins_cross__",
    "__n_noms_cross__",
] + [f"__sweep_{'_'.join(c)}__" for c in SWEEP_COMBOS]
N_ENG = len(ENG_NAMES)


# ── Feature construction ─────────────────────────────────────────────────────
def build_row_features(codes, yr, cat_key, f_idx, n_base):
    """Build feature vector: binary award indicators + engineered features."""
    x = np.zeros(n_base + N_ENG, dtype=np.float32)
    yr_feats = year_code_features.get(yr, {})

    active = set()
    for code in codes:
        for feat in yr_feats.get(code, set()):
            active.add(feat)
            if feat in f_idx:
                x[f_idx[feat]] = 1

    # Aggregate counts + sweep tracking
    n_wi = n_ni = n_wc = n_nc = 0
    in_cat_win_awards = set()  # full award names that had in-category wins

    for feat in active:
        parts = feat.split("::")
        if len(parts) != 3:
            continue
        event, fcat, status = parts
        if fcat == cat_key:
            if status == "W":
                n_wi += 1
                if event in KEY_AWARD_NAMES:
                    in_cat_win_awards.add(event)
            else:
                n_ni += 1
        else:
            if status == "W":
                n_wc += 1
            else:
                n_nc += 1

    x[n_base + 0] = n_wi
    x[n_base + 1] = n_ni
    x[n_base + 2] = n_wc
    x[n_base + 3] = n_nc

    # Sweep indicators
    for j, combo in enumerate(SWEEP_COMBOS):
        if all(KEY_AWARDS[abbr] in in_cat_win_awards for abbr in combo):
            x[n_base + 4 + j] = 1

    return x


def build_feature_matrix(all_items, cat_key, f_idx, n_base):
    """Build full (n_items, n_base + N_ENG) feature matrix."""
    n = len(all_items)
    X = np.zeros((n, n_base + N_ENG), dtype=np.float32)
    y = np.zeros(n, dtype=np.int32)
    years = np.zeros(n, dtype=np.int32)
    for i, (yr, codes, lbl, _) in enumerate(all_items):
        X[i] = build_row_features(codes, yr, cat_key, f_idx, n_base)
        y[i] = lbl
        years[i] = yr
    return X, y, years


# ── Model candidates ──────────────────────────────────────────────────────────
def get_candidates():
    """Return list of (name, factory_fn, is_linear) tuples."""
    cands = []
    for alpha in [0.001, 0.005, 0.01, 0.05, 0.1]:
        for l1 in [0.1, 0.5, 0.9]:
            cands.append((
                f"enet_a{alpha}_l{l1}",
                lambda a=alpha, l=l1: SGDClassifier(
                    loss="log_loss", penalty="elasticnet", l1_ratio=l,
                    alpha=a, max_iter=500, random_state=42, class_weight="balanced"),
                True,
            ))
    for n_est, depth, lr in [(50, 2, 0.1), (100, 1, 0.05)]:
        cands.append((
            f"gbt_d{depth}_n{n_est}",
            lambda ne=n_est, d=depth, l=lr: GradientBoostingClassifier(
                n_estimators=ne, max_depth=d, learning_rate=l,
                random_state=42, subsample=0.8, min_samples_leaf=3),
            False,
        ))
    return cands


CANDIDATES = get_candidates()


# ── LOO-CV for one candidate ─────────────────────────────────────────────────
def loo_cv(X_all, y_all, year_arr, factory_fn, is_linear):
    """Run LOO-CV, return (pooled_auc, loo_by_year dict)."""
    years = sorted(set(year_arr))
    all_probs, all_labels = [], []
    loo_by_year = {}

    for lo_year in years:
        tr = year_arr != lo_year
        te = year_arr == lo_year
        Xtr, ytr = X_all[tr], y_all[tr]
        Xte, yte = X_all[te], y_all[te]

        if len(np.unique(ytr)) < 2 or Xtr.shape[0] < 3 or yte.sum() == 0:
            continue

        try:
            clf = factory_fn()
            if is_linear:
                clf.fit(Xtr, ytr)
            else:
                sw = compute_sample_weight("balanced", ytr)
                clf.fit(Xtr, ytr, sample_weight=sw)
            probs = clf.predict_proba(Xte)[:, 1]
        except Exception:
            continue

        all_probs.extend(probs.tolist())
        all_labels.extend(yte.tolist())

        n_noms = len(yte)
        actual_idxs = np.where(yte == 1)[0]
        if len(actual_idxs) > 0:
            actual_idx = int(actual_idxs[0])
            ranked = sorted(range(n_noms), key=lambda x: -probs[x])
            rank = ranked.index(actual_idx) + 1
            predicted_idx = ranked[0]
            loo_by_year[lo_year] = {
                "correct": bool(predicted_idx == actual_idx),
                "rank_of_winner": rank,
                "n_nominees": n_noms,
                "winner_prob": round(float(probs[actual_idx]), 4),
                "confidence": round(float(probs[predicted_idx]), 4),
            }

    if len(set(all_labels)) < 2 or len(all_labels) < 5:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(all_labels, all_probs))

    return auc, loo_by_year


# ── Per-category: model selection + prediction ────────────────────────────────
print(f"\n{'Category':<40} {'N':>4} {'Best model':<20} {'AUC':>6} {'Acc':>5}")
print("-" * 80)

predictions = {}
val_by_cat = {}
cat_meta = {}

for cat_data in NOMINEES:
    cat_name = cat_data["category"]
    cat_key = OSCAR_KEY.get(cat_name, norm_cat(cat_name))
    if cat_key is None:
        predictions[cat_name] = []
        continue

    all_items = hist_by_cat.get(cat_key, [])
    if len(all_items) < 10:
        predictions[cat_name] = []
        cat_meta[cat_key] = {"auc": None, "accuracy": None, "best_model": None,
                             "top_features": [], "n_features": 0, "n_nonzero": 0}
        continue

    # ── Build feature vocabulary ─────────────────────────────────────────
    train_feats_set = set()
    for (yr, codes, _, _) in all_items:
        yr_feats = year_code_features.get(yr, {})
        for code in codes:
            train_feats_set.update(yr_feats.get(code, set()))
    train_feats = sorted(train_feats_set)
    f_idx = {f: i for i, f in enumerate(train_feats)}
    n_base = len(train_feats)

    if n_base == 0:
        predictions[cat_name] = []
        cat_meta[cat_key] = {"auc": None, "accuracy": None, "best_model": None,
                             "top_features": [], "n_features": 0, "n_nonzero": 0}
        continue

    all_feat_names = train_feats + ENG_NAMES
    n_total = n_base + N_ENG

    # ── Build enhanced feature matrix ────────────────────────────────────
    X_all, y_all, year_arr = build_feature_matrix(all_items, cat_key, f_idx, n_base)

    if len(np.unique(y_all)) < 2:
        predictions[cat_name] = []
        cat_meta[cat_key] = {"auc": None, "accuracy": None, "best_model": None,
                             "top_features": [], "n_features": n_total, "n_nonzero": 0}
        continue

    # ── Model selection via LOO-CV ───────────────────────────────────────
    best_auc = -1.0
    best_name = None
    best_factory = None
    best_is_linear = True
    best_loo = {}

    for cand_name, factory_fn, is_linear in CANDIDATES:
        auc, loo = loo_cv(X_all, y_all, year_arr, factory_fn, is_linear)
        if not math.isnan(auc) and auc > best_auc:
            best_auc = auc
            best_name = cand_name
            best_factory = factory_fn
            best_is_linear = is_linear
            best_loo = loo

    if best_factory is None:
        predictions[cat_name] = []
        cat_meta[cat_key] = {"auc": None, "accuracy": None, "best_model": None,
                             "top_features": [], "n_features": n_total, "n_nonzero": 0}
        continue

    val_by_cat[cat_key] = best_loo

    # Attach nominee names to loo_by_year
    for lo_year in best_loo:
        te_items = [item for item in all_items if item[0] == lo_year]
        if not te_items:
            continue
        te_mask = year_arr == lo_year
        yte = y_all[te_mask]
        actual_idxs = np.where(yte == 1)[0]
        if len(actual_idxs) > 0:
            actual_idx = int(actual_idxs[0])
            # Re-derive predicted from stored rank info
            ranked_names = sorted(range(len(te_items)),
                                  key=lambda x: -best_loo[lo_year].get("confidence", 0))
            best_loo[lo_year]["predicted"] = te_items[0][3] if te_items else "?"
            best_loo[lo_year]["actual"] = te_items[actual_idx][3] if actual_idx < len(te_items) else "?"
            # Re-predict to get names right
            try:
                clf_tmp = best_factory()
                tr = year_arr != lo_year
                if best_is_linear:
                    clf_tmp.fit(X_all[tr], y_all[tr])
                else:
                    sw = compute_sample_weight("balanced", y_all[tr])
                    clf_tmp.fit(X_all[tr], y_all[tr], sample_weight=sw)
                probs_tmp = clf_tmp.predict_proba(X_all[te_mask])[:, 1]
                ranked = sorted(range(len(yte)), key=lambda x: -probs_tmp[x])
                best_loo[lo_year]["predicted"] = te_items[ranked[0]][3]
                best_loo[lo_year]["actual"] = te_items[actual_idx][3]
            except Exception:
                pass

    # Validation accuracy on VALIDATION_YEARS
    val_preds = {yr: v for yr, v in best_loo.items() if yr in VALIDATION_YEARS}
    val_acc = (sum(1 for v in val_preds.values() if v["correct"]) / len(val_preds)
               if val_preds else None)

    # ── Train final model on ALL data ────────────────────────────────────
    clf_final = best_factory()
    if best_is_linear:
        clf_final.fit(X_all, y_all)
    else:
        sw = compute_sample_weight("balanced", y_all)
        clf_final.fit(X_all, y_all, sample_weight=sw)

    # Top features
    top_feats = []
    n_nonzero = 0
    if best_is_linear and hasattr(clf_final, "coef_") and clf_final.coef_ is not None:
        coefs = clf_final.coef_[0]
        n_nonzero = int(np.sum(coefs != 0))
        top_idxs = np.argsort(coefs)[::-1][:10]
        top_feats = [
            {"feature": all_feat_names[j], "coefficient": round(float(coefs[j]), 4),
             "display": short_event(all_feat_names[j]) if j < n_base else all_feat_names[j]}
            for j in top_idxs if j < n_total and coefs[j] > 0
        ]
    elif not best_is_linear and hasattr(clf_final, "feature_importances_"):
        imps = clf_final.feature_importances_
        n_nonzero = int(np.sum(imps > 0))
        top_idxs = np.argsort(imps)[::-1][:10]
        top_feats = [
            {"feature": all_feat_names[j], "coefficient": round(float(imps[j]), 4),
             "display": short_event(all_feat_names[j]) if j < n_base else all_feat_names[j]}
            for j in top_idxs if j < n_total and imps[j] > 0
        ]

    cat_meta[cat_key] = {
        "auc": round(best_auc, 4) if not math.isnan(best_auc) else None,
        "accuracy": round(val_acc, 3) if val_acc is not None else None,
        "best_model": best_name,
        "top_features": top_feats,
        "n_features": n_total,
        "n_nonzero": n_nonzero,
    }

    # ── Predict 2026 nominees ────────────────────────────────────────────
    pred_noms = cat_data["nominations"]
    if not pred_noms:
        predictions[cat_name] = []
        continue

    X_pred = np.zeros((len(pred_noms), n_total), dtype=np.float32)
    for i, nom in enumerate(pred_noms):
        codes = get_codes_for_nom(nom)
        # Build features across season years
        for yr in SEASON_YEARS:
            row = build_row_features(codes, yr, cat_key, f_idx, n_base)
            X_pred[i] = np.maximum(X_pred[i], row)  # union of features across season years
        # Recompute aggregate counts properly across both season years
        all_active = set()
        for yr in SEASON_YEARS:
            yr_feats = year_code_features.get(yr, {})
            for code in codes:
                all_active.update(yr_feats.get(code, set()))
        n_wi = n_ni = n_wc = n_nc = 0
        in_cat_win_awards = set()
        for feat in all_active:
            parts = feat.split("::")
            if len(parts) != 3:
                continue
            event, fcat, status = parts
            if fcat == cat_key:
                if status == "W":
                    n_wi += 1
                    if event in KEY_AWARD_NAMES:
                        in_cat_win_awards.add(event)
                else:
                    n_ni += 1
            else:
                if status == "W":
                    n_wc += 1
                else:
                    n_nc += 1
        X_pred[i, n_base + 0] = n_wi
        X_pred[i, n_base + 1] = n_ni
        X_pred[i, n_base + 2] = n_wc
        X_pred[i, n_base + 3] = n_nc
        for j, combo in enumerate(SWEEP_COMBOS):
            if all(KEY_AWARDS[abbr] in in_cat_win_awards for abbr in combo):
                X_pred[i, n_base + 4 + j] = 1

    raw_probs = clf_final.predict_proba(X_pred)[:, 1]
    total = raw_probs.sum()
    norm_probs = raw_probs / total if total > 0 else np.ones(len(pred_noms)) / len(pred_noms)

    # Build active features for rationale
    predictions[cat_name] = []
    for i, nom in enumerate(pred_noms):
        active_with_coef = []
        if best_is_linear and hasattr(clf_final, "coef_"):
            coefs = clf_final.coef_[0]
            active_idxs = np.where(X_pred[i] > 0)[0]
            active_with_coef = [
                {"feature": all_feat_names[j], "coefficient": round(float(coefs[j]), 4),
                 "display": short_event(all_feat_names[j]) if j < n_base else all_feat_names[j]}
                for j in active_idxs if j < n_total and coefs[j] != 0
            ]
            active_with_coef.sort(key=lambda x: -x["coefficient"])

        predictions[cat_name].append({
            "name": ", ".join(nom.get("primaryNames", [])),
            "probability": round(float(norm_probs[i]), 4),
            "rationale": {
                "active_features": active_with_coef[:10],
                "top_model_features": top_feats[:5],
                "category_auc": cat_meta[cat_key]["auc"],
                "model_selected": best_name,
            }
        })

    acc_s = f"{val_acc:.0%}" if val_acc is not None else "N/A"
    print(f"{cat_name[:40]:<40} {len(all_items):>4} {best_name:<20} {best_auc:>5.3f} {acc_s:>5}")


# ── Print validation summary ─────────────────────────────────────────────────
print(f"\n{'Category':<25} {'Acc':>5} {'AUC':>6} {'Model':<20} {'Feats':>5} {'NZ':>4}")
print("-" * 70)
for cat_key in [
    "picture", "director", "actor", "actress", "supp_actor", "supp_actress",
    "original_screenplay", "adapted_screenplay", "cinematography", "editing",
    "production_design", "costume", "sound", "makeup", "score", "song",
    "visual_effects", "documentary", "animated", "international",
]:
    if cat_key in cat_meta:
        m = cat_meta[cat_key]
        acc_s = f"{m['accuracy']:.0%}" if m["accuracy"] is not None else "N/A"
        auc_s = f"{m['auc']:.3f}" if m["auc"] is not None else "N/A"
        mdl = m.get("best_model", "?") or "?"
        print(f"{cat_key:<25} {acc_s:>5} {auc_s:>6} {mdl:<20} {m['n_features']:>5} {m['n_nonzero']:>4}")


# ── Build validation output ──────────────────────────────────────────────────
by_year = {}
for cat_key, yr_dict in val_by_cat.items():
    for yr, v in yr_dict.items():
        yr_str = str(yr)
        if yr_str not in by_year:
            by_year[yr_str] = {}
        by_year[yr_str][cat_key] = v

all_val_preds = [
    v for yr, yr_dict in val_by_cat.items()
    for lo_year, v in yr_dict.items() if lo_year in VALIDATION_YEARS
]
overall_acc = sum(1 for v in all_val_preds if v["correct"]) / max(len(all_val_preds), 1)

by_category_val = {}
for cat_key, yr_dict in val_by_cat.items():
    val_only = {yr: v for yr, v in yr_dict.items() if yr in VALIDATION_YEARS}
    if not val_only:
        continue
    acc = sum(1 for v in val_only.values() if v["correct"]) / len(val_only)
    by_category_val[cat_key] = {
        "accuracy": round(acc, 3),
        "avg_winner_prob": round(float(np.mean([v["winner_prob"] for v in val_only.values()])), 3),
        "loo_auc": cat_meta.get(cat_key, {}).get("auc"),
        "best_model": cat_meta.get(cat_key, {}).get("best_model"),
        "n_years": len(val_only),
    }

valid_aucs = [v.get("loo_auc") for v in by_category_val.values() if v.get("loo_auc") is not None]
overall_auc = round(float(np.mean(valid_aucs)), 3) if valid_aucs else None

print(f"\nOverall validation accuracy (2020-2025): {overall_acc:.1%}  ({len(all_val_preds)} cat-years)")
if overall_auc is not None:
    print(f"Overall AUC (mean of per-category LOO-AUC): {overall_auc:.3f}")

# ── Save ─────────────────────────────────────────────────────────────────────
with open(ML_PRED_PATH, "w") as f:
    json.dump({"method": "ml_model", "year": 2026, "categories": predictions}, f, indent=2)
print(f"\nSaved → {ML_PRED_PATH}")

with open(ML_VALID_PATH, "w") as f:
    json.dump({
        "method": "ml_model",
        "validation_years": VALIDATION_YEARS,
        "overall_accuracy": round(overall_acc, 3),
        "overall_auc": overall_auc,
        "n_cat_years": len(all_val_preds),
        "by_year": {yr: {ck: v for ck, v in yr_dict.items()}
                    for yr, yr_dict in by_year.items()
                    if int(yr) in VALIDATION_YEARS},
        "by_category": by_category_val,
        "cat_meta": {k: {"auc": v["auc"], "best_model": v["best_model"],
                         "top_features": v["top_features"],
                         "n_features": v["n_features"], "n_nonzero": v["n_nonzero"]}
                     for k, v in cat_meta.items()},
    }, f, indent=2)
print(f"Saved → {ML_VALID_PATH}")
print("\nML model complete.")


if __name__ == "__main__":
    pass  # script executes on import when run with python -m
