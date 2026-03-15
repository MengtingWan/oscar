# 98th Academy Awards Prediction Pipeline

**Ceremony:** March 15, 2026

---

## Overview

This project predicts winners for all 24 categories of the 98th Academy Awards using four independent methods:

| Method | Validation Accuracy | AUC | Description |
|---|---|---|---|
| **Stats** | 71.7% | 0.854 | Bayesian-calibrated P(Oscar \| season award) weights, softmax-normalized |
| **ML** | 73.2% | 0.816 | Elastic-net / GBT with per-category model selection, LOO-CV (2020-2025) |
| **LLM** | N/A | N/A | Independent Claude Code web research — not contaminated by other model outputs |
| **Gold Derby** | 89.6% | 0.957 | Crowd consensus odds; AUC from full odds (2020-2025) |

Validation is over 2020-2025 Oscar ceremonies. LLM has no historical validation (training data leakage would contaminate retrospective predictions).

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# 1. Build feature lookup (required first)
python -m src.methods.stats.build_features

# 2. Run each method independently
python -m src.methods.stats.run_stats
python -m src.methods.ml_model.run_ml
python -m src.methods.goldderby.run_goldderby
# LLM analysis: see results/llm_analysis/predictions.json (pre-generated)

# 3. Open notebook for ensemble
jupyter notebook pipeline.ipynb

# 4. Generate webpage
python -m src.build_web_data                           # -> data JS
python -m src.generate_pages                           # -> HTML pages
# Output goes to ../mengtingwan.github.io/oscars/ by default
```

---

## Directory Structure

```
oscar/
├── data/
│   ├── raw/html_historical/          <- scraped IMDB event JSON (per event/year)
│   └── processed/
│       ├── df_data_full.csv          <- IMDB award data 1990-2026 (131K rows)
│       ├── oscar2026_nominees.json   <- official 2026 nominees with IMDB codes
│       ├── gold_derby_odds_2026.json <- GD consensus odds (updated March 13, 2026)
│       ├── gd_historical.json        <- GD top-1 picks 2020-2025 (corrected)
│       ├── gd_historical_odds.json   <- GD full odds 2024-2025 (for AUC)
│       ├── calibrated_weights.json   <- Bayesian-estimated season award weights
│       ├── season_wins_2026.json     <- season award results for reference
│       ├── year_features.pkl         <- GITIGNORED: rebuilt by build_features.py
│       └── hist_by_cat.pkl           <- GITIGNORED: rebuilt by build_features.py
│
├── src/
│   ├── config.py                     <- shared constants (PRED_YEAR, paths, CAT_NORM)
│   ├── utils.py                      <- shared helpers (parse_codes, norm_cat, softmax)
│   ├── build_web_data.py             <- merges predictions/validation -> data JS
│   ├── generate_pages.py            <- generates oscar2026.html + oscar2026_cn.html
│   └── methods/
│       ├── stats/
│       │   ├── build_features.py     <- builds year_features.pkl + hist_by_cat.pkl
│       │   └── run_stats.py          <- Bayesian stats model: calibrate + predict + validate
│       ├── ml_model/
│       │   └── run_ml.py             <- ML model: elastic-net/GBT, LOO-CV
│       ├── goldderby/
│       │   └── run_goldderby.py      <- Gold Derby crowd odds + AUC validation
│       └── llm_analysis/             <- (placeholder; predictions pre-generated)
│
├── pipeline.ipynb                    <- ensemble notebook (edit weights in Cell 6)
└── results/
    ├── stats/
    │   ├── predictions.json          <- 2026 stats predictions with rationale
    │   └── validation.json           <- 2020-2025 LOO-CV accuracy + AUC
    ├── ml_model/
    │   ├── predictions.json          <- 2026 ML predictions with active features
    │   └── validation.json           <- 2020-2025 LOO-CV accuracy + AUC
    ├── goldderby/
    │   ├── predictions.json          <- 2026 GD odds-based predictions
    │   └── validation.json           <- 2020-2025 accuracy + AUC (where odds available)
    └── llm_analysis/
        └── predictions.json          <- 2026 independent LLM predictions + reasoning

Webpages are hosted at mengtingwan.github.io/oscars/ (separate repo).
```

---

## Method Details

### Stats Model (Bayesian Calibration)
For each (award, category) pair, estimates P(Oscar win | season award) from 2005-2025 data using Bayesian inference with a Beta(1,1) prior. Posterior mean: P = (k+1)/(n+2), where k = times the award winner also won the Oscar, and n = total occurrences. Win and nomination weights are estimated separately (no arbitrary factors). Each nominee's score = sum of calibrated weights for their season awards, softmax-normalized per category. Calibrated weights saved to `data/processed/calibrated_weights.json`. Validation uses leave-one-year-out calibration to prevent information leakage.

### ML Model (Elastic-Net / GBT)
Per-category model selection via LOO-CV. Candidates: elastic-net grid (alpha x l1_ratio) + gradient boosting (conservative depth). Enhanced features: binary season awards + aggregate counts (in-cat wins/noms, cross-cat wins/noms) + sweep indicators (SAG+BAFTA, BAFTA+GG+CC, etc.). Training data: 2000-2025. Best model picked per category by LOO-AUC.

### LLM Analysis
Independent Claude Code web research conducted March 14, 2026 (one day before the ceremony). No Gold Derby model outputs, stats model outputs, or ML outputs were consulted as primary sources. Research covered: guild awards (DGA, SAG, WGA, PGA, ACE, ASC, CAS, CDG, ADG, VES, MUAHS, Annie), BAFTA, Golden Globes, Critics Choice, Spirit Awards, and trade publications (Variety, IndieWire, Hollywood Reporter, Deadline).

### Gold Derby
Uses `gold_derby_odds_2026.json` for 2026 predictions (numeric consensus odds from expert/editor/user aggregation). For validation: `gd_historical.json` provides top-1 picks for 2020-2025 (accuracy), and `gd_historical_odds.json` provides full odds distributions for 2024-2025 (enabling AUC computation). Note: the historical top-1 data was corrected — the original file had wrong 2025 entries (Brady Corbet as Director, Demi Moore as Actress — both were GD misses; actual winners were Sean Baker and Mikey Madison).

### Ensemble
Open `pipeline.ipynb` (repo root) and edit the weight variables in Cell 6:
```python
W_STATS = 0.25
W_ML    = 0.25
W_LLM   = 0.25
W_GD    = 0.25
```
Re-run all cells to recompute.

---

## Key Design Decisions

- **No data leakage**: Oscar events excluded from season-award features; stats validation uses LOO-calibrated weights
- **Bayesian calibration**: Stats weights estimated from data with Beta(1,1) smoothing, not hand-set
- **Independent LLM**: Claude Code did its own web research without consulting stats/ML/GD outputs
- **Rationale logging**: Each prediction includes a rationale block (season wins, ML top features, LLM reasoning, GD odds rank)
- **Small files tracked in git**: All `.json` result files and processed data tracked; binary `.pkl` intermediates gitignored and regenerated via `build_features.py`
