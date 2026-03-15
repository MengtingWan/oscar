---
name: oscar-predict
description: Predict Academy Awards winners using a four-method ensemble (Stats, ML, LLM research, Crowd consensus). Scrapes historical award data from IMDB, estimates Bayesian weights, trains ML models with full LOO-CV validation, conducts independent LLM web research, and combines everything into a weighted ensemble. Use when the user wants Oscar predictions, award season analysis, or to run the prediction pipeline for any ceremony year.
argument-hint: "[year] [--weights stats=0.25,ml=0.25,llm=0.25,gd=0.25] [--categories picture,director,actor,...]"
---

# Oscar Prediction Pipeline

Build and run a complete Oscar prediction system from scratch: scrape historical data, estimate Bayesian weights from data, train ML models with cross-validation, conduct independent LLM web research, gather crowd consensus odds, and ensemble everything.

## Arguments

- `$0` — Ceremony year (e.g., `2027`). Defaults to next upcoming ceremony.
- `--weights` — Custom ensemble weights as `stats=X,ml=X,llm=X,gd=X` (normalized to 1.0). Default: equal 0.25.
- `--categories` — Comma-separated category keys to focus on. Default: all 24.

## Category Keys

`picture`, `director`, `actor`, `actress`, `supp_actor`, `supp_actress`, `original_screenplay`, `adapted_screenplay`, `cinematography`, `editing`, `production_design`, `costume`, `sound`, `makeup`, `score`, `song`, `visual_effects`, `documentary`, `animated`, `animated_short`, `live_action_short`, `documentary_short`, `international`, `cast`

---

## Step 1: Project Setup

Create the directory structure and install dependencies:

```
oscar/
├── data/processed/          # Scraped + processed data
├── results/stats/           # Stats method outputs
├── results/ml_model/        # ML method outputs
├── results/llm_analysis/    # LLM method outputs
├── results/goldderby/       # Crowd method outputs
├── src/
│   ├── config.py            # Shared constants
│   ├── utils.py             # Shared utilities
│   └── methods/
│       ├── stats/
│       │   ├── build_features.py
│       │   └── run_stats.py
│       ├── ml_model/
│       │   └── run_ml.py
│       └── goldderby/
│           └── run_goldderby.py
└── pipeline.ipynb           # Ensemble notebook
```

Dependencies: `pandas`, `numpy`, `scikit-learn`, `requests`, `beautifulsoup4`, `anthropic`, `jinja2`.

Write a `src/config.py` with:
- `PRED_YEAR`, `CEREMONY_NAME`, `CEREMONY_DATE`
- `OSCAR_EVENT_ID = "ev0000003"` (IMDB event ID for the Oscars)
- `SEASON_YEARS` — the two calendar years spanning the award season (e.g., `[2026, 2027]`)
- `TRAIN_YEARS = list(range(2000, PRED_YEAR))`
- `VALIDATION_YEARS = list(range(PRED_YEAR - 6, PRED_YEAR))`
- All file paths for data and results
- `CAT_NORM` dict — regex patterns mapping freeform award category names to canonical keys (see `reference.md`)
- `OSCAR_KEY` dict — official Oscar category names to canonical keys

Write a `src/utils.py` with:
- `parse_codes(s)` — parse stringified list of IMDB codes to Python list
- `norm_cat(cat_name)` — normalize freeform award category name to canonical key using regex matching against `CAT_NORM`
- `make_feature_name(event_name, cat_key, is_winner)` — build feature string `"<event>::<cat_key>::W|N"`
- `softmax(scores)` — numerically stable softmax (subtract max before exp)
- `get_codes_for_nom(nom)` — extract all IMDB codes from nominee dict
- `fuzzy_match(name, lookup_dict)` — case-insensitive substring match
- `short_event(feat_str)` — shorten feature string for display (e.g., "Screen Actors Guild Awards::actor::W" → "SAG Win")

## Step 2: Scrape Historical Award Data

Write a scraper to build a comprehensive historical dataset of award results. The goal is a CSV (`data/processed/df_data_full.csv`) with columns:

```
eventId, eventName, year, categoryName, primaryNomineeCode, secondaryNomineeCode, primaryNomineeName, isWinner
```

**Data source**: IMDB award event pages. Each event has an IMDB event ID (e.g., `ev0000003` for Oscars). Scrape nominees and winners for **at least 20 years** of data (e.g., 2000–present) from all of these events:

| Event | IMDB ID | What it predicts |
|-------|---------|------------------|
| Academy Awards | ev0000003 | (target labels — NOT features) |
| SAG Awards | ev0000598 | Acting, Ensemble→Picture |
| DGA Awards | ev0000212 | Director |
| PGA Awards | ev0000531 | Picture |
| WGA Awards | ev0000710 | Screenplay |
| BAFTA Awards | ev0000123 | All major categories |
| Golden Globes | ev0000292 | Picture, Director, Acting |
| Critics Choice | ev0000133 | All categories |
| ACE Eddie Awards | ev0000020 | Editing |
| ASC Awards | ev0000043 | Cinematography |
| ADG Awards | ev0000057 | Production Design |
| CDG Awards | ev0000180 | Costume |
| CAS Awards | ev0000157 | Sound |
| VES Awards | ev0000699 | Visual Effects |
| Annie Awards | ev0000048 | Animated Feature |
| Spirit Awards | ev0000349 | Independent film categories |
| Grammy Awards | ev0000302 | Song |

For each event/year, extract every nominee with their IMDB person/title codes, category name, and whether they won. Store primary codes (person or title) and secondary codes (the associated title or person).

**Critical rule**: The Oscar event (`ev0000003`) data is collected for labels only. It must NEVER be used as input features — only as the target variable. This prevents data leakage.

Also search the web for current-year nominee data and save to `data/processed/oscar{YEAR}_nominees.json`:
```json
[
  {
    "category": "Best Motion Picture of the Year",
    "nominations": [
      {
        "primaryNames": ["Film Title"],
        "secondaryNames": ["Producer Name"],
        "primaryCodes": ["tt1234567"],
        "secondaryCodes": ["nm1234567"],
        "imgUrls": []
      }
    ]
  }
]
```

## Step 3: Build Feature Lookups

Write `src/methods/stats/build_features.py` that processes the historical CSV into two data structures:

### 3a: Year-indexed feature lookup (`year_features.pkl`)

Structure: `{year: {imdb_code: {set of feature strings}}}`

For every row in the historical CSV **except Oscar rows** (exclude `eventId == ev0000003`):
1. Normalize the category name to a canonical key via `norm_cat()`
2. Build a feature string: `"{eventName}::{cat_key}::W"` for winners, `"{eventName}::{cat_key}::N"` for nominees
3. Index by year → IMDB code → set of feature strings

Also build a `current_code_features` lookup that unions features across the current season years.

### 3b: Oscar training rows (`hist_by_cat.pkl`)

Structure: `{cat_key: [(year, codes, label, nom_name), ...]}`

For every Oscar row in `TRAIN_YEARS`:
1. Normalize category → canonical key
2. Extract IMDB codes (primary + secondary)
3. Label = 1 if winner, 0 if nominee
4. Group by category key

Save both as pickle files.

## Step 4: Method 1 — Stats (Bayesian-Calibrated Season Award Weights)

Write `src/methods/stats/run_stats.py`. This method estimates weights from data — no hardcoded values.

### Why Bayesian Estimation?

The core question is: given that a nominee won (or was nominated for) a particular season award, how much should that increase our belief they'll win the Oscar? We need to estimate **θ = P(Oscar win | season award signal)** for every (award, category, win/nom) combination.

The naive frequentist estimate θ̂ = k/n (where k = times the signal holder won the Oscar, n = times they were an Oscar nominee) fails for sparse signals. If an obscure award has k=1, n=1, the naive estimate is 1.0 — obviously overfit. We need **regularization toward uncertainty**.

**Bayesian model:**

- **Likelihood**: Each observation is a Bernoulli trial — `k | θ ~ Binomial(n, θ)`
- **Prior**: `θ ~ Beta(1, 1)` — the uniform distribution on [0,1], encoding maximum ignorance (any value of θ equally plausible before seeing data)
- **Posterior** (by Beta-Binomial conjugacy): `θ | k, n ~ Beta(1 + k, 1 + n - k)`
- **Posterior mean**: `E[θ | k, n] = (k + 1) / (n + 2)`

This gives us three critical properties:

1. **Automatic shrinkage toward 0.5**: Sparse signals (small n) are pulled toward 0.5 (maximum uncertainty). Data-rich signals converge to k/n. A DGA→Director weight with k=17, n=20 gets 0.818 (close to the data). An obscure signal with k=1, n=1 gets 0.667 (heavily shrunk from 1.0). An unseen signal defaults to 0.500 (pure prior).

2. **Laplace's rule of succession**: The formula (k+1)/(n+2) answers "if k of n past holders won, what's the probability the next one wins?" — directly the quantity we need for prediction.

3. **Optimal under squared-error loss**: The posterior mean minimizes E[(θ̂ − θ)²], so when we sum weights, the total is the best estimate of cumulative evidence.

**Why sum the weights?** The scoring formula `score = Σ θ̂(event, cat, status)` is a naive Bayes-like decomposition — each signal contributes independently. This is an approximation (in reality, DGA and PGA winners often coincide), but it works well because: (a) Bayesian shrinkage prevents weak signals from dominating, (b) softmax normalization calibrates probabilities relative to competitors, and (c) LOO-CV validation confirms ~72% accuracy, 0.85 AUC. The ML model (Method 2) handles signal interactions properly via sweep indicators and learned coefficients — that's why having both in the ensemble is valuable.

### Weight Calibration Algorithm

For each `(award_event, oscar_category, W/N)` triple, count from historical data:
- `hits` = number of times a nominee with this season award signal also won the Oscar
- `total` = number of times a nominee with this season award signal appeared as an Oscar nominee

Compute the Bayesian posterior mean:

```
P(Oscar win | season award signal) = (hits + 1) / (total + 2)
```

This is estimated separately for wins (`W`) and nominations (`N`) — no arbitrary discount factor.

Implementation:
1. Loop over calibration years (e.g., 2005 to PRED_YEAR-1)
2. For each year, get Oscar nominees and their season features from the year-indexed lookup
3. For each nominee's season features matching the Oscar category, increment `total`; if that nominee won the Oscar, also increment `hits`
4. Compute posterior mean for every `(event, category, status)` triple

### Scoring

For the current year's nominees:
1. Look up each nominee's IMDB codes in `current_code_features`
2. Sum the calibrated weights for all matching in-category season features
3. Add a small epsilon (1e-4) to all scores
4. Apply softmax normalization per category → probabilities

### Validation (Leave-One-Year-Out)

For each year in `VALIDATION_YEARS`:
1. **Re-calibrate weights excluding that year** (LOO: train on all years except the held-out year)
2. Score that year's Oscar nominees using the LOO weights
3. Record: correct/incorrect, confidence, winner probability, rank of actual winner, AUC

Report:
- Per-year accuracy
- Per-category accuracy and AUC
- Overall accuracy and AUC across all category-years

Save:
- `results/stats/predictions.json` — current year predictions with rationale (raw score, season wins, season nominations with weights)
- `results/stats/validation.json` — LOO-CV results by year and by category
- `data/processed/calibrated_weights.json` — all estimated weights with hit counts for transparency

## Step 5: Method 2 — ML Model (Per-Category Model Selection)

Write `src/methods/ml_model/run_ml.py`. This trains actual classifiers from the historical data.

### Feature Construction

For each nominee (identified by IMDB codes) in a given year and Oscar category:

**Base features** (binary, sparse):
- One indicator per unique `"{eventName}::{cat_key}::W|N"` feature string seen in training data
- Feature = 1 if nominee has that season award, 0 otherwise

**Engineered features** (dense):
- `n_wins_in_cat` — count of in-category wins across all events
- `n_noms_in_cat` — count of in-category nominations
- `n_wins_cross` — count of wins in OTHER categories (cross-category signal)
- `n_noms_cross` — count of nominations in other categories
- **Sweep indicators** (binary): SAG+BAFTA, SAG+GG, BAFTA+GG, SAG+BAFTA+GG, BAFTA+GG+CC — set to 1 when a nominee won all awards in that combination for the same Oscar category

### Model Candidates

Build a grid of candidate models:

**Elastic-net logistic regression** (SGDClassifier with log_loss):
- `alpha` in `[0.001, 0.005, 0.01, 0.05, 0.1]`
- `l1_ratio` in `[0.1, 0.5, 0.9]`
- `class_weight="balanced"`, `max_iter=500`, `random_state=42`

**Gradient Boosted Trees** (GradientBoostingClassifier):
- `(n_estimators=50, max_depth=2, learning_rate=0.1)`
- `(n_estimators=100, max_depth=1, learning_rate=0.05)`
- `subsample=0.8`, `min_samples_leaf=3`, `random_state=42`
- Use `compute_sample_weight("balanced", y)` for class imbalance

Total: 15 elastic-net + 2 GBT = 17 candidate models per category.

### Per-Category Model Selection via LOO-CV

For **each Oscar category independently**:

1. Build the feature vocabulary from all training data for that category
2. Construct the full feature matrix `(n_samples, n_base_features + n_engineered)`
3. For each of the 17 candidate models, run leave-one-year-out cross-validation:
   - For each held-out year: train on all other years, predict held-out year
   - Strict temporal split: train only on years < held-out year is NOT required here because we use LOO-year (the point is each year is held out entirely)
   - Compute pooled AUC across all held-out years
4. Select the model with the highest pooled AUC for that category
5. Train the selected model on ALL data for final predictions

**Why per-category?** Different Oscar categories have different predictive structures. Best Director is well-predicted by a single DGA feature; Best Picture requires many signals. Per-category selection adapts the model complexity to each category's structure.

### Prediction

For the current year's nominees:
1. Build feature vectors using features from all season years (union across season years using element-wise max)
2. Recompute aggregate counts across both season years
3. Run `predict_proba` on the final trained model
4. Normalize probabilities per category

### Validation Output

Record per-category:
- LOO-AUC, accuracy, best model selected
- Top features (coefficients for elastic-net, importances for GBT)
- Number of features, number of nonzero features

Save:
- `results/ml_model/predictions.json` — predictions with rationale (active features, coefficients, model selected, category AUC)
- `results/ml_model/validation.json` — full LOO-CV results by year and category, including cat_meta (AUC, best model, top features per category)

## Step 6: Method 3 — LLM Analysis (Independent Web Research)

This is the agent's primary original contribution. Conduct independent web research — do NOT echo the stats/ML outputs.

1. **Search expert predictions** from: Variety, IndieWire, Hollywood Reporter, Deadline, Awards Daily, Gold Derby, The Wrap
2. **Read anonymous voter interviews** in trade publications
3. **Analyze narratives**: comeback stories, overdue wins, campaign momentum, controversy, "it's their time" sentiment
4. **Consider Best Picture ranked-choice voting** (preferential ballot) — broad appeal matters more than passionate minority support
5. **Note late momentum shifts** — SAG/BAFTA results often reshape the race
6. **Assess uncertainty honestly** — short film and craft categories are harder to predict; use flatter distributions

For each nominee produce:
- `probability` — well-calibrated (don't be overconfident)
- `reasoning` — 2-3 sentences on why
- `key_signals` — list of most important evidence
- `uncertainty` — what could cause an upset

Save to `results/llm_analysis/predictions.json`:
```json
{
  "method": "llm_analysis",
  "year": YEAR,
  "note": "Independent LLM web research analysis. Not contaminated by stats/ML outputs.",
  "categories": {
    "Category Name": [
      {
        "name": "Nominee Name",
        "probability": 0.55,
        "rationale": {
          "reasoning": "Swept DGA, PGA, BAFTA...",
          "key_signals": ["Won DGA", "Won PGA", "Won BAFTA Best Film"],
          "uncertainty": "Rival film won SAG Ensemble, creating genuine 50/50 split."
        }
      }
    ]
  }
}
```

## Step 7: Method 4 — Crowd Consensus (Gold Derby)

Search Gold Derby (goldderby.com) for combined expert/editor/user consensus odds.

1. Scrape or search for numeric odds for each nominee in each category
2. Normalize to probabilities per category
3. If Gold Derby is unavailable, aggregate expert picks from multiple prediction sources

Also gather **historical Gold Derby top-1 picks** for validation years to compute accuracy and (where full odds are available) AUC.

Save:
- `results/goldderby/predictions.json` — predictions with rationale (`raw_odds`, `odds_rank`, `n_candidates`)
- `results/goldderby/validation.json` — historical accuracy and AUC

## Step 8: Ensemble

Combine all 4 methods using weighted average:

```python
ensemble_prob(nominee) = w_stats * P_stats + w_ml * P_ml + w_llm * P_llm + w_gd * P_gd
```

Default weights: `w_stats = w_ml = w_llm = w_gd = 0.25`. Normalize per category so probabilities sum to 1.0.

Use fuzzy name matching when looking up nominees across methods (case-insensitive substring match), since different methods may use slightly different name spellings.

## Step 9: Present Results

Print a final table:

```
============================================================
[N]th Academy Awards ([Date]) — Ensemble Predictions
============================================================
Weights: Stats=25%, ML=25%, LLM=25%, GD=25%

Category               Predicted Winner              Prob   2nd Choice                 Prob
------------------------------------------------------------------------------------------
Best Picture           [Film]                         63%   [Film]                      25%
Director               [Name]                         85%   [Name]                       8%
...
```

For **contested categories** (top nominee < 70%), show per-method breakdown:

```
[Category]
  Name                          Ens   Stat    ML   LLM    GD
  -----------------------------------------------------------
  [Nominee 1]                   47%    33%   42%   50%   63%
  [Nominee 2]                   27%    33%   42%    3%   27%
```

Also print a **validation summary** comparing all methods:

```
Method          Accuracy    AUC     N cat-yrs
---------------------------------------------
Stats            72%       0.854    120
ML               73%       0.816    120
Gold Derby       90%       0.957     96
LLM              N/A       N/A      N/A
```

## Step 10: Save All Outputs

Ensure all files are saved:
- `data/processed/df_data_full.csv` — historical award data
- `data/processed/oscar{YEAR}_nominees.json` — current nominees
- `data/processed/year_features.pkl` — year-indexed feature lookup
- `data/processed/hist_by_cat.pkl` — Oscar training rows by category
- `data/processed/calibrated_weights.json` — estimated Bayesian weights
- `results/stats/predictions.json` and `validation.json`
- `results/ml_model/predictions.json` and `validation.json`
- `results/llm_analysis/predictions.json`
- `results/goldderby/predictions.json` and `validation.json`

---

## Key Design Principles

1. **No data leakage** — Oscar event is excluded from features; it is only used as labels. Season award features for year N come from the year-indexed lookup for that year's season.
2. **Weights estimated from data** — The stats method uses Bayesian estimation with a Beta(1,1) prior, NOT hardcoded weights. Every weight has a transparent `(hits+1)/(total+2)` derivation.
3. **Models trained on data** — The ML method trains actual classifiers (elastic-net, GBT) on the historical feature matrix. No approximations or lookup tables.
4. **Per-category model selection** — Different categories have different predictive structures; the pipeline selects the best model for each category via LOO-CV.
5. **Full validation** — Both stats and ML methods report leave-one-year-out accuracy and AUC, broken down by year and by category.
6. **Method independence** — LLM analysis must not copy stats/ML; each method contributes unique signal.
7. **Calibration** — 60% means ~60% win chance; don't be overconfident. Short film / craft categories are noisier; flatten those distributions.
8. **Ranked-choice awareness** — Best Picture uses preferential ballot; broad appeal matters.
9. **Recency** — Late-season results (SAG, BAFTA) outweigh early-season (Gotham, NBR).
10. **Rationale transparency** — Every prediction includes evidence so users can judge quality.
