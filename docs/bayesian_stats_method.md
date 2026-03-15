# Bayesian Stats Method: Theory and Design

This document explains the statistical theory behind the Stats prediction method (`src/methods/stats/run_stats.py`).

## The Prediction Problem

Given a set of season award signals for an Oscar nominee, estimate the probability they win the Oscar. The stats method decomposes this into per-signal weights estimated from historical data, then sums them.

## Why Not Hardcoded Weights?

A naive approach would assign arbitrary weights to each award (e.g., "DGA Win = 0.8, BAFTA Nom = 0.2"). This is fragile:
- How do you choose the numbers?
- How do you handle new awards or categories with different predictive value?
- How do you know your intuition is calibrated?

Instead, we **estimate weights from 20 years of historical data** using Bayesian inference, so every weight has a transparent, data-driven derivation.

## The Bayesian Model

### Setup

For a specific season award signal (e.g., "DGA Win in Director category"), we observe binary outcomes across years:
- **n** = number of times an Oscar nominee in that category held this signal
- **k** = number of those times the nominee actually won the Oscar

We want to estimate **θ = P(Oscar win | this signal)**, the true underlying probability.

### Why Not Use k/n?

The maximum likelihood estimate θ̂ = k/n fails for sparse data:

| Signal | k | n | k/n |
|--------|---|---|-----|
| DGA → Director | 17 | 20 | 0.850 |
| Obscure critics → Actor | 1 | 1 | **1.000** |
| Rare guild → Costume | 0 | 2 | **0.000** |
| Never observed | 0 | 0 | **undefined** |

The obscure signal at k/n = 1.0 is obviously overfit. The zero-count case is undefined. We need **regularization toward uncertainty**.

### Bayesian Inference

**Likelihood**: Each observation (did the signal holder win the Oscar?) is a Bernoulli trial. Over n observations with k successes:

```
P(k | θ, n) = C(n,k) · θ^k · (1-θ)^(n-k)
```

This is the Binomial likelihood.

**Prior**: We place a Beta distribution prior on θ:

```
θ ~ Beta(α₀, β₀)
```

We choose **α₀ = β₀ = 1**, giving the **Beta(1,1) distribution** — the uniform distribution on [0,1]. This encodes **maximum ignorance**: before seeing any data, any value of θ between 0 and 1 is equally plausible. This is the least informative proper prior for a probability parameter.

**Posterior**: The Beta distribution is the **conjugate prior** for the Binomial likelihood, meaning the posterior is also a Beta distribution:

```
θ | k, n ~ Beta(α₀ + k, β₀ + n - k) = Beta(1 + k, 1 + n - k)
```

This is derived from Bayes' theorem:

```
P(θ | k, n) ∝ P(k | θ, n) · P(θ)
             ∝ θ^k · (1-θ)^(n-k) · θ^(α₀-1) · (1-θ)^(β₀-1)
             = θ^(α₀+k-1) · (1-θ)^(β₀+n-k-1)
```

which is the kernel of a Beta(α₀ + k, β₀ + n - k) distribution.

**Posterior mean** (our point estimate for θ):

```
E[θ | k, n] = (α₀ + k) / (α₀ + β₀ + n) = (k + 1) / (n + 2)
```

This is the formula used in the code: `P = (hits + 1) / (total + 2)`.

## Properties of the Estimator

### 1. Automatic Shrinkage Toward 0.5

The posterior mean interpolates between the prior mean (0.5) and the data (k/n), with the mixing determined by sample size:

```
E[θ | k, n] = n/(n+2) · (k/n) + 2/(n+2) · (1/2)
```

- When n is large: the weight on data (k/n) dominates → converges to MLE
- When n is small: the weight on prior (0.5) dominates → shrinks toward uncertainty
- When n = 0: returns 0.5 (pure prior)

| Signal | k | n | k/n (MLE) | (k+1)/(n+2) (Bayesian) | Shrinkage |
|--------|---|---|-----------|------------------------|-----------|
| DGA → Director | 17 | 20 | 0.850 | 0.818 | Minimal |
| SAG → Actor | 18 | 21 | 0.857 | 0.826 | Minimal |
| PGA → Picture | 13 | 20 | 0.650 | 0.636 | Minimal |
| Small critics → Actor | 3 | 5 | 0.600 | 0.571 | Moderate |
| Obscure award | 1 | 1 | 1.000 | 0.667 | Heavy |
| Rare guild (no wins) | 0 | 2 | 0.000 | 0.250 | Heavy |
| Never observed | 0 | 0 | undefined | 0.500 | Full prior |

This is **exactly the behavior we want**: trust data-rich signals like DGA and SAG; be skeptical of data-poor signals; default to maximum uncertainty for unseen combinations.

### 2. Laplace's Rule of Succession

The formula (k+1)/(n+2) is historically known as **Laplace's rule of succession** (1814). Laplace's original question: "If the sun has risen every day for n days, what is the probability it rises tomorrow?" His answer: (n+1)/(n+2).

Our application is directly analogous: "If k out of n past holders of this season award won the Oscar, what is the probability the next holder wins?" The answer: (k+1)/(n+2).

### 3. Optimality Under Squared-Error Loss

The posterior mean minimizes the Bayes risk under squared-error loss:

```
E[(θ̂ - θ)²] is minimized when θ̂ = E[θ | data]
```

Since we're summing weights to produce a total score, and we want the sum to approximate the nominee's true "total evidence," using the optimal point estimate for each component is the right choice.

### 4. Posterior Variance (Uncertainty Quantification)

The posterior Beta distribution also gives us uncertainty. The posterior variance is:

```
Var(θ | k, n) = (k+1)(n-k+1) / ((n+2)²(n+3))
```

For data-rich signals (large n), variance is small — we're confident in the weight. For sparse signals, variance is large. We don't currently use the variance in scoring, but it's available for future extensions (e.g., uncertainty-weighted ensembles).

## From Weights to Predictions

### Scoring: Naive Bayes Decomposition

A nominee's score is the sum of Bayesian weights for all their season awards:

```
score(nominee) = Σ_{signals} θ̂(event, category, status)
```

This is a **naive Bayes-like approximation** — it treats signals as contributing independently. In reality, signals are correlated: a DGA winner is more likely to also be a BAFTA winner. The summation overweights nominees who swept multiple correlated awards.

This approximation is acceptable for three reasons:
1. **Bayesian shrinkage prevents dominance**: Even if a nominee has 5 correlated wins, each weight is modest (0.5–0.85), so the total doesn't explode
2. **Softmax normalization**: Raw scores are converted to probabilities relative to competitors, so absolute scale doesn't matter — only relative ordering
3. **Empirical validation**: LOO-CV shows ~72% accuracy and 0.85 AUC, competitive with more sophisticated methods

The ML model (Method 2) handles interactions properly via sweep indicators and multivariate learned coefficients. Having both methods in the ensemble captures both independent signal strength (Stats) and interaction effects (ML).

### Softmax Normalization

Raw scores are converted to probabilities via softmax:

```
P(nominee_i wins) = exp(score_i) / Σ_j exp(score_j)
```

With numerical stability (subtract max before exponentiating):

```
P(nominee_i) = exp(score_i - max(scores)) / Σ_j exp(score_j - max(scores))
```

Softmax amplifies score differences — a nominee with even a small lead in total weight gets a disproportionate probability share. This matches the winner-take-all nature of Oscar voting.

## Separate Win and Nomination Weights

Wins and nominations are estimated as **separate signals**, not related by an arbitrary discount factor. The data determines the relationship:

- A DGA **Win** in Director has weight (k+1)/(n+2) where k and n come from DGA winners only
- A DGA **Nomination** in Director has its own weight from its own k and n

Typically, win weights are 0.5–0.85 while nomination weights are 0.15–0.35 — but this ratio **emerges from the data**, not from a hand-tuned parameter.

## Validation: Leave-One-Year-Out

To validate without data leakage:

1. For each held-out year Y in the validation window:
   - **Re-estimate all weights** using data from all years except Y
   - Score year Y's Oscar nominees using the LOO weights
   - Record whether the top-scored nominee was the actual winner
   - Compute AUC for the probability ranking

2. Report per-year accuracy, per-category accuracy and AUC, and overall metrics

This is critical: the weights used to predict year Y have **never seen year Y's outcomes**. The LOO re-calibration ensures honest evaluation, even though it's computationally more expensive (requires re-running the calibration for each held-out year).

## Why Beta(1,1) and Not Another Prior?

**Beta(1,1) = Uniform(0,1)**: Maximum ignorance. No prior belief about whether an award is predictive or not.

Alternatives considered:
- **Beta(0.5, 0.5)** (Jeffreys prior): Gives more weight to extreme values (0 or 1). Less appropriate here because we expect most signals to have moderate predictive power.
- **Beta(2, 2)**: Slightly informative, peaks at 0.5. Would shrink estimates more aggressively. Reasonable but less transparent.
- **Empirical Bayes**: Estimate the prior from data. More powerful but adds complexity and risks overfitting the prior.

Beta(1,1) is chosen for **transparency and simplicity**: every weight has a clear (k+1)/(n+2) formula that anyone can verify, and the uniform prior makes the fewest assumptions.
