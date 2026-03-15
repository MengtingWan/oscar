# Oscar Prediction Reference Data

## Official Oscar Category Names → Keys

| Official Name | Key |
|--------------|-----|
| Best Motion Picture of the Year | `picture` |
| Best Performance by an Actor in a Leading Role | `actor` |
| Best Performance by an Actress in a Leading Role | `actress` |
| Best Performance by an Actor in a Supporting Role | `supp_actor` |
| Best Performance by an Actress in a Supporting Role | `supp_actress` |
| Best Achievement in Directing | `director` |
| Best Original Screenplay | `original_screenplay` |
| Best Adapted Screenplay | `adapted_screenplay` |
| Best Achievement in Cinematography | `cinematography` |
| Best Achievement in Film Editing | `editing` |
| Best Achievement in Production Design | `production_design` |
| Best Achievement in Costume Design | `costume` |
| Best Sound | `sound` |
| Best Achievement in Makeup and Hairstyling | `makeup` |
| Best Achievement in Music Written for Motion Pictures (Original Score) | `score` |
| Best Achievement in Music Written for Motion Pictures (Original Song) | `song` |
| Best Achievement in Visual Effects | `visual_effects` |
| Best Documentary Feature Film | `documentary` |
| Best Animated Feature Film | `animated` |
| Best Animated Short Film | `animated_short` |
| Best Live Action Short Film | `live_action_short` |
| Best Documentary Short Film | `documentary_short` |
| Best International Feature Film | `international` |
| Best Casting | `cast` |

## IMDB Award Event IDs for Scraping

| Event | IMDB ID | What it predicts |
|-------|---------|------------------|
| Academy Awards | ev0000003 | TARGET LABELS ONLY — never use as features |
| SAG Awards | ev0000598 | Acting categories, Ensemble → Picture |
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
| MUAHS Awards | ev0000442 | Makeup and Hairstyling |

## Award Abbreviation Map (for `short_event()` display)

```python
{
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
}
```

## Category Normalization Patterns (`CAT_NORM`)

These regex patterns map freeform award category names to canonical keys. Order matters — check specific patterns before generic ones.

```python
CAT_NORM = {
    "picture":             ["best film","best motion picture","outstanding motion picture",
                            "feature film","theatrical motion picture","best picture",
                            "outstanding producer of theatrical motion pictures"],
    "director":            ["best director","directorial achievement",
                            "outstanding directorial achievement in theatrical"],
    "actor":               ["leading actor","actor in a leading","male actor in a leading",
                            "best actor","best male lead","performance by a male actor in a leading"],
    "actress":             ["leading actress","actress in a leading","female actor in a leading",
                            "best actress","best female lead","performance by a female actor in a leading"],
    "supp_actor":          ["supporting actor","male actor in a supporting",
                            "actor in a supporting","best male supporting"],
    "supp_actress":        ["supporting actress","female actor in a supporting",
                            "actress in a supporting","best female supporting"],
    "original_screenplay": ["original screenplay","writing original","screenplay - original",
                            "screenplay (original)","best screenplay.*original"],
    "adapted_screenplay":  ["adapted screenplay","writing adapted","screenplay - adapted",
                            "screenplay (adapted)","best screenplay.*adapted"],
    "screenplay":          ["best screenplay$","best screenplay -"],
    "cinematography":      ["cinematography"],
    "editing":             ["editing","edited feature film","best edited.*feature"],
    "production_design":   ["production design","art direction","period feature film",
                            "contemporary feature film","fantasy feature film"],
    "costume":             ["costume design","excellence in period film",
                            "excellence in contemporary","excellence in fantasy"],
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
                            "best international"],
    "cast":                ["cast in a motion picture","ensemble cast","best casting"],
}
```

**Important pre-checks in `norm_cat()`** (before scanning `CAT_NORM`):
- "directorial achievement" or `\bdirecting\b` → `director`
- "cinematography" → `cinematography`
- "sound mixing" or "sound editing" → `sound`
- `\bbest edited\b` or `\bedited.*feature film\b` → `editing`
- "documentary" (without "short") → `documentary`
- "international" + "feature" (without "short") → `international`
- `animated\s+(?:feature|theatrical|motion)` → `animated`
- `(?:period|contemporary|fantasy)\s+feature\s+film` → `production_design`
- `cast in.*motion picture` → `cast`
- "female actor" or ("actress" + "motion picture") → `actress` or `supp_actress`
- "male actor" or ("actor" + "motion picture") → `actor` or `supp_actor`

## Sweep Combinations for ML Features

```python
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

SWEEP_COMBOS = [
    ("SAG", "BAFTA"),
    ("SAG", "GG"),
    ("BAFTA", "GG"),
    ("SAG", "BAFTA", "GG"),
    ("BAFTA", "GG", "CC"),
]
```

## ML Engineered Feature Names

```python
ENG_NAMES = [
    "__n_wins_in_cat__",
    "__n_noms_in_cat__",
    "__n_wins_cross__",
    "__n_noms_cross__",
    "__sweep_SAG_BAFTA__",
    "__sweep_SAG_GG__",
    "__sweep_BAFTA_GG__",
    "__sweep_SAG_BAFTA_GG__",
    "__sweep_BAFTA_GG_CC__",
]
```

## Historical Data CSV Schema

`data/processed/df_data_full.csv` columns:
- `eventId` — IMDB event identifier (e.g., `ev0000003`)
- `eventName` — full event name (e.g., "Screen Actors Guild Awards")
- `year` — ceremony year
- `categoryName` — freeform category name from IMDB
- `primaryNomineeCode` — stringified list of IMDB codes (e.g., `"['nm0000123']"`)
- `secondaryNomineeCode` — stringified list of associated codes
- `primaryNomineeName` — display name
- `isWinner` — boolean

## Softmax Formula

```python
def softmax(scores):
    s = [float(x) for x in scores]
    m = max(s) if s else 0.0
    e = [math.exp(x - m) for x in s]
    t = sum(e)
    return [x / t for x in e] if t > 0 else [1.0 / len(s)] * len(s)
```

## Bayesian Weight Formula

```
P(Oscar win | season award signal) = (hits + 1) / (total + 2)
```

Where:
- `hits` = times the season award signal holder also won the Oscar
- `total` = times the season award signal holder was an Oscar nominee
- Prior: Beta(1,1) — uniform, no initial preference
- This naturally shrinks toward 0.5 for awards with limited data
