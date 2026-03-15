"""
src/build_web_data.py
=====================
Merge prediction + validation JSONs into a single JS data file for the webpage.

Passes through full rationale objects so the webpage can render method-specific
views without losing detail.

Usage:
    python -m src.build_web_data [--output-dir DIR]

Outputs:
    oscar2026_data.js  — sets window.OSCAR_DATA for the HTML pages
"""

import argparse, json, os

from src.config import (
    PRED_YEAR, CEREMONY_NAME, CEREMONY_DATE, OSCAR_KEY,
    NOMINEES_PATH, STATS_PRED_PATH, STATS_VALID_PATH,
    ML_PRED_PATH, ML_VALID_PATH, LLM_PRED_PATH,
    GD_PRED_PATH, GD_VALID_PATH, CALIB_WEIGHTS_PATH,
)
from src.utils import short_event


def load(p):
    with open(p) as f:
        return json.load(f)


def method_probs(data, cat_name):
    """Return {name: entry_dict} for a category from a method's predictions."""
    return {e['name']: e for e in data.get('categories', {}).get(cat_name, [])}


def fuzzy_get(lookup, name):
    """Fuzzy match name against lookup dict keys (case-insensitive substring)."""
    if name in lookup:
        return lookup[name]
    nl = name.lower()
    for k, v in lookup.items():
        if nl in k.lower() or k.lower() in nl:
            return v
    return None


def build_categories(nominees, stats, ml, llm, gd):
    """Build unified category list with full rationale objects per method."""
    categories = []
    for cat_data in nominees:
        cat_name = cat_data['category']
        cat_key = OSCAR_KEY.get(cat_name, '')
        noms = cat_data['nominations']
        n = len(noms)

        sp = method_probs(stats, cat_name)
        mp = method_probs(ml, cat_name)
        lp = method_probs(llm, cat_name)
        gp = method_probs(gd, cat_name)

        nom_list = []
        for nom in noms:
            name = ', '.join(nom.get('primaryNames', []))
            secondary = ', '.join(nom.get('secondaryNames', []))
            img = (nom.get('imgUrls') or [None])[0]

            se = fuzzy_get(sp, name)
            me = fuzzy_get(mp, name)
            le = fuzzy_get(lp, name)
            ge = fuzzy_get(gp, name)

            default_p = round(1.0 / n, 4) if n > 0 else 0

            # Build season win tags from stats rationale
            s_rat = se.get('rationale', {}) if se else {}
            wins = []
            for w in s_rat.get('season_wins', []):
                wins.append(f"{w['award']}::{w['category']}::W")
            for w in s_rat.get('season_nominations', []):
                wins.append(f"{w['award']}::{w['category']}::N")

            # Trim ML features to top 5
            m_rat = me.get('rationale', {}) if me else {}
            ml_features = m_rat.get('active_features', [])[:5]

            nom_list.append({
                'name': name,
                'secondary': secondary,
                'imageUrl': img,
                'primaryCodes': nom.get('primaryCodes', []),
                'secondaryCodes': nom.get('secondaryCodes', []),
                # Per-method probabilities
                'stats_prob': round(se['probability'], 4) if se else default_p,
                'ml_prob': round(me['probability'], 4) if me else default_p,
                'llm_prob': round(le['probability'], 4) if le else default_p,
                'gd_prob': round(ge['probability'], 4) if ge else default_p,
                # Season award tags
                'wins': wins,
                # Full rationale objects per method
                'stats_rationale': {
                    'score_raw': s_rat.get('score_raw', 0),
                    'season_wins': s_rat.get('season_wins', []),
                    'season_nominations': s_rat.get('season_nominations', []),
                },
                'ml_rationale': {
                    'features': ml_features,
                    'category_auc': m_rat.get('category_auc'),
                    'model_selected': m_rat.get('model_selected', ''),
                },
                'llm_rationale': {
                    'reasoning': (le.get('rationale', {}) if le else {}).get('reasoning', ''),
                    'key_signals': (le.get('rationale', {}) if le else {}).get('key_signals', []),
                    'uncertainty': (le.get('rationale', {}) if le else {}).get('uncertainty', ''),
                },
                'gd_rationale': {
                    'raw_odds': (ge.get('rationale', {}) if ge else {}).get('raw_odds'),
                    'odds_rank': (ge.get('rationale', {}) if ge else {}).get('odds_rank'),
                    'n_candidates': (ge.get('rationale', {}) if ge else {}).get('n_candidates'),
                },
            })

        categories.append({
            'category': cat_name,
            'category_key': cat_key,
            'nominees': nom_list,
        })

    return categories


def clean_pred(s):
    """Clean prediction string like \"['Parasite']\" -> Parasite."""
    if not s:
        return '?'
    return str(s).replace("['", '').replace("']", '').replace("', '", ' / ')


def build_validation(sv, mv, gv):
    """Build merged validation data with full detail."""
    cat_order = [
        'picture', 'director', 'actor', 'actress', 'supp_actor', 'supp_actress',
        'original_screenplay', 'adapted_screenplay', 'cinematography', 'editing',
        'production_design', 'costume', 'sound', 'makeup', 'score', 'song',
        'visual_effects', 'documentary', 'animated', 'international',
    ]

    by_cat = {}
    for ck in cat_order:
        sc = sv.get('by_category', {}).get(ck, {})
        mc = mv.get('by_category', {}).get(ck, {})
        gc = gv.get('by_category', {}).get(ck, {})
        by_cat[ck] = {
            'stats_accuracy': sc.get('accuracy'),
            'stats_auc': sc.get('auc'),
            'ml_accuracy': mc.get('accuracy'),
            'ml_auc': mc.get('loo_auc'),
            'ml_model': mc.get('best_model'),
            'gd_accuracy': gc.get('accuracy'),
            'gd_auc': gc.get('auc'),
        }

    years = sv.get('validation_years', [])
    by_year = {}
    for yr in years:
        yr_s = str(yr)
        s_yr = sv.get('by_year', {}).get(yr_s, {})
        m_yr = mv.get('by_year', {}).get(yr_s, {})
        g_yr = gv.get('by_year', {}).get(yr_s, {})
        yr_data = {}
        all_cats = set(list(s_yr) + list(m_yr) + list(g_yr))
        for ck in all_cats:
            se, me, ge = s_yr.get(ck, {}), m_yr.get(ck, {}), g_yr.get(ck, {})
            yr_data[ck] = {
                'stats_correct': se.get('correct'),
                'stats_pred': clean_pred(se.get('predicted', '')),
                'stats_auc': se.get('auc'),
                'ml_correct': me.get('correct'),
                'ml_pred': clean_pred(me.get('predicted', '')),
                'gd_correct': ge.get('correct'),
                'gd_pred': clean_pred(ge.get('predicted', '')),
                'gd_auc': ge.get('auc'),
                'actual': clean_pred(
                    se.get('actual') or me.get('actual') or ge.get('actual', '')),
            }
        by_year[yr_s] = yr_data

    return {
        'stats': {
            'overall_accuracy': sv['overall_accuracy'],
            'overall_auc': sv.get('overall_auc'),
            'n_cat_years': sv['n_cat_years'],
        },
        'ml': {
            'overall_accuracy': mv['overall_accuracy'],
            'overall_auc': mv.get('overall_auc'),
            'n_cat_years': mv['n_cat_years'],
        },
        'gd': {
            'overall_accuracy': gv['overall_accuracy'],
            'overall_auc': gv.get('overall_auc'),
            'n_cat_years': gv['n_cat_years'],
            'n_auc_cat_years': gv.get('n_auc_cat_years', 0),
        },
        'years': years,
        'cat_order': cat_order,
        'by_category': by_cat,
        'by_year': by_year,
    }


def load_calibrated_weights():
    """Load top calibrated weights for display."""
    if not os.path.isfile(CALIB_WEIGHTS_PATH):
        return []
    with open(CALIB_WEIGHTS_PATH) as f:
        cw = json.load(f)
    # Return top 30 by weight, wins only
    items = []
    for key, v in cw.items():
        if '::W' in key:
            items.append({
                'feature': key,
                'weight': v['weight'],
                'hits': v['hits'],
                'total': v['total'],
            })
    items.sort(key=lambda x: -x['weight'])
    return items[:30]


def main():
    ap = argparse.ArgumentParser(description='Build web data file for Oscar predictions page')
    default_out = os.path.join(os.path.dirname(__file__), '..', '..', 'mengtingwan.github.io', 'oscars')
    ap.add_argument('--output-dir', default=default_out,
                    help='Output directory for data JS (default: ../mengtingwan.github.io/oscars/)')
    args = ap.parse_args()

    # Load all source data
    nominees = load(NOMINEES_PATH)
    stats = load(STATS_PRED_PATH)
    ml = load(ML_PRED_PATH)
    llm = load(LLM_PRED_PATH)
    gd = load(GD_PRED_PATH)
    sv = load(STATS_VALID_PATH)
    mv = load(ML_VALID_PATH)
    gv = load(GD_VALID_PATH)

    categories = build_categories(nominees, stats, ml, llm, gd)
    validation = build_validation(sv, mv, gv)
    top_weights = load_calibrated_weights()

    data = {
        'ceremony': CEREMONY_NAME,
        'date': CEREMONY_DATE,
        'year': PRED_YEAR,
        'ensemble_weights': {'stats': 0.25, 'ml': 0.25, 'llm': 0.25, 'gd': 0.25},
        'categories': categories,
        'validation': validation,
        'top_weights': top_weights,
    }

    os.makedirs(args.output_dir, exist_ok=True)

    # Write data JS
    data_path = os.path.join(args.output_dir, 'oscar2026_data.js')
    with open(data_path, 'w') as f:
        f.write('// Auto-generated by src/build_web_data.py\n')
        f.write('window.OSCAR_DATA = ')
        json.dump(data, f, ensure_ascii=False)
        f.write(';\n')
    print(f'  {data_path} ({os.path.getsize(data_path) / 1024:.0f} KB)')
    print('Done.')


if __name__ == '__main__':
    main()
