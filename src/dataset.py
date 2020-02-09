import json
import gzip
import sys
import re
import numpy as np
import pandas as pd


def load_data(min_year = 2000):
    oscars = []
    with gzip.open('../data/oscars.json.gz') as fin:
        for l in fin:
            d = json.loads(l)
            if d['year'] >= min_year:
                oscars.append(d)

    meta = []
    with gzip.open('../data/meta.json.gz') as fin:
        for l in fin:
            d = json.loads(l)
            meta.append(d)
            
    titleMap = {}
    with open('../data/titles.json') as fin:
        for l in fin:
            d = json.loads(l)
            titleMap[d['const']] = [d['rating_score_critic'], d['rating_score_user'],
                                    d['n_rating_user'], d['n_review_critic'], 
                                    d['n_review_user']]

    featureNameMap = {}
    featureMap = {}
    const_map = {}
    for d in meta:
        _year, _feature_id = [], []
        const_map[d['const']] = {'name': d['name'], 'imageUrl': d['imageUrl']}
        for di in d['awards']:
            tmp = [di['eventName'], di['awardName'], di['categoryName'], str(di['isWinner'])]
            featureName = ' :: '.join(tmp)
            if featureName not in featureNameMap:
                featureNameMap[featureName] = len(featureNameMap)
            _year.append(di['year'])
            _feature_id.append(featureNameMap[featureName])
            if di['isWinner']:
                tmp = [di['eventName'], di['awardName'], di['categoryName'], 'False']
                featureName = ' :: '.join(tmp)
                if featureName not in featureNameMap:
                    featureNameMap[featureName] = len(featureNameMap)
                _year.append(di['year'])
                _feature_id.append(featureNameMap[featureName])        
        featureMap[d['const']] = {'year': np.array(_year), 'feature_id': np.array(_feature_id)}
    _tmp = dict([[v, k] for k, v in featureNameMap.items()])
    featureNames = [_tmp[k] for k in range(len(_tmp))]

    _years, _cateNames, _str, _labels, _features = [], [], [], [], []
    _rating_features = []
    for d in oscars:
        _year, _cateName = d['year'], d['categoryName']
        for ni in d['nominations']:
            _str.append(json.dumps(ni))
            _y = int(ni['isWinner'])
            _ratings = [None, None, None, None, None]
            _feat_ids = []
            for k in ni['primaryNominees']:
                di = featureMap[k]
                _feat_ids.extend(list(di['feature_id'][di['year'] == _year]))
                if k in titleMap:
                    _ratings = titleMap[k]
            for k in ni['secondaryNominees']:
                di = featureMap[k]
                _feat_ids.extend(list(di['feature_id'][di['year'] == _year]))
                if k in titleMap:
                    _ratings = titleMap[k]
            _feat_ids = list(set(_feat_ids))
                    
            _years.append(_year)
            _cateNames.append(_cateName)
            _labels.append(_y)
            _feat = np.zeros(len(featureNames))
            _feat[_feat_ids] = 1
            _features.append(_feat)
            _rating_features.append(_ratings)

    _years = np.array(_years)
    _cateNames = np.array(_cateNames)
    _str = np.array(_str)
    _labels = np.array(_labels)
    _features = np.array(_features)
    _rating_features = np.array(_rating_features)

    df_data = pd.DataFrame(np.array([_years, _cateNames, _str, _labels]).transpose(), 
                           columns = ['year', 'category', 'string', 'label'])
    df_data['year'] = df_data['year'].astype(int)
    df_data['label'] = df_data['label'].astype(int)
    df_data = pd.concat([df_data, pd.DataFrame(_features, columns = featureNames)], axis=1)
    ratingNames = ['rating_score_critic','rating_score_user',
                    'n_rating_user','n_review_critic','n_review_user']
    df_data = pd.concat([df_data, pd.DataFrame(_rating_features, columns = ratingNames)], axis=1)
    df_data[ratingNames] = df_data.groupby(["category"])[ratingNames].transform(lambda x: x.fillna(x.mean()))
    df_data[['n_rating_user','n_review_critic','n_review_user']] = np.log(
        df_data[['n_rating_user','n_review_critic','n_review_user']].fillna(1))
    df_data[ratingNames] = df_data.groupby(["year","category"])[ratingNames].transform(lambda x: x - x.mean())
    df_data[ratingNames] = df_data[ratingNames].fillna(0)
    df_data[ratingNames] = df_data.groupby(["category"])[ratingNames].transform(lambda x: (x - x.min())/(x.max() - x.min() + 1e-5))
    
    featureNames = np.append(featureNames, ratingNames)
    
    return df_data, featureNames, const_map