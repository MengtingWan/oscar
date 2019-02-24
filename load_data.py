import gzip
import json
import numpy as np
import re
from collections import defaultdict
def load_data():
    data_raw = []
    with gzip.open("./data/awards.json.gz") as fin:
        for l in fin:
            data_raw.append(json.loads(l))
    return data_raw

def correct(text):
    name_map = {"Best Art Direction-Set Decoration": "Best Achievement in Production Design", 
                "Best Achievement in Art Direction": "Best Achievement in Production Design",
                "Best Cinematography": "Best Achievement in Cinematography",
                "Best Costume Design": "Best Achievement in Costume Design",
                "Best Director": "Best Achievement in Directing",
                "Best Film Editing": "Best Achievement in Film Editing",
                "Best Makeup": "Best Achievement in Makeup and Hairstyling", 
                "Best Achievement in Makeup": "Best Achievement in Makeup and Hairstyling",
                "Best Music, Original Score": "Best Achievement in Music Written for Motion Pictures, Original Score",
                "Best Music, Original Song": "Best Achievement in Music Written for Motion Pictures, Original Song",
                "Best Effects, Sound Effects Editing": "Best Achievement in Sound Editing", 
                "Best Sound Editing": "Best Achievement in Sound Editing",
                "Best Sound": "Best Achievement in Sound Mixing", 
                "Best Sound Mixing": "Best Achievement in Sound Mixing",
                "Best Visual Effects": "Best Achievement in Visual Effects", 
                "Best Effects, Visual Effects": "Best Achievement in Visual Effects",
                "Best Actor in a Leading Role": "Best Performance by an Actor in a Leading Role",
                "Best Actor in a Supporting Role": "Best Performance by an Actor in a Supporting Role",
                "Best Actress in a Leading Role": "Best Performance by an Actress in a Leading Role",
                "Best Actress in a Supporting Role": "Best Performance by an Actress in a Supporting Role",
                "Best Animated Feature": "Best Animated Feature Film of the Year",
                "Best Animated Short Film": "Best Short Film, Animated",
                "Best Documentary, Features": "Best Documentary, Feature",
                "Best Documentary Feature": "Best Documentary, Feature",
                "Best Documentary, Short Subjects": "Best Documentary, Short Subject", 
                "Best Documentary Short Subject": "Best Documentary, Short Subject",
                "Best Foreign Language Film": "Best Foreign Language Film of the Year",
                "Best Picture": "Best Motion Picture of the Year",
                "Best Short Film, Live Action": "Best Live Action Short Film",
                "Best Writing, Screenplay Based on Material Previously Produced or Published": "Best Writing, Adapted Screenplay",
                "Best Writing, Screenplay Written Directly for the Screen": "Best Writing, Original Screenplay",
                "Best Original Screenplay": "Best Writing, Original Screenplay",
                "Best Adapted Screenplay": "Best Writing, Adapted Screenplay",
                "Best Achievement in Music Written for Motion Pictures (Original Song)": "Best Achievement in Music Written for Motion Pictures, Original Song",
                "Best Achievement in Music Written for Motion Pictures (Original Score)": "Best Achievement in Music Written for Motion Pictures, Original Score",
                "Best Animated Feature Film": "Best Animated Feature Film of the Year"
                }
    if text is None:
        return ''
    else:
        rtn = text
        if text in name_map:
            rtn = name_map[text]
        rtn = re.sub(r'[^\w\s]', '', rtn.lower())
        words = rtn.split(' ')
        remove_set = set(["performance", "in", "a", "an", "by", "of", "the", "year", "achievement"])
        rtn = ' '.join(w.strip() for w in words if w not in remove_set)
        return rtn

def preprocess(data_raw):
    data = []
    featureNames = set()
    const_map = {}
    dataFeature = defaultdict(list)
    data_oscar = []
    for d_raw in data_raw:
        d = {}
        tmp = d_raw["title"].split("_")
        if tmp[0] == "AMPAS":
            d["title"] = tmp[0]
            d["year"] = tmp[1]           
            cate_list = []
            for di in d_raw["data"]:
                catei = {}
                catei["categoryName"] = correct(di['categoryName'])
                nomination_list = []
                for dii in di["nominations"]:
                    c1 = None
                    c2 = None
                    diin1 = dii['primaryNominees']
                    if len(diin1)>0:
                        c1 = diin1[0]['const']
                        const_map[dii['primaryNominees'][0]['const']] = dii['primaryNominees'][0]['name']
                    diin2 = dii['secondaryNominees']
                    if len(diin2)>0:
                        c2 = diin2[0]['const']
                        const_map[dii['secondaryNominees'][0]['const']] = dii['secondaryNominees'][0]['name']
                    nomination_list.append([c1, c2, dii['isWinner']])
                catei["nominations"] = nomination_list
                cate_list.append(catei)
            d["categories"] = cate_list
            data_oscar.append(d)
        else:
            if tmp[0] != "ISA":
                d["title"] = tmp[0]
                d["year"] = tmp[1]
                cate_list = []
                for di in d_raw["data"]:
                    catei = {}
                    catei["categoryName"] = correct(di['categoryName'])
                    catei["featureName"] = d["title"]+"_"+catei["categoryName"]
                    fnameTrue = catei["featureName"]+"_True"
                    fnameFalse = catei["featureName"]+"_False"
                    featureNames.add(fnameTrue)
                    featureNames.add(fnameFalse)
                    nomination_list = []
                    for dii in di["nominations"]:
                        c1 = None
                        c2 = None
                        diin1 = dii['primaryNominees']
                        if len(diin1)>0:
                            c1 = diin1[0]['const']
                            if dii['isWinner']:
                                dataFeature[(c1,tmp[1])].append(fnameTrue)
                            else:
                                dataFeature[(c1,tmp[1])].append(fnameFalse)
                            const_map[dii['primaryNominees'][0]['const']] = dii['primaryNominees'][0]['name']
                        diin2 = dii['secondaryNominees']
                        if len(diin2)>0:
                            c2 = diin2[0]['const']
                            if dii['isWinner']:
                                dataFeature[(c2,tmp[1])].append(fnameTrue)
                            else:
                                dataFeature[(c2,tmp[1])].append(fnameFalse)
                            const_map[dii['secondaryNominees'][0]['const']] = dii['secondaryNominees'][0]['name']
                        nomination_list.append([c1, c2, dii['isWinner']])
                    catei["nominations"] = nomination_list
                    cate_list.append(catei)
                d["categories"] = cate_list
                data.append(d)
    featureNames = list(featureNames)
    dataFeature = dict(dataFeature)
    
    return data, data_oscar, featureNames, dataFeature, const_map

def prepare_raw_features(data, data_oscar, dataFeature):
    X_raw1 = []
    X_raw2 = []
    y = []
    c_raw = []
    n_raw = []
    yr = []
    for d in data_oscar:
        _yr = d["year"]
        for c in d["categories"]:
            cName = c["categoryName"]
            for n in c["nominations"]:
                c1, c2, _y = n
                f1 = []
                f2 = []
                if (c1, _yr) in dataFeature:
                    f1 = dataFeature[(c1, _yr)]
                if (c2, _yr) in dataFeature:
                    f2 = dataFeature[(c2, _yr)]
                y.append(_y)
                yr.append(_yr)
                c_raw.append(cName)
                X_raw1.append(f1)
                X_raw2.append(f2)
                n_raw.append([c1, c2])
    return np.array(X_raw1), np.array(X_raw2), np.array(y), np.array(c_raw), np.array(n_raw), np.array(yr)

def extract_features_per_category(X_raw1, X_raw2, y, c_raw, n_raw, yr, cName):
    X_c_raw1 = X_raw1[c_raw == cName]
    X_c_raw2 = X_raw2[c_raw == cName]
    y_c = y[c_raw == cName].astype(float)
    yr_c = yr[c_raw == cName].astype(float)
    n_c = n_raw[c_raw == cName]
    
    fnames1 = list(set([f for x in X_c_raw1 for f in x]))
    fmap1 = dict(zip(np.array(fnames1), np.arange(len(fnames1))))
    X_mat1 = np.zeros((len(y_c), len(fnames1)))
    for i in range(len(y_c)):
        for f in X_c_raw1[i]:
            X_mat1[i, fmap1[f]] = 1

    fnames2 = list(set([f for x in X_c_raw2 for f in x]))
    fmap2 = dict(zip(np.array(fnames2), np.arange(len(fnames2))))
    X_mat2 = np.zeros((len(y_c), len(fnames2)))
    for i in range(len(y_c)):
        for f in X_c_raw2[i]:
            X_mat2[i, fmap2[f]] = 1
    X_mat = np.append(X_mat1, X_mat2, axis=1)
    featureNames = [f+"_primary" for f in fnames1]
    featureNames.extend([f+"_secondary" for f in fnames2])
    featureNames = np.array(featureNames)
    return y_c, X_mat, yr_c, featureNames, n_c
    
    
    
    