import numpy as np
from sklearn import linear_model
import load_data
import re

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def sigmoid(x):
    p_s = 1.0/(1.0 + np.exp(-x))
    return p_s

def run_each_category(X_raw1, X_raw2, y, c_raw, n_raw, yr, cName, last_yr, eps=1e-10):
    y_c, X_mat, yr_c, featureNames, n_c = load_data.extract_features_per_category(X_raw1, X_raw2, y, c_raw, n_raw, yr, cName)
    x_test = X_mat[yr_c==2019,:]
    n_test = n_c[yr_c==2019,:]
    
    # start leave-one-out training
    lbda_set = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
    l1_set = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    para_list = [(lb,l1) for lb in lbda_set for l1 in l1_set]
    validation_metric = []
    for (lbda, l1_ratio) in para_list:
        _validation_metric = []
        for _yr in range(2010, last_yr):
            train_id = np.where((yr_c!=last_yr)&(yr_c!=_yr))[0]
            validation_id = np.where(yr_c==_yr)[0]
            x_train = X_mat[train_id,:]
            y_train = y_c[train_id]
            x_validation = X_mat[validation_id,:]
            y_validation = y_c[validation_id]
            model = linear_model.SGDClassifier(loss='log', penalty='elasticnet', alpha=lbda,
                                               l1_ratio=l1_ratio, max_iter=100, tol=1e-5)
            model.fit(x_train, y_train)

            y_score_validation = softmax(model.decision_function(x_validation))
            ce_validation = np.sum(y_validation*np.log(y_score_validation+eps))
            _validation_metric.append(ce_validation)
        validation_metric.append(_validation_metric)
    validation_metric = np.array(validation_metric)
    vali_mean = validation_metric.mean(axis=1)
    i_best = np.argmax(vali_mean)
    lbda_best, l1_best = para_list[i_best]

    
    train_id = np.where(yr_c!=last_yr)[0]
    x_train = X_mat[train_id,:]
    y_train = y_c[train_id]
    y_score_test = []
    coef_list = []
    for k in range(50):
        model = linear_model.SGDClassifier(loss='log', penalty='elasticnet', alpha=lbda_best,
                                           l1_ratio=l1_best, max_iter=100, tol=1e-5)
        model.fit(x_train, y_train)
        y_score_test_ = model.decision_function(x_test)
        y_score_test.append(y_score_test_)
        coef_list.append(model.coef_[0,:])
    
    # softmax normalization
    y_prob_test = softmax(np.array(y_score_test).mean(axis=0))
    # get sigmoid first, then normalize the probability
#    y_prob_test = sigmoid(np.array(y_score_test).mean(axis=0))
#    y_prob_test /= np.sum(y_prob_test)
    coef = np.array(coef_list).mean(axis=0)
    signals = [(featureNames[k], coef[k]) for k in np.argsort(coef)[-10:][::-1] if coef[k]>0]
    return n_test, y_prob_test, signals
    

def run_all():
    data_raw = load_data.load_data()
    data, data_oscar, featureNames, dataFeature, const_map = load_data.preprocess(data_raw)
    X_raw1, X_raw2, y, c_raw, n_raw, yr = load_data.prepare_raw_features(data, data_oscar, dataFeature)
    
    # predicting year 2019
    last_yr = 2019
    
    cName_list = list(set(c_raw))
    results = []
    print("start training ...")
    for cName in cName_list:
        print("current category: ", cName)
        n_test, y_prob_test, signals = run_each_category(X_raw1, X_raw2, y, c_raw, n_raw, yr, cName, last_yr)
        res = {}
        res["category"] = cName
        prediction = []
        for i in range(len(y_prob_test)):
            c1 = ""
            c2 = ""
            if n_test[i][0] in const_map:
                c1 = re.sub(r'[^\w\s]', '', const_map[n_test[i][0]])
            if n_test[i][1] in const_map:
                c2 = re.sub(r'[^\w\s]', '', const_map[n_test[i][1]])
            prediction.append([c1, c2, y_prob_test[i]])
        res["prediction"] = prediction
        res["evidence"] = signals
        results.append(res)
    print("done!")
    
    results_flat = [["category", "primary nomination", "secondary nomination", "chance of winning"]]
    results_flat.extend([[res["category"], pred[0], pred[1], np.round(pred[2], decimals=6)] for res in results for pred in res["prediction"]])
    evidence_flat = [["category", "supporting feature", "coefficient"]]
    evidence_flat.extend([[res["category"], s[0], np.round(s[1], decimals=6)] for res in results for s in res["evidence"]])
    np.savetxt("results_flat.csv", results_flat, fmt="%s", delimiter=", ")
    np.savetxt("evidence_flat.csv", evidence_flat, fmt="%s", delimiter=", ")
    
if __name__== "__main__":
    run_all()
    