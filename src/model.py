import numpy as np
from sklearn import linear_model, metrics
from dataset import load_data
import pandas as pd
import re
import sys
import json


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def sigmoid(x):
    p_s = 1.0/(1.0 + np.exp(-x))
    return p_s


def run_each_category(df_cate, year_pred, featureNames, eps=1e-10):

    x_test = df_cate[df_cate['year'] == year_pred][featureNames].values
    nomination_test = df_cate[df_cate['year'] == year_pred]['string'].apply(lambda x : json.loads(x)).values

    print('leave-one-out training ...')
    sys.stdout.flush()
    lbda_set = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
    l1_set = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    para_list = [(lb, l1) for lb in lbda_set for l1 in l1_set]
    validation_accr = []
    validation_auc = []
    for (lbda, l1_ratio) in para_list:
        _validation_accr = []
        _validation_auc = []
        for _yr in range(2010, year_pred):
            sys.stdout.flush()
            idx_train = (df_cate['year'] != _yr) & (df_cate['year'] != year_pred)
            idx_validation = (df_cate['year'] == _yr)
            x_train = df_cate[idx_train][featureNames].values
            y_train = df_cate[idx_train]['label'].values
            x_validation = df_cate[idx_validation][featureNames].values
            y_validation = df_cate[idx_validation]['label'].values

            model = linear_model.SGDClassifier(loss='log', penalty='elasticnet', alpha=lbda,
                                            l1_ratio=l1_ratio, tol=1e-3)
            model.fit(x_train, y_train)
            
            y_score_validation = model.decision_function(x_validation)
            j = np.argmax(y_score_validation)
            auc_validation = metrics.roc_auc_score(y_validation, y_score_validation)

            _validation_accr.append(y_validation[j])
            _validation_auc.append(auc_validation)

        validation_accr.append(_validation_accr)
        validation_auc.append(_validation_auc)

    validation_accr = np.array(validation_accr)
    validation_auc = np.array(validation_auc)
    vali_mean = validation_auc.mean(axis=1)
    i_best = np.argmax(vali_mean)
    validation_auc_best = validation_auc[i_best,:].mean()
    validation_accr_best = validation_accr[i_best,:].mean()
    lbda_best, l1_best = para_list[i_best]
    
    print('selected hyper-parameters: lbda={0}, l1={1}'.format(lbda_best, l1_best)) 
    print("best avg. AUC="+(str(validation_auc_best.round(2))), '; best avg. Accr='+str(validation_accr_best.round(2)))
    print('generating predictions ...')
    sys.stdout.flush()
    x_train = df_cate[df_cate['year'] != year_pred][featureNames].values
    y_train = df_cate[df_cate['year'] != year_pred]['label'].values
    y_score_test = []
    coef_list = []
    n_sample = 20
    for k in range(n_sample):
        model = linear_model.SGDClassifier(loss='log', penalty='elasticnet', alpha=lbda_best,
                                           l1_ratio=l1_best, tol=1e-4)
        model.fit(x_train, y_train)
        y_score_test_ = model.decision_function(x_test)
        y_score_test.append(y_score_test_)
        coef_list.append(model.coef_[0, :])

    y_test = np.array(y_score_test).mean(axis=0)
    y_test = y_test - y_test.mean()
    y_prob_test = softmax(y_test)
    y_test_se = np.array(y_score_test).std(axis=0)/np.sqrt(n_sample-1)
    coef = np.array(coef_list).mean(axis=0)
    

    signals = [(featureNames[k], coef[k])
               for k in np.argsort(coef)[-10:][::-1] if coef[k] > 0]
    print('done!')
    sys.stdout.flush()
    return nomination_test, y_test, y_test_se, y_prob_test, signals, validation_auc_best, validation_accr_best


def run_all(min_year = 2000, year_pred = 2020):
    df_data, featureNames, const_map = load_data(min_year)
    df_data = df_data[df_data['year'] >= min_year]
    cateNames = df_data['category'].unique()
    
    print('Predicting Year of', year_pred)
    print('===========================')

    results = []
    for _category in cateNames:
        print("[Current Category]: ", _category)
        sys.stdout.flush()

        df_cate = df_data[df_data['category'] == _category]
        nomination_test, y_test, y_test_se, y_prob_test, signals, validation_auc_best, validation_accr_best = run_each_category(df_cate, year_pred, featureNames)

        res = {}
        res["category"] = _category
        prediction = []
        for i in range(len(y_prob_test)):
            c1 = []
            url = []
            code1 = nomination_test[i]['primaryNominees']
            for _code in code1:
                if _code in const_map:
                    c1.append(const_map[_code]['name'])
                    url.append(const_map[_code]['imageUrl'])
            c2 = []
            code2 = nomination_test[i]['secondaryNominees']
            for _code in code2:
                if _code in const_map:
                    c2.append(const_map[_code]['name'])
            prediction.append(['; '.join(c1), '; '.join(c2), code1, code2, url[0], validation_auc_best, validation_accr_best, y_test[i], y_test_se[i], y_prob_test[i]])
        res["prediction"] = prediction
        res["evidence"] = signals
        i_max = np.argmax(y_prob_test)
        print('Winner:', prediction[np.argmax(y_prob_test)][0], '({0});'.format(prediction[i_max][1]), 'Chance:', y_prob_test[i_max].round(3))
        print('===========================')
        sys.stdout.flush()
        results.append(res)
    print("done!")

    with open('../results/from_'+str(min_year)+'.predict_'+str(year_pred)+'.results.json', 'w') as fout:
        for res in results:
            fout.write(json.dumps(res)+'\n')

    results_flat = pd.DataFrame([[res["category"], pred[0], pred[1], pred[-1]] for res in results for pred in res["prediction"]],
                            columns = ["category", "primary nomination", "secondary nomination", "chance of winning"])
    evidence_flat = pd.DataFrame([[res["category"], s[0], np.round(
        s[1], decimals=6)] for res in results for s in res["evidence"]],
                                columns= ["category", "supporting feature", "coefficient"])
    results_flat.to_csv('../results/results_flat.csv', index=False)
    evidence_flat.to_csv('../results/evidence_flat.csv', index=False)


if __name__ == "__main__":
    run_all()
