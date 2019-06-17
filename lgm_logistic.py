#%%
from statsmodels.discrete.discrete_model import Logit
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import os
from datetime import datetime
import statsmodels.api as sm


#%%
def naive_bayes_integ(target, oof, trained):
    lgodds = np.ones(oof.shape[0]) * np.log(1 / 9)
    preds_lgodds = lgodds.copy()
    for j in range(oof.shape[1]):
        if roc_auc_score(target, oof[:, j]) >= .5000:
            lgodds += np.log(oof[:, j]) - np.log((1. - oof[:, j]))
            preds_lgodds += np.log(trained[:, j]) - np.log((1 - trained[:, j]))
    oofLog_auc = roc_auc_score(target, lgodds)
    print('validate auc (train): %.4f' % oofLog_auc)
    return preds_lgodds

def weight_inter(target, data, train):
    weight = []
    for col in range(200):
        if roc_auc_score(target, data[:, col]) >= 0.5:
            weight.append(roc_auc_score(target, data[:, col]))
        else:
            weight.append(0)

    weight = np.array(weight)
    weight = weight / weight.mean()

    oof_pres = (data * weight).sum(axis=1) / data.shape[1]
    print('Validate AUC (weight):', roc_auc_score(target, oof_pres))
    train_pres = (train * weight).sum(axis=1) / data.shape[1]
    return train_pres


if __name__ == '__main__':
    #%%
    dt = datetime.now().strftime('%m-%d-%y-%H-%M')
    print('Time now: ', dt)

    #%%
    test = pd.read_csv(r'./test.csv')
    train = pd.read_csv(r'./train.csv')

    #%%
    oof = pd.read_csv(r'./002_lgm_200_oof_06-16-19-10-47.csv', header=None, prefix='var_')
    trained = pd.read_csv(r'./002_lgm_200_train_06-16-19-10-47.csv', header=None, prefix='var_')

    #%% Log
    # model = Logit(train['target'], oof)
    # # model = model.fit(method='lbfgs')
    logit_model = sm.Logit(train['target'], oof)
    logit_model = logit_model.fit(method='minimize')  # disp=0 Don't show convergence message.
    oofLog = logit_model.predict(oof)
    oofLog_auc = roc_auc_score(train['target'], oofLog)
    print('validate auc (train): %.3f' % oofLog_auc)

    #%% pred
    trained_predations = naive_bayes_integ(train['target'], oof.values, trained.values)
    # lgodds = np.ones(oof.shape[0]) * np.log(1 / 9)
    # for var in range(oof.shape[1]):
    #     if roc_auc_score(train['target'], oof.iloc[:, var].values) >= 0.500:
    #         lgodds += np.log(oof.iloc[:, var].values) - np.log((1 - oof.iloc[:, var].values))
    #
    # oofLog_auc = roc_auc_score(train['target'], lgodds)
    # print('validate auc (train): %.4f' % oofLog_auc)

    #%% Weight
    def weight_inter(target, data, train):
        weight = []
        for col in range(200):
            if roc_auc_score(target, data[:, col]) >= 0.5:
                weight.append(roc_auc_score(target, data[:, col]))
            else:
                weight.append(0)

        weight = np.array(weight)
        weight = weight / weight.mean()

        oof_pres = (data * weight).sum(axis=1) / data.shape[1]
        print('Validate AUC (weight):', roc_auc_score(target, oof_pres))
        train_pres = (train * weight).sum(axis=1) / data.shape[1]
        return train_pres

    trained_predations = weight_inter(train['target'], oof.values, trained.values)
    #%%
    output = pd.DataFrame()
    output['ID_code'] = test['ID_code']
    output['target'] = trained_predations
    output.to_csv(r'./lgm_wight_%s.csv' % dt, index=False)



    # weight = []
    #
    # for col in range(200):
    #     if roc_auc_score(train['target'], oof.iloc[:, col]) >= 0.5:
    #         weight.append(roc_auc_score(train['target'], oof.iloc[:, col]))
    #     else:
    #         weight.append(0)
    #
    # weight = np.array(weight)
    # weight = weight/weight.mean()
    #
    # oof_pres = (oof * weight).sum(axis=1)/200
    # print('AUC (weight):', roc_auc_score(train['target'], oof_pres))
