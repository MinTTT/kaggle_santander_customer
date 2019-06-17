#!/home/panchu/anaconda3/bin/python
#%%
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
import data_process as dp
import lgm_logistic as integ

#%%
def add_feature(df_in, fea, unique=False):
    df = df_in.copy()
    if not unique:
        for fe_na in tqdm(fea):
            fe_vals = df[fe_na].copy()
            unique_counts = fe_vals.value_counts()
            fe_vals = fe_vals.map(unique_counts)
            df['%s_count' % fe_na] = fe_vals.values
            df['%s_count' % fe_na].astype('int')
    else:
        unique_indx, _ = dp.find_unique_data(df[fea].values)
        for fe_na in tqdm(fea):
            fe_vals = df[fe_na].copy()
            unique_counts = fe_vals.iloc[unique_indx].value_counts()
            fe_vals = fe_vals.map(unique_counts)
            df['%s_count' % fe_na] = fe_vals.values
            df['%s_count' % fe_na].astype('int')
    return df


#%%
test = pd.read_csv(r'./test.csv')
train = pd.read_csv(r'./train.csv')

#%% ADD NEW FEATURES (COUNTS)
#unique_indx, _ = dp.find_unique_data(test.iloc[:, 1:].values)
#unique_test = test.iloc[unique_indx, :]
ori_feature = np.array([fe for fe in train.columns.values if fe not in ['ID_code', 'target']])
test['target'] = -1
all_data = pd.concat([train, test], ignore_index=True, sort=True)
print('Add counts features to all sets')
all_data_c = add_feature(all_data, ori_feature, unique=True)
train_c = all_data_c.iloc[range(len(train)), :]
test_c = all_data_c.iloc[np.arange(len(train)) + len(train), :]

#%% TRAIN
#%% TRAIN
# https://www.kaggle.com/felipemello/step-by-step-guide-to-the-magic-lb-0-922
param = {'bagging_fraction': 0.5166,
         'bagging_freq': 3,
         'lambda_l1': 3.968,
         'lambda_l2': 1.263,
         'learning_rate': 0.00141,
         'max_depth': 3,
         'min_data_in_leaf': 17,
         'min_gain_to_split': 0.2525,
         'min_sum_hessian_in_leaf': 19.55,
         'num_leaves': 20,
         'feature_fraction': 1.,
         'save_binary': True,
         'seed': 2319,
         'feature_fraction_seed': 2319,
         'bagging_seed': 2319,
         'drop_seed': 2319,
         'data_random_seed': 2319,
         'objective': 'binary',
         'boosting_type': 'gbdt',
         'verbosity': -1,
         'metric': 'auc',
         'is_unbalance': True,
         'boost_from_average': 'false'}

num_folds = 4
#train_c = train_c.sample(frac=1)
oof = np.zeros((len(train_c), len(ori_feature)))
preds = np.zeros((len(test), len(ori_feature)))

#%%
folds = StratifiedKFold(n_splits=num_folds)

#%%
for j, fea_na in enumerate(ori_feature):
    for i, (trn_idx, val_idx) in enumerate(
            folds.split(train_c[fea_na],
                        train_c['target'])):
        feas_loop = [fea_na, fea_na+'_count']
        x_tran = train_c.iloc[trn_idx][feas_loop]
        y_tran = train_c.iloc[trn_idx]['target']
        x_val = train_c.iloc[val_idx][feas_loop]
        y_val = train_c.iloc[val_idx]['target']
        trn_data = lgb.Dataset(x_tran, label=y_tran)
        val_data = lgb.Dataset(x_val, label=y_val)
        print('Features: ', j+1, 'Folds idx: ', i+1)
        model = lgb.train(param, trn_data, 1260,
                          valid_sets=[trn_data, val_data],
                          verbose_eval=-1)
        oof[val_idx, j] = model.predict(train_c.iloc[val_idx][feas_loop],
                                        num_iteration=model.best_iteration)
        preds[:, j] += model.predict(test_c[feas_loop],
                                     num_iteration=model.best_iteration) / folds.n_splits

#%%
dt = datetime.now().strftime('%m-%d-%y-%H-%M')
print('Time now:', dt)

#%%
out1 = pd.DataFrame(preds)
out2 = pd.DataFrame(oof)
out1.to_csv(r'./003_lgm_200_train_%s.csv' % dt, index=False, header=False)
out2.to_csv(r'./003_lgm_200_oof_%s.csv' % dt, index=False, header=False)

#%%
logit_model = sm.Logit(train_c['target'], oof)
logit_model = logit_model.fit(method='newton')  # disp=0 Don't show convergence message.

#%%
oofLog = logit_model.predict(oof)
oofLog_auc = roc_auc_score(train_c['target'], oofLog)
print('validate auc (train): %.3f' % oofLog_auc)
predsLog = logit_model.predict(preds)

#%%
output = pd.DataFrame()
output['ID_code'] = test['ID_code']
output['target'] = predsLog
output.to_csv(r'./003_lgm_200_log_%s.csv' % dt, index=False)
#%%
preds_naive = integ.naive_bayes_integ(train_c['target'], oof, preds)
output2 = pd.DataFrame({'ID_code': test['ID_code'].values,
                        'target': preds_naive})
output2.to_csv(r'./003_lgm_200_naive_%s.csv' % dt, index=False)
#%%
preds_weight = integ.weight_inter(train_c['target'], oof, preds)
output3 = pd.DataFrame({'ID_code': test['ID_code'].values,
                        'target': preds_weight})
output3.to_csv(r'./003_lgm_200_weight_%s.csv' % dt, index=False)
print('Finished !')

#%%
#TEST
# a = np.array([]).astype('int')
# for i, (trn_idx, val_idx) in enumerate(
#             folds.split(train_c[fea_na],
#                         train_c['target'])):
#     a = np.append(a, val_idx)