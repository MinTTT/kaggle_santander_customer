#%%
import lightgbm as lgb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
#%%
test = pd.read_csv(r'./test.csv')
train = pd.read_csv(r'./train.csv')

#%%
# for testing the code, choose part of data
train2 = train.sample(frac=1, random_state=0)
target = train2['target']
#%%
param = {
    'bagging_freq': 5,
    'bagging_fraction': 0.335,
    'boost_from_average': 'false',
    'boost': 'gbdt',
    'feature_fraction': 0.041,
    'learning_rate': 0.04,
    'max_depth': -1,
    'metric': 'auc',
    'min_data_in_leaf': 80,
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 10,
    'num_threads': 60,
    'tree_learner': 'serial',
    'objective': 'binary',
    'verbosity': -1
}
evals_result = {}
num_vars = 200


#%%
num_folds = 6
features = [c for c in train2.columns if c not in ['ID_code', 'target']]

folds = StratifiedKFold(n_splits=num_folds, random_state=2222)
oof = np.zeros(len(train2))
getVal = np.zeros(len(train2))
predictions = np.zeros(len(test))
feature_importance_df = pd.DataFrame()

#%%
print('Light GBM Model')
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train2.values, target.values)):
    X_train, y_train = train2.iloc[trn_idx][features], target.iloc[trn_idx]
    X_valid, y_valid = train2.iloc[val_idx][features], target.iloc[val_idx]

    #X_tr, y_tr = augment(X_train.values, y_train.values)
    #X_tr = pd.DataFrame(X_tr)

    print("Fold idx:{}".format(fold_ + 1))
    trn_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_valid, label=y_valid)

    clf = lgb.train(param, trn_data, 1000000, valid_sets=[trn_data, val_data], verbose_eval=5000,
                    early_stopping_rounds=4000)
    oof[val_idx] = clf.predict(train2.iloc[val_idx][features], num_iteration=clf.best_iteration)
    getVal[val_idx] += oof[val_idx] / folds.n_splits

    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    predictions += clf.predict(test[features], num_iteration=clf.best_iteration) / folds.n_splits

print("CV score: {:<8.5f}".format(roc_auc_score(target, getVal)))

#%%
from datetime import datetime

dt = datetime.now().strftime('%m-%d-%y-%H-%M')
print(dt)
#%%
output = pd.DataFrame()
output['ID_code'] = test['ID_code']
output['target'] = predictions
output.to_csv(r'./lgm_%s.csv' % dt, index=False)
print('Finished !')
