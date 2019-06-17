#%%
from statsmodels.discrete.discrete_model import Logit
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import os
dt = datetime.now().strftime('%m-%d-%y-%H')

#%% LOAD DATA
train = pd.read_csv(r'./train.csv')

#%%
logit_model = Logit(train['target'], train.iloc[:, 2:-1])
logit_model = logit_model.fit(disp=0)  # disp=0 Don't show convergence message.
#%%
predsLog = logit_model.predict(train.iloc[:, 2:-1])
predsLog_auc = roc_auc_score(train['target'], predsLog)
print('========================================================')
print('Val_AUC = ', round(predsLog_auc, 5))
print('========================================================')

#%% predict test
test = pd.read_csv(r'./test.csv')
predsLog2 = logit_model.predict(test.iloc[:, 1:-1])
#%% output
testoutput = pd.DataFrame()
testoutput['ID_code'] = test['ID_code'].copy()
testoutput['target'] = predsLog2
testoutput.to_csv(r'./logistic_out_%s.csv' % dt, index=False)
#%%
from datetime import datetime
from pan_plot import aucplot
import matplotlib.pyplot as plt

folderna = 'AUC' + dt
os.makedirs(folderna)

#%%
fig_1 = plt.figure(figsize=(6, 6))
auc = aucplot(train['target'], predsLog)
plt.savefig(r'./%s/logsmol.png' % folderna)
plt.show()
