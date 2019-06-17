#%%
import numpy as np
import pandas as pd

#lg_preds = pd.read_csv(r'./logistc_out_06-11-19-15.csv')
lgm_preds = pd.read_csv(r'./lgm_06-13-19-07-57.csv')
#lgm_preds = pd.read_csv(r'./lgm_wight_06-16-19-11-28.csv')
bayes_preds = pd.read_csv(r'./modified_naive_bayes_06-13-19-00.csv')

#%%

out = pd.DataFrame()
out['target'] = np.mean(np.column_stack([lgm_preds['target'], bayes_preds['target']]), axis=1)
out['ID_code'] = lgm_preds['ID_code']
out = out.iloc[:, -1::-1]
#%%
from datetime import datetime

dt = datetime.now().strftime('%m-%d-%y-%H-%M')
print(dt)

out.to_csv(r'./integration_%s.csv' % dt, index=False)