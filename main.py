#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as mas
mas.use('ggplot')
import seaborn as sns
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm

dul_color = ['#C05649', '#49B3C0']
tri_color = ['#EC7E47', '#47EC7E', '#7E47EC']

#%%
train = pd.read_csv(r'./train.csv')
#%%
train_des = train.describe()
train_des.to_csv(r'./data_description.csv')
    #%%
count_1 = len(train.loc[lambda df: df['target'] == 1]) / len(train)
count_0 = len(train.loc[lambda df: df['target'] == 0]) / len(train)
fig_1, ax1 = plt.subplots(figsize=(6, 6))
ax1.bar([0, 1], [count_0, count_1], color=dul_color, alpha=.5)
ax1.set_xticks([0, 1])
ax1.set_xlabel('Target')
ax1.set_ylabel('Percentage')
plt.show()

#%% correlation between variables
train_data = train.loc[lambda df: df.columns != 'ID_code']
train_corr = train_data.corr()
fig_2, ax = plt.subplots(1, 2, figsize=(13, 6))
sns.distplot(train_corr.iloc[0, 1:], ax=ax[0], color=dul_color[0])
sns.distplot([i for i in train_corr.values.flatten() if i != 1.],
             ax=ax[1],
             color=dul_color[1])
ax[0].set_title('Distribution of Correlation Coefficients \n between Target and Variables')
ax[0].set_xlabel('Targets - Variables')
ax[1].set_xlabel('Variables')
ax[1].set_title('Distribution of Correlation Coefficients \n Variables')
plt.show()
#%% PCA
PCA_data = train.loc[:, [True if i not in ['target', 'ID_code']
                         else False for i in train.columns]]
n = 5
svd = TruncatedSVD(n_components=n, n_iter=18, random_state=42)
svd.fit(PCA_data.values)
#%%
print(svd.explained_variance_ratio_.sum())
fig_3, ax = plt.subplots(figsize=(6, 6))
plt.plot(np.arange(n), svd.singular_values_)
plt.show()
#%%
svd = TruncatedSVD(n_components=2, n_iter=18, random_state=42)
svd.fit(PCA_data.values)
reduced = svd.transform(PCA_data.values)
reduced = pd.DataFrame({'component 1': reduced[:, 0],
                        'component 2': reduced[:, 1],
                        'target': train['target'].values})
#%%
fig_4, ax = plt.subplots(figsize=(6, 6))
sns.scatterplot(x='component 1', y='component 2',
                data=reduced[reduced['target'] == 0],
                color=dul_color[0], alpha=.5,
                label='$\mathrm{Target} = 0$')
sns.scatterplot(x='component 1', y='component 2',
                data=reduced[reduced['target'] == 1],
                color=dul_color[1], alpha=.5,
                label='$\mathrm{Target} = 1$')
plt.show()

#%%
var_list = np.array([i for i in train.columns if i not in ['target', 'ID_code']])
var_list = np.reshape(var_list, (20, 10))
# 20 * 10 plots
fig_5, ax = plt.subplots(20, 10, figsize=(6*10, 20*6))
for i in tqdm(range(20)):
    for j in range(10):
        var_na = var_list[i, j]
        sns.distplot(train.loc[lambda df: df['target'] == 1][var_na].values,
                     label='Target = 1', ax=ax[i, j])
        sns.distplot(train.loc[lambda df: df['target'] == 0][var_na].values,
                     label='Target = 0', ax=ax[i, j])
        ax[i, j].set_title(var_na)
        ax[i, j].legend()
plt.savefig(r'./data_eda.pdf')
plt.show()