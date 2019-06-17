import matplotlib.pyplot as plt
from sklearn import metrics


def aucplot(target, pred, ax=None):
    """
    :param target:
    :param pred:
    :param ax:
    :return: auc:
    """
    if not ax:
        ax = plt.gca()

    # auc
    fpr, tpr, threshold = metrics.roc_curve(target, pred)
    roc_auc = metrics.auc(fpr, tpr)

    # plot
    auc = ax.plot(fpr, tpr, 'b', label='Val AUC = %0.3f' % roc_auc)
    plt.legend(loc='lower right')
    ax.plot([0, 1], [0, 1], 'r--')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')

    return auc

#%%
if __name__ == '__main__':
    from statsmodels.discrete.discrete_model import Logit
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score


    train = pd.read_csv(r'./train.csv')

    logit_model = Logit(train['target'], train.iloc[:, 2:-1])
    logit_model = logit_model.fit(disp=0)  # disp=0 Don't show convergence message.

    predsLog = logit_model.predict(train.iloc[:, 2:-1])
    #%%
    fig_1 = plt.figure(figsize=(6, 6))
    aucplot(train['target'], predsLog)
    plt.show()