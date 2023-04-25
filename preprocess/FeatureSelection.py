# Feature Selection Method

import numpy as np
from scipy.stats import norm


def z_test(features, label):
    f_name = list(features.columns)
    data = features.join(label)
    for f_n in f_name:
        p_mean=data.loc[data['label']==1, f_n].mean()
        n_mean=data.loc[data['label']==0, f_n].mean()
        p_std=data.loc[data['label'] == 1, f_n].std()
        n_std=data.loc[data['label'] == 0, f_n].std()
        no_of_p = data.loc[data['label'] == 1, f_n].count()
        no_of_n = data.loc[data['label'] == 0, f_n].count()
        pooledSE = np.sqrt(p_std**2/no_of_p+n_std**2/no_of_n)+1e-12
        z = (p_mean-n_mean)/pooledSE
        pval = 2*(1-norm.cdf(abs(z)))
        # print(pval)
        if pval>0.05:
            data=data.drop(columns=[f_n])
    features = data.drop(['label'],axis=1)
    return features