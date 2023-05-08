# SHAP related methods

import shap
from itertools import repeat, chain
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

# Changable part:
# Change the path to the folder with the whole directory
sys.path.append('/Users/goddessoffailures/Desktop/Mock-Up_sample/Mock-Up_sample/REVITALISE')

# Change the name of data set you analyse with SHAP analysis
global dataset
dataset = 'POM'
# Change the name of dimension you analyse with SHAP analysis
global rate_type
rate_type = 'persuasiveness'





revert_dict = lambda d: dict(chain(*[zip(val, repeat(key)) for key, val in d.items()]))

# calculate importance of each category
def SVMInterpret(model, X, groups, features_name):
    group_shap = grouped_shap(model, X, X, groups, features_name)
    txtForSave = open('./results/' + dataset + '/' + rate_type + '/mean_interpret.txt', 'w')

    for col_0 in group_shap.columns:
        tmp = group_shap[col_0].values
        tmp = np.abs(tmp)
        txtForSave.write(col_0 + ": {}".format(tmp.mean()) + '\n')
    txtForSave.close()


def setGlobal(dataset_pr, rate_type_pr):
    global dataset
    dataset = 'POM'
    global rate_type
    rate_type = 'gffg'
    dataset = dataset_pr
    rate_type = rate_type_pr



def grouped_shap(model, background, target, groups, features_name):
    
    explainer = shap.KernelExplainer(model.predict_proba, background)
    shap_values = explainer.shap_values(target)
    shap.summary_plot(shap_values[1], target, max_display=10, auto_size_plot=True)
    shap_0 = shap_values[1]
    groupmap = revert_dict(groups)
    shap_Tdf = pd.DataFrame(shap_0, columns=pd.Index(features_name, name='features')).T
    shap_Tdf['group'] = shap_Tdf.reset_index().features.map(groupmap).values
    shap_grouped = shap_Tdf.groupby('group').sum().T

    return shap_grouped

