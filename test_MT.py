from sklearn.model_selection import train_test_split
# from skmultilearn.model_selection import iterative_train_test_split
import pandas as pd
from preprocess.FeatureSelection import z_test
import json
from feedback.SHAP import SVMInterpret, setGlobal
import numpy as np
import scipy.stats as st
from feedback.feedback_generator import feedback
from feedback.SHAP import grouped_shap
from joblib import dump,load
from Models.ML_Model import definedSVM, SupportVectorMachine
import random
from sklearn.metrics import f1_score


global dataset
dataset = 'POM'

global rate_type
rate_type = 'confidence'

with open('./categories.json','r') as f:
    categories = json.load(f)



def loadFeturesByCategories():
    audio_category = categories['audio_category']
    # visual_category = categories['visual_category']
    filler_category = categories['filler_category']
    text_linking_rate_category = categories['text_linking_rate_category']
    text_synonyms_rate_category = categories['text_synonyms_rate_category']
    text_div_category = categories['text_div_category']
    text_dens_category = categories['text_dens_category']
    text_disc_category = categories['text_disc_category']
    text_ref_category = categories['text_ref_category']
    # text_polarity= categories['text_polarity']

    group_by_category = {**text_linking_rate_category, **text_synonyms_rate_category, **text_div_category, **text_dens_category, **text_disc_category, **text_ref_category, **filler_category}
    # group_by_category = {**text_rate_category, **text_embedding_category, **text_div_category}
    # group_by_category = {**audio_category, **filler_category, **visual_category}

    # audio = pd.read_csv('./data/acoustic.csv', index_col=0)
    # visual = pd.read_csv('./data/visual.csv', index_col=0)
    filler = pd.read_csv('./data/' + dataset + '/filler.csv', index_col=0)
    text_linking_rate =  pd.read_csv('./data/' + dataset + '/text_linking_rate.csv', index_col=0)
    text_synonym_rate =  pd.read_csv('./data/' + dataset + '/text_synonyms_rate.csv', index_col=0)
    # text_embeddings =  pd.read_csv('./data/' + dataset + '/text_embeddings.csv', index_col=0)
    text_div =  pd.read_csv('./data/' + dataset + '/text_diversity.csv', index_col=0)
    text_dens =  pd.read_csv('./data/' + dataset + '/text_density.csv', index_col=0)
    text_disc =  pd.read_csv('./data/' + dataset + '/text_discource.csv', index_col=0)
    text_ref =  pd.read_csv('./data/' + dataset + '/text_reference.csv', index_col=0)
    # text_polar =  pd.read_csv('./data/' + dataset + '/textPolarity.csv', index_col=0)


    # X = text_linking_rate.loc[(text_linking_rate.index).isin(foller_index)].join(text_synonym_rate, how = 'left').join(text_div, how = 'left').join(text_dens, how = 'left').join(text_disc, how = 'left').join(text_ref, how = 'left') #.join(text_linking_rate, how = 'left')
    X = text_linking_rate.join(text_synonym_rate, how = 'left').join(text_div, how = 'left').join(text_dens, how = 'left').join(text_disc, how = 'left').join(text_ref, how = 'left') #.join(text_linking_rate, how = 'left')
    return X, group_by_category

def  loadRatings():
    print(rate_type)
    Y = pd.read_csv('./data/' + dataset + '/labels/' + rate_type + 'Label.csv', index_col=0)
    return Y


def readDataFromFile():
    X = pd.read_csv('./demo/' + dataset +  '/background.csv', index_col=0)
    return X


def dataPreprocessing(X, Y):
    # split to train / test
    # from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    # msss = MultilabelStratifiedShuffleSplit(n_splits=2, test_size=0.02, random_state=42)
    # for train_index, test_index in msss.split(X, Y):
    #     x_train, x_test_temp = X[train_index], X[test_index]
    #     y_train, y_test_temp = Y[train_index], Y[test_index]


    # audio = z_test(audio,Y)
    # visual = z_test(visual,Y)
    # text_embeddings = z_test(text_embeddings, Y)
    target = X.loc[[136647]]
    # target = X.loc[['AMU08']]
    feature_name = list(X.columns)

    Y = Y.loc[(Y.index).isin(X.index)]
    X = X.loc[(X.index).isin(Y.index)]
    columns = X.columns
    index= X.index

    # import pandas as pd
    # from sklearn import preprocessing
    # df = X.values #returns a numpy array
    # min_max_scaler = preprocessing.MinMaxScaler()
    # x_scaled = min_max_scaler.fit_transform(df)
    # X = pd.DataFrame(x_scaled, index = index, columns = columns)

    X.to_csv('./demo/' + dataset + '/' + rate_type + '/background.csv')
    target.to_csv('./demo/' + dataset + '/' + rate_type + '/target.csv')

    return X, Y, target, feature_name


def getDimentions():
    POM_labels = pd.read_csv('./datastats/POM_labels.csv', sep=",", index_col=0, skiprows=1, dtype=np.float64)
    dimentions = POM_labels.columns
    return dimentions


def calculate_ci(data):
    mean = np.mean(data)
    ci = st.norm.interval(alpha=0.95, loc=mean, scale=st.sem(data))
    return mean, ci


def choseBestParametersForClassification(X, Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0, stratify=Y)
    rf_param, rf_train, rf_test, svm = SupportVectorMachine(X, Y, X_test, y_test)
    print('best_param', rf_param)
    print('best_train_score', rf_train)
    txtForSave = open('./results/' + dataset + '/' + rate_type + '/best_parameters_svm.txt', 'w')
    txtForSave.write("best parameters: {}".format(rf_param) + " best_train_score: {}".format(rf_train) + "\n")
    txtForSave.close()
    return rf_param



def averageF1Score(X, Y, best_param):
    test_mean = []
    for i in range(0, 300):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=i, stratify=Y)
        rf_test, svm = definedSVM(X_train, y_train, X_test, y_test,kernel=best_param['kernel'],probability=best_param['probability'],gamma=best_param['gamma'],C=best_param['C'])
        test_mean.append(rf_test)

    mean, ci = calculate_ci(test_mean)
    print(mean, ci)
    txtForSave = open('./results/' + dataset + '/' + rate_type + '/F1_score.txt', 'w')
    txtForSave.write("mean: {}".format(mean) + " confidence interval: {}".format(ci) + "\n")
    txtForSave.close()
    dump(svm, './demo/' + dataset + '/' + rate_type + '/svm.joblib')

    return svm, X_train, X_test


def randomClassifier(X, Y):
    test_mean = []
    for i in range(0, 300):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=i, stratify=Y)
        random_list  = random.choices(range(0, 2), k = len(y_train.index))
        f1 = f1_score(y_train['label'], random_list)
        # y_true = [0, 1, 1]
        # y_pred = [1, 0, 0]
        # f1 = f1_score(y_true, y_pred)
        print(f1)
        test_mean.append(f1)

    mean, ci = calculate_ci(test_mean)
    print(mean, ci)
    txtForSave = open('./results/' + dataset + '/' + rate_type + '/random_F1_score.txt', 'w')
    txtForSave.write("mean: {}".format(mean) + " confidence interval: {}".format(ci) + "\n")
    txtForSave.close()
    return None



def shapAnalysis(svm, X_train, X_test, target, group_by_category, feature_name):
    feedback_generator = feedback('./results/' + dataset + '/' + rate_type + '/')
    group_shap = grouped_shap(svm, X_test, target, group_by_category, feature_name)
    # group_shap = grouped_shap(svm,X_train,X_test,group_by_category,feature_name)
    feedback_generator.ABS_SHAP(group_shap)
    SVMInterpret(svm, X_test, group_by_category, feature_name)
    return None



if __name__ == "__main__":
    dimentions = getDimentions()


    for dim in dimentions:
        print("------------------" + dim + "------------------")
        rate_type = dim
        setGlobal(dataset, dim)
        Y = loadRatings()
        X, group_by_category = loadFeturesByCategories()
        X, Y, target, feature_name = dataPreprocessing(X, Y)
        best_param = choseBestParametersForClassification(X, Y)
        print(best_param)
        svm, X_train, X_test = averageF1Score(X, Y, best_param)
        randomClassifier(X, Y)
        # shapAnalysis(svm, X_train, X_test, target, group_by_category, feature_name)









