from sklearn.model_selection import train_test_split, LeaveOneOut
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
from Models.ML_Model import definedSVM, SupportVectorMachine#, simpleDNN
import random
from sklearn.metrics import f1_score
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix


# Changeable variables. Please specify data set name: MT or POM
global dataset
dataset = 'MT'
# You can specify this dimension but program will analyse every dimension by the default
global rate_type
rate_type = 'confidence'


global threshold
threshold = 0.10

with open('./categories.json','r') as f:
    categories = json.load(f)




def featureSelection(X, Y, target):
    data = X.join(Y, how='left')


    corr = data.corr(method='spearman')
    thresholded_corr = corr[abs(corr['label']) > threshold]
    X = X[thresholded_corr.index[:-2]]
    if (dataset == "POM"):
        target = X.loc[[136647]]
    elif (dataset == "MT"):
        target = X.loc[['BOD09']]
    feature_name = list(X.columns)

    return X, Y, target, feature_name



def loadFeturesByCategories():
    # audio_category = categories['audio_category']
    # visual_category = categories['visual_category']
    filler_category = categories['filler_category']
    text_linking_rate_category = categories['text_linking_rate_category']
    text_synonyms_rate_category = categories['text_synonyms_rate_category']
    text_div_category = categories['text_div_category']
    text_dens_category = categories['text_dens_category']
    text_disc_category = categories['text_disc_category']
    text_ref_category = categories['text_ref_category']
    text_polarity= categories['text_polarity']

    group_by_category = {**text_linking_rate_category, **text_synonyms_rate_category, **text_div_category, **text_dens_category, **text_disc_category, **text_ref_category, **filler_category, **text_polarity}
    # group_by_category = {**text_rate_category, **text_embedding_category, **text_div_category}
    # group_by_category = {**audio_category, **filler_category, **visual_category}

    # audio = pd.read_csv('./data/acoustic.csv', index_col=0)
    # visual = pd.read_csv('./data/visual.csv', index_col=0)
    if (dataset == "POM"):
        filler = pd.read_csv('./data/' + dataset + '/filler.csv', index_col=0)
        text_polar = pd.read_csv('./data/' + dataset + '/LIWC_eng_dict_2007_Analysis.csv', index_col=0)
    else:
        text_polar = pd.read_csv('./data/' + dataset + '/LIWC_fr_dict_2007_Analysis.csv', index_col=0)
    text_linking_rate =  pd.read_csv('./data/' + dataset + '/text_linking_rate.csv', index_col=0)
    text_synonym_rate =  pd.read_csv('./data/' + dataset + '/text_synonyms_rate.csv', index_col=0)
    text_div =  pd.read_csv('./data/' + dataset + '/text_diversity.csv', index_col=0)
    text_dens =  pd.read_csv('./data/' + dataset + '/text_density.csv', index_col=0)
    text_disc =  pd.read_csv('./data/' + dataset + '/text_discource.csv', index_col=0)
    text_ref =  pd.read_csv('./data/' + dataset + '/text_reference.csv', index_col=0)


    if (dataset == "POM"):
        X = filler.join(text_linking_rate, how='left').join(text_polar, how='left').join(text_synonym_rate,how='left').join(text_div,how='left').join(text_dens, how='left').join(text_disc, how='left').join(text_ref,how='left')

    else:
        X = text_linking_rate.join(text_polar, how = 'left').join(text_synonym_rate, how = 'left').join(text_div, how = 'left').join(text_dens, how = 'left').join(text_disc, how = 'left').join(text_ref, how = 'left')


    columns = X.columns
    index = X.index

    # This part may perform normalization but we did not found any effectiveness in it

    # df = X.values #returns a numpy array
    # min_max_scaler = preprocessing.MinMaxScaler()
    # x_scaled = min_max_scaler.fit_transform(df)
    # X = pd.DataFrame(x_scaled, index = index, columns = columns)

    return X, group_by_category

def  loadRatings():
    Y = pd.read_csv('./data/' + dataset + '/labels/' + rate_type + 'Label.csv', index_col=0)
    return Y


def readDataFromFile():
    X = pd.read_csv('./demo/' + dataset +  '/background.csv', index_col=0)
    return X


def dataPreprocessing(X, Y):
    Y = Y.loc[(Y.index).isin(X.index)]
    X = X.loc[(X.index).isin(Y.index)]

    if (dataset == "POM"):
        target = X.loc[[136647]]
    elif (dataset == "MT"):
        target = X.loc['BOD09']
    feature_name = list(X.columns)

    X.to_csv('./demo/' + dataset + '/' + rate_type + '/background.csv')
    target.to_csv('./demo/' + dataset + '/' + rate_type + '/target.csv')

    return X, Y, target, feature_name


def getDimentions(dataset):
    if (dataset == "MT"):
        dimentions = ["confidence", "persuasiveness", "engagement", "global"]
    elif (dataset == "POM"):

        POM_labels = pd.read_csv('./datastats/POM_labels.csv', sep=",", index_col=0, skiprows=1, dtype=np.float64)
        dimentions = POM_labels.columns
        # If you want to analyse all dimensions of POM comment the line below
        dimentions = [ "confident", "persuasion"]

    return dimentions


def calculate_ci(data):
    mean = np.mean(data)
    ci = st.norm.interval(alpha=0.95, loc=mean, scale=st.sem(data))
    return mean, ci


def choseBestParametersForClassification(X, Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0, stratify=Y)
    rf_param, rf_train, rf_test, svm = SupportVectorMachine(X_train, y_train, X_test, y_test)
    txtForSave = open('./results/' + dataset + '/' + rate_type + '/best_parameters_svm.txt', 'w')
    txtForSave.write("best parameters: {}".format(rf_param) + " best_train_score: {}".format(rf_train) + "\n")
    txtForSave.close()
    return rf_param, svm



def averageF1Score(X, Y, best_param):
    test_mean = []
    for i in range(0, 50):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
        rf_test, svm = definedSVM(X_train, y_train, X_test, y_test,kernel=best_param['kernel'],probability=best_param['probability'],gamma=best_param['gamma'],C=best_param['C'])
        test_mean.append(rf_test)

    mean, ci = calculate_ci(test_mean)
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
        # f1 = f1_score(y_train['label'], random_list)
        acc = accuracy_score(y_train['label'], random_list)
        test_mean.append(acc)

    mean, ci = calculate_ci(test_mean)
    txtForSave = open('./results/' + dataset + '/' + rate_type + '/random_acc_score.txt', 'w')
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


def leaveOneOutTrain(X, Y, best_param, best_model):
    from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
    loo = LeaveOneOut()
    test_mean = []
    for i, (train_index, test_index) in enumerate(loo.split(X)):
        X_train = X.loc[X.index[train_index]]
        X_test = X.loc[X.index[test_index]]
        y_train = Y.loc[Y.index[train_index]]
        y_test = Y.loc[Y.index[test_index]]
        rf_test, svm = definedSVM(X_train, y_train, X_test, y_test, kernel=best_param['kernel'],
                                  probability=best_param['probability'], gamma=best_param['gamma'], C=best_param['C'])
        test_mean.append(rf_test)

    mean, ci = calculate_ci(test_mean)
    txtForSave = open('./results/' + dataset + '/' + rate_type + '/lvo_accuracy_score.txt', 'w')
    txtForSave.write("mean: {}".format(mean) + " confidence interval: {}".format(ci) + "\n")
    txtForSave.close()

    return None


def calcWRTcategories(featuresCorr, group_by_category, type):

    groupmap = revert_dict(group_by_category)
    featureCorr_df = pd.DataFrame(featuresCorr, index=pd.Index(featuresCorr.index, name='features'))
    featureCorr_df['group'] = featureCorr_df.index.map(groupmap).values
    featureCorr_grouped = featureCorr_df.groupby('group').agg({'label': lambda x: abs(x).mean()})

    plt.cla()
    plt.title("Mean abs {}'s correlaltion w.r.t. categories".format(type))
    plt.bar(featureCorr_grouped.index, featureCorr_grouped['label'], color="orange")
    plt.xticks(featureCorr_grouped.index, featureCorr_grouped.index, rotation=25, fontsize='x-small')
    plt.draw()
    plt.savefig('./results/' + dataset + '/' + rate_type + '/' + type + '_correlation_by_group.png')

    txtForSave = open('./results/' + dataset + '/' + rate_type + '/' + type +'_mean_abs_by_category.txt', 'w')
    txtForSave.write("mean absolute correlation by groups \n: {}".format(featureCorr_grouped) + " \n with correlation coefficients:\n {}".format(
        featuresCorr) + "\n")
    txtForSave.close()



from itertools import repeat, chain
revert_dict = lambda d: dict(chain(*[zip(val, repeat(key)) for key, val in d.items()]))

def spearmanCorr(data, type):
    corr = data.corr(method=type)

    print("__________________________" + type + "__________________________")
    thresholded_corr = corr[abs(corr['label']) > threshold]
    if (len(thresholded_corr.index) > 1) and (rate_type != 'humerous')and (rate_type != 'lazy'):
        plt.cla()
        plt.title("{}'s correlaltion w.r.t. textual features".format(type))
        plt.bar(thresholded_corr.index[:-2], thresholded_corr['label'][:-2], color="blue")
        plt.xticks(thresholded_corr.index[:-2], thresholded_corr.index[:-2], rotation=25, fontsize='x-small')
        plt.draw()
        plt.savefig('./results/' + dataset + '/' + rate_type + '/' + type +'_correlation.png')
        txtForSave = open('./results/' + dataset + '/' + rate_type + '/' + type +'_top.txt', 'w')
        txtForSave.write("threshold: {}".format(threshold) + " top features and correlation coefficients:\n {}".format(
            thresholded_corr['label'][:-2]) + "\n")
        txtForSave.close()

    return corr.loc['label'][:-1], thresholded_corr['label'][:-2]







def correlation(X, Y, group_by_category):
    type = 'spearman'
    data = X.join(Y, how = 'left')
    allCorrelationCoefff, topCorrelationCoefff = spearmanCorr(data, type)
    calcWRTcategories(allCorrelationCoefff, group_by_category, type)
    type = 'pearson'
    allCorrelationCoefff, topCorrelationCoefff = spearmanCorr(data, type)
    print(topCorrelationCoefff)
    calcWRTcategories(allCorrelationCoefff, group_by_category, type)


if __name__ == "__main__":
    dimentions = getDimentions(dataset)

    for dim in dimentions:
        print("------------------" + dim + "------------------")
        rate_type = dim
        setGlobal(dataset, dim)
        X, group_by_category = loadFeturesByCategories()
        Y = loadRatings()
        X, Y, target, feature_name = dataPreprocessing(X, Y)
        print("*********************** CalculCorr ***********************")
        correlation(X, Y, group_by_category)
        print("*********************** featureSelection ***********************")
        X, Y, target, feature_name = featureSelection(X, Y, target)
        if (len(X.columns) > 0):
            print("*********************** Chose Best Parameters ***********************")
            best_param, best_model = choseBestParametersForClassification(X, Y)
            print("*********************** LeaveOneOut ***********************")
            leaveOneOutTrain(X, Y, best_param, best_model)
            print("*********************** AverageF1 ***********************")
            svm, X_train, X_test = averageF1Score(X, Y, best_param)
            print("*********************** Random ***********************")
            randomClassifier(X, Y)
            print("*********************** SHAP ***********************")
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
            best_model.fit(X_train, y_train.values.ravel())
            shapAnalysis(best_model, X_train, X_test, target, group_by_category, feature_name)









