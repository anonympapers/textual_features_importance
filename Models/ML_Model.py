# related machine learning models

from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


def SupportVectorMachine(X_train, Y_train, X_test, Y_test):
    svc_param_grid = {'kernel': ['rbf'],
                      'probability':[True],
                      'gamma': [0.001, 0.01, 0.1, 1,'auto'],
                      'C': [1, 10, 20]}
    SVMC = svm.SVC()
    gsSVMC = GridSearchCV(SVMC, param_grid=svc_param_grid, cv=10, scoring="f1", n_jobs=-1, verbose=1)
    gsSVMC.fit(X_train, Y_train)
    SVMC_best = gsSVMC.best_estimator_
    best_param = gsSVMC.best_params_
    train_score = gsSVMC.best_score_
    test_score = SVMC_best.score(X_test, Y_test)
    return best_param, train_score, test_score, SVMC_best


def definedSVM(X_train, Y_train, X_test, Y_test,kernel='rbf',probability=True,gamma=0.01,C=20):

    # Confidence parameters POM:
    # best_param
    # {'C': 20, 'gamma': 0.01, 'kernel': 'rbf', 'probability': True}
    # best_train_score
    # 0.45559991628822455

    # Persuasiveness parameters POM:
    # kernel = 'rbf', probability = True, gamma = 0.001, C = 1

    SVM = svm.SVC(kernel=kernel, probability=probability, gamma=gamma, C=C)
    SVM.fit(X_train,Y_train)
    test_score = SVM.score(X_test, Y_test)
    f1 = f1_score(Y_test, SVM.predict(X_test))
    return f1, SVM

def RandomForest(X_train, Y_train, X_test, Y_test):
    RFC = RandomForestClassifier()
    ## Search grid for optimal parameters
    rf_param_grid = {"max_depth": [None],
                     "max_features": [10,20,'auto'],
                     "min_samples_split": [2, 3, 10],
                     "min_samples_leaf": [1, 3, 10],
                     "bootstrap": [True],
                     "n_estimators": [300,400],
                     "criterion": ["gini"]}
    gsRFC = GridSearchCV(RFC, param_grid=rf_param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1)
    gsRFC.fit(X_train, Y_train)
    RFC_best = gsRFC.best_estimator_
    best_param = gsRFC.best_params_
    train_score = gsRFC.best_score_
    test_score = RFC_best.score(X_test, Y_test)
    importance = RFC_best.feature_importances_
    return best_param, train_score, test_score, importance