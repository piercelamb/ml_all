import sys
import pandas as pd
import numpy as np
import matplotlib
from scipy import sparse
from scipy import linalg
from sklearn.decomposition import PCA, FastICA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from numpy import mean
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn import tree as dt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, v_measure_score, homogeneity_completeness_v_measure, rand_score, \
    adjusted_rand_score, adjusted_mutual_info_score, homogeneity_score, completeness_score, fowlkes_mallows_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_confusion_matrix
from imblearn.pipeline import Pipeline as imbalancePipeline
from sklearn.pipeline import Pipeline as sklearnPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import learning_curve
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from collections import Counter
from sklearn import preprocessing
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import time
import random
from functools import wraps
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.cluster import KMeans
from sklearn import mixture
from sklearn.metrics import silhouette_samples, silhouette_score
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import InterclusterDistance
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.base import ClusterMixin
from sklearn.mixture import GaussianMixture
from yellowbrick.cluster import KElbow
import statistics
from itertools import product

RANDOM_STATE = 1337

def timeit(func):
    @wraps(func)
    def timed_function(*args, **kwargs):
        start = time.time()
        output = func(*args, **kwargs)
        end = time.time()
        print('%s execution time: %f secs' % (func.__name__, end - start))
        return output
    return timed_function

def scale_continuous_data(X_train, X_test):
    categorical = ['E3', 'E7', 'E8', 'V5']
    numerical = [x for x in X_train.columns if x not in categorical]
    scaler = preprocessing.StandardScaler().fit(
        X_train.loc[:, numerical])
    X_train.loc[:, numerical] = scaler.transform(
        X_train.loc[:, numerical])
    X_test.loc[:, numerical] = scaler.transform(
        X_test.loc[:, numerical])
    return X_train, X_test

def preprocess_ford_data(X_train, X_test):
    X_train, X_test = scale_continuous_data(X_train, X_test)

    return X_train, X_test

def get_data_ford(dataroot, dataset_sample, is_rfc):
    df_train = pd.read_csv(dataroot + 'fordTrain.csv')
    df_test = pd.read_csv(dataroot + 'fordTest.csv')
    df_pred = pd.read_csv(dataroot + 'Solution.csv')
    y_train = df_train['IsAlert']
    y_test = df_pred['Prediction']
    X_train, X_test = preprocess_ford_data(df_train.drop(['IsAlert', 'TrialID', 'ObsNum'], axis=1), df_test.drop(['IsAlert', 'TrialID', 'ObsNum'], axis=1))

    if dataset_sample != 0:
        print("Sampling the training set at: "+str(dataset_sample))
        X_train, X_test_new, y_train, y_test_new = train_test_split(X_train, y_train, train_size=dataset_sample,
                                                            random_state=RANDOM_STATE)
        print("Sampled number of instances: " + str(len(X_train.index)))

    num_features = len(X_train.columns)
    print("FAD has " + str(num_features) + " features after preprocessing")
    if is_rfc:
        X_train = X_train.set_axis(range(0, num_features), axis=1)
        X_test = X_test.set_axis(range(0, num_features), axis=1)
    return X_train, y_train, X_test, y_test, num_features

@timeit
def calc_testing(clf, X_train, y_train, X_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred

def get_confusion_matrix(optimal_clf, X_test, y_test, title, filename):
    confusion_matrix = plot_confusion_matrix(
        optimal_clf,
        X_test,
        y_test,
        cmap=plt.cm.Blues,
        normalize='true'
    )
    confusion_matrix.ax_.set_title(title)
    plt.savefig(filename)
    plt.clf()

def plot_learning_curve(is_iterative, model,title,X_train,y_train,cv, filename,scoring):
    plt.figure()
    plt.title(title)
    plt.xlabel("Num Samples")
    plt.ylabel(scoring)
    step=np.linspace(1/cv,1.0,cv)
    train_sizes,train_scores,test_scores = learning_curve(model,X_train,y_train,cv=cv,train_sizes=step, scoring=scoring, n_jobs=-1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1,
                     color='red')
    plt.fill_between(train_sizes,train_scores_mean-train_scores_std,train_scores_mean+train_scores_std,alpha=0.1,color='purple')
    plt.plot(train_sizes,test_scores_mean,'o-',color='red',label="Cross-validation Score")
    plt.plot(train_sizes,train_scores_mean,'o-',color='purple',label="Training Score")
    plt.legend()
    plt.savefig(filename)
    plt.clf()

def plot_loss_curve(model, title, filename):
    plt.xlabel("Num Iterations")
    plt.ylabel("Loss")
    plt.title(title)
    plt.plot(model.loss_curve_)
    plt.savefig(filename)
    plt.clf()

def final_plots(run_type, clf, clf_type, cv, score, X_train, y_train, X_test, y_test, y_pred, alg_name, is_iterative):

    print(clf_type + " on scoring method " + score + " Accuracy score: " + str(accuracy_score(y_test, y_pred)))
    print(clf_type + " on scoring method " + score + " f1 score: " + str(f1_score(y_test, y_pred)))
    get_confusion_matrix(clf, X_test, y_test, run_type+' '+alg_name+' Confusion Matrix (' + clf_type + ')',
                         run_type+'_'+alg_name+'_'+clf_type + '_' + score + '_confusion_matrix.png')
    plot_learning_curve(is_iterative, clf, run_type+' '+alg_name+' Learning Curve (' + clf_type + ')', X_train, y_train, cv=cv, scoring=score,
                        filename=run_type+'_'+alg_name+'_'+clf_type + '_' + score + '_Learning_Curve.png')
    if is_iterative:
        plot_loss_curve(clf, run_type+' '+alg_name+' Loss Curve (' + clf_type + ')',
                        run_type+'_'+alg_name+'_'+clf_type + '_' + score + '_Loss_Curve.png')


def rfc_determine_components(run_type, X_train, y_train, X_test, y_test, num_features):
    print("Running RFC")
    model = RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=RANDOM_STATE,n_jobs=-2)
    model.fit(X_train, y_train)
    sorted_features = model.feature_importances_.argsort()
    best_features_first = sorted_features[::-1]
    print(best_features_first)
    scoring_df = pd.DataFrame(columns=['NN accuracy'])
    kwargs = {
        'hidden_layer_sizes':(150,),
        'activation':'tanh'
    }

    for k in range(1, num_features):
        select_k_features = best_features_first[:k].tolist()
        X_Train_new = X_train[select_k_features]
        X_test_new = X_test[select_k_features]
        clf = MLPClassifier(**kwargs)
        print("Fitting for k: "+str(k))
        y_pred = calc_testing(clf, X_Train_new, y_train, X_test_new)
        score = accuracy_score(y_test, y_pred)
        print("Accuracy for k "+str(k)+" is: "+str(score))
        scoring_df.at[k, 'NN accuracy'] = score

    print(scoring_df)
    max_acc = scoring_df.max()
    print("The maximum accuracy was: "+str(max_acc))
    best_feature_length = scoring_df.idxmax()
    print("Best feature length: "+str(best_feature_length))
    best_features = best_features_first[:best_feature_length]
    print("Best features: "+ str(best_features))

import warnings
warnings.filterwarnings("ignore")
def run_DR(run_type, X_train, y_train, X_test, y_test, kwargs, cv):
    scoring = 'accuracy'
    smote = False
    is_iterative = True
    DR_algs = {
        'PCA': PCA(n_components=12, random_state=RANDOM_STATE),
        'ICA': FastICA(n_components=29, random_state=RANDOM_STATE),
        'RP': GaussianRandomProjection(n_components=18, random_state=RANDOM_STATE),
        'RFC': [29,17,5,16,19,13]
    }


    for alg_name, alg in DR_algs.items():
        print("Running "+alg_name)
        if alg_name != 'RFC':
            X_Train_new = alg.fit_transform(X_train)
            X_test_new = alg.fit_transform(X_test)
        else:
            X_Train_new = X_train[alg]
            X_test_new = X_test[alg]

        print("Starting GridSearch")
        start = time.time()
        gs = GridSearchCV(MLPClassifier(), kwargs, cv=cv, n_jobs=-2)
        gs.fit(X_Train_new, y_train)
        end = time.time()
        print('%s execution time: %f secs' % (alg_name, end - start))
        best_nn = gs.best_estimator_
        best_score = gs.best_score_
        best_params = gs.best_params_
        y_pred = calc_testing(best_nn, X_Train_new, y_train, X_test_new)
        print(alg_name+" best training score was: "+str(best_score))
        print(alg_name+" best params: "+str(best_params))

        final_plots(run_type, best_nn, 'NN', cv, scoring, X_Train_new, y_train, X_test_new, y_test, y_pred, alg_name, is_iterative)

def run_and_apply_clustering(run_type, X_train, y_train, X_test, y_test, cv):
    scoring = 'accuracy'
    smote = False
    is_iterative = True
    DR_algs = {
        'PCA': PCA(n_components=12, random_state=RANDOM_STATE),
        'ICA': FastICA(n_components=29, random_state=RANDOM_STATE),
        'RP': GaussianRandomProjection(n_components=18, random_state=RANDOM_STATE),
        'RFC': [29, 17, 5, 16, 19, 13]
    }
    clustering_algs = {
        'kmeans': KMeans(n_clusters=2, random_state=RANDOM_STATE),
        'em': GaussianMixture(n_components=2, random_state=RANDOM_STATE)
    }

    nn_args = {
        'PCA': {'hidden_layer_sizes':[(100,200,400)], 'activation':['tanh']},
        'ICA': {'hidden_layer_sizes':[(100,200,400)], 'activation':['tanh']},
        'RP': {'hidden_layer_sizes':[(100,200,400)], 'activation':['tanh']},
        'RFC':{'hidden_layer_sizes':[(100,200,400)], 'activation':['tanh']}
    }

    for alg_name, alg in DR_algs.items():
        print("Running " + alg_name)
        if alg_name != 'RFC':
            X_Train_new = alg.fit_transform(X_train)
            X_test_new = alg.fit_transform(X_test)
            X_Train_new = pd.DataFrame(X_Train_new, columns=range(0, alg.n_components_))
            X_test_new = pd.DataFrame(X_test_new, columns=range(0, alg.n_components_))
        else:
            X_Train_new = X_train[alg]
            X_test_new = X_test[alg]
        for cluster_alg, runner in clustering_algs.items():
            print("Running clusterer: "+cluster_alg)
            X_train_labels = runner.fit_predict(X_Train_new)
            X_test_labels = runner.fit_predict(X_test_new)
            X_Train_new[alg.n_components_] = X_train_labels.tolist()
            X_test_new[alg.n_components_] = X_test_labels.tolist()

            print("Starting GridSearch")
            start = time.time()
            gs = GridSearchCV(MLPClassifier(), nn_args[alg_name], cv=cv, n_jobs=-2)
            gs.fit(X_Train_new, y_train)
            end = time.time()
            print('%s execution time: %f secs' % (alg_name, end - start))
            best_nn = gs.best_estimator_
            best_score = gs.best_score_
            best_params = gs.best_params_
            y_pred = calc_testing(best_nn, X_Train_new, y_train, X_test_new)
            print(alg_name + " best training score was: " + str(best_score))
            print(alg_name + " best params: " + str(best_params))

            final_plots(run_type, best_nn, 'NN_'+cluster_alg, cv, scoring, X_Train_new, y_train, X_test_new, y_test, y_pred,
                        alg_name, is_iterative)




def run_nn_DR(dataroot):
    smote = (False, 0.6)
    is_rfc = True
    training_sample = 0.3
    run_type = 'FAD'
    cv = 5
    kwargs = {
        'hidden_layer_sizes':[(100,), (150,), (100,200), (100,200,400)],
        'activation':['tanh', 'logistic'],
    }
    X_train, y_train, X_test, y_test, num_features = get_data_ford(dataroot, training_sample, is_rfc)
    #rfc_determine_components(run_type,  X_train, y_train, X_test, y_test, num_features)
    #print("Running dimensionality reduction on FAD dataset")
    #run_DR(run_type, X_train, y_train, X_test, y_test, kwargs, cv)
    print("Running clustering and applying results as new features")
    run_and_apply_clustering(run_type, X_train, y_train, X_test, y_test, cv)

if __name__ == "__main__":
    passed_arg = sys.argv[1]
    if passed_arg.startswith('/'):
        dataroot = passed_arg
    else:
        dataroot = '/Users/plamb/Documents/Personal/Academic/Georgia Tech/Classes/ML/hw/dimensionality_reduction/data/'
    if passed_arg == 'nn':
        run_nn_DR(dataroot)
    # elif passed_arg == 'ford':
    #     run_ford(dataroot)
    else:
        print("please run with an absolute path to the data")
        exit(146)