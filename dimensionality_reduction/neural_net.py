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

def calc_testing(clf, X_train, y_train, X_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred

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

def run_nn_DR(dataroot):
    smote = (False, 0.6)
    is_rfc = True
    training_sample = 0.3
    run_type = 'FAD'
    X_train, y_train, X_test, y_test, num_features = get_data_ford(dataroot, training_sample, is_rfc)
    rfc_determine_components(run_type,  X_train, y_train, X_test, y_test, num_features)
    print("Running dimensionality reduction on FAD dataset")
    #
    print("Applying reduced dataset to Neural Net")

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