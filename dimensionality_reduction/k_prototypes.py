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
# Import module for k-protoype cluster
from kmodes.kprototypes import KPrototypes
# Ignore warnings
import warnings
warnings.filterwarnings('ignore', category = FutureWarning)
# Format scientific notation from Pandas
pd.set_option('display.float_format', lambda x: '%.3f' % x)
# Import module for data visualization
from plotnine import *
import plotnine
from sklearn import metrics
from sklearn.pipeline import make_pipeline

RANDOM_STATE = 1337

def bench_k_means(alg, name, data, labels, df, cat):
    """Benchmark to evaluate the KMeans initialization methods.

    Parameters
    ----------
    kmeans : KMeans instance
        A :class:`~sklearn.cluster.KMeans` instance with the initialization
        already set.
    name : str
        Name given to the strategy. It will be used to show the results in a
        table.
    data : ndarray of shape (n_samples, n_features)
        The data to cluster.
    labels : ndarray of shape (n_samples,)
        The labels used to compute the clustering metrics which requires some
        supervision.
    """
    t0 = time.time()
    estimator = alg.fit(data, categorical=cat)
    fit_time = time.time() - t0
    results = [name, fit_time]
    labels_pred = estimator.predict(data, categorical=cat)

    # Define the metrics which require only the true labels and estimator
    # labels
    clustering_metrics = {
        'homo':metrics.homogeneity_score,
        'compl':metrics.completeness_score,
        'v-meas':metrics.v_measure_score,
        'RI':metrics.rand_score,
        'AMI':metrics.adjusted_mutual_info_score,
    }
    for scoring, m in clustering_metrics.items():
        score = m(labels, labels_pred)
        df.at[name, scoring] = score

    results += [m(labels, labels_pred) for scoring, m in clustering_metrics.items()]


    # The silhouette score requires the full dataset
    # results += [
    #     metrics.silhouette_score(data, labels_pred,
    #                              metric="euclidean", sample_size=300,)
    # ]

    # Show the results
    formatter_result = ("{:9s}\t{:.3f}s\t{:.3f}\t{:.3f}"
                        "\t{:.3f}\t{:.3f}\t{:.3f}")
    print(formatter_result.format(*results))


def preprocess_shoppers(dataroot):
    run_type = 'OSI'
    datapath = dataroot + 'online_shoppers_intention.csv'
    df = pd.read_csv(datapath)
    y_train = df['Revenue'].astype(int)
    X_train = df.drop('Revenue', axis=1)
    string_columns = ['Month', 'VisitorType', 'Weekend']
    string_cleanup = {
        "Month": {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "June": 6, "Jul": 7, "Aug":8, "Sep":9, "Oct":10, "Nov":11, "Dec":12},
        "VisitorType":{"Returning_Visitor":1, "New_Visitor":2, "Other":3},
    }
    #convert strings to nums
    X_train.replace(string_cleanup, inplace=True)
    X_train['Weekend'] = X_train['Weekend'].astype(int)
    categorical_cols = [
        'Administrative', 'Informational', 'ProductRelated', 'SpecialDay', 'Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend'
    ]
    numerical = [x for x in X_train.columns if x not in categorical_cols]
    scaler = preprocessing.MinMaxScaler().fit(
        X_train.loc[:, numerical])
    X_train.loc[:, numerical] = scaler.transform(
        X_train.loc[:, numerical])
    catColumnsPos = [X_train.columns.get_loc(col) for col in categorical_cols]
    print('Categorical columns           : {}'.format(categorical_cols))
    print('Categorical columns position  : {}'.format(catColumnsPos))
    #print(X_train.info())
    #exit(1)
    X_train = X_train.to_numpy()
    #calc_prototypes(X_train, catColumnsPos)
    benchmark_prototypes(X_train, y_train, catColumnsPos)

def benchmark_prototypes(X_train, y_train, catColumnsPos):
    scoring_df = pd.DataFrame(columns=['homo', 'compl', 'v-meas', 'RI', 'AMI'])
    print(82 * '_')
    print('run\t\ttime\thomo\tcompl\tv-meas\tRI\tAMI')
    alg = KPrototypes(n_jobs = -1, n_clusters=2, init='Huang', random_state=RANDOM_STATE)
    bench_k_means(alg=alg, name='k_prototypes', data=X_train, labels=y_train, df=scoring_df, cat=catColumnsPos)
    print(82 * '_')

def preprocess_ford(dataroot, dataset_sample):
    run_type = 'FAD'
    df_train = pd.read_csv(dataroot + 'fordTrain.csv')
    y_train = df_train['IsAlert']
    X_train = df_train.drop(['IsAlert', 'TrialID', 'ObsNum'], axis=1)
    if dataset_sample != 0:
        print("Sampling the training set at: "+str(dataset_sample))
        X_train, X_test_new, y_train, y_test_new = train_test_split(X_train, y_train, train_size=dataset_sample,
                                                            random_state=RANDOM_STATE)
    print("Sampled number of instances: "+str(len(X_train.index)))
    X_train = X_train.drop(['P8', 'V7', 'V9'], axis=1) #drop because single value
    categorical_cols = [
        'P3','P4','P6','P7','E4','E5','E6','E3', 'E7', 'E8', 'E9', 'E10','E11','V2','V3','V4','V5','V10'
    ]
    catColumnsPos = [X_train.columns.get_loc(col) for col in categorical_cols]
    print('Categorical columns           : {}'.format(categorical_cols))
    print('Categorical columns position  : {}'.format(catColumnsPos))
    X_train = X_train.to_numpy()
    benchmark_prototypes(X_train, y_train, catColumnsPos)

def calc_prototypes(dfMatrix, catColumnsPos):
    # Choose optimal K using Elbow method
    cost = []
    for cluster in range(1, 5):
        print("Trying k=" + str(cluster))
        kprototype = KPrototypes(n_jobs=-1, n_clusters=cluster, init='Huang', random_state=RANDOM_STATE)
        kprototype.fit_predict(dfMatrix, categorical=catColumnsPos)
        cost.append(kprototype.cost_)
        print('Cluster initiation: {}'.format(cluster))

    #Converting the results into a dataframe and plotting them
    df_cost = pd.DataFrame({'Cluster': range(1, len(cost)+1), 'Cost': cost})

    # Data viz
    print("Plotting elbow")
    plotnine.options.figure_size = (8, 4.8)
    (
            ggplot(data=df_cost) +
            geom_line(aes(x='Cluster',
                          y='Cost')) +
            geom_point(aes(x='Cluster',
                           y='Cost')) +
            geom_label(aes(x='Cluster',
                           y='Cost',
                           label='Cluster'),
                       size=10,
                       nudge_y=1000) +
            labs(title='Optimal number of cluster with Elbow Method') +
            xlab('Number of Clusters k') +
            ylab('Cost') +
            theme_bw()

    ).save(filename='lol.png', format='png')

def run_clustering(dataroot):
    training_sample = 0.3
    print("Calculating elbow for shoppers")
    preprocess_shoppers(dataroot)#
    #print("Caluclating elbow for ford")
    #preprocess_ford(dataroot, training_sample)

if __name__ == "__main__":
    passed_arg = sys.argv[1]
    if passed_arg.startswith('/'):
        dataroot = passed_arg
    else:
        dataroot = '/Users/plamb/Documents/Personal/Academic/Georgia Tech/Classes/ML/hw/dimensionality_reduction/data/'
    if passed_arg == 'clustering':
        run_clustering(dataroot)
    # if passed_arg == 'dr':
    #     run_dim_reduction(dataroot)
    # elif passed_arg == 'ford':
    #     run_ford(dataroot)
    else:
        print("please run with an absolute path to the data")
        exit(146)