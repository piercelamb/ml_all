import sys
import pandas as pd
import numpy as np
import matplotlib
from scipy import sparse
from scipy import linalg
from time import time
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
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
from sklearn.ensemble import AdaBoostClassifier
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
def get_smote_pipeline(smote):
    smote_sample = smote[1]
    over = SMOTE(sampling_strategy=smote_sample, random_state=RANDOM_STATE)
    steps = [('over', over)]
    if len(smote) == 3:
        under_sample = smote[2]
        under = RandomUnderSampler(sampling_strategy=under_sample, random_state=RANDOM_STATE)
        steps.append(('under', under))

    return imbalancePipeline(steps=steps)

class GaussianMixtureCluster(GaussianMixture, ClusterMixin):
    """Subclass of GaussianMixture to make it a ClusterMixin."""

    def fit(self, X):
        super().fit(X)
        self.labels_ = self.predict(X)
        return self

    def get_params(self, **kwargs):
        output = super().get_params(**kwargs)
        output["n_clusters"] = output.get("n_components", None)
        return output

    def set_params(self, **kwargs):
        kwargs["n_components"] = kwargs.pop("n_clusters", None)
        return super().set_params(**kwargs)

def run_clustering_algs(run_type, k_clusters, metrics, X):
    clustering_algs = [
        'kmeans',
         'em'
    ]
    k_results = []
    for alg in clustering_algs:
        print("**************************")
        print("Running "+alg)
        for metric in metrics:
            print("Using metric: "+metric)
            if alg == 'kmeans':
                #clusterer = KMeans(n_clusters=k, random_state=RANDOM_STATE)
                visualizer = KElbowVisualizer(KMeans(random_state=RANDOM_STATE), k=k_clusters, metric=metric)
            else:
                #clusterer = mixture.GaussianMixture(n_components=k, random_state=RANDOM_STATE)
                visualizer = KElbowVisualizer(GaussianMixtureCluster(random_state=RANDOM_STATE), k=k_clusters, metric=metric, force_model=True)
            print("Fitting "+metric)
            visualizer.fit(X)
            best_k = visualizer.elbow_value_
            best_k_score = visualizer.elbow_score_
            k_results.append(best_k)
            print("Metric "+metric+" best k: "+str(best_k)+" with score "+str(best_k_score))
            visualizer.finalize()
            visualizer.show(outpath=run_type+"_"+alg+"_"+metric+"_kelbow.png")
            plt.clf()


    data = Counter(k_results)
    print(k_results)
    print(data.most_common())  # Returns all unique items and their counts
    best_k_overall = data.most_common(1)[0][0]  # Returns the highest occurring item
    print("The most chosen k is: "+str(best_k_overall))
    for alg in clustering_algs:
        print("plotting InterClusterDistance for: " + alg)
        if alg == 'kmeans':
            model = KMeans(best_k_overall, random_state=RANDOM_STATE)
            viz = InterclusterDistance(model)
            viz.fit(X)
            viz.finalize()
            viz.show(outpath=run_type + "_" + alg + "_interclusterdistance.png")
            plt.clf()
            viz = SilhouetteVisualizer(model)
            viz.fit(X)
            viz.finalize()
            viz.show(outpath=run_type + "_" + alg + "_silhouetteViz.png")
            plt.clf()





def get_data_shoppers(dataroot, smote, scaler):
    run_type = 'OSI'
    datapath = dataroot + 'online_shoppers_intention.csv'
    df = pd.read_csv(datapath)
    target = df['Revenue'].astype(int)
    attributes = df.drop('Revenue', axis=1, )
    string_columns = ['Month', 'VisitorType', 'Weekend']
    numerical_columns = ['Administrative', 'Administrative_Duration', 'Informational',
       'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
       'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay',
       'OperatingSystems', 'Browser', 'Region', 'TrafficType',]
    column_trans = make_column_transformer(
        (OneHotEncoder(handle_unknown='ignore'), string_columns),
        (RobustScaler(), numerical_columns)
         #remainder='passthrough'
    )
    clean_attrs = column_trans.fit_transform(attributes)
    num_features = len(clean_attrs[0])
    print("OSI has " + str(num_features) + " features after preprocessing")
    clean_attrs = pd.DataFrame(clean_attrs, columns=range(0, num_features))
    # scaler = RobustScaler()
    # clean_attrs = scaler.fit_transform(clean_attrs)

    # X_train, X_test, y_train, y_test = train_test_split(clean_attrs, target, stratify=target, test_size=0.3,
    #                                                     random_state=RANDOM_STATE)
    # is_smote = smote[0]
    # if is_smote:
    #     print("Smote resampling initiated, Counter before: "+str(Counter(y_train)))
    #     imbPipeline = get_smote_pipeline(smote)
    #     X_train, y_train = imbPipeline.fit_resample(X_train, y_train)
    #     print("Smote resampling complete, Counter after: "+str(Counter(y_train)))
    return run_type, clean_attrs, target

def drop_correlated_columns(X_train):
    # correlation = X_train.corr().abs()
    # get_graph = c.unstack()
    # final = get_graph.order(kind="quicksort", na_last=False)[::-1]
    X_train = X_train.drop(['P4', 'V6', 'V10', 'E9', 'E2'], axis=1)
    return X_train

def scale_variant_columns(X_train):

    X_train.loc[X_train['E7'] > 4, 'E7'] = 4
    X_train.loc[X_train['E8'] > 4, 'E8'] = 4

    return X_train

def scale_continuous_data(X_train):
    categorical = ['E3', 'E7', 'E8', 'V5']
    numerical = [x for x in X_train.columns if x not in categorical]
    scaler = preprocessing.RobustScaler().fit(
        X_train.loc[:, numerical])
    X_train.loc[:, numerical] = scaler.transform(
        X_train.loc[:, numerical])

    return X_train

def preprocess_ford_data(X_train):
    X_train = drop_correlated_columns(X_train)
    X_train = scale_variant_columns(X_train)
    X_train = scale_continuous_data(X_train)

    return X_train

import warnings
warnings.filterwarnings("ignore")
def get_data_ford(dataroot, dataset_sample, scaler):
    run_type = 'FAD'
    df_train = pd.read_csv(dataroot + 'fordTrain.csv')
    y_train = df_train['IsAlert']
    X_train = df_train.drop(['IsAlert', 'TrialID', 'ObsNum'], axis=1)
    if dataset_sample != 0:
        print("Sampling the training set at: " + str(dataset_sample))
        X_train, X_test_new, y_train, y_test_new = train_test_split(X_train, y_train, train_size=dataset_sample,
                                                                    random_state=RANDOM_STATE)
    print("Sampled number of instances: " + str(len(X_train.index)))
    print("Scaling data to range 0 -> 1")
    # scaler = MaxAbsScaler()
    # X_train = scaler.fit_transform(X_train)
    X_train = scale_continuous_data(X_train)
    num_features = len(X_train.columns)
    print("FAD has " + str(num_features) + " features after preprocessing")
    X_train = X_train.set_axis(range(0, num_features), axis=1)

    return run_type, X_train, y_train

# def run_clustering(dataroot):
#     smote = (False, 0.6)
#     training_sample = 0.3
#     k_clusters = (2,25)
#     metrics = ['distortion', 'silhouette', 'calinski_harabasz']
#     print("Running clustering on OCI dataset")
#     run_type, X_train, y_train = get_data_shoppers(dataroot, smote)
#     run_clustering_algs(run_type, k_clusters, metrics, X_train)
#     print("\n----------------------------------\n")
#     print("Running clustering on FAD dataset")
#     run_type, X_train, y_train = get_data_ford(dataroot, training_sample)
#     run_clustering_algs(run_type, k_clusters, metrics, X_train


def bench_k_means(alg, name, data, labels, df):
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
    estimator = make_pipeline(StandardScaler(), alg).fit(data)
    fit_time = time.time() - t0
    results = [name, fit_time]
    labels_pred = estimator.predict(data)

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


def dimensionality_reduction(run_type, explained_variance, X_train, y_train, num_clusters, num_components):
    algs = {
        'kmeans': KMeans(random_state=RANDOM_STATE),
        'em': GaussianMixture(random_state=RANDOM_STATE)
    }
    for alg_name, alg in algs.items():
        scoring_df = pd.DataFrame(columns=['homo','compl','v-meas','RI','AMI'])
        print(82 * '_')
        print('run\t\ttime\thomo\tcompl\tv-meas\tRI\tAMI')
        alg.set_params(n_clusters=num_clusters) if alg_name == 'kmeans' else alg.set_params(n_components=num_components)
        bench_k_means(alg=alg, name=alg_name, data=X_train, labels=y_train, df=scoring_df)

        n = 14 if run_type == 'OSI' else 12
        pca_trans = PCA(n_components=n, random_state=RANDOM_STATE).fit_transform(X_train)
        alg.set_params(n_clusters=num_clusters) if alg_name == 'kmeans' else alg.set_params(n_components=num_components)
        bench_k_means(alg=alg, name="PCA", data=pca_trans, labels=y_train, df=scoring_df)

        n = 26 if run_type == 'OSI' else 29
        ica_trans = FastICA(n_components=n, random_state=RANDOM_STATE).fit_transform(X_train)
        alg.set_params(n_clusters=num_clusters) if alg_name == 'kmeans' else alg.set_params(n_components=num_components)
        bench_k_means(alg=alg, name="ICA", data=ica_trans, labels=y_train, df=scoring_df)

        n = 23 if run_type == 'OSI' else 18
        rp_trans = GaussianRandomProjection(n_components=n, random_state=RANDOM_STATE).fit_transform(X_train)
        alg.set_params(n_clusters=num_clusters) if alg_name == 'kmeans' else alg.set_params(n_components=num_components)
        bench_k_means(alg=alg, name="RP", data=rp_trans, labels=y_train, df=scoring_df)

        if alg == 'kmeans':
            if run_type == 'OSI':
                X_new = X_train[[23]]
            else:
                X_new = X_train[[29, 17, 5, 19, 14, 13, 15, 16]]
        else:
            if run_type == 'OSI':
                X_new = X_train[[23, 22]]
            else:
                X_new = X_train[[29, 17, 5, 19, 14, 13, 15, 16, 6, 4, 24, 12, 0, 22, 28]]
        alg.set_params(n_clusters=num_clusters) if alg_name == 'kmeans' else alg.set_params(n_components=num_components)
        bench_k_means(alg=alg, name="RFC", data=X_new, labels=y_train, df=scoring_df)

        print(82 * '_')
        plot = scoring_df.plot(kind='bar', rot=0, ylabel="Score", title=run_type + " " + alg_name + " DR Scoring")
        fig = plot.get_figure()
        fig.savefig('bar_chart/'+run_type + "_" + alg_name + "_scoring_bar_chart.png")
        plt.clf()

def run_dim_reduction(dataroot):
    scaler = RobustScaler()
    smote = (False, 0.6)
    training_sample = 0.3
    explained_variance = 0.95
    num_clusters = 2
    n_components = 2
    print("Running dimensionality reduction on OCI dataset")
    run_type, X_train, y_train = get_data_shoppers(dataroot, smote, scaler)
    dimensionality_reduction(run_type, explained_variance, X_train, y_train, num_clusters, n_components)
    print("\n----------------------------------\n")
    print("Running dimensionality reduction on FAD dataset with num clusters: "+str(num_clusters))
    run_type, X_train, y_train = get_data_ford(dataroot, training_sample, scaler)
    dimensionality_reduction(run_type, explained_variance, X_train, y_train, num_clusters, n_components)



if __name__ == "__main__":
    passed_arg = sys.argv[1]
    if passed_arg.startswith('/'):
        dataroot = passed_arg
    else:
        dataroot = '/Users/plamb/Documents/Personal/Academic/Georgia Tech/Classes/ML/hw/dimensionality_reduction/data/'
    if passed_arg == 'clustering':
        run_clustering(dataroot)
    if passed_arg == 'dr':
        run_dim_reduction(dataroot)
    # elif passed_arg == 'ford':
    #     run_ford(dataroot)
    else:
        print("please run with an absolute path to the data")
        exit(146)