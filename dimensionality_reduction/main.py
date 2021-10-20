import sys
import pandas as pd
import numpy as np
import matplotlib
from sklearn.decomposition import PCA, FastICA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from numpy import mean
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn import tree as dt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, v_measure_score, homogeneity_completeness_v_measure
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





def get_data_shoppers(dataroot, smote):
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
        (MinMaxScaler(), numerical_columns)
        # remainder='passthrough'
    )
    clean_attrs = column_trans.fit_transform(attributes)

    # X_train, X_test, y_train, y_test = train_test_split(clean_attrs, target, stratify=target, test_size=0.3,
    #                                                     random_state=RANDOM_STATE)
    # is_smote = smote[0]
    # if is_smote:
    #     print("Smote resampling initiated, Counter before: "+str(Counter(y_train)))
    #     imbPipeline = get_smote_pipeline(smote)
    #     X_train, y_train = imbPipeline.fit_resample(X_train, y_train)
    #     print("Smote resampling complete, Counter after: "+str(Counter(y_train)))
    return run_type, clean_attrs, target

def get_data_ford(dataroot, dataset_sample):
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
    scaler = MinMaxScaler()
    X_train_rescaled = scaler.fit_transform(X_train)
    return run_type, X_train_rescaled, y_train

def run_clustering(dataroot):
    smote = (False, 0.6)
    training_sample = 0.3
    k_clusters = (2,25)
    metrics = ['distortion', 'silhouette', 'calinski_harabasz']
    print("Running clustering on OCI dataset")
    run_type, X_train, y_train = get_data_shoppers(dataroot, smote)
    run_clustering_algs(run_type, k_clusters, metrics, X_train)
    print("\n----------------------------------\n")
    print("Running clustering on FAD dataset")
    run_type, X_train, y_train = get_data_ford(dataroot, training_sample)
    run_clustering_algs(run_type, k_clusters, metrics, X_train)

def pca_determine_components(run_type, explained_variance, X_train):
    pca = PCA().fit(X_train)

    fig, ax = plt.subplots()
    y = np.cumsum(pca.explained_variance_ratio_)
    y_len = len(y)
    xi = np.arange(1, y_len + 1, step=1)

    plt.ylim(0.0, 1.1)
    plt.plot(xi, y, marker='o', linestyle='--', color='b')

    plt.xlabel('Number of Components')
    plt.xticks(np.arange(0, y_len + 1, step=1))  # change from 0-based array index to 1-based human-readable label
    plt.ylabel('Cumulative variance (%)')
    plt.title('The number of components needed to explain variance')

    plt.axhline(y=explained_variance, color='r', linestyle='-')
    plt.text(0.5, 0.85, str(explained_variance) + ' cut-off threshold', color='red', fontsize=16)

    ax.grid(axis='x')
    plt.savefig(run_type + "_pca_optimization.png")
    plt.clf()
    exit(1)

def ica_determine_components(run_type, X_train, y_train):
    n_components_list = range (1, 31)
    fastICA = FastICA(random_state=RANDOM_STATE)
    kurtosis = {}
    for n in n_components_list:
        print("Testing components: "+str(n))
        fastICA.set_params(n_components=n)
        ica_results = fastICA.fit_transform(X_train)
        df = pd.DataFrame(ica_results)
        df = df.kurt(axis=0)
        avg_kurt = df.abs().mean()
        print("Average kurtosis: "+str(avg_kurt))
        kurtosis[n] = avg_kurt

    max_n = max(kurtosis, key=kurtosis.get)
    print("The max kurtosis is: "+str(kurtosis[max_n])+" with n of: "+str(max_n))

    kurtosis = pd.Series(kurtosis)
    plot = kurtosis.plot(
        xlabel="Num of Components", ylabel="Average Kurtosis", title=run_type+" Kurtosis over Independent Components"
    )
    fig = plot.get_figure()
    fig.savefig(run_type+"_ICA_kurtosis_components.png")
    plt.clf()

def compare_labelings(dr_type, run_type, X_train, y_train):
    algs = ['kmeans', 'em']
    true_labels = y_train.to_numpy()
    kmeans = KMeans(n_clusters=2, random_state=RANDOM_STATE)
    em = GaussianMixture(n_components=2, random_state=RANDOM_STATE)
    for alg in algs:
        if dr_type == 'PCA':
            pca_trans_data = PCA(n_components=15).fit_transform(X_train)
            if alg == 'kmeans':
                print("Testing PCA on "+alg)
                kmeans_fitted = kmeans.fit(X_train)
                non_pca_labels = kmeans_fitted.labels_
                PCA_kmeans_fitted = kmeans.fit(pca_trans_data)
                pca_labels = PCA_kmeans_fitted.labels_
            else:
                print("Testing PCA on " + alg)
                em_fitted = em.fit(X_train)
                non_pca_labels = em_fitted.predict(X_train)
                PCA_em_fitted = em.fit(pca_trans_data)
                pca_labels = PCA_em_fitted.predict(pca_trans_data)
            print(homogeneity_completeness_v_measure(non_pca_labels, true_labels))
            print(homogeneity_completeness_v_measure(pca_labels, true_labels))
        if dr_type == 'ICA':
            n = 26 if run_type == 'OSI' else 29
            ica_trans_data = FastICA(n_components=n).fit_transform(X_train)
            if alg == 'kmeans':
                print("Testing ICA on " + alg)
                kmeans = KMeans(n_clusters=2, random_state=RANDOM_STATE)
                kmeans_fitted = kmeans.fit(X_train)
                non_ica_labels = kmeans_fitted.labels_
                ICA_kmeans_fitted = kmeans.fit(ica_trans_data)
                ica_labels = ICA_kmeans_fitted.labels_
            else:
                print("Testing ICA on " + alg)
                em_fitted = em.fit(X_train)
                non_ica_labels = em_fitted.predict(X_train)
                ICA_em_fitted = em.fit(ica_trans_data)
                ica_labels = ICA_em_fitted.predict(ica_trans_data)
            print(homogeneity_completeness_v_measure(non_ica_labels, true_labels))
            print(homogeneity_completeness_v_measure(ica_labels, true_labels))

                # viz = InterclusterDistance(kmeans)
                # viz.fit(X_train)
                # viz.finalize()
                # viz.show(outpath=run_type + "_kmeans_no_PCA_interclusterdistance.png")
                # plt.clf()
                #
                # viz = InterclusterDistance(PCA_kmeans)
                # viz.fit(pca_trans_data)
                # viz.finalize()
                # viz.show(outpath=run_type + "_kmeans_" + dr_type + "_interclusterdistance.png")



def dimensionality_reduction(run_type, explained_variance, X_train, y_train):
    print("Running PCA")
    # pca_determine_components(run_type, explained_variance, X_train)
    compare_labelings('PCA', run_type, X_train, y_train)
    #print("Running ICA")
    #ica_determine_components(run_type, X_train, y_train)
    #compare_labelings('ICA', run_type, X_train, y_train)
    #print("Running RP")
    #run_RP()
    #print("Running LDA")
    #run_LDA()

def run_dim_reduction(dataroot):
    smote = (False, 0.6)
    training_sample = 0
    explained_variance = 0.95
    print("Running dimensionality reduction on OCI dataset")
    run_type, X_train, y_train = get_data_shoppers(dataroot, smote)
    dimensionality_reduction(run_type, explained_variance, X_train, y_train)
    print("\n----------------------------------\n")
    print("Running dimensionality reduction on FAD dataset")
    run_type, X_train, y_train = get_data_ford(dataroot, training_sample)
    dimensionality_reduction(run_type, explained_variance, X_train, y_train)



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