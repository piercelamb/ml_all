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





def get_data_shoppers(dataroot, smote, is_rfc):
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
    num_features = len(clean_attrs[0])
    print("OSI has "+str(num_features)+" features after preprocessing")
    if is_rfc:
        clean_attrs = pd.DataFrame(clean_attrs, columns=range(0, num_features))
    # X_train, X_test, y_train, y_test = train_test_split(clean_attrs, target, stratify=target, test_size=0.3,
    #                                                     random_state=RANDOM_STATE)
    # is_smote = smote[0]
    # if is_smote:
    #     print("Smote resampling initiated, Counter before: "+str(Counter(y_train)))
    #     imbPipeline = get_smote_pipeline(smote)
    #     X_train, y_train = imbPipeline.fit_resample(X_train, y_train)
    #     print("Smote resampling complete, Counter after: "+str(Counter(y_train)))
    return run_type, clean_attrs, target, num_features

def get_data_ford(dataroot, dataset_sample, is_rfc):
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
    num_features = len(X_train_rescaled[0])
    print("FAD has " + str(num_features) + " features after preprocessing")
    if is_rfc:
        X_train_rescaled = pd.DataFrame(X_train_rescaled, columns=range(0, num_features))
    return run_type, X_train_rescaled, y_train, num_features

def run_clustering(dataroot):
    smote = (False, 0.6)
    training_sample = 0.3
    k_clusters = (2,25)
    metrics = ['distortion', 'silhouette', 'calinski_harabasz']
    print("Running clustering on OCI dataset")
    run_type, X_train, y_train, num_features = get_data_shoppers(dataroot, smote)
    run_clustering_algs(run_type, k_clusters, metrics, X_train)
    print("\n----------------------------------\n")
    print("Running clustering on FAD dataset")
    run_type, X_train, y_train, num_features = get_data_ford(dataroot, training_sample)
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

def rp_determine_components(run_type, X_train, y_train):
    n_components = range(2, 30)
    seeds = range(1,11)

    RPs = ['sparse', 'gaussian']
    sparse_df = pd.DataFrame(index=n_components)
    gaussian_df = pd.DataFrame(index=n_components)
    print("Exexcuting seeds and components")
    for component in n_components:
        for seed in seeds:
            for rp in RPs:
                if rp == 'sparse':
                    model = SparseRandomProjection(random_state=seed, n_components=component)
                    df = sparse_df
                else:
                    model = GaussianRandomProjection(n_components=component, random_state=seed)
                    df = gaussian_df
                model.fit(X_train)
                #get reconstruction error
                components = model.components_
                if sparse.issparse(components):
                    components = components.todense()
                inverse = linalg.pinv(components)
                recons = np.matmul(np.matmul(inverse,components),X_train.T).T
                error_distance = np.square(X_train-recons)
                mean_error = np.nanmean(error_distance)
                df.at[component, 'seed'+str(seed)] = mean_error

    sparse_df['average_error'] = sparse_df.mean(axis=1)
    gaussian_df['average_error'] = gaussian_df.mean(axis=1)
    plot = sparse_df.plot(y='average_error', ylabel="Average Reconstruction Error", xlabel="Number of Components", title="Reconstruction Error over Components" )
    plt.axhline(y=0.1, color='r', linestyle='-')
    plt.text(0.5, 0.85, str(0.1) + ' cut-off threshold', color='red', fontsize=16)
    fig = plot.get_figure()
    fig.savefig(run_type+"_RP_sparse_components.png")
    plt.clf()
    plot = gaussian_df.plot(y='average_error', ylabel="Average Reconstruction Error",
                          xlabel="Number of Components", title="Reconstruction Error over Components")
    plt.axhline(y=0.1, color='r', linestyle='-')
    plt.text(0.5, 0.85, str(0.1) + ' cut-off threshold', color='red', fontsize=16)
    fig = plot.get_figure()
    fig.savefig(run_type+"_RP_gaussian_components.png")
    plt.clf()

def compare_labelings(dr_type, run_type, X_train, y_train):
    algs = ['kmeans', 'em']
    scoring_methods = ['adj_rand_index', 'adj_mutual_info', 'homogeneity', 'completeness', 'v_measure', 'FM']
    true_labels = y_train.to_numpy()
    kmeans = KMeans(n_clusters=2, random_state=RANDOM_STATE)
    em = GaussianMixture(n_components=2, random_state=RANDOM_STATE)
    scoring_df = pd.DataFrame()
    for alg in algs:
        if dr_type == 'PCA':
            n = 14 if run_type == 'OSI' else 12
            pca_trans_data = PCA(n_components=n, random_state=RANDOM_STATE).fit_transform(X_train)
            if alg == 'kmeans':
                print("Testing PCA on "+alg)
                non_dr_labels = kmeans.fit_predict(X_train)
                dr_labels = kmeans.fit_predict(pca_trans_data)
            else:
                print("Testing PCA on " + alg)
                non_dr_labels = em.fit_predict(X_train)
                dr_labels = em.fit_predict(pca_trans_data)
        if dr_type == 'ICA':
            n = 26 if run_type == 'OSI' else 29
            ica_trans_data = FastICA(n_components=n, random_state=RANDOM_STATE).fit_transform(X_train)
            if alg == 'kmeans':
                print("Testing ICA on " + alg)
                non_dr_labels = kmeans.fit_predict(X_train)
                dr_labels = kmeans.fit_predict(ica_trans_data)
            else:
                print("Testing ICA on " + alg)
                non_dr_labels = em.fit_predict(X_train)
                dr_labels = em.fit_predict(ica_trans_data)
        if dr_type == 'RP':
            n = 6 if run_type == 'OSI' else 12
            rp_trans_data = GaussianRandomProjection(n_components=n, random_state=RANDOM_STATE).fit_transform(X_train)
            if alg == 'kmeans':
                print("Testing RP on " + alg)
                non_dr_labels = kmeans.fit_predict(X_train)
                dr_labels = kmeans.fit_predict(rp_trans_data)
            else:
                print("Testing RP on " + alg)
                non_dr_labels = em.fit_predict(X_train)
                dr_labels = em.fit_predict(rp_trans_data)
        if dr_type == 'RFC':
            pass
        else:
            pass


        for i, scoring in enumerate(scoring_methods):
            if scoring == 'adj_rand_index':
                no_DR = adjusted_rand_score(true_labels, non_dr_labels)
                DR = adjusted_rand_score(true_labels,dr_labels)
            elif scoring == 'adj_mutual_info':
                no_DR = adjusted_mutual_info_score(true_labels, non_dr_labels)
                DR = adjusted_mutual_info_score(true_labels,dr_labels)
            elif scoring == 'homogeneity':
                no_DR = homogeneity_score(true_labels, non_dr_labels)
                DR = homogeneity_score(true_labels,dr_labels)
            elif scoring == 'completeness':
                no_DR = completeness_score(true_labels, non_dr_labels)
                DR = completeness_score(true_labels, dr_labels)
            elif scoring == 'v_measure':
                no_DR = v_measure_score(true_labels, non_dr_labels)
                DR = v_measure_score(true_labels, dr_labels)
            elif scoring == 'FM':
                no_DR = fowlkes_mallows_score(true_labels, non_dr_labels)
                DR = fowlkes_mallows_score(true_labels, dr_labels)


            scoring_df.at[scoring, alg+'_no_DR'] = no_DR
            scoring_df.at[scoring, alg+'_DR'] = DR
            # print(run_type + " " + scoring + " for " + alg + " with no DR: " + str(no_DR))
            # print(run_type + " " + scoring + " for " + alg + " using " + dr_type + ": " + str(DR))
    print(scoring_df)
    plot = scoring_df.plot(kind='bar', rot=0, ylabel="Score", title=run_type+" "+dr_type+" Scoring")
    fig = plot.get_figure()
    fig.savefig(run_type+"_"+dr_type+"_scoring_bar_chart.png")
    plt.clf()
def rfc_determine_components(run_type, X_train, y_train, num_features):
    model = RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=RANDOM_STATE,n_jobs=-2)
    model.fit(X_train, y_train)
    sorted_features = model.feature_importances_.argsort()
    best_features_first = sorted_features[::-1]
    algs = [
        'kmeans',
        'em'
    ]
    from sklearn import metrics
    clustering_metrics = {
        'homo': metrics.homogeneity_score,
        'compl': metrics.completeness_score,
        'v-meas': metrics.v_measure_score,
        'RI': metrics.rand_score,
        'AMI': metrics.adjusted_mutual_info_score,
    }
    scoring_df = pd.DataFrame()
    for alg in algs:
        print("Running "+alg)
        for k in range(1, num_features):
            select_k_features = best_features_first[:k].tolist()
            X_new = X_train[select_k_features]
            if alg == 'kmeans':
                labels_pred = KMeans(n_clusters=2, random_state=RANDOM_STATE).fit_predict(X_new)
            if alg == 'em':
                labels_pred = GaussianMixture(n_components=2, random_state=RANDOM_STATE).fit_predict(X_new)
            for scoring, m in clustering_metrics.items():
                score = m(y_train, labels_pred)
                scoring_df.at[k, scoring] = score
        best_features_df = scoring_df.idxmax().to_frame().T
        best_feature_length = best_features_df.mode(axis=1).at[0, 0]
        best_features = best_features_first[:best_feature_length]
        print(run_type+" "+alg+" best feature length: "+str(best_feature_length))
        print(run_type + " " + alg + " best features: " + str(best_features))
        print(run_type + " " + alg + " scores for best features: ")
        print(scoring_df.iloc[best_feature_length])



def dimensionality_reduction(run_type, explained_variance, X_train, y_train, num_features):
    #print("Running PCA")
    # pca_determine_components(run_type, explained_variance, X_train)
    #compare_labelings('PCA', run_type, X_train, y_train)
    #print("Running ICA")
    #ica_determine_components(run_type, X_train, y_train)
    #compare_labelings('ICA', run_type, X_train, y_train)
    #print("Running RP")
    #rp_determine_components(run_type, X_train, y_train)
    #compare_labelings('RP', run_type, X_train, y_train)
    print("Running RFC")
    rfc_determine_components(run_type, X_train, y_train, num_features)
    #compare_labelings('RFC', run_type, X_train, y_train)

def run_dim_reduction(dataroot):
    smote = (False, 0.6)
    is_rfc = True
    training_sample = 0.3
    explained_variance = 0.95
    print("Running dimensionality reduction on OCI dataset")
    run_type, X_train, y_train, num_features = get_data_shoppers(dataroot, smote, is_rfc)
    dimensionality_reduction(run_type, explained_variance, X_train, y_train, num_features)
    print("\n----------------------------------\n")
    print("Running dimensionality reduction on FAD dataset")
    run_type, X_train, y_train, num_features = get_data_ford(dataroot, training_sample, is_rfc)
    dimensionality_reduction(run_type, explained_variance, X_train, y_train, num_features)



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