import sys
import pandas as pd
import numpy as np
import matplotlib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from numpy import mean
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn import tree as dt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
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
import matplotlib.cm as cm

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
    for alg in clustering_algs:
        print("**************************")
        print("Running "+alg)
        df = pd.DataFrame(index=k_clusters, columns=metrics)
        for metric in metrics:
            print("Using metric: "+metric+" across "+str(len(k_clusters))+" different k values")
            for k in k_clusters:
                # Create a subplot with 1 row and 2 columns
                fig, ax1 = plt.subplots()

                # The 1st subplot is the silhouette plot
                # The silhouette coefficient can range from -1, 1 but in this example all
                # lie within [-0.1, 1]
                ax1.set_xlim([-0.1, 1])
                # The (n_clusters+1)*10 is for inserting blank space between silhouette
                # plots of individual clusters, to demarcate them clearly.
                ax1.set_ylim([0, len(X) + (k + 1) * 10])

                # Initialize the clusterer with n_clusters value and a random generator
                # seed of 10 for reproducibility.
                if alg == 'kmeans':
                    clusterer = KMeans(n_clusters=k, random_state=RANDOM_STATE)
                else:
                    clusterer = GaussianMixture(n_components=k, random_state=RANDOM_STATE)
                cluster_labels = clusterer.fit_predict(X)

                # The silhouette_score gives the average value for all the samples.
                # This gives a perspective into the density and separation of the formed
                # clusters
                silhouette_avg = silhouette_score(X, cluster_labels, metric=metric)
                # print("For n_clusters =", k,
                #       "The average silhouette_score is :", silhouette_avg)

                df.at[k, metric] = silhouette_avg
                # Compute the silhouette scores for each sample
                sample_silhouette_values = silhouette_samples(X, cluster_labels)

                y_lower = 10
                for i in range(k):
                    # Aggregate the silhouette scores for samples belonging to
                    # cluster i, and sort them
                    ith_cluster_silhouette_values = \
                        sample_silhouette_values[cluster_labels == i]

                    ith_cluster_silhouette_values.sort()

                    size_cluster_i = ith_cluster_silhouette_values.shape[0]
                    y_upper = y_lower + size_cluster_i

                    color = cm.nipy_spectral(float(i) / k)
                    ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                      0, ith_cluster_silhouette_values,
                                      facecolor=color, edgecolor=color, alpha=0.7)

                    # Label the silhouette plots with their cluster numbers at the middle
                    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                    # Compute the new y_lower for next plot
                    y_lower = y_upper + 10  # 10 for the 0 samples

                ax1.set_title("The silhouette plot for the various clusters.")
                ax1.set_xlabel("The silhouette coefficient values")
                ax1.set_ylabel("Cluster label")

                # The vertical line for average silhouette score of all the values
                ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

                ax1.set_yticks([])  # Clear the yaxis labels / ticks
                ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

                # # 2nd Plot showing the actual clusters formed
                # colors = cm.nipy_spectral(cluster_labels.astype(float) / k)
                # ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                #             c=colors, edgecolor='k')
                #
                # # Labeling the clusters
                # centers = clusterer.cluster_centers_
                # # Draw white circles at cluster centers
                # ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                #             c="white", alpha=1, s=200, edgecolor='k')
                #
                # for i, c in enumerate(centers):
                #     ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                #                 s=50, edgecolor='k')
                #
                # ax2.set_title("The visualization of the clustered data.")
                # ax2.set_xlabel("Feature space for the 1st feature")
                # ax2.set_ylabel("Feature space for the 2nd feature")

                plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                              "with n_clusters = %d" % k),
                             fontsize=14, fontweight='bold')

                plt.savefig('metrics/'+run_type+"_"+alg+"_"+metric+"_"+str(k)+".png")
                plt.clf()
                # time.sleep(5)
                # plt.close('all')
                # fig.gcf()

        print("Printing average silhouette scores for "+alg)
        print(df)



def run_clustering_shoppers(dataroot, k_clusters, metrics, smote):
    run_type = 'OSI'
    datapath = dataroot + 'online_shoppers_intention.csv'
    df = pd.read_csv(datapath)
    target = df['Revenue']
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

    run_clustering_algs(run_type, k_clusters, metrics, clean_attrs)

def run_clustering_ford(dataroot, k_clusters, metrics, dataset_sample):
    run_type = 'FAD'
    df_train = pd.read_csv(dataroot + 'fordTrain.csv')
    y_train = df_train['IsAlert']
    X_train = df_train.drop(['IsAlert', 'TrialID', 'ObsNum'], axis=1)
    if dataset_sample != 0:
        print("Sampling the training set at: "+str(dataset_sample))
        X_train, X_test_new, y_train, y_test_new = train_test_split(X_train, y_train, train_size=dataset_sample,
                                                            random_state=RANDOM_STATE)
    print("Sampled number of instances: "+str(len(X_train.index)))
    run_clustering_algs(run_type, k_clusters, metrics, X_train)


def run_clustering(dataroot):
    training_sample = 0.3
    smote = (False, 0.6)
    k_clusters = [2,3,4,5]
    metrics = ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']
    print("Running clustering on OCI dataset")
    run_clustering_shoppers(dataroot, k_clusters, metrics, smote)
    print("\n----------------------------------\n")
    print("Running clustering on FAD dataset")
    run_clustering_ford(dataroot, k_clusters, metrics, training_sample)


if __name__ == "__main__":
    passed_arg = sys.argv[1]
    if passed_arg.startswith('/'):
        dataroot = passed_arg
    else:
        dataroot = '/Users/plamb/Documents/Personal/Academic/Georgia Tech/Classes/ML/hw/dimensionality_reduction/data/'
    if passed_arg == 'clustering':
        run_clustering(dataroot)
    # if passed_arg == 'shoppers':
    #     run_shoppers(dataroot)
    # elif passed_arg == 'ford':
    #     run_ford(dataroot)
    else:
        print("please run with an absolute path to the data")
        exit(146)