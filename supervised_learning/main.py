import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn import tree as dt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.compose import make_column_transformer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_confusion_matrix
from imblearn.pipeline import Pipeline as imbalancePipeline
from sklearn.pipeline import Pipeline as sklearnPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import learning_curve
from collections import Counter
import time
from functools import wraps
from sklearn.feature_selection import SequentialFeatureSelector

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

def plot_accuracy_scores(ccp_alphas, train_scores, test_scores, figure):
    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs alpha for training and testing sets")
    ax.plot(ccp_alphas, train_scores, marker='o', label="train",
            drawstyle="steps-post")
    ax.plot(ccp_alphas, test_scores, marker='o', label="test",
            drawstyle="steps-post")
    ax.legend()
    plt.savefig(figure)
    plt.clf()

def plot_learning_curve(model,title,X_train,y_train,cv=None):
    plt.figure()
    plt.title(title)
    plt.xlabel("Num Samples")
    plt.ylabel("Accuracy")
    step=np.linspace(1/cv,1.0,cv)
    train_sizes,train_scores,test_scores = learning_curve(model,X_train,y_train,cv=cv,train_sizes=step)
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
    plt.savefig('OSI_Learning_Curve.png')
    return plt

@timeit
def get_cross_validated_clf(pipeline, parameters_to_tune, X_train, y_train, cv):
    clf = GridSearchCV(
        pipeline,
        param_grid=parameters_to_tune,
        cv=cv,
        #verbose=3,
        n_jobs=-1)
    clf.fit(X=X_train, y=y_train)
    #best_estimate_.steps returns:
    # [('over', SMOTE(sampling_strategy=0.3)), ('under', RandomUnderSampler(sampling_strategy=0.5)),
    #  ('model', DecisionTreeClassifier(ccp_alpha=0.0005, criterion='entropy', max_depth=5,
    #                                   min_samples_split=4, random_state=1337))]
    #tree_model = clf.best_estimator_.steps[2][1]
    tree_model = clf.best_estimator_
    print(clf.best_score_, clf.best_params_)
    print('Accuracy Score on cross-validated DT: ' + str(clf.best_score_))
    print('Size of cross-validated DT: ' + str(tree_model.tree_.node_count))
    return tree_model, clf.best_params_

def get_pruned_clf(tree_model, best_params, X_train, X_test, y_train, y_test):
    path = tree_model.cost_complexity_pruning_path(X_train, y_train)
    potential_alphas, impurities = path.ccp_alphas, path.impurities
    criterion = best_params['criterion']
    max_depth = best_params['max_depth']
    splitter = best_params['splitter']
    potential_clfs = {}
    train_scores = []
    test_scores = []
    plotted_alphas = []
    for ccp_alpha in potential_alphas:
        clf = dt.DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, splitter=splitter, random_state=RANDOM_STATE, ccp_alpha=ccp_alpha)
        # create the DT
        clf.fit(X_train, y_train)
        # Get the accuracy score
        test_score = clf.score(X_test, y_test)
        train_score = clf.score(X_train, y_train)
        #if ccp_alpha < 0.0175:
        test_scores.append(test_score)
        train_scores.append(train_score)
        plotted_alphas.append(ccp_alpha)
        potential_clfs[test_score] = clf

    max_score = max(potential_clfs.keys())

    plot_accuracy_scores(plotted_alphas, train_scores, test_scores, 'figure_2.png')
    return max_score, potential_clfs[max_score]

@timeit
def get_max_score(clf, X_test, y_test):
    return clf.score(X_test, y_test)

def find_optimal_clf(pipeline, parameters_to_tune, cv, X_train, X_test, y_train, y_test):
    tree_model, best_params = get_cross_validated_clf(pipeline, parameters_to_tune, X_train, y_train, cv)
    #max_score, optimal_clf = get_pruned_clf(tree_model, best_params, X_train, X_test, y_train, y_test)
    max_score = get_max_score(tree_model, X_test, y_test)
    return max_score, tree_model

#tried 0.3, 0.5 (89.6%), 0.4, 0.5 (89.8), 0.4, 0.7 (89.2), 0.5/0.8 (87), 0.5/0.5 (90) 0.6/0.6 (87) 0.4/0.4 (89.91)
def get_imbalanced_data_pipeline():
    model = dt.DecisionTreeClassifier()
    over = SMOTE(sampling_strategy=0.5, random_state=RANDOM_STATE)
    under = RandomUnderSampler(sampling_strategy=0.5, random_state=RANDOM_STATE)
    steps = [('over', over), ('under', under), ('model', model)]
    pipeline = imbalancePipeline(steps=steps)

    parameters_to_tune = {
        'model__max_depth': range(3, 20),
        'model__criterion': ('gini', 'entropy'),
        'model__ccp_alpha': [0.0, 0.000005, 0.00005, 0.0005, 0.005],
        # 'model__min_samples_split':range(2,10),
        # 'model__min_samples_leaf':range(1,5),
        # 'model__max_features': ['sqrt', 'log2', None],
        'model__random_state':[RANDOM_STATE]
    }
    return pipeline, parameters_to_tune

def run_shoppers(dataroot):
    datapath = dataroot + 'online_shoppers_intention.csv'
    df = pd.read_csv(datapath)
    # df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
    target = df['Revenue']
    attributes = df.drop('Revenue', axis=1,)
    string_columns = ['Month', 'VisitorType', 'Weekend']
    # numerical_columns = ['Administrative', 'Administrative_Duration', 'Informational',
    #    'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
    #    'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay',
    #    'OperatingSystems', 'Browser', 'Region', 'TrafficType',]
    #TODO: maybe can eek more performance out by passing numerical columns through RobustScalar (remove passthrough)
    #TODO: https://stackoverflow.com/a/52801019
    #TODO: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html
    column_trans = make_column_transformer(
        (OneHotEncoder(handle_unknown='ignore'), string_columns),
        #(RobustScaler(), numerical_columns)
        remainder='passthrough'
    )
    clean_attrs = column_trans.fit_transform(attributes)
    #TODO: we could maybe eek more accuracy out of the test_size param
    #TODO: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    #imbalanced_data_pipeline, parameters_to_tune = get_imbalanced_data_pipeline()
    imbalanced_data_pipeline = dt.DecisionTreeClassifier()
    parameters_to_tune = {
        'max_depth': range(3, 20),
        'criterion': ('gini', 'entropy'),
        'ccp_alpha': [0.0, 0.000005, 0.00005, 0.0005, 0.005],
        # 'model__min_samples_split':range(2,10),
        # 'model__min_samples_leaf':range(1,5),
        # 'model__max_features': ['sqrt', 'log2', None],
        'random_state': [RANDOM_STATE]
    }
    X_train, X_test, y_train, y_test = train_test_split(clean_attrs, target, stratify=target, test_size=0.3, random_state=RANDOM_STATE)
    #print(Counter(y_train))
    #exit(1)
    cv = 10
    max_score, optimal_clf = find_optimal_clf(imbalanced_data_pipeline, parameters_to_tune, cv, X_train, X_test, y_train, y_test)
    print('Accuracy Score on test Data: '+str(max_score))
    confusion_matrix = plot_confusion_matrix(
        optimal_clf,
        X_test,
        y_test,
        cmap=plt.cm.Blues,
        normalize='true'
    )
    confusion_matrix.ax_.set_title('Confusion Matrix for predictions')
    plt.savefig('OSI_confusion_matrix.png')
    plt.clf()
    plot_learning_curve(optimal_clf, 'Decision Tree Learning Curve', X_train, y_train, cv=cv)

def feature_selection_get_cols(direction, dataframe, mask):
    num_columns = len(dataframe.columns)
    num_bools = len(mask)

    if direction == 'forward':
        difference = num_columns - num_bools
        for i in range(0, difference):
            mask.append(False)
        return dataframe.loc[:, mask]
    elif direction == 'backward':
        print("uh oh backwards direction")
        exit(1)

def run_ford(dataroot):
    df_train = pd.read_csv(dataroot + 'fordTrain.csv')
    df_test = pd.read_csv(dataroot + 'fordTest.csv')
    y_train = df_train['IsAlert']
    X_train = df_train.drop('IsAlert', axis=1)
    x_test = df_test.drop('IsAlert', axis=1)
    y_test = df_test['IsAlert']
    cross_validation_folds = 10
    sfs_direction='forward'
    base_clf = dt.DecisionTreeClassifier()
    #a float for n_features_to_select = % of features to select
    sfs_results = SequentialFeatureSelector(estimator=base_clf, direction=sfs_direction, scoring='accuracy', n_features_to_select=0.5, n_jobs=-1, cv=cross_validation_folds)

    best_features_mask = sfs_results.get_support()
    selected_feature_names = feature_selection_get_cols(sfs_direction, X_train, best_features_mask)
    print('Original feature set: ', X_train.columns.values)
    print('Best features:', selected_feature_names.columns.values)
    print('X_train shape before: '+str(X_train.shape))
    optimal_X_train = sfs_results.transform(X_train)
    print('X_train shape after: '+str(optimal_X_train.shape))
    print("Starting gridsearch...")
    parameters_to_tune = {
        'estimator__max_depth': range(3, 20),
        'estimator__criterion': ('gini', 'entropy'),
        'estimator__ccp_alpha': [0.0, 0.000005, 0.00005, 0.0005, 0.005],
        # 'model__min_samples_split':range(2,10),
        # 'model__min_samples_leaf':range(1,5),
        # 'model__max_features': ['sqrt', 'log2', None],
        'estimator__random_state': [RANDOM_STATE]
    }

    gs = GridSearchCV(
        sfs_results.estimator,
        param_grid=parameters_to_tune,
        cv=cross_validation_folds,
        scoring='accuracy',
        #verbose=3,
        n_jobs=-1)
    gs.fit(X=optimal_X_train, y=y_train)
    clf = gs.best_estimator_
    print("Best parameters via GridSearch", gs.best_params_)
    print('Best cross-validation score:', gs.best_score_)

    test_score = get_max_score(clf, x_test, y_test)
    print("Test data score: ", test_score)


if __name__ == "__main__":
    passed_arg = sys.argv[1]
    if passed_arg.startswith('/'):
        dataroot = passed_arg
    else:
        dataroot = '/Users/plamb/Documents/Personal/Academic/Georgia Tech/Classes/ML/hw/supervised_learning/data/'
    if passed_arg == 'shoppers':
        run_shoppers(dataroot)
    elif passed_arg == 'ford':
        run_ford(dataroot)
    else:
        print("please run with an absolute path to the data")
        exit(146)