import sys
import pandas as pd
import numpy as np
import matplotlib
from sklearn.neural_network import MLPClassifier
from numpy import mean
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn import tree as dt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
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

def plot_learning_curve(model,title,X_train,y_train,cv, filename,scoring='Accuracy'):
    plt.figure()
    plt.title(title)
    plt.xlabel("Num Samples")
    plt.ylabel(scoring)
    step=np.linspace(1/cv,1.0,cv)
    train_sizes,train_scores,test_scores = learning_curve(model,X_train,y_train,cv=cv,train_sizes=step, scoring=scoring)
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
    return plt

@timeit
def get_cross_validated_clf(pipeline, parameters_to_tune, X_train, y_train, cv, clf_type, scoring):
    clf = GridSearchCV(
        pipeline,
        param_grid=parameters_to_tune,
        cv=cv,
        #verbose=3,
        scoring=scoring,
        n_jobs=-1)
    clf.fit(X=X_train, y=y_train)
    #best_estimate_.steps returns:
    # [('over', SMOTE(sampling_strategy=0.3)), ('under', RandomUnderSampler(sampling_strategy=0.5)),
    #  ('model', DecisionTreeClassifier(ccp_alpha=0.0005, criterion='entropy', max_depth=5,
    #                                   min_samples_split=4, random_state=1337))]
    #tree_model = clf.best_estimator_.steps[2][1]
    tree_model = clf.best_estimator_
    print(clf.best_score_, clf.best_params_)
    print(scoring+' Score on cross-validated '+clf_type+': ' + str(clf.best_score_))
    #print('Size of cross-validated '+clf_type+': ' + str(tree_model.tree_.node_count))
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
def get_max_score(clf, X_test, y_test, scoring):
    y_pred = clf.predict(X_test)
    if scoring == 'precision':
        score = precision_score(y_test, y_pred)
    elif scoring == 'recall':
        score = recall_score(y_test, y_pred)
    else:
        score = clf.score(X_test, y_test)
    return score

def get_precision_recall(classifier, X_train, y_train, X_test, y_test, clf_type, filename):
    classifier.fit(X_train, y_train)
    #y_score = classifier.decision_function(X_test)
    y_score = classifier.predict(X_test)
    average_precision = average_precision_score(y_test, y_score)
    print('Average precision-recall score: {0:0.2f}'.format(
        average_precision))
    disp = plot_precision_recall_curve(classifier, X_test, y_test)
    disp.ax_.set_title('Precision-Recall curve ('+clf_type+'): '
                       'AP={0:0.2f}'.format(average_precision))
    disp.ax_.get_legend().remove()
    disp.figure_.savefig(filename)

def find_optimal_clf(pipeline, parameters_to_tune, cv, X_train, X_test, y_train, y_test, clf_type, scoring):
    tree_model, best_params = get_cross_validated_clf(pipeline, parameters_to_tune, X_train, y_train, cv, clf_type, scoring)
    #max_score, optimal_clf = get_pruned_clf(tree_model, best_params, X_train, X_test, y_train, y_test)
    max_score = get_max_score(tree_model, X_test, y_test, scoring)
    #get_precision_recall(tree_model, X_train, y_train, X_test, y_test, clf_type, clf_type+'_Precision_Recall_Curve.png')
    return max_score, tree_model

#tried 0.3, 0.5 (89.6%), 0.4, 0.5 (89.8), 0.4, 0.7 (89.2), 0.5/0.8 (87), 0.5/0.5 (90) 0.6/0.6 (87) 0.4/0.4 (89.91)
def get_imbalanced_data_pipeline(model):
    model = model
    smote_sample = 0.4
    under_sample = 0.5
    print(smote_sample)
    print(under_sample)
    over = SMOTE(sampling_strategy=smote_sample, random_state=RANDOM_STATE)
    under = RandomUnderSampler(sampling_strategy=under_sample, random_state=RANDOM_STATE)
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

def get_validation_curve(param, options, score, clf_type, clf, X_train, y_train, X_test, y_test):

        score_training = []
        score_testing  = []
        for option in options:
            print("Trying option: "+str(option))
            if param == 'n_estimators':
                clf=AdaBoostClassifier(n_estimators=option, learning_rate=0.1)
            elif param == 'learning_rate':
                clf = AdaBoostClassifier(n_estimators=35, learning_rate=option)
            train_score = cross_validate(clf,X_train, y_train, cv=10, scoring=score, return_estimator=True, n_jobs=-1)
            #get the estimator with the highest score
            max_score = max(train_score['test_score'])
            index = np.where(train_score['test_score'] == max_score)[0][0]
            clf = train_score['estimator'][index]
            y_pred = clf.predict(X_test)
            test_score = recall_score(y_test, y_pred) if score == 'recall' else precision_score(y_test, y_pred)
            score_training.append(clf.score(X_train,y_train))
            score_testing.append(test_score)

        plt.figure()
        plt.xlabel(param)
        plt.ylabel(score)
        plt.plot(options, score_training,'o-',color='g',label="Training Sample")
        plt.plot(options,score_testing,'o-',color='b',label="Testing Sample")
        plt.legend(loc='best')
        plt.savefig(param+'_'+score+'_validation_curve_'+clf_type+'.png')

        max_score = max(score_testing)
        index = score_testing.index(max_score)
        best_max_depth = options[index]
        print(clf_type+" on scoring method "+score+" "+ param+ " : "+str(best_max_depth))
        tree = AdaBoostClassifier(n_estimators=35, learning_rate=0.1)
        tree.fit(X_train, y_train)
        y_pred = tree.predict(X_test)
        print(clf_type+" on scoring method "+score+" Accuracy score: " +str(accuracy_score(y_test, y_pred)))
        print(clf_type + " on scoring method " + score + " score: " + str(f1_score(y_test, y_pred)))
        get_confusion_matrix(tree, X_test, y_test, 'OSI Confusion Matrix (' + clf_type + ')',
                             clf_type + '_'+score+'_OSI_confusion_matrix.png')
        plot_learning_curve(tree, 'Learning Curve (' + clf_type + ')', X_train, y_train, cv=10, scoring=score,
                            filename=clf_type + '_'+score+'_OSI_Learning_Curve.png')


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
    column_trans = make_column_transformer(
        (OneHotEncoder(handle_unknown='ignore'), string_columns),
        #(RobustScaler(), numerical_columns)
        remainder='passthrough'
    )
    clean_attrs = column_trans.fit_transform(attributes)
    imbalanced_data_pipeline, parameters_to_tune = get_imbalanced_data_pipeline(model=dt.DecisionTreeClassifier(max_depth=4))
    #imbalanced_data_pipeline = dt.DecisionTreeClassifier()
    X_train, X_test, y_train, y_test = train_test_split(clean_attrs, target, stratify=target, test_size=0.3, random_state=RANDOM_STATE)
    #print(Counter(y_train))
    #exit(1)
    # scores = cross_val_score(imbalanced_data_pipeline, X_train, y_train, scoring='recall', cv=10, n_jobs=-1)
    # print(mean(scores))
    # exit(1)

    cv = 10
    classifiers_and_parameters = {
        'decision_tree' : {
            imbalanced_data_pipeline:
                {
                'max_depth': [4],
                'criterion': ['entropy'],
                #'model__ccp_alpha': [0.0, 0.000005, 0.00005, 0.0005, 0.005], #validation_curve
                # 'model__min_samples_split':range(2,10),
                # 'model__min_samples_leaf':range(1,5),
                # 'model__max_features': ['sqrt', 'log2', None],
                'random_state': [RANDOM_STATE]
            }
        },
        'ada_boost' : {
            AdaBoostClassifier():
            {
                'n_estimators': [50, 100, 200, 500, 1000],#validation_curve
                'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1.0], #validation_curve
                'random_state': [RANDOM_STATE]
            }
        },
        'neural_network' : {
            MLPClassifier():
            {
                'hidden_layer_sizes': [(10,30,10),(20,)], #validation_curve
                'activation': ['logistic', 'tanh', 'relu'], #validation_curve
                'solver': ['sgd', 'adam'],
                'alpha': [0.0001, 0.05],
                'learning_rate': ['constant','adaptive'],
                'warm_start': [True],
                'random_state': [RANDOM_STATE]
            }
        },

    }

    type_to_run = 'ada_boost'
    scoring = ['f1']
    #scoring = ['precision', 'recall']
    validation_curves_ada = {
        #'n_estimators': [25,75,125,175,225,275,325,375],
        #'n_estimators': [25,35,45,55,65,75,85,95,100],
        'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1.0]
    }
    validation_curves_dt = {
        'max_depth': [1,2,3,4,5,6,7,8,9,10,15,20]
    }
    use_all_scoring_methods = True
    execute_validation_curves = True
    for clf_type, clf_info in classifiers_and_parameters.items():
        for clf, params in clf_info.items():
            if clf_type == type_to_run:
                print('Starting GridSearch for: '+clf_type+'...')
                if use_all_scoring_methods:
                    for score in scoring:
                        print('Executing on '+score)
                        if execute_validation_curves:
                            for param, options in validation_curves_ada.items():
                                print("Working on param: "+param)
                                get_validation_curve(param, options, score, clf_type, clf, X_train, y_train, X_test, y_test)

                        # max_score, optimal_clf = find_optimal_clf(clf, params, cv, X_train, X_test, y_train, y_test, clf_type, score)
                        # print(score+' Score for '+clf_type+' on test Data: '+str(max_score))
                    exit(1)
                else:
                    max_score, optimal_clf = find_optimal_clf(clf, params, cv, X_train, X_test, y_train, y_test, clf_type, scoring[0])
                    print(scoring+' Score for '+clf_type+' on test Data: '+str(max_score))

                get_confusion_matrix(optimal_clf, X_test, y_test, 'OSI Confusion Matrix ('+clf_type+')', clf_type+'_OSI_confusion_matrix.png')
                plot_learning_curve(optimal_clf, 'Learning Curve ('+clf_type+')', X_train, y_train, cv=cv, filename= clf_type+'_OSI_Learning_Curve.png')

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

def feature_selection_get_cols(direction, dataframe, mask):
    num_columns = len(dataframe.columns)
    num_bools = len(mask)

    difference = num_columns - num_bools
    for i in range(0, difference):
        mask.append(False)
    return dataframe.loc[:, mask]

def runSFS(base_clf, sfs_direction, feature_percent, cv, X_train, y_train):
    print('Starting Sequential Feature Selection with direction: ', sfs_direction)
    sfs_results = SequentialFeatureSelector(estimator=base_clf, direction=sfs_direction, scoring='accuracy',
                                            n_features_to_select=feature_percent, n_jobs=-1, cv=cv)
    sfs_results.fit(X_train, y_train)
    best_features_mask = sfs_results.get_support()
    selected_feature_names = feature_selection_get_cols(sfs_direction, X_train, best_features_mask)
    print('Original feature set: ', X_train.columns.values)
    print('Best features:', selected_feature_names.columns.values)
    print('X_train shape before: ' + str(X_train.shape))
    optimal_X_train = sfs_results.transform(X_train)
    print('X_train shape after: ' + str(optimal_X_train.shape))

    return optimal_X_train

@timeit
def runFADGridSearch(base_clf, parameters_to_tune, cross_validation_folds, training_features, feature_option, y_train):
    gs = GridSearchCV(
        base_clf,
        param_grid=parameters_to_tune,
        cv=cross_validation_folds,
        scoring='accuracy',
        # verbose=3,
        n_jobs=-1)
    gs.fit(X=training_features, y=y_train)
    clf = gs.best_estimator_
    print('Results for ' + str(feature_option) + '% option')
    print("Best parameters via GridSearch", gs.best_params_)
    print('Best cross-validation score:', gs.best_score_)
    return clf


def run_ford(dataroot):
    df_train = pd.read_csv(dataroot + 'fordTrain.csv')
    df_test = pd.read_csv(dataroot + 'fordTest.csv')
    df_pred = pd.read_csv(dataroot + 'Solution.csv')
    y_train = df_train['IsAlert']
    y_test = df_pred['Prediction']
    X_train = df_train.drop(['IsAlert', 'TrialID', 'ObsNum'], axis=1)
    #X_train = df_train.drop(['IsAlert', 'TrialID', 'ObsNum', 'P4', 'V6', 'V10', 'E9', 'E2', 'P8', 'V7', 'V9'], axis=1)
    X_trains_my_deduction = {
        'kek': df_train[['E8', 'E9', 'V10']]
    }
    X_trains_forward = {
        '10': df_train[['P6', 'E9', 'V10']],
        #'30': df_train[['P6','P7', 'P8', 'E8', 'E9', 'E11', 'V7', 'V9', 'V10']],
        #'50': df_train[['P2', 'P6', 'P7', 'P8', 'E3', 'E4', 'E5', 'E7', 'E8', 'E9', 'E11', 'V2', 'V5', 'V7','V9', 'V10']],
    }
    X_trains_forward_big_drop = {
        #'10': df_train[['E7','E8']],
        '30': df_train[['E3', 'E7', 'E8', 'E11', 'V5', 'V8']]
    }
    X_trains_backward_big_drop = {
        '10': df_train[['E8', 'V1']]
    }
    X_trains_backward = {
        '10': df_train[['E8', 'E9', 'V1']],
    }
    X_trains_custom_merged = {
        '17': df_train[['P6', 'E8', 'E9', 'V1', 'V10']]
    }

    X_trains_custom_similar = {
        '3': df_train['E9']
    }
    X_trains_custom_merged_no_p = {
        '13': df_train[['E8', 'E9', 'V1', 'V10']]
    }

    x_tests_forward = {
        '10': df_test[['P6', 'E9', 'V10']],
        #'30': df_test[['P6','P7', 'P8', 'E8', 'E9', 'E11', 'V7', 'V9', 'V10']],
        #'50': df_test[['P2', 'P6', 'P7', 'P8', 'E3', 'E4', 'E5', 'E7', 'E8', 'E9', 'E11', 'V2', 'V5', 'V7','V9', 'V10']],
    }
    x_tests_backward = {
        '10': df_test[['E8', 'E9', 'V1']],
    }
    x_tests_custom_merged = {
        '17': df_test[['P6', 'E8', 'E9', 'V1', 'V10']]
    }

    x_tests_custom_similar = {
        '3': df_test['E9']
    }
    x_tests_custom_merged_no_p = {
        '13': df_test[['E8', 'E9', 'V1', 'V10']]
    }
    x_tests_forward_big_drop = {
        #'10': df_test[['E7','E8']],
        '30': df_test[['E3', 'E7', 'E8', 'E11', 'V5', 'V8']]
    }

    x_tests_backward_big_drop = {
        '10': df_test[['E8', 'V1']]
    }

    x_tests_my_deduction = {
        'kek': df_test[['E8', 'E9', 'V10']]
    }

    #x_test = df_test.drop(['IsAlert', 'TrialID', 'ObsNum'], axis=1)
    cross_validation_folds = 10
    sfs_direction='forward'
    base_clf = dt.DecisionTreeClassifier()
    #optimal_X_train = runSFS(base_clf,sfs_direction,0.3,cross_validation_folds,X_train,y_train)
    #exit(1)

    print("Starting gridsearch...")
    parameters_to_tune = {
        'max_depth': range(3, 20),
        'criterion': ('gini', 'entropy'),
        'ccp_alpha': [0.0, 0.000005, 0.00005, 0.0005, 0.005],
        #'min_samples_split':range(2,10),
        #'min_samples_leaf':range(1,5),
        #'max_features': ['sqrt', 'log2', None],
        'random_state': [RANDOM_STATE]
    }

    for feature_option, training_features in X_trains_custom_merged.items():
        print('Running gridsearch on ' + str(feature_option) + '% feature results')
        optimal_clf = runFADGridSearch(base_clf, parameters_to_tune, cross_validation_folds, training_features, feature_option, y_train)
        test_score = get_max_score(optimal_clf, x_tests_custom_merged[feature_option], y_test)
        print("Test data score: ", test_score)

    get_confusion_matrix(optimal_clf, x_tests_custom_merged['17'], y_test, 'Confusion Matrix for DT on Ford', 'DT_ford_confusion_matrix.png')
    plot_learning_curve(optimal_clf, 'Decision Tree Learning Curve (Ford)', X_trains_custom_merged['17'], y_train, cv=cross_validation_folds,filename='DT_Ford_Learning_Curve.png')


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