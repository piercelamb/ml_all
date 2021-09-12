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

def get_data_OSI():
    datapath = dataroot + 'online_shoppers_intention.csv'
    df = pd.read_csv(datapath)
    target = df['Revenue']
    attributes = df.drop('Revenue', axis=1, )
    string_columns = ['Month', 'VisitorType', 'Weekend']
    column_trans = make_column_transformer(
        (OneHotEncoder(handle_unknown='ignore'), string_columns),
        remainder='passthrough'
    )
    clean_attrs = column_trans.fit_transform(attributes)
    X_train, X_test, y_train, y_test = train_test_split(clean_attrs, target, stratify=target, test_size=0.3,
                                                        random_state=RANDOM_STATE)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, y_train, X_test, y_test

def get_data_ford(dataset_sample, num_features):
    df_train = pd.read_csv(dataroot + 'fordTrain.csv')
    df_test = pd.read_csv(dataroot + 'fordTest.csv')
    df_pred = pd.read_csv(dataroot + 'Solution.csv')
    y_train = df_train['IsAlert']
    y_test = df_pred['Prediction']
    if num_features == 'full':
        print("Selecting all features")
        X_train = df_train.drop(['IsAlert', 'TrialID', 'ObsNum'], axis=1)
        X_test = df_test.drop(['IsAlert', 'TrialID', 'ObsNum'], axis=1)

    else:
        #Best Feature Selection
        print("Selecting SFS Forward features")
        X_train = df_train[['P6', 'E9', 'V10']]
        X_test = df_test[['P6', 'E9', 'V10']]
    if dataset_sample != 0:
        print("Sampling the training set at: "+str(dataset_sample))
        X_train, X_test_new, y_train, y_test_new = train_test_split(X_train, y_train, train_size=dataset_sample,
                                                            random_state=RANDOM_STATE)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, y_train, X_test, y_test

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


def get_estimator_with_highest_score(train_score):
    max_score = max(train_score['test_score'])
    index = np.where(train_score['test_score'] == max_score)[0][0]
    return train_score['estimator'][index]

def get_best_param_value(score_testing, options, score, clf_type):
    max_score = max(score_testing)
    index = score_testing.index(max_score)
    best_param_value = options[index]
    return best_param_value

def plot_validation_curve(run_type, clf_type, param, score, score_training, score_testing, options,smote):
    smote_text = "SMOTE" if smote[0] else 'reg'
    plt.figure()
    plt.xlabel(param)
    plt.ylabel(score)
    #for MLPClassifier, turns the passed tuples into strings for label display in the graph
    options = [''.join(str(x)) for x in options] if all(isinstance(item, tuple) for item in options) else options
    plt.plot(options, score_training, 'o-', color='g', label="Training Sample")
    plt.plot(options, score_testing, 'o-', color='b', label="Testing Sample")
    plt.legend(loc='best')
    plt.title(run_type+' '+smote_text+' '+param+ ' '+'Validation Curve')
    plt.savefig(run_type+'_'+smote_text+'_'+clf_type + '_' +param + '_' + score + '_validation_curve.png')
    plt.clf()

def plot_loss_curve(model, title, filename):
    plt.xlabel("Num Iterations")
    plt.ylabel("Loss")
    plt.title(title)
    plt.plot(model.loss_curve_)
    plt.savefig(filename)
    plt.clf()

def final_plots(run_type, clf, clf_type, cv, score, X_train, y_train, X_test, y_test, y_pred, smote, is_iterative):
    smote_text = "SMOTE" if smote[0] else 'reg'
    print(clf_type + " on scoring method " + score + " Accuracy score: " + str(accuracy_score(y_test, y_pred)))
    print(clf_type + " on scoring method " + score + " f1 score: " + str(f1_score(y_test, y_pred)))
    get_confusion_matrix(clf, X_test, y_test, run_type+' '+smote_text+' Confusion Matrix (' + clf_type + ')',
                         run_type+'_'+smote_text+'_'+clf_type + '_' + score + '_confusion_matrix.png')
    plot_learning_curve(is_iterative, clf, run_type+' '+smote_text+' Learning Curve (' + clf_type + ')', X_train, y_train, cv=cv, scoring=score,
                        filename=run_type+'_'+smote_text+'_'+clf_type + '_' + score + '_Learning_Curve.png')
    if is_iterative:
        plot_loss_curve(clf, run_type+' '+smote_text+' Loss Curve (' + clf_type + ')',
                        run_type+'_'+smote_text+'_'+clf_type + '_' + score + '_Loss_Curve.png')

def initialize_a_param(clf, param, X_train, y_train, cv, clf_type, scoring, is_smote):
    param_name = list(param)[0]
    print("Initializing "+param_name+" with GridSearch")
    if is_smote:
        smote_param_name = "model__" + param_name
        param[smote_param_name] = param.pop(param_name)
        param_name = smote_param_name
    clf = GridSearchCV(
        clf,
        param_grid=param,
        cv=cv,
        scoring=scoring,
        n_jobs=-1)
    clf.fit(X=X_train, y=y_train)
    initial_value = clf.best_params_[param_name]
    print('Initial value for '+param_name+' = '+str(initial_value))
    return initial_value

@timeit
def calc_testing(clf, X_train, y_train, X_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred

def get_smote_pipeline(clf, smote):
    smote_sample = smote[1]
    under_sample = smote[2]
    over = SMOTE(sampling_strategy=smote_sample, random_state=RANDOM_STATE)
    under = RandomUnderSampler(sampling_strategy=under_sample, random_state=RANDOM_STATE)
    steps = [('over', over), ('under', under), ('model', clf)]
    return imbalancePipeline(steps=steps)

def discover_classifier(clf_type, kwargs, smote):
    is_smote = smote[0]
    if clf_type == 'decision_tree':
        return get_smote_pipeline(dt.DecisionTreeClassifier(**kwargs), smote) if is_smote else dt.DecisionTreeClassifier(**kwargs)
    elif clf_type == 'ada_boost':
        return get_smote_pipeline(AdaBoostClassifier(**kwargs), smote) if is_smote else AdaBoostClassifier(**kwargs)
    elif clf_type == 'neural_network':
        return get_smote_pipeline(MLPClassifier(**kwargs), smote) if is_smote else MLPClassifier(**kwargs)
    elif clf_type == 'svc':
        return get_smote_pipeline(SVC(**kwargs), smote) if is_smote else SVC(**kwargs)
    elif clf_type == 'knn':
        if 'random_state' in kwargs: del kwargs['random_state']
        return get_smote_pipeline(KNeighborsClassifier(**kwargs), smote) if is_smote else KNeighborsClassifier(**kwargs)
    else:
        print("YOU HAVENT DEFINED "+clf_type+" IN discover_classifier")

def get_classifier(clf_type, param_name, option, param2, smote):
    if param2 != None:
        param2_name = list(param2)[0]
        param2_val = param2[param2_name]
        kwargs = {param_name: option, 'random_state': RANDOM_STATE, param2_name: param2_val}
    else:
        kwargs = {param_name: option, 'random_state': RANDOM_STATE}

    return discover_classifier(clf_type, kwargs, smote)

@timeit
def calc_validation_curve(clf_type, score, cv, X_train, y_train, X_test, y_test, param_name, param_options, param2, smote):
    score_training = []
    score_testing = []
    for option in param_options:
        print("Executing with option: "+str(option))
        clf = get_classifier(clf_type, param_name, option, param2, smote)
        train_score = cross_validate(clf, X_train, y_train, cv=cv, scoring=score, return_estimator=True, n_jobs=-1)
        clf = get_estimator_with_highest_score(train_score)
        y_pred = clf.predict(X_test)
        #test_score = recall_score(y_test, y_pred) if score == 'recall' else precision_score(y_test, y_pred)
        test_score = f1_score(y_test, y_pred) if score == 'f1' else accuracy_score(y_test, y_pred)
        score_training.append(clf.score(X_train, y_train))
        score_testing.append(test_score)


    return score_training, score_testing


def get_validation_curve(run_type, score, cv, clf_type, X_train, y_train, X_test, y_test, param1, param2, smote):
    param_name = list(param1)[0]
    if smote[0] and param_name.startswith('model__'):
        fixed_param_name = param_name[len('model__'):]
        param1[fixed_param_name] = param1.pop(param_name)
        param_name = fixed_param_name
    param_options = param1[param_name]
    print('\n----------------------\n')
    print('Executing validation curve for '+param_name)
    score_training, score_testing = calc_validation_curve(clf_type, score, cv, X_train, y_train, X_test, y_test, param_name, param_options, param2, smote)
    print(clf_type + ' on scoring method ' + score + ' best CV score: ' + str(max(score_training)))
    plot_validation_curve(run_type, clf_type, param_name, score, score_training, score_testing, param_options, smote)
    best_param_value = get_best_param_value(score_testing, param_options, score, clf_type)
    print(clf_type + " on scoring method " + score + " " + param_name + " : " + str(best_param_value))
    print('\n----------------------\n')
    return best_param_value

def validate_two_params(param1, param2, clf, run_type, clf_type, scoring, cv,X_train, y_train, X_test, y_test, smote):
    param2_name = list(param2)[0]
    param1_name = list(param1)[0]
    param_2_initial_val = initialize_a_param(clf, param2, X_train, y_train, cv, clf_type, scoring, smote[0])
    param2_initialized = {param2_name: param_2_initial_val}
    best_param1_value = get_validation_curve(run_type, scoring, cv, clf_type, X_train, y_train, X_test, y_test,
                                             param1, param2_initialized, smote)
    param1_best = {param1_name: best_param1_value}
    best_param2_value = get_validation_curve(run_type, scoring, cv, clf_type, X_train, y_train, X_test, y_test,
                                             param2, param1_best, smote)
    if param_2_initial_val != best_param2_value:
        print("Initialized value: "+str(param_2_initial_val)+" != best value:"+str(best_param2_value))
        param2_best = {param2_name: best_param2_value}
        best_param1_value = get_validation_curve(run_type, scoring, cv, clf_type, X_train, y_train, X_test, y_test,
                                                 param1, param2_best, smote)
        param1_best = {param1_name: best_param1_value}
        best_param2_value = get_validation_curve(run_type, scoring, cv, clf_type, X_train, y_train, X_test, y_test,
                                                 param2, param1_best, smote)
        if param2_best[list(param2_best)[0]] != best_param2_value:
            print("2nd CV iteration value: " + str(param2_best[list(param2_best)[0]]) + " != best value:" + str(best_param2_value))

    kwargs = {param1_name: best_param1_value, param2_name: best_param2_value, 'random_state': RANDOM_STATE}
    return kwargs

def validate_single_param(param1, param2, run_type, clf_type, scoring, cv,X_train, y_train, X_test, y_test, smote):
    best_param_value = get_validation_curve(run_type, scoring, cv, clf_type, X_train, y_train, X_test, y_test, param1,
                                            param2, smote)
    param_name = list(param1)[0]
    kwargs = {param_name: best_param_value, 'random_state': RANDOM_STATE}
    return kwargs

def get_optimal_hyperparameters(param2, clf_type, smote, param1, run_type, scoring, cv,X_train,y_train,X_test,y_test):
    print("Executing " + clf_type + ' with smote: ' + str(smote[0]))
    if param2 != None:
        clf = discover_classifier(clf_type, {}, smote)
        kwargs = validate_two_params(param1,param2,clf,run_type,clf_type,scoring,cv,X_train,y_train,X_test,y_test, smote)
    else:
        kwargs = validate_single_param(param1, param2, run_type,clf_type,scoring,cv,X_train,y_train,X_test,y_test, smote)
    return kwargs

def run_dt(X_train, y_train, X_test, y_test, scoring, cv, smote, run_type, param1, param2):
    clf_type = 'decision_tree'
    is_iterative = False
    kwargs = get_optimal_hyperparameters(param2, clf_type, smote, param1, run_type, scoring, cv,X_train,y_train,X_test,y_test)
    clf = dt.DecisionTreeClassifier(**kwargs)
    y_pred = calc_testing(clf, X_train, y_train, X_test)
    final_plots(run_type, clf, clf_type, cv, scoring, X_train, y_train, X_test, y_test, y_pred, smote, is_iterative)


def run_ada(X_train, y_train, X_test, y_test, scoring, cv, smote, run_type, param1, param2):
    clf_type = 'ada_boost'
    is_iterative = False
    kwargs = get_optimal_hyperparameters(param2, clf_type, smote, param1, run_type, scoring, cv, X_train, y_train,
                                         X_test, y_test)
    clf = AdaBoostClassifier(**kwargs)
    y_pred = calc_testing(clf, X_train, y_train, X_test)
    final_plots(run_type, clf, clf_type, cv, scoring, X_train, y_train, X_test, y_test, y_pred, smote, is_iterative)

def run_nn(X_train, y_train, X_test, y_test, scoring, cv, smote, run_type, param1, param2):
    clf_type = 'neural_network'
    is_iterative = True
    kwargs = get_optimal_hyperparameters(param2, clf_type, smote, param1, run_type, scoring, cv, X_train, y_train,
                                         X_test, y_test)
    clf = MLPClassifier(**kwargs)
    y_pred = calc_testing(clf, X_train, y_train, X_test)
    final_plots(run_type, clf, clf_type, cv, scoring, X_train, y_train, X_test, y_test, y_pred, smote, is_iterative)

def run_svc(X_train, y_train, X_test, y_test, scoring, cv, smote, run_type, param1, param2):
    clf_type = 'svc'
    is_iterative = True
    kwargs = get_optimal_hyperparameters(param2, clf_type, smote, param1, run_type, scoring, cv, X_train, y_train,
                                         X_test, y_test)
    clf = SVC(**kwargs)
    y_pred = calc_testing(clf, X_train, y_train, X_test)
    final_plots(run_type, clf, clf_type, cv, scoring, X_train, y_train, X_test, y_test, y_pred, smote, is_iterative)

def run_knn(X_train, y_train, X_test, y_test, scoring, cv, smote, run_type, param1, param2):
    clf_type = 'knn'
    is_iterative = False
    kwargs = get_optimal_hyperparameters(param2, clf_type, smote, param1, run_type, scoring, cv, X_train, y_train,
                                         X_test, y_test)
    if 'random_state' in kwargs: del kwargs['random_state']
    clf = KNeighborsClassifier(**kwargs)
    y_pred = calc_testing(clf, X_train, y_train, X_test)
    final_plots(run_type, clf, clf_type, cv, scoring, X_train, y_train, X_test, y_test, y_pred, smote, is_iterative)


def run_shoppers(dataroot):
    X_train, y_train, X_test, y_test = get_data_OSI()
    scoring = 'f1'
    cross_val_folds = 10
    smote = (False,0.3,0.6)
    run_type = 'OSI'
    # run_dt(
    #     X_train,
    #     y_train,
    #     X_test,
    #     y_test,
    #     scoring,
    #     cross_val_folds,
    #     smote,
    #     run_type,
    #     {'max_depth': [1,2,3,4,5,6,7,8,9,10,15,20]},
    #     None
    # )
    # run_nn(
    #     X_train,
    #     y_train,
    #     X_test,
    #     y_test,
    #     scoring,
    #     cross_val_folds,
    #     smote,
    #     run_type,
    #     #(10,10,10) = 3 hidden layers with 10 units, (10,) = 1 hidden layer with 10 units
    #     #had to play with the below numbers a lot
    #     {'hidden_layer_sizes': [(100,), (150,), (200,), (100,100), (100,100,100)]},  # validation_curve
    #     {'activation': ['logistic', 'tanh', 'relu', 'identity']},  # validation_curve
    # )
    #Weak learner puts more weight on the examples its getting wrong (showing they're the 'harder' examples)
    # and as long as you're still getting better than chance error, it guarantees you'll get some of the harder examples right
    # run_ada(
    #     X_train,
    #     y_train,
    #     X_test,
    #     y_test,
    #     scoring,
    #     cross_val_folds,
    #     smote,
    #     run_type,
    #     {'n_estimators': [50, 100, 150, 200]},
    #     {'learning_rate': [0.0001, 0.001, 0.01]}
    # )
    # run_svc(
    #     X_train,
    #     y_train,
    #     X_test,
    #     y_test,
    #     scoring,
    #     cross_val_folds,
    #     smote,
    #     run_type,
    #     {'C': [1,2,3,4,5,6,7,8,9,10]},  # validation_curve
    #     #{'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']},  # validation_curve
    #     {'kernel': ['rbf', 'sigmoid']},  # validation_curve
    # )
    #preference bias
    #Locality: Near points are similar
    #smoothness: averaging over nearest neighbors
    #all features matter eqaully
    # run_knn(
    #     X_train,
    #     y_train,
    #     X_test,
    #     y_test,
    #     scoring,
    #     cross_val_folds,
    #     smote,
    #     run_type,
    #     {'n_neighbors': [3,4,5,6,7,8,9,10]},  # validation_curve
    #     {'weights': ['uniform', 'distance']},  # validation_curve
    # )

#TODO: See if the distribution of trues to falses is the same between training and testing
def run_ford(dataroot):
    training_sample = 0.3
    num_features = 'full'
    X_train, y_train, X_test, y_test = get_data_ford(training_sample, num_features)
    scoring = 'accuracy'
    cross_val_folds = 10
    smote = (False, None, None)
    run_type = 'FAD'
    # run_dt(
    #     X_train,
    #     y_train,
    #     X_test,
    #     y_test,
    #     scoring,
    #     cross_val_folds,
    #     smote,
    #     run_type,
    #     {'max_depth': [3,5,9,10,12,14,16,18,20]},
    #     {'criterion': ['gini', 'entropy']}
    # )
    run_nn(
        X_train,
        y_train,
        X_test,
        y_test,
        scoring,
        cross_val_folds,
        smote,
        run_type,
        #(10,10,10) = 3 hidden layers with 10 units, (10,) = 1 hidden layer with 10 units
        #had to play with the below numbers a lot
        {'hidden_layer_sizes': [(100,), (150,), (200,), (100,100), (100,100,100)]},  # validation_curve
        {'activation': ['logistic', 'tanh', 'relu', 'identity']},  # validation_curve
    )
    # run_ada(
    #     X_train,
    #     y_train,
    #     X_test,
    #     y_test,
    #     scoring,
    #     cross_val_folds,
    #     smote,
    #     run_type,
    #     {'n_estimators': [50, 100, 150, 200]},
    #     {'learning_rate': [0.0001, 0.001, 0.01]}
    # )
    # run_knn(
    #     X_train,
    #     y_train,
    #     X_test,
    #     y_test,
    #     scoring,
    #     cross_val_folds,
    #     smote,
    #     run_type,
    #     {'n_neighbors': [3,4,5,6,7,8,9,10]},  # validation_curve
    #     {'weights': ['uniform', 'distance']},  # validation_curve
    # )
    # run_svc(
    #     X_train,
    #     y_train,
    #     X_test,
    #     y_test,
    #     scoring,
    #     cross_val_folds,
    #     smote,
    #     run_type,
    #     {'C': [1,2,3,4,5,6,7,8,9,10]},  # validation_curve
    #     #{'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']},  # validation_curve
    #     {'kernel': ['rbf', 'sigmoid']},  # validation_curve
    # )
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