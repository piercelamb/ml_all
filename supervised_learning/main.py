import sys
import pandas as pd
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

RANDOM_STATE = 1337

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

def get_cross_validated_clf(X_train, y_train):
    parameters_to_tune = {
        'max_depth': range(3, 20),
        'criterion': ('gini', 'entropy'),
        'ccp_alpha': [0.0, 0.000005, 0.00005, 0.0005, 0.005],
        'min_samples_split':range(2,10),
        'min_samples_leaf':range(1,5),
        'random_state':[RANDOM_STATE]
    }
    clf = GridSearchCV(
        dt.DecisionTreeClassifier(),
        param_grid=parameters_to_tune,
        cv=5,
        #verbose=3,
        n_jobs=-1)
    clf.fit(X=X_train, y=y_train)
    tree_model = clf.best_estimator_
    print(clf.best_score_, clf.best_params_)
    print('Accuracy Score on cross-validated DT: ' + str(clf.best_score_))
    print('Size of cross-validated DT: ' + str(tree_model.tree_.node_count))
    # res = pd.DataFrame(clf.cv_results_)
    # print(res)
    # exit(1)
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

def find_optimal_clf(X_train, X_test, y_train, y_test):
    tree_model, best_params = get_cross_validated_clf(X_train, y_train)
    #max_score, optimal_clf = get_pruned_clf(tree_model, best_params, X_train, X_test, y_train, y_test)
    max_score = tree_model.score(X_test, y_test)
    return max_score, tree_model



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
        #remainder='passthrough'
    )
    clean_attrs = column_trans.fit_transform(attributes)
    #TODO: we could maybe eek more accuracy out of the test_size param
    #TODO: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

    X_train, X_test, y_train, y_test = train_test_split(clean_attrs, target, stratify=target, test_size=0.3, random_state=RANDOM_STATE)
    max_score, optimal_clf = find_optimal_clf(X_train, X_test, y_train, y_test)
    print('Accuracy Score on pruned DT: '+str(max_score))
    print('Size of pruned DT: ' +str(optimal_clf.tree_.node_count))
    exit(1)


def run_ford(dataroot):
    df_train = pd.read_csv(dataroot + 'fordTrain.csv')
    df_test = pd.read_csv(dataroot + 'fordTest.csv')
    y_train = df_train['IsAlert']
    X_train = df_train.drop('IsAlert', axis=1)
    x_test = df_test.drop('IsAlert', axis=1)

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