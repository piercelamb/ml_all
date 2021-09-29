import sys
import mlrose_hiive as rose
from mlrose_hiive import simulated_annealing as sa, random_hill_climb as rhc, genetic_alg as ga, mimic
from mlrose_hiive import RHCRunner, SARunner, GARunner, NNGSRunner, ExpDecay, NNClassifier
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import time
import random
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import make_column_transformer
import sklearn.metrics as skmt
from functools import partial

RANDOM_STATE = 1337

def get_alg(alg_name, kwargs, schedule, mutation_prob, keep_pct, vector_length):
    if alg_name == 'hill_climb':
        initial_state = np.random.randint(2, size=vector_length)
        kwargs['init_state'] = initial_state
        return rose.random_hill_climb(**kwargs)
    elif alg_name == 'annealing':
        initial_state = np.random.randint(2, size=vector_length)
        kwargs['init_state'] = initial_state
        kwargs['schedule'] = schedule
        return rose.simulated_annealing(**kwargs)
    elif alg_name == 'genetic':
        kwargs['mutation_prob'] = mutation_prob
        return rose.genetic_alg(**kwargs)
    elif alg_name == 'mimic':
        kwargs['keep_pct'] = keep_pct
        return rose.mimic(**kwargs)

def vary_problem_size(alg_name,time_results, curve_results, func_name, algs, fitness_func, state_vector_sizes, max_attempts, max_iters, mutation_prob, keep_pct, maximize, schedule, curve, const_problem, const_iters):
    print("Iters constant is: " + str(const_iters))
    curve_results[alg_name] = []
    for vector_length in state_vector_sizes:
        opt_prob = rose.DiscreteOpt(fitness_fn=fitness_func, maximize=maximize, length=vector_length)
        kwargs = {'problem': opt_prob,
                  'max_attempts': max_attempts,
                  'max_iters': const_iters,
                  'random_state': RANDOM_STATE,
                  'curve': curve}

        start_time = time.time()
        best_state, best_fitness, fitness_curve = get_alg(alg_name, kwargs, schedule, mutation_prob, keep_pct,
                                                          vector_length)
        end_time = time.time()
        total_time = end_time - start_time
        time_results.at[vector_length, alg_name] = total_time
        curve_results[alg_name].append(best_fitness)
        print("Problem size " + str(vector_length) + " took " + str(total_time) + " seconds")
        kwargs.clear()

    return time_results, curve_results

def vary_iteration_size(alg_name,time_results, curve_results, func_name, algs, fitness_func, state_vector_sizes, max_attempts, max_iters, mutation_prob, keep_pct, maximize, schedule, curve, const_problem, const_iters):
    print("Problem constant is: "+str(const_problem))
    for iters in max_iters:
        opt_prob = rose.DiscreteOpt(fitness_fn=fitness_func, maximize=maximize, length=const_problem)
        kwargs = {'problem': opt_prob,
                  'max_attempts': max_attempts,
                  'max_iters': iters,
                  'random_state': RANDOM_STATE,
                  'curve': curve}

        start_time = time.time()
        best_state, best_fitness, fitness_curve = get_alg(alg_name, kwargs, schedule, mutation_prob, keep_pct, const_problem)
        end_time = time.time()
        total_time = end_time - start_time
        print("Iterations " + str(iters) + " took " + str(total_time) + " seconds")

        time_results.at[iters, alg_name] = total_time
        curve_results[alg_name] = fitness_curve
        kwargs.clear()

    return time_results, curve_results

def get_and_plot_alg_results(func_name, algs, fitness_func, state_vector_sizes, max_attempts, max_iters, mutation_prob, keep_pct, maximize, schedule, curve, const_problem, const_iters):
    experiments = {
        'iterations':vary_iteration_size,
        'problem_size': vary_problem_size
    }
    for experiment_name, experiment in experiments.items():
        print("------------")
        print("Varying "+experiment_name)
        time_results = pd.DataFrame(None, columns=algs)
        curve_results = {}
        for alg_name in algs:
            print("Executing alg: "+alg_name)
            time_results, curve_results = experiment(alg_name, time_results, curve_results, func_name, algs, fitness_func, state_vector_sizes, max_attempts, max_iters, mutation_prob, keep_pct, maximize, schedule, curve, const_problem, const_iters)

        plot_results(func_name, experiment_name, time_results, curve_results, max_iters,state_vector_sizes)




def plot_results(name, experiment_name, time_results, curve_results, max_iters, state_vector_sizes):
    print("Plotting time results for "+name)
    ax = plt.gca()
    plt.grid(True)
    plt.title(name+" problem time results ("+experiment_name+")")
    plt.xlabel(experiment_name)
    plt.ylabel('Time')
    for column in time_results:
        time_results.plot(kind='line', y=column, ax=ax, marker='o')
    plt.legend(loc="best")
    plt.savefig(name+"_problem_time_results_"+experiment_name+".png")
    plt.clf()

    print("Plotting "+experiment_name+" results for "+name)
    plt.grid(True)
    plt.title(name + " problem "+experiment_name+" results")
    plt.xlabel(experiment_name)
    plt.ylabel('Fitness')
    if experiment_name == 'iterations':
        for alg_name, curve in curve_results.items():
            iters = range(1, len(curve)+1) #required because GA returns very odd iteration numbers
            plt.plot(iters,curve[:,0], label=alg_name)
    else:
        for alg_name, curve in curve_results.items():
            plt.plot(state_vector_sizes,curve, label=alg_name)
    plt.legend(loc="best")
    plt.savefig(name + "_problem_"+experiment_name+"_results.png")
    plt.clf()



def perform_experiments():
    #before mimic choked:
    # problem_sizes = [100, 250, 500, 750]
    # const_problem = 200
    # max_attempts = 100
    # max_iters = [100, 1000, 5000, 10000]
    problem_sizes = [50, 100, 150, 200]
    const_problem = 100
    max_attempts = 100
    max_iters = [50, 200, 400, 600]
    const_iters = 100
    mutation_prob = 0.5
    keep_pct = 0.5
    maximize = True
    curve = True
    schedule = rose.ExpDecay()
    print('Constructing Fitness Functions')
    fitness_funcs = {
                        'One_Max': rose.OneMax(),
                        #'Four_Peaks': rose.FourPeaks(),
                        #'Flip_Flop': rose.FlipFlop()
    }
    algs = [
        'hill_climb',
        'annealing',
        'genetic',
        'mimic'
    ]
    for name, fitness_func in fitness_funcs.items():
        print("\n-------------------------------------\n")
        print("Running fitness_func: "+name)
        get_and_plot_alg_results(
            name,
            algs,
            fitness_func,
            problem_sizes,
            max_attempts,
            max_iters,
            mutation_prob,
            keep_pct,
            maximize,
            schedule,
            curve,
            const_problem,
            const_iters
        )

def drop_correlated_columns(X_train, X_test):
    # correlation = X_train.corr().abs()
    # get_graph = c.unstack()
    # final = get_graph.order(kind="quicksort", na_last=False)[::-1]

    X_test = X_test.drop(['P4', 'V6', 'V10', 'E9', 'E2'], axis=1)
    X_train = X_train.drop(['P4', 'V6', 'V10', 'E9', 'E2'], axis=1)
    return X_train, X_test

def scale_variant_columns(X_train, X_test):

    X_train.loc[X_train['E7'] > 4, 'E7'] = 4
    X_train.loc[X_train['E8'] > 4, 'E8'] = 4
    X_test.loc[X_test['E7'] > 4, 'E7'] = 4
    X_test.loc[X_test['E8'] > 4, 'E8'] = 4
    return X_train, X_test

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
    X_train, X_test = drop_correlated_columns(X_train, X_test)
    X_train, X_test = scale_variant_columns(X_train, X_test)
    X_train, X_test = scale_continuous_data(X_train, X_test)

    return X_train, X_test

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

    elif num_features == 'small':
        #Best Feature Selection
        print("Selecting SFS Forward features")
        X_train = df_train[['P6', 'E9', 'V10']]
        X_test = df_test[['P6', 'E9', 'V10']]
    elif num_features == 'munge':
        print("Preprocessing ford data")
        X_train, X_test = preprocess_ford_data(df_train.drop(['IsAlert', 'TrialID', 'ObsNum'], axis=1), df_test.drop(['IsAlert', 'TrialID', 'ObsNum'], axis=1))

    if dataset_sample != 0:
        print("Sampling the training set at: "+str(dataset_sample))
        X_train, X_test_new, y_train, y_test_new = train_test_split(X_train, y_train, train_size=dataset_sample,
                                                            random_state=RANDOM_STATE)
    return X_train, y_train, X_test, y_test

def get_data_OSI(smote):
    datapath = dataroot + 'online_shoppers_intention.csv'
    df = pd.read_csv(datapath)
    target = df['Revenue'] = df['Revenue'].astype(int)
    attributes = df.drop('Revenue', axis=1, )
    string_columns = ['Month', 'VisitorType', 'Weekend']
    numerical_columns = ['Administrative', 'Administrative_Duration', 'Informational',
       'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
       'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay',
       'OperatingSystems', 'Browser', 'Region', 'TrafficType',]
    column_trans = make_column_transformer(
        (OneHotEncoder(handle_unknown='ignore'), string_columns),
        (MinMaxScaler(), numerical_columns)
        #remainder='passthrough'
    )
    clean_attrs = column_trans.fit_transform(attributes)

    X_train, X_test, y_train, y_test = train_test_split(clean_attrs, target, stratify=target, test_size=0.3,
                                                        random_state=RANDOM_STATE)
    # is_smote = smote[0]
    # if is_smote:
    #     print("Smote resampling initiated, Counter before: "+str(Counter(y_train)))
    #     imbPipeline = get_smote_pipeline(smote)
    #     X_train, y_train = imbPipeline.fit_resample(X_train, y_train)
    #     print("Smote resampling complete, Counter after: "+str(Counter(y_train)))

    return X_train, y_train, X_test, y_test

def get_the_runner(alg, X_train, y_train, X_test, y_test, grid_search_parameters, max_iters):
    default_params = {
        'x_train':X_train,
        'y_train':y_train,
        'x_test':X_test,
        'y_test':y_test,
        'experiment_name':alg,
        #'iteration_list':[1, 10, 50, 100, 250, 500, 1000, 10000],
        'iteration_list': [max_iters],
        'bias':True,
        'early_stopping':True,
        'clip_max':5,
        'max_attempts':500,
        'n_jobs':-2,
        'seed':RANDOM_STATE,
        'output_directory':None
    }
    if alg == 'random_hill_climb':
        custom_params = {
            'restarts': [3, 5],
            'algorithm': rose.algorithms.rhc.random_hill_climb,
        }
    elif alg == 'simulated_annealing':
        custom_params = {
            'schedule': [ExpDecay(1),ExpDecay(10),ExpDecay(25)],
            'algorithm': rose.algorithms.sa.simulated_annealing,
        }
    elif alg == 'genetic_alg':
        custom_params = {
            "pop_size": [5, 10, 25],
            "mutation_prob": [0.001, 0.01, 1],
            'algorithm': rose.algorithms.ga.genetic_alg,
        }
    else:
        #gradient descent
        custom_params = {
            "learning_rate_init": [0.0002],
            'algorithm': rose.algorithms.gd.gradient_descent,
        }
    final_params ={
        **default_params,
        **custom_params,
        'grid_search_parameters': grid_search_parameters,
        #'grid_search_scorer_method': partial(skmt.f1_score, average="binary")
    }

    return NNGSRunner(**final_params)

def get_loss_curve(alg, curves, min_row):
    if alg == 'random_hill_climb':
        best_fitness_restart = min_row.iloc[0]['current_restart']
        curves = curves[curves.current_restart == best_fitness_restart]
        curves.reset_index(inplace=True, drop=True)
    curves.plot(title=alg+" NN Fitness over Iterations", xlabel="Iterations", ylabel="Fitness", x="Iteration",
                y="Fitness")
    plt.savefig(alg+"_nn_fitness.png")
    plt.clf()

    return curves['Fitness']

def get_analysis(alg, run_stats, curves):
    run_stats = run_stats[run_stats['Iteration'] != 0]
    min_row = run_stats[run_stats.Fitness == run_stats.Fitness.min()]
    best_fitness = min_row['Fitness']
    print(alg + " best fitness: " + str(best_fitness))
    loss_curve = get_loss_curve(alg, curves, min_row)

    return best_fitness, loss_curve

def perform_nn(dataroot):
    # training_sample = 0.3
    # num_features = 'munge'
    # X_train, y_train, X_test, y_test = get_data_ford(training_sample, num_features)
    # scoring = 'accuracy'
    # cross_val_folds = 10
    # smote = (False, None, None)
    # run_type = 'FAD'

    smote = (True, 0.6)
    X_train, y_train, X_test, y_test = get_data_OSI(smote)
    scoring = 'f1'
    cross_val_folds = 10
    run_type = 'OSI'

    #assignment1 FAD params: 1 layer, 150 neurons, tanh as activation
    #assignment1 OSI params: 2 layers, 100 neurons, sigmoid
    algs = [
        'gradient_descent',
        'random_hill_climb',
        'simulated_annealing',
        'genetic_alg'
    ]

    all_fitness = {}
    all_time = {}
    all_scoring = {}
    all_curves = {}
    for alg in algs:
        print("\n-------------------------\n")
        print("Executing NN with "+alg)
        max_iters = 1000
        grid_search_parameters = {
            'hidden_layer_sizes':[[100], [100, 200]],
            'max_iters': [max_iters],
            'learning_rate': [0.00001, 0.0001, 0.001, 0.01],
            'activation': [rose.sigmoid],
        }

        runner = get_the_runner(alg, X_train, y_train, X_test, y_test, grid_search_parameters, max_iters)

        start_time = time.time()
        run_stats, curves, cv_results, grid_search_cv = runner.run()
        total_time = time.time() - start_time
        print(alg+" complete, running time: "+str(total_time))
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        print(cv_results)
        print(grid_search_cv)
        print(grid_search_cv.best_score_)
        print(grid_search_cv.best_params_)
        best_fitness, loss_curve = get_analysis(alg, run_stats, curves)
        all_curves[alg] = loss_curve
        all_fitness[alg] = best_fitness
        all_time[alg] = total_time
        all_scoring[alg] = grid_search_cv.best_score_
        print(alg + " best score " + str(grid_search_cv.best_score_))
        exit(1)

    pd.DataFrame(all_curves).plot(title="Neural Network Convergence", ylabel="Fitness", xlabel="Iterations")
    plt.savefig("neural_network_convergence.png")
    plt.clf()




if __name__ == "__main__":
    passed_arg = sys.argv[1]
    if passed_arg.startswith('/'):
        dataroot = passed_arg
    else:
        dataroot = '/Users/plamb/Documents/Personal/Academic/Georgia Tech/Classes/ML/hw/unsupervised_learning/data/'
    if passed_arg == 'experiments':
        perform_experiments()
    elif passed_arg == 'nn':
        perform_nn(dataroot)
    else:
        print("please run with an absolute path to the data")
        exit(146)
