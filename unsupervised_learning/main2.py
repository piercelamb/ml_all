import statistics
import sys
import mlrose_hiive as rose
from mlrose_hiive import simulated_annealing as sa, random_hill_climb as rhc, genetic_alg as ga, mimic
from mlrose_hiive import RHCRunner, SARunner, GARunner, NNGSRunner, ExpDecay, NNClassifier, GeomDecay, MIMICRunner
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import time as time_keeper
import random
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import learning_curve
from collections import Counter
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import make_column_transformer
import sklearn.metrics as skmt
from functools import partial
from itertools import islice

RANDOM_STATE = 1337

def plot_results(func_name, alg_name, final_df, problem_size):
    final_df.plot(title=alg_name+" Avg Fitness over Iterations PR="+str(problem_size), xlabel="Iterations", ylabel="Fitness",
                y="Mean_Fitness")
    plt.savefig(func_name+"_"+ alg_name +"_"+ str(problem_size)+"_avg_fitness_iterations.png")
    plt.clf()
    final_df.plot(title=alg_name+" Avg Fitness over FEvals PR="+str(problem_size), xlabel="Function Evaluations", ylabel="Fitness",
                y="Mean_Fitness", x="Mean_FEvals")
    plt.savefig(func_name+"_"+ alg_name +"_"+ str(problem_size)+"_avg_fitness_fevals.png")
    plt.clf()
    # final_df.plot(title=alg_name+" Avg Time over Iterations", xlabel="Function Evaluations", ylabel="Time",
    #             y="Mean_Time", x="Mean_FEvals")
    # plt.savefig(alg_name + "_avg_time_iterations.png")
    # plt.clf()

def plot_combined_results(func_name,combined_df, problem_size):
    cols = combined_df.columns
    fitness_cols = filter(lambda x: x.endswith('_fitness'), cols)
    fitness_only = combined_df[fitness_cols]
    fitness_only.plot(title=func_name+ " Avg Fitness over Iterations PR="+str(problem_size), xlabel="Iterations", ylabel="Fitness")
    plt.savefig(func_name +"_"+ str(problem_size)+ "_COMBINED_avg_fitness_iterations.png")
    plt.clf()

    # fevals_cols = filter(lambda x: x.endswith('_fevals'), cols)
    # fevals_only = combined_df[fevals_cols]
    # mean_fevals = pd.DataFrame()
    # mean_fevals['Mean_FEvals'] = fevals_only.mean(axis=1)
    # fitness_only['FEvals'] = mean_fevals
    # combined_df.plot(title=func_name+" Avg Fitness over Avg FEvals", xlabel="Function Evaluations", ylabel="Fitness",
    #             y=fitness_cols, x=fevals_cols)
    # plt.savefig(func_name + "_COMBINED_avg_fitness_fevals.png")
    # plt.clf()
    # exit(1)

def get_analysis(alg,run_stats, curves, seed):
    #run_stats = run_stats[run_stats['Iteration'] != 0]
    max_row = run_stats[run_stats.Fitness == run_stats.Fitness.max()]
    if max_row.size > 1:
        #occasionally multiple runs get to the same maximum fitness
        max_row = max_row[max_row.Time == max_row.Time.min()]
    best_fitness = max_row.iloc[0]['Fitness']
    print(alg + " best fitness: " + str(best_fitness))

    if alg == 'hill_climb':
        best_fitness_current_restart = max_row.iloc[0]['current_restart']
        best_fitness_restart = max_row.iloc[0]['Restarts']
        curves = curves[curves.current_restart == best_fitness_current_restart]
        curves = curves[curves.Restarts == best_fitness_restart]
        curves.reset_index(inplace=True, drop=True)
    if alg == 'annealing':
        best_fitness_temp = max_row.iloc[0]['Temperature']
        curves = curves[curves.Temperature == best_fitness_temp]
        curves.reset_index(inplace=True, drop=True)
    if alg == 'genetic':
        best_mutation = max_row.iloc[0]['Mutation Rate']
        best_pop = max_row.iloc[0]['Population Size']
        curves = curves[curves['Mutation Rate'] == best_mutation]
        curves = curves[curves['Population Size'] == best_pop]
        curves.reset_index(inplace=True, drop=True)
    if alg == 'mimic':
        best_keep = max_row.iloc[0]['Keep Percent']
        best_pop = max_row.iloc[0]['Population Size']
        curves = curves[curves['Keep Percent'] == best_keep]
        curves = curves[curves['Population Size'] == best_pop]
        curves.reset_index(inplace=True, drop=True)

    id = alg+str(seed)
    curves.rename(columns={'Iteration': id+'_iteration', 'Time': id+'_time', 'FEvals': id+'_fevals', 'Fitness':id+'_fitness'}, inplace=True)
    time = curves[id+'_time']
    fevals = curves[id+'_fevals']
    fitness = curves[id+'_fitness']
    return best_fitness, time, fevals, fitness

def get_and_plot_alg_results(func_name, algs, fitness_func, total_time, state_vector_sizes, max_attempts, max_iters, mutation_prob,
                             keep_pct, maximize, schedule, curve, const_problem, const_iters, seeds, problem_size):
    combined_df = pd.DataFrame()
    for alg_name in algs:
        avg_time = pd.DataFrame()
        avg_fevals = pd.DataFrame()
        avg_fitness = pd.DataFrame()
        final_df = pd.DataFrame()
        best_fitness_arr = []
        start_time = time_keeper.time()
        for seed in seeds:
            print("Executing alg: " + alg_name+" with seed: "+str(seed))
            opt_prob = rose.DiscreteOpt(fitness_fn=fitness_func, maximize=maximize, length=problem_size)
            runner = get_alg(alg_name, opt_prob, func_name, seed)

            df_run_stats, df_run_curves = runner.run()

            best_fitness, time, fevals, fitness = get_analysis(alg_name,df_run_stats, df_run_curves, seed)
            id=alg_name+str(seed)
            avg_time[id] = time
            avg_fevals[id] = fevals
            avg_fitness[id] = fitness
            best_fitness_arr.append(best_fitness)

        end_time = time_keeper.time()
        print("Total time to learn across all seeds: "+str(end_time-start_time))
        avg_time['Mean_Time'] = avg_time.mean(axis=1)
        avg_fevals['Mean_FEvals'] = avg_fevals.mean(axis=1)
        avg_fitness['Mean_Fitness'] = avg_fitness.mean(axis=1)
        final_df['Mean_Time'] = avg_time['Mean_Time']
        final_df['Mean_FEvals'] = avg_fevals['Mean_FEvals']
        final_df['Mean_Fitness'] = avg_fitness['Mean_Fitness']
        #pd.set_option("display.max_rows", None, "display.max_columns", None)

        calc_time = final_df[final_df['Mean_Fitness'] < statistics.mean(best_fitness_arr)]
        mean_time = calc_time['Mean_Time'].sum()
        print(alg_name + " Mean Time: " + str(mean_time))
        total_time.at[func_name, alg_name] = mean_time
        plot_results(func_name, alg_name, calc_time, problem_size)
        combined_df[alg_name+"_fitness"] = final_df['Mean_Fitness']
        combined_df[alg_name+"_fevals"] = final_df['Mean_FEvals']

    plot_combined_results(func_name, combined_df, problem_size)
    html = total_time.to_html(index=True)
    with open("total_time.html", 'w') as fp:
        fp.write(html)

def get_alg(alg_name, opt_prob, func_name, seed):
    default_params = {'problem': opt_prob,
              'experiment_name': func_name,
              'max_attempts': 200,
              'iteration_list':[50, 200, 400],
              'seed': seed,
              }
    if alg_name == 'hill_climb':
        custom_params = {
            'restart_list': [3, 5, 10],
        }
        return RHCRunner(**default_params, **custom_params)
    elif alg_name == 'annealing':
        custom_params = {
            'decay_list':[ExpDecay, GeomDecay],
            'temperature_list':[1, 5, 10, 50]
        }
        return SARunner(**default_params, **custom_params)
    elif alg_name == 'genetic':
        custom_params = {
            "population_sizes": [50, 100, 150, 200],
            "mutation_rates": [0.001, 0.01, 1],
        }
        return GARunner(**default_params, **custom_params)
    else:
        #MIMiC
        custom_params = {
            "population_sizes": [100, 250, 400],
            "keep_percent_list": [0.1, 0.25, 0.5, 0.75],
            'use_fast_mimic': True
        }
        return MIMICRunner(**default_params, **custom_params)

def perform_experiments():
    #before mimic choked:
    # problem_sizes = [100, 250, 500, 750]
    # const_problem = 200
    # max_attempts = 100
    # max_iters = [100, 1000, 5000, 10000]
    problem_sizes = [100, 200]
    const_problem = [100, 200]
    max_attempts = 100
    max_iters = [50, 200, 400]
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
        #'genetic',
        #'mimic'
    ]
    random_seeds = [
        1,2,3
    ]
    for problem_size in problem_sizes:
        for name, fitness_func in fitness_funcs.items():
            print("\n-------------------------------------\n")
            print("Running fitness_func: "+name+" with problem size: "+str(problem_size) )
            get_and_plot_alg_results(
                name,
                algs,
                fitness_func,
                pd.DataFrame(index=fitness_funcs.keys()),
                problem_sizes,
                max_attempts,
                max_iters,
                mutation_prob,
                keep_pct,
                maximize,
                schedule,
                curve,
                const_problem,
                const_iters,
                random_seeds,
                problem_size
            )

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