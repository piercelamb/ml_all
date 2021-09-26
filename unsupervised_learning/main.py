import sys
import mlrose_hiive as rose
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import time
import random
from functools import wraps

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
    plt.ylabel('Fitness results')
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
    problem_sizes = [100, 250, 500, 750]
    const_problem = 200
    max_attempts = 100
    max_iters = [100, 1000, 5000, 10000]
    const_iters = 150
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



if __name__ == "__main__":
    perform_experiments()
    #test_nn()
    # passed_arg = sys.argv[1]
    # if passed_arg.startswith('/'):
    #     dataroot = passed_arg
    # else:
    #     dataroot = '/Users/plamb/Documents/Personal/Academic/Georgia Tech/Classes/ML/hw/unsupervised_learning/data/'
    # if passed_arg == 'shoppers':
    #     run_shoppers(dataroot)
    # elif passed_arg == 'ford':
    #     run_ford(dataroot)
    # else:
    #     print("please run with an absolute path to the data")
    #     exit(146)