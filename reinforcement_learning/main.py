import sys
import pprint
import hiive.mdptoolbox as mdp
from hiive.mdptoolbox.example import forest
from hiive.mdptoolbox.mdp import ValueIteration, PolicyIteration, QLearning
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
import gym
from gym.envs.toy_text.frozen_lake import generate_random_map
import time
from itertools import cycle, islice

RANDOM_SEED = 1337

def plot_env(env, figsize, title, policy=[]):
    print("Plotting lake with figsize: "+str(figsize))
    color_map = {
        'S': 'c',
        'F': 'w',
        'H': 'r',
        'G': 'g'
    }
    tiles = env.ncol
    figure = plt.figure(figsize=(figsize, figsize))
    plt.axis('off')
    ax = figure.add_subplot(111, xlim=(-.01, tiles + 0.01), ylim=(-.01, tiles + 0.01))

    for n in range(0, tiles):
        for x in range(0, tiles):
            y = tiles - 1 - n
            plot = plt.Rectangle([x, y], 1, 1, edgecolor='k', linewidth=1)
            tile_letter = env.desc[n, x].decode("utf-8")
            plot.set_facecolor(color_map[tile_letter])
            ax.add_patch(plot)

            if policy:
               pass

    plt.savefig(title)

def get_transitions_rewards(env, map_size):
    np_size = map_size * map_size
    transitions = np.zeros((4, np_size, np_size))
    rewards = np.zeros((4, np_size, np_size))
    old_state = 0
    tiles = env.P
    for tile in tiles:
        actions = env.P[tile]
        for action in actions:
            possible_transitions = tiles[tile][action]
            for i in range(len(possible_transitions)):
                probability = possible_transitions[i][0]
                reward = possible_transitions[i][2]
                new_state = possible_transitions[i][1]
                if new_state != old_state:
                    transitions[action][tile][new_state] = probability
                    rewards[action][tile][new_state] = reward
                else:
                    transitions[action][tile][new_state] = transitions[action][tile][old_state] + probability
                    rewards[action][tile][new_state] = rewards[action][tile][old_state] + reward
                old_state = new_state
    return transitions, rewards

def set_run_data(runner, run_df, run_data, discount, epsilon=None):
    best_run = run_data[-1]

    if epsilon:
        col = str(discount)+'_'+str(epsilon)
    else:
        col = str(discount)
    run_df.at['time', col] = best_run['Time']
    run_df.at['iterations', col] = best_run['Iteration']
    run_df.at['reward', col] = best_run['Reward']
    run_df.at['Max V', col] = best_run['Max V']
    run_df.at['Mean V', col] = best_run['Mean V']
    run_df.at['Error', col] = best_run['Error']
    run_df.at['policy', col] = runner.policy

#derived from
#https://tinyurl.com/3n8vdj8
def get_policy_results(env, policy):
    num_misses = 0
    steps = []
    for episode in range(0, 1000):
        step=0
        obs = env.reset()
        while True:
            action = policy[obs]
            obs, reward, done, non = env.step(action)
            step = step + 1
            if done and reward == 0:
                num_misses = num_misses + 1
                break
            elif done and reward == 1:
                steps.append(step)
                break

    failed = (num_misses/1000) * 100
    avg_steps = np.mean(steps)

    # print('----------------------------------------------')
    # print('You took an average of {:.0f} steps to get the frisbee'.format(avg_steps))
    # print('And you fell in the hole {:.2f} % of the times'.format(failed))
    # print('----------------------------------------------')

    return failed, avg_steps

def q_create_df(discounts, epsilons, alphas, alpha_decays, epsilon_decays, max_iters):
    index = [
        'time',
        'iterations',
        'reward',
        'Max V',
        'Mean V',
        'Error',
        'avg_steps',
        'success',
        'policy'
    ]
    columns = []
    for discount in discounts:
        for epsilon in epsilons:
            for epsilon_decay in epsilon_decays:
                for alpha in alphas:
                    for alpha_decay in alpha_decays:
                        for max_iter in max_iters:
                            columns.append(str(discount)+'_'+str(epsilon)+'_'+str(epsilon_decay)+'_'+str(alpha)+'_'+str(alpha_decay)+'_'+str(max_iter))
    return pd.DataFrame(columns=columns, index=index)



def set_run_data_q(runner, run_df, run_data, discount, epsilon, alpha, alpha_decay, epsilon_decay, max_iter):
    best_run = run_data[-1]
    col = str(discount)+'_'+str(epsilon)+'_'+str(epsilon_decay)+'_'+str(alpha)+'_'+str(alpha_decay)+'_'+str(max_iter)
    run_df.at['time', col] = best_run['Time']
    run_df.at['iterations', col] = best_run['Iteration']
    run_df.at['reward', col] = best_run['Reward']
    run_df.at['Max V', col] = best_run['Max V']
    run_df.at['Mean V', col] = best_run['Mean V']
    run_df.at['Error', col] = best_run['Error']
    run_df.at['policy', col] = runner.policy

def run_Q(transitions, rewards, map_size, discounts, epsilons, alphas, alpha_decays, epsilon_decays, max_iters, env=None):
    print("Running QLearning with map size "+str(map_size))
    run_df = q_create_df(discounts, epsilons, alphas, alpha_decays, epsilon_decays, max_iters)
    start = time.time()
    for d in discounts:
        print("trying discount: "+str(d))
        for e in epsilons:
            print("trying epsilon: "+str(e))
            for e_d in epsilon_decays:
                print("trying e_decay: "+str(e_d))
                for a in alphas:
                    print("trying alpha: "+str(a))
                    for a_d in alpha_decays:
                        print("trying a_decay: "+str(a_d))
                        for i in max_iters:
                            print("trying iters: "+str(i))
                            runner = QLearning(transitions, rewards, gamma=d, epsilon=e, epsilon_decay=e_d, alpha=a, alpha_decay=a_d, n_iter=i)
                            run_data = runner.run()
                            set_run_data_q(runner, run_df, run_data, d, e, a, a_d, e_d, i)

    finish = time.time() - start
    print("QLearning completed in: " + str(finish))
    if env:
        print("Getting policy scores")
        for col, val in run_df.loc['policy'].items():
            failed, avg_steps = get_policy_results(env, val)
            run_df.at['success', col] = 100 - failed
            run_df.at['avg_steps', col] = avg_steps

    return run_df

def run_lake():
    map_size = 4
    game = 'lake'
    #TODO realized for lake that higher discounts and lower epsilons were better early on
    # discounts = [0.5, 0.6, 0.7, 0.8, 0.9]
    # epsilons = [0.00001, 0.0000001, 0.000000001]
    # alphas = [0.01, 0.1, 0.2]
    # alpha_decays = [0.7, 0.8, 0.9]
    # epsilon_decays = [0.7, 0.8, 0.9]
    # max_iters = [10000, 100000]

    discounts = [0.9]
    epsilons = [0.0000001]
    alphas = [0.01]
    alpha_decays = [0.9]
    epsilon_decays = [0.9]
    max_iters = [1000000]

    print("Running lake problem with map size "+str(map_size))
    # random_map = generate_random_map(size=map_size, p=0.8)
    # env = gym.make("FrozenLake-v1", desc=random_map).unwrapped
    env = gym.make('FrozenLake-v1').unwrapped
    env.max_episode_steps=300
    plot_env(env, map_size, title='lake_'+str(map_size)+'.png')
    transitions, rewards = get_transitions_rewards(env, map_size)
    type = 'value'
    #value_res = run_mdp(type, transitions, rewards, discounts, map_size, env=env, epsilons=epsilons)
    #analyze_mdp(game, type, value_res)
    # type = 'policy'
    # policy_res = run_mdp(type, transitions, rewards, discounts, map_size, env=env)
    # analyze_mdp(game, type, policy_res)
    q_res = run_Q(transitions, rewards, map_size, discounts, epsilons, alphas, alpha_decays, epsilon_decays, max_iters, env)
    type = 'QL'
    analyze_mdp(game, type, q_res)

def run_mdp(type, transitions, rewards, discounts, map_size, env=None, epsilons=None):
    print("Running "+type+" iterations with map size "+str(map_size))
    index = [
        'time',
        'iterations',
        'reward',
        'Max V',
        'Mean V',
        'Error',
        'avg_steps',
        'success',
        'policy'
    ]
    columns = []
    for discount in discounts:
        if epsilons:
            for epsilon in epsilons:
                columns.append(str(discount)+'_'+str(epsilon))
        else:
            columns.append(str(discount))
    run_df = pd.DataFrame(columns=columns, index=index)
    start = time.time()

    for discount in discounts:
        if epsilons:
            for epsilon in epsilons:
                #converges when either max iter hit or the maximum change in value function falls
                #below the passed epsilon value
                runner = ValueIteration(transitions, rewards, epsilon=epsilon, gamma=discount)
                run_data = runner.run()
                set_run_data(runner, run_df, run_data, discount, epsilon)
        else:
            runner = PolicyIteration(transitions, rewards, gamma=discount)
            run_data = runner.run()
            set_run_data(runner, run_df, run_data, discount)

    finish = time.time() - start
    print(type+" finished in: "+str(finish))
    if env:
        print("Getting policy scores")
        for col, val in run_df.loc['policy'].items():
            failed, avg_steps = get_policy_results(env, val)
            run_df.at['success', col] = 100 - failed
            run_df.at['avg_steps', col] = avg_steps

    return run_df

def analyze_mdp(game, type, df):
    if game == 'lake':
        sorted_df = df.sort_values(['success', 'avg_steps'], ascending=[True, False], axis=1)
    else:
        sorted_df = df.sort_values('reward', axis=1)
    top_5 = sorted_df[sorted_df.columns[-5:]]
    print(top_5.drop('policy'))
    print(top_5.columns)
    best_run = top_5.iloc[:,-1:]
    print("The best policy was: ")
    print(best_run.loc['policy', :].tolist())
    top_5 = top_5.T
    if type == 'value':
        xlabel = "Discount and Epsilon with '_' seperator"
        plot_metadata = {
            'success':{'xlabel':xlabel, 'title':"Success vs Discount & Epsilon", 'ylabel':"Value iter, success % over 1000 episodes"},
            'avg_steps':{'xlabel':xlabel, 'title':"Avg Steps vs Discount & Epsilon", 'ylabel':"Value iter, avg steps over 1000 episodes"},
            'time':{'xlabel':xlabel, 'title':"Time vs Discount & Epsilon", 'ylabel':"Value Iter time to converge"},
            'iterations': {'xlabel': xlabel, 'title': "Iterations vs Discount & Epsilon", 'ylabel': "Value Iterations to converge "},
            'reward':{'xlabel':xlabel, 'ylabel': "Reward", 'title': 'Value Iter Top 5 Reward'}
        }
    elif type == 'policy':
        xlabel = "Discount Values"
        plot_metadata = {
            'success':{'xlabel':xlabel, 'title':"Success vs Discount", 'ylabel':"Policy iter, success % over 1000 episodes"},
            'avg_steps':{'xlabel':xlabel, 'title':"Avg Steps vs Discount", 'ylabel':"Policy iter, avg steps over 1000 episodes"},
            'time': {'xlabel': xlabel, 'title': "Time vs Discount",'ylabel': "Policy iter, time to converge"},
            'iterations': {'xlabel': xlabel,'title': "Iterations vs Discount", 'ylabel': "Policy iterations to converge"},
            'reward': {'xlabel': xlabel, 'ylabel': "Reward", 'title': 'Policy iter, top 5 Reward'}
        }
    else:
        xlabel='All hyper params'
        plot_metadata = {
            'success':{'xlabel':xlabel, 'title':"Success vs Hyper Params", 'ylabel':"QL, Success % over 1000 episodes"},
            'avg_steps':{'xlabel':xlabel, 'title':"Avg Steps vs Hyper Params", 'ylabel':"QL, Avg steps over 1000 episodes"},
            'time': {'xlabel':xlabel, 'title': "Time vs Hyper Params",'ylabel': "QL, Time to converge"},
            'iterations': {'xlabel': xlabel,'title': "Iterations vs Hyper Params", 'ylabel': "QL, Iterations to converge"},
            'reward': {'xlabel': xlabel, 'ylabel': "Reward", 'title': 'QL, top 5 Reward'}
        }
    for y, metadata in plot_metadata.items():
        if type == 'QL':
            top_5.plot(y=y, xlabel=metadata['xlabel'], ylabel=metadata['ylabel'], title=metadata['title'], kind='bar')
        else:
            top_5.plot(y=y, xlabel=metadata['xlabel'], ylabel=metadata['ylabel'], title=metadata['title'], kind='bar', rot=0)
        plt.savefig(game+'_'+type+'_'+y+'.png')
        plt.clf()

def run_forest():
    map_size = 700
    game = 'forest'
    #TODO realized for lake that higher discounts and lower epsilons were better early on
    discounts = [
        # 0.6,
        # 0.7,
        # 0.8,
        # 0.9,
        0.99,
        0.999,
        #0.9999
         #0.99,
         #0.999
                 ]
    epsilons = [
        0.00001,
        # 0.0000001,
        # 0.000000001,
        0.0000000000001
    ]
    alphas = [0.01, 0.1]
    alpha_decays = [0.7, 0.9]
    epsilon_decays = [0.7, 0.9]
    max_iters = [10000000]


    print("Running forest problem with map size "+str(map_size))
    transitions, rewards = forest(S=map_size)
    type = 'value'
    value_res = run_mdp(type, transitions, rewards, discounts, map_size, epsilons=epsilons)
    analyze_mdp(game, type, value_res)
    type = 'policy'
    policy_res = run_mdp(type, transitions, rewards, discounts, map_size)
    analyze_mdp(game, type, policy_res)
    q_res = run_Q(transitions, rewards, map_size, discounts, epsilons, alphas, alpha_decays, epsilon_decays, max_iters)
    type='QL'
    analyze_mdp(game, type, q_res)

if __name__ == "__main__":
    passed_arg = sys.argv[1]
    if passed_arg == 'lake':
        run_lake()
    elif passed_arg == 'forest':
        run_forest()
    else:
        print("please run with either 'value', 'policy' or 'q'")
        exit(146)