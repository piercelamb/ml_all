import sys
import pprint
import hiive.mdptoolbox as mdp
from hiive.mdptoolbox.mdp import ValueIteration, PolicyIteration, QLearning
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
import gym
from gym.envs.toy_text.frozen_lake import generate_random_map

RANDOM_SEED = 1337

def plot_env(env, figsize, policy=[]):
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

    plt.savefig('lake_'+str(figsize)+'.png')

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

def run_value(transitions, rewards, discounts, epsilons, map_size):
    print("Running value iterations with map size "+str(map_size))
    columns = ['gamma', 'epsilon', 'time', 'iterations', 'reward', 'average_steps', 'steps_stddev', 'success_pct',
               'policy', 'mean_rewards', 'max_rewards', 'error']
    data = pd.DataFrame(0.0, index=np.arange(len(discounts) * len(epsilons)), columns=columns)

    print('Gamma,\tEps,\tTime,\tIter,\tReward')
    print(80*'_')

    testNum = 0
    for discount in discounts:
        print("Testing discount: "+str(discount))
        for epsilon in epsilons:
            print("Testing epsilon: "+str(epsilon))
            #converges when either max iter hit or the maximum change in value function falls
            #below the passed epsilon value
            runner = ValueIteration(transitions, rewards, epsilon=epsilon, gamma=discount)
            run_data = runner.run()
            # for dict in run_data:
            #     print(dict)'
            best_run = run_data[-1]

            Time = best_run["Time"]
            iters = best_run["Iteration"]
            maxR = best_run["Max V"]

            max_rewards, mean_rewards, errors = [], [], []
            for run in run_data:
                max_rewards.append(run['Max V'])
                mean_rewards.append(run['Mean V'])
                errors.append(run['Error'])

            data['gamma'][testNum] = discount
            data['epsilon'][testNum] = epsilon
            data['time'][testNum] = Time
            data['iterations'][testNum] = iters
            data['reward'][testNum] = maxR
            data['mean_rewards'][testNum] = {tuple(mean_rewards)}
            data['max_rewards'][testNum] = {tuple(max_rewards)}
            data['error'][testNum] = {tuple(errors)}
            data['policy'][testNum] = {runner.policy}

            print('%.2f,\t%.0E,\t%.2f,\t%d,\t%f' % (discount, epsilon, Time, iters, maxR))

            testNum = testNum + 1

    print(data)
    exit(1)
def run_lake():
    map_size = 6
    discounts = [0.1, 0.3, 0.5, 0.8]
    epsilons = [0.001, 0.00001, 0.0000001, 0.000000001]
    print("Running lake problem with map size "+str(map_size))
    random_map = generate_random_map(size=map_size, p=0.8)

    env = gym.make("FrozenLake-v1", desc=random_map).unwrapped
    env.max_episode_steps=300
    plot_env(env, map_size)
    transitions, rewards = get_transitions_rewards(env, map_size)
    run_value(transitions, rewards, discounts, epsilons, map_size)
    #run_policy()
    #run_Q()

if __name__ == "__main__":
    passed_arg = sys.argv[1]
    if passed_arg == 'lake':
        run_lake()
    # elif passed_arg == 'forest':
    #     run_forest()
    else:
        print("please run with either 'value', 'policy' or 'q'")
        exit(146)