import os
import pickle
import time
from argparse import ArgumentParser
from collections import defaultdict

import gymnasium as gym
import lithops
import matplotlib.pyplot as plt
import numpy as np

os.environ['LITHOPS_CONFIG_FILE'] = os.path.join(os.getcwd(), 'config.yaml')


def initial_policy():
    q = defaultdict(float)
    n = defaultdict(int)
    return q, n


def policy(q, state, epsilon):
    values = np.array([q[state, action] for action in actions])
    probs = (values == np.max(values)).astype(float)
    probs /= np.sum(probs)
    probs = probs * (1 - epsilon) + epsilon / len(actions)
    return probs


def play_episodes(q, epsilon):
    trajectories = []

    for episode in range(n_episodes):
        trajectories.append([])
        state, _ = env.reset()

        while True:
            state = tuple(map(lambda x: np.digitize(*x), zip(state, bins)))

            probs = policy(q, state, epsilon)
            action = np.random.choice(actions, p=probs).item()
            next_state, reward, terminated, truncated, _ = env.step(action)

            trajectories[-1].append((state, action, reward))
            state = next_state

            if terminated or truncated:
                break

    return trajectories


def calculate_updates(trajectories):
    updates = []
    returns = []

    for trajectory in trajectories:
        first_occurrences = defaultdict(int)
        for i, (state, action, _) in enumerate(trajectory):
            if (state, action) not in first_occurrences:
                first_occurrences[state, action] = i

        updates.append({})
        g = 0

        for i, (state, action, reward) in enumerate(reversed(trajectory)):
            g = gamma * g + reward

            if i == first_occurrences[state, action]:
                updates[-1][state, action] = g

        returns.append(g)

    return updates, returns


def update_policy(q, n, updates):
    for update in updates:
        for (state, action), g in update.items():
            n[state, action] += 1
            q[state, action] += (g - q[state, action]) / n[state, action]

    return q, n


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('-p', '--parallelism', type=int, required=True)
    args = args.parse_args()

    global actions, bins, env, gamma, n_episodes

    # environment definition
    env = gym.make('CartPole-v1', render_mode=None)
    actions = list(range(env.action_space.n))
    bins = [
        np.linspace(-4.8, 4.8, 5),
        np.linspace(-2, 2, 7),
        np.linspace(-0.419, 0.419, 11),
        np.linspace(-2, 2, 15)
    ]
    gamma = 1.0

    # number of steps and episodes
    total_steps = 100
    improvement_steps = 10
    parallelism = args.parallelism
    n_episodes = total_steps // improvement_steps // parallelism

    # decaying epsilon
    epsilon = np.logspace(0, -2, improvement_steps, base=10)

    # initial policy
    q, n = initial_policy()

    # distributed training with Lithops
    start = time.time()
    returns = []

    with lithops.ServerlessExecutor() as executor:
        for i in range(improvement_steps):
            params = [(q, epsilon[i])] * parallelism
            results = (executor
                       .map(play_episodes, params, runtime_memory=2048)
                       .map(calculate_updates, runtime_memory=2048)
                       .get_result())

            updates = []

            for elem in results:
                if elem is not None:
                    updates += elem[0]
                    returns += elem[1]

            q, n = update_policy(q, n, updates)

            print(f"Iteration {i + 1} - epsilon: {epsilon[i]}, mean return: {np.mean(returns[-parallelism * n_episodes:])}")

        executor.wait()
        executor.plot()
        plt.tight_layout()
        plt.savefig(f'lithops_{parallelism}.pdf')
        plt.clf()

    end = time.time()

    # save training time
    with open(f'time_{parallelism}.txt', 'w') as f:
        f.write(f'{end - start}')

    # save policy
    with open(f'q_{parallelism}.pkl', 'wb') as f:
        pickle.dump(q, f)

    with open(f'n_{parallelism}.pkl', 'wb') as f:
        pickle.dump(n, f)

    # plot training results
    plt.style.use('default')
    plt.plot(np.array(returns).reshape(-1, 1000).mean(axis=1))
    plt.xlabel(r'Step ($\times 1000$)')
    plt.ylabel('Average return')
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'returns_{parallelism}.pdf')
    plt.clf()
