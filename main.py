import os
os.environ['LITHOPS_CONFIG_FILE'] = os.path.join(os.getcwd(), 'config.yaml')

from collections import defaultdict

import gymnasium as gym
import lithops
import matplotlib.pyplot as plt
import numpy as np


def initial_policy():
    q = defaultdict(float)
    n = defaultdict(int)
    return q, n


def policy(q, state, epsilon):
    values = np.array([q[state, action] for action in actions])
    probs = (values == np.max(values)).astype(float)
    probs /= np.sum(probs)
    probs = probs * (1 - epsilon) + epsilon / len(actions)
    return dict(zip(actions, probs))


def play_episodes(q, epsilon):
    trajectories = []

    for episode in range(n_episodes):
        trajectories.append([])
        state, _ = env.reset()

        while True:
            state = tuple(map(lambda x: np.digitize(*x), zip(state, bins)))
            probs = policy(q, state, epsilon)
            action = np.random.choice(actions, p=list(probs.values())).item()

            env.render()
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

    # number of episodes and improvement steps
    n_episodes = 100
    improvement_steps = 100
    parallelism = 10
    returns = []

    # decaying epsilon
    epsilon = np.logspace(0, -2, improvement_steps, base=10)

    # initial policy
    q, n = initial_policy()

    # distributed training with Lithops
    with lithops.FunctionExecutor() as executor:
        for i, e in enumerate(epsilon):
            params = [(q, e)] * parallelism
            futures = executor.map_reduce(play_episodes, params, calculate_updates)
            results = [future.result() for future in futures]

            returns += [np.mean(g) for _, g in results]
            updates = [u for upd, _ in results for u in upd]
            q, n = update_policy(q, n, updates)

            print(f"Iteration {i + 1} - epsilon: {e}, mean return: {returns[-1]}")

    # plot training results
    plt.plot(returns)
    plt.xlabel('Improvement step')
    plt.ylabel('Mean return')
    plt.grid()
    plt.tight_layout()
    plt.savefig('return.pdf')
    plt.show()

    # test policy
    env = gym.make('CartPole-v1', render_mode='human')
    play_episodes(q, 0.)
