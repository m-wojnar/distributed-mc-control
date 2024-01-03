from collections import defaultdict

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def initial_q():
    return defaultdict(float)


def policy(q, state, actions, epsilon):
    values = np.array([q[state, action] for action in actions])
    probs = (values == np.max(values)).astype(float)
    probs /= np.sum(probs)
    probs = probs * (1 - epsilon) + epsilon / len(actions)
    return dict(zip(actions, probs))


def play_episodes(env, q, epsilon, n_episodes, bins, actions):
    trajectories = []
    returns = []

    for episode in range(n_episodes):
        trajectories.append([])
        state, _ = env.reset()

        while True:
            state = tuple(map(lambda x: np.digitize(*x), zip(state, bins)))
            probs = policy(q, state, actions, epsilon)
            action = np.random.choice(actions, p=list(probs.values()))

            env.render()
            next_state, reward, terminated, truncated, _ = env.step(action)

            trajectories[-1].append((state, action, reward))
            state = next_state

            if terminated or truncated:
                break

        returns.append(sum(map(lambda x: x[2], trajectories[-1])))

    return trajectories, returns


def improve_policy(q, trajectories, gamma):
    q = q.copy()

    for trajectory in trajectories:
        g = 0

        for i in range(len(trajectory) - 1, -1, -1):
            state, action, reward = trajectory[i]
            g = gamma * g + reward

            for j in range(i):
                if trajectory[j][0] == state and trajectory[j][1] == action:
                    break
            else:
                q[state, action] += g

    return q


def combine_policies(qs):
    q = defaultdict(float)
    n = defaultdict(int)

    for q_ in qs:
        for (state, action), value in q_.items():
            q[state, action] += value
            n[state, action] += 1

    for state, action in q:
        q[state, action] /= n[state, action]

    return q


if __name__ == '__main__':
    # --- MASTER ---

    # environment definition
    env = gym.make('CartPole-v1', render_mode=None)
    actions = list(range(env.action_space.n))
    bins = [
        np.linspace(-4.8, 4.8, 5),
        np.linspace(-4, 4, 10),
        np.linspace(-0.419, 0.419, 20),
        np.linspace(-2, 2, 15)
    ]
    gamma = 1.0

    # initial policy
    q = initial_q()

    # number of episodes and improvement steps
    n_episodes = 100
    k = 100
    returns = []

    # decaying epsilon
    epsilon = np.logspace(0, -2, k, base=10)

    for e in (pbar := tqdm(epsilon)):
        # --- MONTE CARLO NODES ---
        trajectories = []

        for _ in range(1):  # <- in parallel
            new_trajectories, new_returns = play_episodes(env, q, e, n_episodes, bins, actions)

            trajectories.append(new_trajectories)
            returns.append(np.mean(new_returns))

            pbar.set_description(f'epsilon: {e:.3f}, mean return: {returns[-1]:.3f}')

        # --- REDUCER NODES ---
        qs = []

        for t in trajectories:  # <- in parallel
            qs.append(improve_policy(q, t, gamma))

        # --- MASTER ---
        q = combine_policies(qs)

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
    play_episodes(env, q, 0.0, 10, bins, actions)
