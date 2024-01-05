from collections import defaultdict

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def initial_policy():
    q = defaultdict(float)
    n = defaultdict(int)
    return q, n


def policy(q, state, actions, epsilon):
    values = np.array([q[state, action] for action in actions])
    probs = (values == np.max(values)).astype(float)
    probs /= np.sum(probs)
    probs = probs * (1 - epsilon) + epsilon / len(actions)
    return dict(zip(actions, probs))


def play_episodes(env, q, epsilon, n_episodes, bins, actions):
    trajectories = []

    for episode in range(n_episodes):
        trajectories.append([])
        state, _ = env.reset()

        while True:
            state = tuple(map(lambda x: np.digitize(*x), zip(state, bins)))
            probs = policy(q, state, actions, epsilon)
            action = np.random.choice(actions, p=list(probs.values())).item()

            env.render()
            next_state, reward, terminated, truncated, _ = env.step(action)

            trajectories[-1].append((state, action, reward))
            state = next_state

            if terminated or truncated:
                break

    return trajectories


def calculate_updates(trajectories, gamma):
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
    # --- MASTER ---

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

    # initial policy
    q, n = initial_policy()

    # number of episodes and improvement steps
    n_episodes = 100
    improvement_steps = 100
    parallelism = 5
    returns = []

    # decaying epsilon
    epsilon = np.logspace(0, -2, improvement_steps, base=10)

    for e in (pbar := tqdm(epsilon)):
        # --- MONTE CARLO NODES ---
        trajectories = []

        for _ in range(parallelism):  # <- in parallel
            trajectories.append(play_episodes(env, q, e, n_episodes, bins, actions))

        # --- REDUCER NODES ---
        updates = []

        for trajectory in trajectories:  # <- in parallel
            new_updates, new_returns = calculate_updates(trajectory, gamma)
            updates += new_updates
            returns.append(np.mean(new_returns))

        pbar.set_description(f'epsilon: {e:.3f}, return: {returns[-1]:.3f}')

        # --- MASTER ---
        q, n = update_policy(q, n, updates)

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
