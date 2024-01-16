# Distributed Monte Carlo Control

This is a project written for the "Big Data" course (pl. "Analiza dużych nauk danych") at the AGH University of Krakow in the winter semester 2023/2024.

The repository contains an implementation of a distributed reinforcement learning algorithm for training policies in the gymnasium environment using the Lithops serverless computing framework. It leverages parallelism to efficiently generate trajectories and update policy.

## Features

- **Distributed training**: Utilizes the Lithops serverless computing framework to distribute the training process across multiple nodes.
- **CartPole environment**: Applies the CartPole environment from the Gymnasium library, providing a simple yet effective scenario for reinforcement learning.
- **Policy iteration updates**: Utilizes policy iteration updates with the Monte Carlo Control method. **Attention!** The agent uses an on-policy learning, so with a high step-to-update ratio, you can expect unstable training.
- **Decaying $\varepsilon$-greedy policy**: Implements an epsilon-greedy exploration strategy with a decaying epsilon value over iterations to balance exploration and exploitation.

## Structure

The file is structured as follows:

- `initial_policy`: Initializes the Q-table and visitation counts for the $\varepsilon$-greedy policy.
- `policy`: Defines the epsilon-greedy policy for action selection based on the Q-table.
- `play_episodes`: Executes episodes with the current policy and returns the trajectories.
- `calculate_updates`: Computes Monte Carlo updates and returns the total returns for each trajectory.
- `update_policy`: Updates the Q-table and visitation counts based on the Monte Carlo returns.
- The file orchestrates the distributed training using Lithops, saves the final policy, and plots training results.

## Usage

1. Ensure you have Lithops installed (`pip install lithops`). This repository uses Google Cloud Functions, for this you need to install the appropriate version of Lithops (`pip install "lithops[gcp]"`).
2. Adjust the desired parallelism level using the `-p` or `--parallelism` argument.
3. Run the script:

```bash
python main.py -p <parallelism_level>
```

## Additional Notes

- The CartPole environment is defined with specific bin configurations for state discretization, influencing the agent's observations. **Attention!:** If you want to use a different continuous state space environment, you need to adjust the buckets accordingly.
- Lithops requires a backend from a cloud service provider to fully utilize its capabilities. To use this implementation, you must provide project configuration from your cloud provider.
