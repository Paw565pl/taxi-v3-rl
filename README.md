# Taxi-v3 - Reinforcement Learning

This repository contains implementations of various reinforcement learning algorithms applied to the [Taxi V3](https://gymnasium.farama.org/environments/toy_text/taxi) environment. The goal is to train an AI agent to successfully navigate a taxi through a grid world, pick up passengers, and drop them off at their destinations. The included algorithms allow the agent to learn optimal policies for maximizing rewards in this task.

## Which algorithms were used?

| Results     | Genetic Algorithm | Q-Learning | SARSA | PPO  |
| ----------- | ----------------- | ---------- | ----- | ---- |
| Mean reward | -294.44           | 7.74       | 7.63  | 7.80 |
| Std reward  | 221.17            | 2.82       | 2.82  | 2.23 |

**Genetic algorithm** really struggles in this environment. Agent keeps driving in circle or performing illegal drop-off.
**Q-Learning** and **Sarsa** manage to learn an optimal policy for this problem and solve it effectively.
**Proximal Policy Optimization (PPO)** stands out as the most effective by a small margin.

## How to run it locally?

1. **Clone the repository**
2. **Install project dependencies**

```shell
uv sync
```

or

```shell
pip install -r requirements.txt
```

You will find directory for each of the algorithms. There are separate modules for training
each model and using it on the environment with the display of model's actions in the graphical representation.
