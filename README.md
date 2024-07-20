# Taxi V3 - reinforcement learning

This repository contains implementations of various reinforcement learning algorithms applied to
the [Taxi V3](https://gymnasium.farama.org/environments/toy_text/taxi)
environment. The goal is to train an AI agent to successfully navigate a taxi through a grid world, pick up passengers,
and drop them off at their destinations. The included algorithms allow the agent to
learn optimal policies for maximizing rewards in this task.

### Which algorithms were used?

- Genetic algorithm
- Proximal Policy Optimization (PPO)
- Q-Learning

Among the reinforcement learning algorithms, **Q-Learning** stands out as the most effective. While the **genetic
algorithm** manages to solve the problem, its primary limitation lies in its ability to perform only on a specific world
map.
Unfortunately, **Proximal Policy Optimization (PPO)** struggles to address this particular challenge.

### How to run it locally?

1. **Clone the repository**
2. **Install project dependencies**

```shell
poetry install
```

You will find directory for each of the algorithms. There are separate modules for training
the model and using it on the environment with the display of model's actions in the graphical representation.
