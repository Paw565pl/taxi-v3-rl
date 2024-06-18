import gymnasium as gym
import numpy as np


def use_solution():
    solution = np.load("solution.npy")

    env = gym.make("Taxi-v3", render_mode="human")
    env.reset(seed=123)

    for action in solution:
        _, reward, terminated, _, _ = env.step(int(action))
        env.render()

        if terminated == 1:
            break

    env.close()


if __name__ == "__main__":
    use_solution()
