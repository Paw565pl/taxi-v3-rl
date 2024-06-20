import gymnasium as gym
import numpy as np


def use_model(episodes: int):
    env = gym.make("Taxi-v3", render_mode="human")

    # load model
    q_table = np.load("q_table.npy")

    for _ in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False
        rewards = 0

        while not terminated and not truncated:
            action = np.argmax(q_table[state, :])
            new_state, reward, terminated, truncated, _ = env.step(action)
            rewards += reward
            state = new_state

    env.close()


if __name__ == "__main__":
    use_model(10)
