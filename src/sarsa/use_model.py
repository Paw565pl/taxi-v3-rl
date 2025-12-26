import gymnasium as gym
import numpy as np


def use_model(episodes: int) -> None:
    sarsa_q_table = np.load("src/sarsa/sarsa_q_table.npy")
    env = gym.make("Taxi-v3", render_mode="human")

    for episode in range(episodes):
        state, _ = env.reset()
        state = int(state)

        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = int(np.argmax(sarsa_q_table[state]))

            state, reward, terminated, truncated, _ = env.step(action)
            state = int(state)

            done = terminated or truncated
            total_reward += float(reward)
            steps += 1

        print(f"episode: {episode + 1} - total reward: {total_reward} - steps: {steps}")

    env.close()


if __name__ == "__main__":
    use_model(10)
