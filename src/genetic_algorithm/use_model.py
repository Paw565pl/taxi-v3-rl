import gymnasium as gym
import numpy as np

MAX_STEPS = 50


def use_model(episodes: int) -> None:
    model = np.load("src/genetic_algorithm/model.npy")
    env = gym.make("Taxi-v3", render_mode="human")

    for episode in range(episodes):
        state, _ = env.reset()
        state = int(state)

        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = int(model[state])

            state, reward, terminated, truncated, _ = env.step(action)
            state = int(state)

            done = terminated or truncated
            total_reward += float(reward)
            steps += 1

            if steps > MAX_STEPS:
                print("steps limit exceeded")
                break

        print(f"episode: {episode + 1} - total reward: {total_reward} - steps: {steps}")

    env.close()


if __name__ == "__main__":
    use_model(10)
