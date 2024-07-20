import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt


def train_model(episodes: int):
    env = gym.make("Taxi-v3", render_mode=None)

    q_table = np.zeros((env.observation_space.n, env.action_space.n))  # noqa

    learning_rate_a = 0.95
    discount_factor_g = 0.95
    epsilon = 1
    epsilon_decay_rate = 0.0001
    rng = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False

        rewards = 0
        while not terminated and not truncated:
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state, :])

            new_state, reward, terminated, truncated, _ = env.step(action)

            rewards += reward

            q_table[state, action] = q_table[state, action] + learning_rate_a * (
                float(reward)
                + discount_factor_g * np.max(q_table[new_state, :])
                - q_table[state, action]
            )

            state = new_state

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        if epsilon == 0:
            learning_rate_a = 0.0001

        rewards_per_episode[i] = rewards

    env.close()

    sum_rewards = np.zeros(episodes)
    for episode_index in range(episodes):
        sum_rewards[episode_index] = np.sum(
            rewards_per_episode[max(0, episode_index - 100) : (episode_index + 1)]
        )

    mean_reward = np.mean(rewards_per_episode)
    print(f"Mean reward: {mean_reward}")

    # create rewards plot
    plt.figure(figsize=(16, 8))
    plt.plot(rewards_per_episode)
    plt.title("Rewards per episode")
    plt.xlabel("Episode")
    plt.ylabel("Sum rewards")
    plt.savefig("rewards_plot.png")

    # save model
    np.save("q_table.npy", q_table)


if __name__ == "__main__":
    print("started training model...")
    train_model(2000)
    print("done! model has been saved to file")
