import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt

EPISODES = 10_000
SEED = 123


def train_model() -> None:
    print("started training model...")

    env = gym.make("Taxi-v3", render_mode=None)
    n_states = int(env.observation_space.n)  # type: ignore
    n_actions = int(env.action_space.n)  # type: ignore

    sarsa_q_table = np.zeros((n_states, n_actions))

    learning_rate_a = 0.1
    discount_factor_g = 0.99
    epsilon = 1.0
    epsilon_decay_rate = 0.9995
    epsilon_min = 0.01

    rng = np.random.default_rng(seed=SEED)

    rewards_per_episode = np.zeros(EPISODES)
    for i in range(EPISODES):
        state, _ = env.reset(seed=SEED + i)
        state = int(state)

        if rng.random() < epsilon:
            action = rng.integers(0, n_actions)
        else:
            action = int(np.argmax(sarsa_q_table[state, :]))

        terminated = False
        truncated = False
        total_reward = 0

        while not terminated and not truncated:
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = int(next_state)

            total_reward += float(reward)

            if rng.random() < epsilon:
                next_action = rng.integers(0, n_actions)
            else:
                next_action = int(np.argmax(sarsa_q_table[next_state, :]))

            target = (
                float(reward)
                + discount_factor_g * sarsa_q_table[next_state, next_action]
            )
            sarsa_q_table[state, action] = sarsa_q_table[
                state, action
            ] + learning_rate_a * (target - sarsa_q_table[state, action])

            state = next_state
            action = next_action

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay_rate

        rewards_per_episode[i] = total_reward

    env.close()

    last_100_avg = np.mean(rewards_per_episode[-100:])
    last_100_std = np.std(rewards_per_episode[-100:])
    print(f"Mean reward (last 100 episodes): {last_100_avg:.2f}")
    print(f"Std reward (last 100 episodes): {last_100_std:.2f}")

    mean_reward = np.mean(rewards_per_episode)
    std_reward = np.std(rewards_per_episode)
    print(f"\nMean reward (total): {mean_reward:.2f}")
    print(f"Std reward (total): {std_reward:.2f}")

    # create rewards plot
    window_size = 50
    moving_avg = np.convolve(
        rewards_per_episode, np.ones(window_size) / window_size, mode="same"
    )

    plt.figure(figsize=(16, 8))
    plt.plot(rewards_per_episode, label="Raw Reward", alpha=0.6)
    plt.plot(moving_avg, label="Moving Average (50)", color="green", linewidth=2)
    plt.title("Rewards per episode (SARSA)")
    plt.xlabel("Episode")
    plt.ylabel("Sum rewards")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.savefig("src/sarsa/rewards_plot.png")

    # save model
    np.save("src/sarsa/sarsa_q_table.npy", sarsa_q_table)

    print("done! model has been saved to file")


if __name__ == "__main__":
    train_model()
