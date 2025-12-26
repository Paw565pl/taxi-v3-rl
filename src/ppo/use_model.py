from make_env import make_env
from stable_baselines3 import PPO


def use_model(episodes: int) -> None:
    env = make_env(render_mode="human")
    model = PPO.load("src/ppo/models/best_model")

    for episode in range(episodes):
        state, _ = env.reset()

        done = False
        total_reward = 0
        steps = 0

        while not done:
            action, _ = model.predict(state, deterministic=True)
            action = int(action.item())

            state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated
            total_reward += float(reward)
            steps += 1

        print(f"episode: {episode + 1} - total reward: {total_reward} - steps: {steps}")

    env.close()


if __name__ == "__main__":
    use_model(10)
