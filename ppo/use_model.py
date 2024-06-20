import gymnasium as gym
from stable_baselines3 import PPO


def use_model(episodes: int):
    env = gym.make("Taxi-v3", render_mode="human")
    model = PPO.load("best_model", env=env)

    vec_env = model.get_env()
    observation = vec_env.reset()
    for _ in range(episodes):
        terminated = False
        truncated = False

        while not terminated and not truncated:
            action, _states = model.predict(observation)
            observation, rewards, done, info = vec_env.step(action)
            vec_env.render("human")

    vec_env.close()


if __name__ == "__main__":
    use_model(10)
