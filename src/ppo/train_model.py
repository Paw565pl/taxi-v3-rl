import numpy as np
from make_env import make_env
from matplotlib import pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy

TRAIN_SEED = 123
TEST_SEED = 999


def train_model() -> None:
    print("started training model...")

    vec_env = make_vec_env(make_env, n_envs=4, seed=TRAIN_SEED)

    eval_env = make_env()
    eval_env.reset(seed=TEST_SEED)

    model = PPO(
        MlpPolicy,
        vec_env,
        learning_rate=lambda progress_remaining: progress_remaining * 3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        policy_kwargs={"net_arch": {"pi": [256, 256], "vf": [256, 256]}},
        verbose=1,
        seed=TRAIN_SEED,
        device="cpu",
    )

    eval_callback = EvalCallback(
        eval_env,
        eval_freq=10_000,
        log_path="src/ppo/logs",
        best_model_save_path="src/ppo/models",
    )

    model.learn(
        total_timesteps=500_000,
        callback=[eval_callback],
        progress_bar=True,
    )

    best_model = PPO.load("src/ppo/models/best_model", env=eval_env)

    mean_reward, std_reward = evaluate_policy(best_model, eval_env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward:.2f}")
    print(f"Std reward: {std_reward:.2f}")

    if len(eval_callback.evaluations_results) < 1:
        return

    timesteps = eval_callback.evaluations_timesteps
    rewards = eval_callback.evaluations_results

    # Calculate mean reward for every timestep
    mean_rewards = np.mean(np.array(rewards), axis=1)

    # Create plot
    plt.figure(figsize=(16, 8))
    plt.plot(timesteps, mean_rewards)
    plt.title("Training Progress (PPO)")
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Reward")
    plt.grid(True, alpha=0.4)
    plt.savefig("src/ppo/training_progress.png")

    print("done! model has been saved to file")


if __name__ == "__main__":
    train_model()
