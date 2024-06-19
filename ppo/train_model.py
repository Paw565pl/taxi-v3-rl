import numpy as np
from matplotlib import pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy


def train_model():
    vec_env = make_vec_env("Taxi-v3", n_envs=4)

    model = PPO(
        MlpPolicy,
        vec_env,
        learning_rate=1e-4,
        n_steps=1024,
        batch_size=32,
        n_epochs=5,
        gamma=0.95,
        gae_lambda=0.9,
        clip_range=0.1,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=dict(pi=[16, 16, 32, 32], vf=[16, 16, 32, 32])),
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path="./logs",
        name_prefix="model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    stop_train_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=5, min_evals=10, verbose=1
    )
    eval_callback = EvalCallback(
        vec_env,
        callback_after_eval=stop_train_callback,
        eval_freq=500,
        log_path="./logs",
        best_model_save_path=".",
    )
    model.learn(
        total_timesteps=100_000,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    mean_reward, _ = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print(f"Mean reward: {mean_reward}")

    timesteps = eval_callback.evaluations_timesteps
    rewards = eval_callback.evaluations_results

    # Calculate mean reward for every timestep
    mean_rewards = np.mean(np.array(rewards), axis=1)

    # Create plot
    plt.figure(figsize=(12, 8))
    plt.plot(timesteps, mean_rewards)
    plt.title("Training Progress")
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Reward")
    plt.savefig("training_progress.png")


if __name__ == "__main__":
    print("started training model...")
    train_model()
    print("done! model has been saved to file")
