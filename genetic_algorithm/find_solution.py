import gymnasium as gym


def fitness_func(ga_instance, solution, solution_idx):
    env = gym.make("Taxi-v3", render_mode=None)
    env.reset(seed=123)
    total_reward = 0

    for action in solution:
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        done = terminated or truncated
        if done:
            break

    env.close()

    return total_reward
