import gymnasium as gym
import numpy as np
import numpy.typing as npt
import pygad

ENV_NAME = "Taxi-v3"
MAX_STEPS_PER_EPISODE = 50
TRAINING_SEEDS = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
EPISODES_PER_FITNESS = len(TRAINING_SEEDS)


def fitness_func(
    ga_instance: pygad.GA, solution: npt.NDArray[np.int_], solution_idx: int
) -> float:
    env = gym.make(ENV_NAME, render_mode=None)
    total_reward = 0

    for seed in TRAINING_SEEDS:
        state, _ = env.reset(seed=seed)
        state = int(state)

        done = False
        episode_reward = 0
        steps = 0

        while not done and steps < MAX_STEPS_PER_EPISODE:
            action = int(solution[state])

            next_state, reward, terminated, truncated, _ = env.step(action)
            state = int(next_state)

            done = terminated or truncated
            episode_reward += float(reward)
            steps += 1

        total_reward += episode_reward

    env.close()

    return total_reward / EPISODES_PER_FITNESS


def evaluate_solution(
    solution: npt.NDArray[np.int_], episodes: int = 100
) -> tuple[float, float]:
    TEST_SEED = 123

    env = gym.make(ENV_NAME, render_mode=None)
    rewards: list[float] = []

    for i in range(episodes):
        state, _ = env.reset(seed=TEST_SEED + i)
        state = int(state)

        done = False
        episode_reward = 0
        steps = 0

        while not done and steps < MAX_STEPS_PER_EPISODE:
            action = int(solution[state])

            next_state, reward, terminated, truncated, _ = env.step(action)
            state = int(next_state)

            done = terminated or truncated
            episode_reward += float(reward)
            steps += 1

        rewards.append(episode_reward)

    env.close()

    return float(np.mean(rewards)), float(np.std(rewards))


def train_model() -> None:
    print("started training model...")

    env = gym.make(ENV_NAME, render_mode=None)

    num_genes = int(env.observation_space.n)  # type: ignore
    num_actions = int(env.action_space.n)  # type: ignore

    env.close()

    ga_instance = pygad.GA(
        num_generations=100,
        num_parents_mating=20,
        fitness_func=fitness_func,
        sol_per_pop=200,
        num_genes=num_genes,
        gene_type=int,
        gene_space={"low": 0, "high": num_actions},
        parent_selection_type="tournament",
        K_tournament=10,
        crossover_type="uniform",
        mutation_percent_genes=10,
        keep_parents=20,
        parallel_processing=("process", None),
    )

    ga_instance.run()
    solution, solution_fitness, _ = ga_instance.best_solution()

    print(f"best fitness: {solution_fitness:.2f}")

    mean_rewards, std_rewards = evaluate_solution(solution)
    print(f"mean rewards: {mean_rewards:.2f}")
    print(f"std rewards: {std_rewards:.2f}")

    ga_instance.plot_fitness(save_dir="src/genetic_algorithm/fitness_plot.png")
    np.save("src/genetic_algorithm/model.npy", solution)

    print("done! model has been saved to file")


if __name__ == "__main__":
    train_model()
