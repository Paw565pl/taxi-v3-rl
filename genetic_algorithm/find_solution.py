import gymnasium as gym
import numpy as np
import pygad


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


def find_solution():
    gene_space = [0, 1, 2, 3, 4, 5]
    # 0: Move south(down)
    # 1: Move north(up)
    # 2: Move east(right)
    # 3: Move west(left)
    # 4: Pickup passenger
    # 5: Drop off passenger

    ga_instance = pygad.GA(
        num_generations=200,
        num_parents_mating=10,
        fitness_func=fitness_func,
        sol_per_pop=500,
        num_genes=50,
        gene_space=gene_space,
        parent_selection_type="sss",
        keep_parents=5,
        crossover_type="single_point",
        mutation_type="random",
        mutation_percent_genes=10,
        parallel_processing=("process", None),
    )
    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print(f"Parameters of the best solution: {solution}")
    print(f"Fitness value of the best solution = {solution_fitness}")

    ga_instance.plot_fitness(save_dir="fitness_plot.png")

    np.save("solution.npy", solution)


if __name__ == "__main__":
    print("searching for optimal solution...")
    find_solution()
    print("done! solution has been saved to file")
