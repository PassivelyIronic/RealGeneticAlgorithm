from algorithms.config import config
from algorithms.fitness import evaluate_fitness
from algorithms.population import Population
from algorithms.BatchTester import BatchTester
from apps.plotter import Plotter
from algorithms.batch_plotter import show_results_window
import time
import pandas as pd


def run_genetic_algorithm():
    bounds = (config.range_start, config.range_end)
    plotter = Plotter()
    start_time = time.time()

    population = Population(config)

    population.evaluate_all(evaluate_fitness, bounds)

    plotter.update_history(population.best_individual.fitness, population.avg_fitness_history[0])
    print(f"Epoka 0, najlepsze przystosowanie: {population.best_individual.fitness}")

    best_individual, best_fitness_history, avg_fitness_history = population.run(evaluate_fitness, bounds)

    for i in range(1, len(best_fitness_history)):
        plotter.update_history(best_fitness_history[i], avg_fitness_history[i])

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Czas wykonania algorytmu: {execution_time:.2f} sekund")

    # plotter.show_results_window(best_individual, execution_time)
    return best_individual, execution_time, plotter


def run_batch():
    param_grid_ga = {
        "range_start": [-20],
        "range_end": [20],
        "epochs": [100,200,500],
        "population_size": [50, 100, 150],
        "precision": [5],
        "num_variables": [2],
        "selection_method": ["tournament"],
        "best_selection_amount": [2],
        "tournament_size": [3],
        "crossover_method": ["arithmetic"],
        "crossover_probability": [0.7],
        "mutation_method": ["uniform"],
        "mutation_probability": [0.05],
        "optimization_type": ["min"],
    }

    param_grid_gjo = {
        "range_start": [-20],
        "range_end": [20],
        "epochs": [100,200,500],
        "population_size": [50, 100, 150],
        "precision": [5],
        "num_variables": [2],
        "optimization_type": ["min"],
    }

    tester = BatchTester(plotter_class=Plotter)
    tester.run_batch(param_grid_ga, param_grid_gjo, repeats=5)

    df = pd.DataFrame(tester.results)
    df.to_csv("results/batch_results.csv", index=False)
    print(df.groupby(["algorithm", "population_size"])["best_fitness"].agg(["mean", "std"]))

    show_results_window(tester.results)
