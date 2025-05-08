from algorithms.config import config
from algorithms.fitness import evaluate_fitness
from algorithms.population import Population
from apps.plotter import Plotter
import time


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

    plotter.show_results_window(best_individual, execution_time)
    return best_individual, execution_time, plotter
