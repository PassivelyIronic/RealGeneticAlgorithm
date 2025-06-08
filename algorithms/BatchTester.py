import itertools
import time

class BatchTester:
    def __init__(self, plotter_class):
        self.plotter_class = plotter_class
        self.results = []

    def run_batch(self, param_grid_ga, param_grid_gjo, repeats=3):
        ga_configs = list(itertools.product(*param_grid_ga.values()))
        gjo_configs = list(itertools.product(*param_grid_gjo.values()))

        for config in ga_configs:
            params = dict(zip(param_grid_ga.keys(), config))
            for _ in range(repeats):
                result = self.run_ga(params)
                self.results.append({**params, "algorithm": "GA", **result})

        for config in gjo_configs:
            params = dict(zip(param_grid_gjo.keys(), config))
            for _ in range(repeats):
                result = self.run_gjo(params)
                self.results.append({**params, "algorithm": "GJO", **result})

    def run_ga(self, params):
        from algorithms.config import config
        from algorithms.genetic import run_genetic_algorithm

        config.range_start = params["range_start"]
        config.range_end = params["range_end"]
        config.epochs = params["epochs"]
        config.population_size = params["population_size"]
        config.precision = params["precision"]
        config.num_variables = params["num_variables"]
        config.selection_method = params["selection_method"]
        config.best_selection_amount = params["best_selection_amount"]
        config.tournament_size = params["tournament_size"]
        config.crossover_method = params["crossover_method"]
        config.crossover_probability = params["crossover_probability"]
        config.mutation_method = params["mutation_method"]
        config.mutation_probability = params["mutation_probability"]
        config.optimization_type = params.get("optimization_type", "min")

        best_solution, execution_time, plotter = run_genetic_algorithm()
        return {
            "best_fitness": best_solution.fitness,
            "best_solution": best_solution.chromosome_values,
            "execution_time": execution_time,
            "best_fitness_history": plotter.best_fitness_history.copy(),
            "avg_fitness_history": plotter.avg_fitness_history.copy(),
        }

    def run_gjo(self, params):
        from algorithms.config import config
        from algorithms.gjo import run_gjo_algorithm

        config.range_start = params["range_start"]
        config.range_end = params["range_end"]
        config.epochs = params["epochs"]
        config.population_size = params["population_size"]
        config.precision = params["precision"]
        config.num_variables = params["num_variables"]
        config.optimization_type = params.get("optimization_type", "min")


        best_solution, execution_time, plotter = run_gjo_algorithm()
        return {
            "best_fitness": best_solution.fitness,
            "best_solution": best_solution.chromosome_values,
            "execution_time": execution_time,
            "best_fitness_history": plotter.best_fitness_history.copy(),
            "avg_fitness_history": plotter.avg_fitness_history.copy(),
        }