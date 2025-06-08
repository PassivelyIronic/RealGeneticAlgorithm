import numpy as np
import random
import math
from algorithms.individual import Individual
from algorithms.fitness import evaluate_fitness
from algorithms.config import config
import time


class GoldenJackal:

    def __init__(self, num_variables, bounds):
        self.num_variables = num_variables
        self.bounds = bounds
        self.position = [random.uniform(bounds[0], bounds[1]) for _ in range(num_variables)]
        self.fitness = float('inf')
        
    def evaluate(self, fitness_func):
        individual = Individual(num_variables=self.num_variables, precision=5, random_init=False)
        individual.chromosome_values = self.position.copy()
        self.fitness = fitness_func(individual)
        return self.fitness
    
    def update_position(self, new_position, bounds):
        self.position = [max(bounds[0], min(bounds[1], pos)) for pos in new_position]


class GJOAlgorithm:

    
    def __init__(self, config_obj=config):
        self.config = config_obj
        self.population_size = config_obj.population_size
        self.num_variables = config_obj.num_variables
        self.max_iterations = config_obj.epochs
        self.bounds = (config_obj.range_start, config_obj.range_end)

        self.C1 = 0.5
        self.C2 = 1.5

        self.population = []
        self.best_jackal = None
        self.second_best_jackal = None

        self.best_fitness_history = []
        self.avg_fitness_history = []
        
    def initialize_population(self):
        self.population = []
        for _ in range(self.population_size):
            jackal = GoldenJackal(self.num_variables, self.bounds)
            self.population.append(jackal)
            
    def evaluate_population(self, fitness_func):
        for jackal in self.population:
            jackal.evaluate(fitness_func)

        if self.config.optimization_type == "max":
            self.population.sort(key=lambda x: x.fitness, reverse=True)
        else:
            self.population.sort(key=lambda x: x.fitness)

        self.best_jackal = self.population[0]
        self.second_best_jackal = self.population[1]

        avg_fitness = sum(jackal.fitness for jackal in self.population) / self.population_size
        self.best_fitness_history.append(self.best_jackal.fitness)
        self.avg_fitness_history.append(avg_fitness)
        
    def levy_flight(self, beta=1.5):
        """
        Generowanie kroku Levy flight dla eksploracji
        """
        sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
                (math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2)))) ** (1 / beta)
        
        u = np.random.normal(0, sigma)
        v = np.random.normal(0, 1)
        step = u / abs(v) ** (1 / beta)
        return step
        
    def update_positions(self, iteration):

        a = 2 - 2 * iteration / self.max_iterations
        
        for i, jackal in enumerate(self.population):
            if i == 0 or i == 1:
                continue
                
            new_position = [0] * self.num_variables
            
            for j in range(self.num_variables):
                r1, r2, r3, r4 = [random.random() for _ in range(4)]

                if random.random() < 0.5:

                    if random.random() < 0.5:
                        levy_step = self.levy_flight()
                        new_position[j] = self.best_jackal.position[j] + levy_step * abs(
                            self.best_jackal.position[j] - jackal.position[j])
                    else:
                        A1 = 2 * a * r1 - a
                        C1 = 2 * r2
                        D_prey1 = abs(C1 * self.best_jackal.position[j] - jackal.position[j])
                        X1 = self.best_jackal.position[j] - A1 * D_prey1
                        
                        A2 = 2 * a * r3 - a
                        C2 = 2 * r4
                        D_prey2 = abs(C2 * self.second_best_jackal.position[j] - jackal.position[j])
                        X2 = self.second_best_jackal.position[j] - A2 * D_prey2
                        
                        new_position[j] = (X1 + X2) / 2
                else:
                    if abs(a) < 1:
                        RL = 0.05 * random.gauss(0, 1)  # Levy random walk
                        new_position[j] = (self.best_jackal.position[j] + 
                                         self.second_best_jackal.position[j]) / 2 + RL
                    else:
                        rand_jackal = random.choice(self.population)
                        new_position[j] = rand_jackal.position[j] + random.uniform(-1, 1) * abs(
                            rand_jackal.position[j] - jackal.position[j])
            

            jackal.update_position(new_position, self.bounds)
            
    def run(self, fitness_func=evaluate_fitness):

        print("Rozpoczynanie algorytmu Golden Jackal Optimization...")
        start_time = time.time()

        self.initialize_population()

        self.evaluate_population(fitness_func)
        print(f"Iteracja 0, najlepsze przystosowanie: {self.best_jackal.fitness}")

        for iteration in range(1, self.max_iterations + 1):
            self.update_positions(iteration)

            self.evaluate_population(fitness_func)

            if iteration % 10 == 0:
                print(f"Iteracja {iteration}, najlepsze przystosowanie: {self.best_jackal.fitness}")
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"Algorytm GJO zakończony. Czas wykonania: {execution_time:.2f} sekund")

        best_individual = Individual(num_variables=self.num_variables, precision=5, random_init=False)
        best_individual.chromosome_values = self.best_jackal.position.copy()
        best_individual.fitness = self.best_jackal.fitness
        
        return best_individual, self.best_fitness_history, self.avg_fitness_history, execution_time


def run_gjo_algorithm():

    from apps.plotter import Plotter
    
    bounds = (config.range_start, config.range_end)
    plotter = Plotter()

    gjo = GJOAlgorithm(config)
    best_individual, best_fitness_history, avg_fitness_history, execution_time = gjo.run(evaluate_fitness)

    for i in range(len(best_fitness_history)):
        plotter.update_history(best_fitness_history[i], 
                             avg_fitness_history[i] if i < len(avg_fitness_history) else None)
    
    print(f"Najlepsze rozwiązanie: {best_individual.chromosome_values}")
    print(f"Wartość funkcji celu: {best_individual.fitness}")

    # plotter.show_results_window(best_individual, execution_time)
    
    return best_individual, execution_time, plotter


if __name__ == "__main__":
    config.function = "Martin and Gaddy"
    config.population_size = 30
    config.epochs = 100
    config.num_variables = 2
    config.range_start = -20
    config.range_end = 20
    config.optimization_type = "min"
    
    best_solution, exec_time, plotter = run_gjo_algorithm()