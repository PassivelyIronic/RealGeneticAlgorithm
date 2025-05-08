import random
from algorithms.individual import Individual
from algorithms.selection import best_selection, roulette_selection, tournament_selection
from algorithms.crossover import arithmetic_crossover, linear_crossover, blend_alpha_crossover, \
    blend_alpha_beta_crossover, averaging_crossover
from algorithms.config import config
from algorithms.fitness import evaluate_fitness


class Population:
    def __init__(self, config=config):
        self.config = config
        self.size = config.population_size
        self.num_variables = config.num_variables
        self.precision = config.precision
        self.generation = 0
        self.individuals = [
            Individual(num_variables=self.num_variables, precision=self.precision)
            for _ in range(self.size)
        ]
        self.best_individual = None
        self.best_fitness_history = []
        self.avg_fitness_history = []

    def evaluate_all(self, fitness_func=evaluate_fitness, bounds=None):
        if bounds is None:
            bounds = (self.config.range_start, self.config.range_end)

        for individual in self.individuals:
            individual.evaluate(fitness_func, bounds)

        if self.config.optimization_type == "max":
            self.best_individual = max(self.individuals, key=lambda ind: ind.fitness)
        else:
            self.best_individual = min(self.individuals, key=lambda ind: ind.fitness)

        avg_fitness = sum(ind.fitness for ind in self.individuals) / self.size
        self.best_fitness_history.append(self.best_individual.fitness)
        self.avg_fitness_history.append(avg_fitness)

    def select_parents(self):
        method = self.config.selection_method

        if method == "tournament":
            selected = tournament_selection(self.individuals, tournament_size=self.config.tournament_size)
        elif method == "best":
            selected = best_selection(self.individuals, self.config.best_selection_amount)
        elif method == "roulette":
            selected = roulette_selection(self.individuals, self.size)
        else:
            raise ValueError(f"Unknown selection method: {method}")

        return selected

    def crossover(self, parent1, parent2):
        method = self.config.crossover_method

        child1 = Individual(num_variables=self.num_variables, precision=self.precision, random_init=False)
        child2 = Individual(num_variables=self.num_variables, precision=self.precision, random_init=False)

        parent1_values = parent1.chromosome_values
        parent2_values = parent2.chromosome_values

        if method == "arithmetic":
            child1_values, child2_values = arithmetic_crossover(parent1_values, parent2_values)
        elif method == "linear":
            child1_values, child2_values = linear_crossover(parent1_values, parent2_values)
        elif method == "blend_alpha":
            child1_values, child2_values = blend_alpha_crossover(parent1_values, parent2_values)
        elif method == "blend_alpha_beta":
            child1_values, child2_values = blend_alpha_beta_crossover(parent1_values, parent2_values)
        elif method == "averaging":
            child1_values, child2_values = averaging_crossover(parent1_values, parent2_values)
        else:
            child1_values, child2_values = arithmetic_crossover(parent1_values, parent2_values)

        child1.chromosomes[0].set_chromosome(child1_values)
        child2.chromosomes[0].set_chromosome(child2_values)

        return child1, child2

    def evolve(self):
        selected = self.select_parents()
        if self.config.optimization_type == "max":
            elites = sorted(self.individuals, key=lambda ind: ind.fitness, reverse=True)[
                     :self.config.best_selection_amount]
        else:
            elites = sorted(self.individuals, key=lambda ind: ind.fitness)[:self.config.best_selection_amount]

        new_population = []

        while len(new_population) < self.size - len(elites):
            parent1 = random.choice(selected)
            parent2 = random.choice(selected)

            if random.random() < self.config.crossover_probability:
                child1, child2 = self.crossover(parent1, parent2)
            else:
                child1 = Individual(num_variables=self.num_variables, precision=self.precision, random_init=False)
                child2 = Individual(num_variables=self.num_variables, precision=self.precision, random_init=False)
                child1.chromosomes[0].set_chromosome(parent1.chromosome_values)
                child2.chromosomes[0].set_chromosome(parent2.chromosome_values)

            child1.apply_mutation(self.config.mutation_probability, method=self.config.mutation_method)
            child2.apply_mutation(self.config.mutation_probability, method=self.config.mutation_method)

            if len(new_population) < self.size - len(elites):
                new_population.append(child1)
            if len(new_population) < self.size - len(elites):
                new_population.append(child2)

        new_population.extend(elites)
        self.individuals = new_population
        self.generation += 1

    def run(self, fitness_func=evaluate_fitness, bounds=None):
        for gen in range(self.config.epochs):
            self.evaluate_all(fitness_func, bounds)
            self.evolve()

            if gen % 10 == 0:
                print(f"Epoka {gen}, najlepsze przystosowanie: {self.best_individual.fitness}")

        self.evaluate_all(fitness_func, bounds)
        return self.best_individual, self.best_fitness_history, self.avg_fitness_history
