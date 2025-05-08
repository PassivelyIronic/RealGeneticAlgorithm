from algorithms.chromosome import Chromosome
from algorithms.config import config
import random
from algorithms.mutation import mutate_uniform, mutate_gaussian


class Individual:
    def __init__(self, num_variables=2, precision=10, random_init=True):
        self.num_variables = num_variables
        self.precision = precision
        self.chromosomes = [Chromosome(num_genes=self.num_variables, random_init=random_init)]
        self.fitness = float('inf')
        self.chromosome_values = []

    def evaluate(self, fitness_func=None, bounds=None):
        self.chromosome_values = []

        for i in range(self.num_variables):
            value = self.chromosomes[0][i]
            self.chromosome_values.append(value)

        if fitness_func:
            self.fitness = fitness_func(self)
        else:
            from algorithms.fitness import evaluate_fitness
            self.fitness = evaluate_fitness(self)

        return self.fitness

    def apply_mutation(self, mutation_rate=0.3, method="uniform"):
        chromosome = self.chromosomes[0].chromosome

        for i in range(len(chromosome)):
            if random.random() < mutation_rate:
                if method == "uniform":

                    mutate_uniform(chromosome, i)
                elif method == "gaussian":

                    mutate_gaussian(chromosome, i)
                else:
                    print(f"Ostrzeżenie: Nieznana metoda mutacji: {method}")