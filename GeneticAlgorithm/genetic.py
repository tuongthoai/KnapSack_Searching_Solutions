import random
import heapq
import matplotlib.pyplot as plt  # draw chart
import tracemalloc  # memory
from time import perf_counter_ns  # time

inp = int(input(f"Input file: "))
with open("./TEST_CASE/INPUT_"+str(inp)+".txt", "r") as f:
    max_weight = float(f.readline().strip())
    num_classes = int(f.readline().strip())
    weights = list(map(float, f.readline().strip().split(", ")))
    values = list(map(int, f.readline().strip().split(", ")))
    class_labels = list(map(int, f.readline().strip().split(", ")))


class GeneticAlgorithm:
    POPULATION_SIZE: int = 1000
    MAX_GEN: int = 100

    def __init__(self):
        self.max_weight = max_weight
        self.num_classes = num_classes
        self.weights = weights
        self.values = values
        self.class_labels = class_labels
        self.n = len(self.weights)
        self.best_fitness = 0
        self.best_chromosome = []
        self.population_size = self.POPULATION_SIZE if self.n < 1000 else 10*self.POPULATION_SIZE
        tmp = [self.create_chromosome()
               for _ in range(10000)]
        self.population = sorted(tmp, key=lambda i: self.fitness(i), reverse=True)[
            :self.population_size]

    def create_chromosome(self):
        return [random.randint(0, 1) for _ in range(len(self.weights))]

    def fitness(self, chromosome):
        total_weight = sum([self.weights[i]
                           for i, j in enumerate(chromosome) if j != 0])
        total_value = sum([self.values[i]
                          for i, j in enumerate(chromosome) if j != 0])
        type_of_class = set([self.class_labels[i]
                            for i, j in enumerate(chromosome) if j != 0])
        return 0 if total_weight > self.max_weight or len(type_of_class) < self.num_classes else total_value

    def selection(self, population):
        """FPS"""
        fitness_values = [(self.fitness(chromosome), chromosome)
                          for chromosome in population]
        return [chromosome for _, chromosome in heapq.nlargest(2, fitness_values)]

    def crossover(self, parent1, parent2):
        """Crossover two chromosome to produce offspring."""
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2

    def mutation(self, solution):
        """BIT FLIP MUTATION"""
        while(self.fitness(solution) == 0):
            mutation_point = random.randint(1, len(solution) - 1)
            solution[mutation_point] = 1 - solution[mutation_point]
        return solution

        # USE BIT MASK
        # mutated_solution = solution.copy()  # create a copy of the original solution
        # # create a bit mask with all bits set to 1
        # mask = (1 << len(solution)) - 1
        # # compute the number of bits to flip
        # flip_mask = int(0.1 * len(solution))
        # # generate a random bit mask to flip bits
        # flip_bits = random.randint(0, mask) & ((1 << flip_mask) - 1)
        # for i in range(len(solution)):
        #     if flip_bits & (1 << i):
        #         # flip the i-th bit if it is set in the flip_bits mask
        #         mutated_solution[i] = 1 - solution[i]
        # return mutated_solution

    def print(self):
        with open("./RESULT/OUTPUT_"+str(inp)+".txt", "w") as f:
            f.write(str(best_fitness) + "\n")
            f.write(", ".join(str(x) for x in best_chromosome))

    def get_best(self):
        """Get best solution"""
        for chromosome in self.population:
            tmp = self.fitness(chromosome)
            if tmp > self.best_fitness:
                self.best_fitness = tmp
                self.best_chromosome = chromosome

    def evolve(self, trace: 'list[int]' = None):
        """Evolve the population for a certain number of generations."""
        for _ in range(self.MAX_GEN):
            # Generate the offspring
            parents = self.selection(self.population)
            if trace is not None:
                trace.append(self.best_fitness)
            offspring = []
            while len(offspring) < self.population_size:
                # Select the parents
                parent1, parent2 = parents[0], parents[1]
                child1, child2 = self.crossover(parent1, parent2)
                if random.uniform(0, 1) < 0.1:  # MUTATION RATE
                    child1 = self.mutation(child1)
                if random.uniform(0, 1) < 0.1:
                    child2 = self.mutation(child2)
                offspring.append(child1)
                offspring.append(child2)
            self.population = self.selection(parents+offspring)
            self.get_best()


if __name__ == '__main__':
    timeAvg = []
    memoryAvg = []
    for _ in range(3):  # run 3 times
        tracemalloc.start()  # "get memory" start here
        time_start = perf_counter_ns()  # "get time" start here
        problem = GeneticAlgorithm()
        trace = []
        problem.evolve(trace)
        best_chromosome = problem.best_chromosome
        best_fitness = problem.best_fitness
        print(f"Best solution found:")
        print(f"Total value:", best_fitness)
        print(best_chromosome)
        time_end = perf_counter_ns()  # "get time" end here
        memory = tracemalloc.get_traced_memory()[1]  # "get memory" end here
        tracemalloc.stop()
        timeAvg.append((time_end - time_start) / 10**6)
        memoryAvg.append(memory / 1024**2)
        plt.plot(trace)
        problem.print()
    print(f'Running time (average of 3 runs): {sum(timeAvg) / 3:.2f} ms.')
    print(f'Consumed memory (average of 3 runs): {sum(memoryAvg) / 3:.4f} MB.')
    plt.title("Knapsack Problem")
    plt.xlabel("Generations")
    plt.ylabel("Best fitness")
    plt.show()
