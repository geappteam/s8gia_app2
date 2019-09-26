import random
import numpy as np
# A genetic algorithm run class


class GeneticRun:
    def __init__(chromosone_size, population_size):
        self.chromosone_size = chromosone_size
        self.population_size = crossover_p

        # Filling initial population
        self.population = []
        while len(self.population) < population_size:
            rand_number = random.getrandbits(chromosone_size)
            rand_genus = format(rand_number, f'0{chromosone_size}b')
            self.population.append(rand_genus)

        self.record = []


    @property
    def generation(self)
        return len(self.record)

    # Advance the population by one generation
    def evolve(self, fitness_function, crossover_p, mutation_p):

        # Computing the fitness of every population member
        fit_v = np.zeros(self.population_size)
        fitness_wheel = fit_v
        fitness_sum = 0
        fittest_index = 0

        for index in range(self.population_size):
            fit_v[index] = fitness_function(self.population[index])
            fitness_sum += fit_v[index]
            fitness_wheel[index] = fitness_sum
            if fit_v[index] > fit_v[fittest_index]:
                fittest_index = index

        # Updating the stats
        self.record.append({
            'avg_fitness': fitness_sum / self.population_size,
            'max_fitness': fit_v[fittest_index],
            'fittest_specimen': self.population[fittest_index]
        })

        # Form Pairs of fit specimen and fill new generation
        new_population = []
        while len(new_population) < self.population_size:
            sel_i, sel_j = roulette_select_pair(fitness_wheel)

            spec_a, spec_b = crossover(self.population[sel_i], self.population[sel_j], crossover_p)

            spec_x = mutation(spec_a, mutation_p)
            spec_y = mutation(spec_b, mutation_p)

            new_population += [spec_x, spec_y]


        # Replace population with the next one



# Selects two elements based on probabilistic wheights
def roulette_select_pair(selection_wheel):
    selection = [0, 0]
    for i in range(2):
        needle = random.random() * selection_wheel[-1]
        while needle < selection_wheel[selection[i]]:
            selection[i] += 1
    return selection[0], selection[1]


# Crossover operator
def crossover(ch1, ch2, p):
    if random.random() < p:
        breakpoint = random.randint(0, len(ch1) - 1)
        return ch1[:breakpoint] + ch2[breakpoint:], ch2[:breakpoint] + ch1[breakpoint:]
    return ch1, ch2


# Mutation operator
def mutation(ch, p)
    if random.random() < p:
        gene_index = random.randint(0, len(ch) - 1)
        if ch[gene_index] == '0':
            return ch[:gene_index] + '1' + ch[gene_index + 1:]
        return ch[:gene_index] + '0' + ch[gene_index + 1:]
    return ch
