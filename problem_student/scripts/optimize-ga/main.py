#!/usr/bin/python3

from fitness import FitnessEvaluator
from genetic_algorithm import GeneticRun
import gene_codec as gc



def optimisation(N = 100, G = 100, p_c = 0.7, p_m = 0.001, log = False, gui = False):

    fit_eval = FitnessEvaluator()

    if log:
        print('Performance optimisation')
        print('------------------------')
        print('Genetic algorithm:')
        print('Population:  N = ', N)
        print('Generations: G = ', G)
        print('Crossover probability: p_c = ', p_c)
        print('Mutation probability:  p_m = ', p_m)


    # Generates random population
    perf_run = GeneticRun(gc.length(gc.default_gene_format), N)
    if log:
        print('\nGenerated initial population')
        print('Generation: 0\n')


    # Evolution loop
    while perf_run.generation < G:
        perf_run.evolve(fit_eval.performance, p_c, p_m)
        if log:
            print(f'Generation: {perf_run.generation}')
            print(f'Average fitness: {perf_run.record[-1]["avg_fitness"]}')
            print(f'Max fitness: {perf_run.record[-1]["max_fitness"]}\n')

    # Return genetic run
    return perf_run



if __name__ == '__main__':

    try:
        optimisation(log = True, gui = True)
    except KeyboardInterrupt:
        pass
