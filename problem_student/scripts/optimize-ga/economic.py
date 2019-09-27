#!/usr/bin/python3

from fitness import FitnessEvaluator
from genetic_algorithm import GeneticRun
import gene_codec as gc
import pprint as pp
import performance_graph as p_graph



def optimisation(N = 80, G = 100, p_c = 0.7, p_m = 0.001, log = False, gui = False):

    fit_eval = FitnessEvaluator()

    if log:
        print('Efficiency optimisation')
        print('------------------------')
        print('Genetic algorithm:')
        print('Population:  N = ', N)
        print('Generations: G = ', G)
        print('Crossover probability: p_c = ', p_c)
        print('Mutation probability:  p_m = ', p_m)

    if gui:
        p_graph.init('Economic profile', f'p_c = {p_c}   p_m = {p_m}   N = {N}   G = {G}')


    # Generates random population
    perf_run = GeneticRun(gc.length(gc.default_gene_format), N)
    if log:
        print('\nGenerated initial population')
        print('Generation: 0\n')

    # Evolution loop
    while perf_run.generation < G:
        perf_run.evolve(fit_eval.economic, p_c, p_m)
        if log:
            print(f'Generation: {perf_run.generation}')
            print(f'Average fitness: {perf_run.record[-1]["avg_fitness"]}')
            print(f'Max fitness: {perf_run.record[-1]["max_fitness"]}\n')
        if gui:
            p_graph.update(perf_run.record)

    # Present best specimen so far
    best_specimen_ch = max(perf_run.record, key = lambda x: x['max_fitness'])['fittest_specimen']
    best_specimen = gc.decode(best_specimen_ch)
    distance, fuel, speed = fit_eval.simulate(best_specimen_ch, tries = 10)
    if log:
        print('-------------')
        print('Best specimen')
        print('-------------')
        print('Best chromosones:\n', best_specimen_ch)
        print('Best parameters:')
        pp.pprint(best_specimen)
        print('Specimen performance:')
        print(' Distance (m):', distance)
        print(' Fuel use (l):', fuel)
        print(' Speed (km/h):', speed)

    if gui:
        p_graph.join()


    # Return best specimen
    return best_specimen



if __name__ == '__main__':

    try:
        optimisation(log = True, gui = True)
    except KeyboardInterrupt:
        pass
