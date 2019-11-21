# Copyright (c) 2019, Simon Brodeur
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  - Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#  - Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#  - Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES LOSS OF USE, DATA,
# OR PROFITS OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# Author: Simon Brodeur <simon.brodeur@usherbrooke.ca>
# Université de Sherbrooke, APP3 S8GIA, A2019

import numpy as np
import matplotlib.pyplot as plt

########################
# Define helper functions here
########################


# usage: FITNESS = evaluateFitness(X)
#
# Evaluate the number of errors on constraints
#
# Input:
# - X, a square grid
#
# Output:
# - FITNESS, 1 / (1 + number of errors)
#
def evaluateFitness(x):

    # Compute the magic constant
    n = x.shape[0]
    magicConstant = int(n * (n ** 2 + 1) / 2)

    # Accumulate the sum of absoluate errors from the magic number
    error = 0

    # Sum of each row and column
    error += np.sum(np.abs(magicConstant - np.sum(x, axis=0)))
    error += np.sum(np.abs(magicConstant - np.sum(x, axis=1)))

    # Sum of the diagonals
    error += np.abs(magicConstant - np.sum(x.diagonal()))
    error += np.abs(magicConstant - np.sum(np.fliplr(x).diagonal()))

    return 1 / (1 + error)


# usage: POPULATION = initializePopulation(POPSIZE, N)
#
# Initialize the population as a tensor, where each individual is a square grid of integers representing magic square.
#
# Input:
# - POPSIZE, the population size.
# - N, the size of the square grid
#
# Output:
# - POPULATION, an integer tensor whose second and third dimensions correspond to encoded individuals as square grids.
#
def initializePopulation(popsize, n):
    population = np.zeros((popsize, n, n), dtype=np.int)

    # TODO: initialize the population

    return population


# usage: PAIRS = doSelection(POPULATION, FITNESS, NUMPAIRS)
#
# Select pairs of individuals from the population.
#
# Input:
# - POPULATION, an integer tensor whose second and third dimensions correspond to encoded individuals as square grids.
# - FITNESS, a vector of fitness values for the population.
# - NUMPAIRS, the number of pairs of individual to generate.
#
# Output:
# - PAIRS, an array of tuples containing pairs of individuals.
#
def doSelection(population, fitness, numPairs):

    # TODO: select pairs of individual in the population
    pairs = []
    for _ in range(numPairs):
        idx1 = np.random.randint(0, len(population))
        idx2 = np.random.randint(0, len(population))
        pairs.append((population[idx1], population[idx2]))

    return pairs


# usage: [NIND1,NIND2] = doCrossover(IND1, IND2, CROSSOVERPROB)
#
# Perform a crossover operation between two individuals, with a given probability.
#
# Input:
# - IND1, an integer matrix encoding the first individual as a square grid.
# - IND2, an integer matrix encoding the second individual as a square grid.
# - CROSSOVERPROB, the crossover probability.
#
# Output:
# - NIND1, an integer matrix encoding the first new individual as a square grid.
# - NIND2, an integer matrix encoding the second new individual as a square grid.
#
def doCrossover(ind1, ind2, crossoverProb):

    # TODO: Perform a crossover between two individuals
    nind1 = ind1.copy()
    nind2 = ind2.copy()
    return nind1, nind2


# usage: [NPOPULATION] = doMutation(POPULATION, MUTATIONPROB)
#
# Perform a mutation operation over the entire population.
#
# Input:
# - POPULATION, an integer tensor whose second and third dimensions correspond to encoded individuals as square grids.
# - MUTATIONPROB, the mutation probability.
#
# Output:
# - NPOPULATION, the new population.
#
def doMutation(population, mutationProb):
    # TODO: Apply mutation to the population
    npopulation = population.copy()
    return npopulation


########################
# Define code logic here
########################


def main():

    # Fix random number generator seed for reproducible results
    np.random.seed(0)

    # Difficulty of the problem
    # TODO: vary the level of difficulty between 3 and 8
    n = 5

    # TODO : adjust population size
    popsize = 10
    population = initializePopulation(popsize, n)

    # TODO : Adjust optimization meta-parameters
    numGenerations = 100
    mutationProb = 0.0
    crossoverProb = 0.0

    bestIndividual = []
    bestIndividualFitness = -1e10
    maxFitnessRecord = np.zeros((numGenerations,))
    overallMaxFitnessRecord = np.zeros((numGenerations,))
    avgMaxFitnessRecord = np.zeros((numGenerations,))

    for i in range(numGenerations):

        # Evaluate fitness function for all individuals in the population
        fitness = np.zeros((popsize,))
        for p in range(popsize):
            # Calculate fitness
            fitness[p] = evaluateFitness(population[p])

        # Save best individual across all generations
        bestFitness = np.max(fitness)
        if bestFitness > bestIndividualFitness:
            bestIndividual = population[fitness == np.max(fitness)][0]
            bestIndividualFitness = bestFitness

        # Record progress information
        maxFitnessRecord[i] = np.max(fitness)
        overallMaxFitnessRecord[i] = bestIndividualFitness
        avgMaxFitnessRecord[i] = np.mean(fitness)

        # Display progress information
        print('Generation no.%d: best fitness is %f, average is %f' %
              (i, maxFitnessRecord[i], avgMaxFitnessRecord[i]))
        print('Overall best fitness is %f' % bestIndividualFitness)

        newPopulation = []
        numPairs = int(popsize / 2)
        pairs = doSelection(population, fitness, numPairs)
        for ind1, ind2 in pairs:
            # Perform a cross-over and place individuals in the new population
            nind1, nind2 = doCrossover(ind1, ind2, crossoverProb)
            newPopulation.extend([nind1, nind2])
        newPopulation = np.array(newPopulation)

        # Apply mutation to all individuals in the population
        newPopulation = doMutation(newPopulation, mutationProb)

        # Replace current population with the new one
        population = newPopulation.copy()

    # Display best individual
    print('#########################')
    print('Best individual: \n', bestIndividual)
    print('#########################')

    # Display plot of fitness over generations
    fig = plt.figure()
    n = np.arange(numGenerations)
    ax = fig.add_subplot(111)
    ax.plot(n, maxFitnessRecord, '-r', label='Generation Max')
    ax.plot(n, overallMaxFitnessRecord, '-b', label='Overall Max')
    ax.plot(n, avgMaxFitnessRecord, '--k', label='Generation Average')
    ax.set_title('Fitness value over generations')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness value')
    ax.legend()
    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
