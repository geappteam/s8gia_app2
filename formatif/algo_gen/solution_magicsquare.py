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

    # Validation
    numDigits = np.prod(x.shape)
    assert np.all(x >= 0) and len(np.unique(x) == numDigits)
    assert np.all(x >= 0) and len(np.unique(x) == numDigits)

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
    numDigits = np.prod((n, n))
    population = np.zeros((popsize, n, n), dtype=np.int)
    for p in range(len(population)):

        # Fill with random permutations
        population[p] = np.random.permutation(n * n).reshape((n, n)) + 1

        # Validation
        assert len(np.unique(population[p]) == numDigits)

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
    # Compute selection probability distribution
    assert np.all(fitness >= 0.0)
    selectProb = np.cumsum(fitness) / np.sum(fitness)

    # Perform a roulette-wheel selection
    pairs = []
    for _ in range(numPairs):
        idx1 = np.argwhere(selectProb > np.random.uniform())[0][0]
        idx2 = np.argwhere(selectProb > np.random.uniform())[0][0]
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
    numDigits = np.prod(ind1.shape)
    nind1 = ind1.copy().ravel()
    nind2 = ind2.copy().ravel()
    if crossoverProb > np.random.uniform():
        # Initialize offsprings as unassigned
        nind1 = -1 * np.ones_like(ind1.ravel())
        nind2 = -1 * np.ones_like(ind2.ravel())

        # Select a random crossover point
        idx1 = np.random.randint(numDigits)

        # Swap the segments and assign to offsprings
        nind1[idx1:] = ind2.ravel()[idx1:]
        nind2[idx1:] = ind1.ravel()[idx1:]

        # Find unused values in parents
        mask1 = np.in1d(ind1.ravel(), nind1[idx1:], invert=True)  # First offspring
        uind1 = ind1.ravel()[np.where(mask1)]
        mask2 = np.in1d(ind2.ravel(), nind2[idx1:], invert=True)  # Second offspring
        uind2 = ind2.ravel()[np.where(mask2)]

        # Assigned unused values to offsprings, conserving the original order
        nind1[:idx1] = uind1  # First offspring
        nind2[:idx1] = uind2  # Second offspring

        # Validation
        assert np.all(nind1 >= 0) and len(np.unique(nind1) == numDigits)
        assert np.all(nind2 >= 0) and len(np.unique(nind2) == numDigits)

    # Reshape to square
    nind1 = nind1.reshape(ind1.shape)
    nind2 = nind2.reshape(ind2.shape)

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
    npopulation = population.copy()
    for p in range(len(population)):
        if mutationProb > np.random.uniform():
            # Select random digits to swap
            idx1, idx2 = np.random.randint(population.shape[1], size=(2, 2))

            # Perform the swap
            tmp = npopulation[(p, *idx1)]
            npopulation[(p, *idx1)] = npopulation[(p, *idx2)]
            npopulation[(p, *idx2)] = tmp

        # Validation
        numDigits = np.prod(npopulation.shape[1:])
        assert len(np.unique(npopulation[p]) == numDigits)

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

    # The parameters for encoding the population
    popsize = 100
    population = initializePopulation(popsize, n)

    # The parameters for the optimisation
    numGenerations = 500
    elitismPerc = 0.1
    mutationProb = 0.5
    crossoverProb = 0.1

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

        # Replace current population with the new one, keeping a given proportion of the elites
        indices = np.argsort(fitness)
        fitness = fitness[indices]
        population = population[indices]
        nbElites = int(np.ceil(elitismPerc * len(population)))
        population = np.concatenate([population[len(population) - nbElites:], newPopulation[nbElites:]])

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
