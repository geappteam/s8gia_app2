# This file contains the fitness functions to run the genetic algorithm

import sys
import gene_codec as gc

sys.path.append('../..')
from torcs.optim.core import TorcsOptimizationEnv


# Class with fitness calculation
class FitnessEvaluator:
    def __init__(self, sim_time_s = 40):
        self.race_track = TorcsOptimizationEnv(sim_time_s)

    # defined by the distance travelled at the end of the simulation
    def performance(
            self,
            genome,
            tries = 1,
            genome_format = gc.default_gene_format):
        distance, _ = self._simulate(genome, tries, genome_format)
        return distance

    # ratio of the distance traveled over the fuel consumed
    def economic(
            self,
            genome,
            tries = 1,
            genome_format = gc.default_gene_format):
        distance, fuel = self._simulate(genome, tries, genome_format)
        return distance / fuel

    # perform track simulation of the given genome to help evaluate fitness
    def _simulate(
            self,
            genome,
            tries = 1,
            genome_format = gc.default_gene_format):

        assert tries > 0

        parameters = gc.decode(genome)

        distanceSum = 0
        fuelSum = 0
        for i in range(tries):
            observation, _, _, _ = self.race_track.step(parameters)
            distanceSum += observation['distRaced'][0]
            fuelSum += observation['fuelUsed'][0]
        distance = distanceSum / tries
        fuel = fuelSum / tries

        return distance, fuel
