#!/usr/bin/python3

import os
import logging
from fitness import FitnessEvaluator
import sample_params
import gene_codec as gc

CDIR = os.path.dirname(os.path.realpath(__file__))

logger = logging.getLogger(__name__)


def main():
    fitness = FitnessEvaluator()
    genome = gc.encode(sample_params.default)
    print(f'Perf fitness of default : {fitness.performance(genome)}')
    print(f'Eco fitness of default : {fitness.economic(genome)}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    try:
        main()

    except KeyboardInterrupt:
        pass

    logger.info('All done.')
