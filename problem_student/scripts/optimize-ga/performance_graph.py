import sys
from threading import Thread
import queue
import matplotlib.pyplot as plt


# Plotting functions


def init(title, description):
    plt.ion()
    plt.suptitle(title)
    plt.title(description)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')


def update(record):
    data = {'avg':[], 'max':[], 'gen': range(1, 1+len(record))}
    for gen in record:
        data['avg'].append(gen['avg_fitness'])
        data['max'].append(gen['max_fitness'])

    plt.plot(data['gen'], data['avg'], 'k',  label = 'Average')
    plt.plot(data['gen'], data['max'], 'k:', label = 'Best')
    plt.legend(('Average', 'Best'))

    # Don't look too close ( ͡° ͜ʖ ͡°)
    try:
        plt.pause(0.05)
    except:
        pass


def join():
    plt.ioff()
    plt.show()
