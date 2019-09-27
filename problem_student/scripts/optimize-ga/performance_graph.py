from threading import Thread
import queue
import matplotlib.pyplot as plt


# A simple perfomance graph class to dynamically at each generation

class PerfGraph (Thread):
    def __init__(self, title, description):
        Thread.__init__(self)
        self.title = title
        self.description = description
        self.closed = False
        self.queue = queue.LifoQueue(1)


    # Feed data from genetic run
    def update(self, record):
        data = {'avg':[], 'max':[], 'gen': range(1, 1+len(record))}
        for gen in record:
            data['avg'].append(gen['avg_fitness'])
            data['max'].append(gen['max_fitness'])
        if not self.closed:
            self.queue.put(data)


    # Gui thread function
    def run(self):

        plt.ion()
        fig = plt.figure()
        fig.canvas.mpl_connect('close_event', self._handle_close)
        plt.suptitle(self.title)
        plt.title(self.description)
        plt.xlabel('Generation')
        plt.ylabel('Fitness')

        while not self.closed:

            try:
                data = self.queue.get_nowait()
                plt.plot(data['gen'], data['avg'], 'k',  label = 'Average')
                plt.plot(data['gen'], data['max'], 'k:', label = 'Best')
                plt.legend(('Average', 'Best'))
            except queue.Empty:
                pass

            try:
                plt.pause(0.2)
            except:
                pass


    def _handle_close(self, event):
        self.closed = True
