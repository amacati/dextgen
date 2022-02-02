import multiprocessing as mp
import threading
import logging
import numpy as np


logger = logging.getLogger(__name__)


class SharedReplayBuffer:
    
    def __init__(self, n_samples: int, n_processes: int):
        self._mp_mgr = mp.Manager()
        self.queue = self._mp_mgr.Queue(n_samples)  # Avoid deadlocking on process join due to open pipes to _queue
        self._prod_events = [mp.Event() for _ in range(n_processes)]
        # Store data in lists instead of dataframes because frequent appends would make it extremely slow.
        self.buffer = [[],[],[],[],[]]  # List of 5 lists for s, a, r, s_next, done
        self._cons_thread = None
        self._cons_stop = threading.Event()

    def _consume(self):
        while True:
            while not self.queue.empty():
                sample = self.queue.get()
                for i in range(5):
                    self.buffer[i].append(sample[i])
            # New sample could have arrived between loop and stop event
            if self._cons_stop.is_set() and self.queue.empty():
                logger.debug("Consumer thread loop shutdown")
                break

    def start(self):
        self._cons_thread = threading.Thread(target=self._consume)
        self._cons_stop.clear()
        self._cons_thread.start()
        logger.debug("Consumer thread started")

    def join(self):
        for event in self._prod_events:
            event.wait()
            event.clear() # Reset flags for the next run
        logger.debug("All worker events set, shutting down consumer thread")
        self.stop()

    def stop(self):
        self._cons_stop.set()
        logger.debug("Set consumer stop event")
        self._cons_thread.join()
        logger.debug("Consumer joined main thread")

    def sample(self, n: int):
        if n > len(self.buffer[0]):
            raise RuntimeError("Tried to create batch with more samples than available!")
        samples = np.random.choice(len(self.buffer[0]), n, replace=False)
        return (np.array([self.buffer[i][j] for j in samples], dtype=np.float32) for i in range(5))
