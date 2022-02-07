import multiprocessing as mp
from collections import deque
import threading
import logging
from typing import Tuple
import numpy as np


logger = logging.getLogger(__name__)


class SharedReplayBuffer:
    """A shared replay buffer to share experience across multiple processes.
    
    The main process shares the ``queue`` with worker processes and assigns each one an event from 
    ``_prod_events``. Processes signal finish through their event. During sample insertion, the 
    consumer thread extracts experience samples from the queue and writes them to local deques. A 
    thread suffices since the main thread sleeps while waiting for all worker process events. Once
    each process has signaled its readyness, the main thread in the main process lets the consumer 
    thread finish processing the queue and closes it.
    """
    
    def __init__(self, n_samples: int, n_processes: int):
        """Initializes the mp.Manager and generates the queue, events and buffers.
        
        Args:
            n_samples (int): Maximum number of samples that are kept in the buffer.
            n_processes (int): Number of worker threads that fill the ``queue``.
        """
        self._mp_mgr = mp.Manager()
        # Manager avoids deadlocking on process join due to open pipes to self.queue
        self.queue = self._mp_mgr.Queue()
        self._prod_events = [mp.Event() for _ in range(n_processes)]
        # Store data in deques instead of dataframes because frequent appends to df are slow. 5 
        # deques for state, action, reward, next_state, done. Deques of length n_samples drops old 
        # experience in favor of new samples. Order between deques is guaranteed by having only one
        # consumer that writes to the deques
        self.buffer = [deque(maxlen=n_samples) for _ in range(5)]  
        self._cons_thread = None
        self._cons_stop = threading.Event()  # Signal to stop the consumer thread

    def _consume(self):
        while True:
            while not self.queue.empty():
                # Consume samples from the queue and store them to the local deques
                sample = self.queue.get()
                for i in range(5):
                    self.buffer[i].append(sample[i])
            # New sample could have arrived between loop and stop event
            if self._cons_stop.is_set() and self.queue.empty():
                logger.debug("Consumer thread loop shutdown")
                break

    def start(self):
        """Starts the consumer thread for the shared mp.Queue.
        """
        self._cons_thread = threading.Thread(target=self._consume)
        self._cons_stop.clear()
        self._cons_thread.start()
        logger.debug("Consumer thread started")

    def join(self):
        """Waits for all registered processes to signal task completion and subsequently shuts down 
        the consumer thread.
        """
        for event in self._prod_events:
            event.wait()
            event.clear() # Reset flags for the next run
        logger.debug("All worker events set, shutting down consumer thread")
        self.stop()

    def stop(self):
        """Stops the consumer thread.
        """
        self._cons_stop.set()
        logger.debug("Set consumer stop event")
        self._cons_thread.join()
        logger.debug("Consumer joined main thread")

    def sample(self, n: int) -> Tuple[np.ndarray, ...]:
        """Sample a batch of replay experience from the deques.
        
        Args:
            n: Number of samples in one batch.
            
        Returns:
            A 5 elements tuple of numpy arrays containing states, actions, rewards, next states and
            dones as float32.
        
        Raises:
            RuntimeError: If the number of available samples is smaller than the desired batch size.
        """
        if n > len(self.buffer[0]):
            raise RuntimeError("Tried to create batch with more samples than available!")
        samples = np.random.choice(len(self.buffer[0]), n, replace=False)
        return (np.array([self.buffer[i][j] for j in samples], dtype=np.float32) for i in range(5))
