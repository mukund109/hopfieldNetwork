import numpy as np
import logging

logging.basicConfig(level=logging.WARNING)

class hopfieldPopulation(np.ndarray):
    def __init__(self, size):
        np.ndarray(size, dtype= np.int8)
        self.fill(-1)
    
    def update(self, action_potential, update_type):
        if update_type == 'simultaneous':
            logging.info("Action potential: {}".format(action_potential))
            a = action_potential
            a[a<=0]=-1
            a[a>0] = 1
            np.copyto(self, a)
    
    def apply_pattern(self, pattern):
        np.copyto(self, pattern)
