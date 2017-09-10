import numpy as np
np.set_printoptions(threshold=np.nan)
import Utility
from multiprocessing.dummy import Pool
from Population import hopfieldPopulation

verbose=False
v_print = print if verbose else lambda *a, **k: None



class network:
    """In the network constructor, the size refers to the total number of 
    neurons, 'ptype' refers to the population type. The following types are 
    currently supported:
        
    1. 'discrete'
    """
    def __init__(self, size=1024, ptype='discrete'):
        self.size = size
        self.ptype= 'discrete'
        self.learnt_patterns= []
        
        if ptype=='discrete':
            self.units = hopfieldPopulation(size)
            self._uweights = np.zeros((size,size), dtype=np.float64)
        
        v_print("Network initialized")
    
    def _learn_(self, learning_rule, *args):
        if len(args)==0:
            raise TypeError("_learn_() is missing *args")

        if learning_rule=='hebb':
            for pat in args:
                self._uweights= np.add(self._uweights, np.outer(pat, pat))

        
        elif learning_rule=='sumthing':
            for pat in args:                
                temp = hopfieldPopulation(self.size)
                temp.apply_pattern(self.units)
                self.units.apply_pattern(pat)
                self.update_all()
                to_learn = np.subtract(pat, self.units) #need to fix the difference problem
                to_learn = to_learn/2
                self._learn_('hebb', to_learn)
                self.units = temp
                
        elif learning_rule=='ortho_hebb':
            for pat in args:                
                ap = np.matmul(self.get_weights()/self.size, pat)
                to_learn = np.subtract(pat, ap) #need to fix the difference problem
                self._learn_('hebb', to_learn)

        
        else:
            raise ValueError("learning_rule can only be 'hebb'/'ortho_hebb'")
                
        np.fill_diagonal(self._uweights, 0)   
        v_print("{0} patterns learnt using {1} rule".format(len(args),
                learning_rule))
        
        
        

    def get_weights(self):
        
        """Returns a 2D numpy ndarray, where the element(i,j) corresponds to 
        the weight of the synapse connecting the i-th and j-th neuron
        """
        norm = len(self.learnt_patterns)
        
        if norm==0:
            return self._uweights
        else:
            return self._uweights/norm
    
    
    

    def learn_patterns(self, pattern='random', 
                       prob=0.5, learning_rule='hebb', nb=10 , to_learn=[]):
        
        """Can be used to learn a pattern  using the specified learning rule
        by modifying the weights of network, if no args are given then the 
        network learns 'nb' random patterns. The following learning rules are 
        currently supported:
            1. 'hebb'
            2. 'ortho_hebb'
        """
        
        patterns= None
        
        if bool(to_learn):
            patterns = to_learn
            v_print("Learning patterns")
            if len(self.learnt_patterns)==0 and learning_rule=='ortho_hebb':
                self._learn_('hebb', patterns[0])
            self._learn_(learning_rule, *patterns)
            self.learnt_patterns = self.learnt_patterns + patterns
            
        elif pattern is 'random':
            v_print("Generating random pattern")
            def ran():
                p = np.random.binomial(1, prob, self.size)
                p[p==0]=-1
                return p
            patterns = [ran() for i in range(nb)]
            v_print("Learning generated patterns")
            
            if len(self.learnt_patterns)==0 and learning_rule=='ortho_hebb':
                self._learn_('hebb', patterns[0])
            self._learn_(learning_rule, *patterns)
            self.learnt_patterns = self.learnt_patterns + patterns
         
        else:
            raise ValueError("Provide patterns to learn or set pattern='random'")
            
#    """returns the action potential of the specified neuron"""
#    def action_potential_obsolete(self, neuron_index):
#        w = self.get_weights()
#        h = np.dot(w[neuron_index,:], self.units)
#        return h
    
    

    def update_all(self, update_type='simultaneous'):
        """updates all neurons simultaneously, i.e 
        neuron_states(t+1) = SIGN(neuron_states(t) * Weight_matrix)
        """  
        v_print("Updating all neuron states simultaneously")
        ap = np.matmul(self.get_weights(), self.units)
        self.units.update(ap, update_type)
        v_print("Update complete")
          
    def update_all_parallel_todo(self, update_type='simultaneous'):
        if update_type=='simultaneous':
            v_print("Updating all neuron states simultaneously")
            new_units = list(self.units)
            def update_unit(info):
                index, unit = info
                unit.update(self.action_potential(index))
                return unit
            pool = Pool(4)
            new_units = pool.map(update_unit, list(enumerate(new_units)))
            self.units= new_units
            
    
    def correlation(self, with_learnt_pattern=True, pattern_index=0,
                        other_pattern=None, print_corr=True):
        if with_learnt_pattern:
            corr= np.dot(self.units, 
                         self.learnt_patterns[pattern_index])/self.size
        else:
            corr = np.dot(self.units, other_pattern)/self.size
        if print_corr:
            v_print(corr)
        return corr
    
    def correlations(self):
        corr = []
        for i in range(len(self.learnt_patterns)):
            corr.append(self.correlation(pattern_index=i, print_corr=False))
        return corr
        
    def overlap(self, p0_index, p1_index):
        return np.dot(self.learnt_patterns[p0_index],
                      self.learnt_patterns[p1_index])/self.size
        
    def energy(self, print_energy=True):
        e = np.matmul(self.units.transpose(), self.get_weights())
        e = np.matmul(e, self.units)
        v_print("Energy: {0:0.2f}".format(-e))
        return -e
                         
    def apply_pattern(self, pattern_index=0):
        self.units.apply_pattern(self.learnt_patterns[pattern_index])

           
    #UTILITY
    def show_weights(self, colourMap='nipy_spectral', 
                     save=False, directory=None, title='default'):
        Utility.display_weights(self, colourMap, save, directory, title)
    
    def show_current_state(self):
        Utility.display_states(self)

    def show_learnt_states(self, index=None):
        if index==None:
            for pattern in self.learnt_patterns:
                Utility.display_states(self, pattern)
        else:
            Utility.display_states(self, self.learnt_patterns[index])
            
    

if __name__=='__main__':
    po = network(size=1000)
    po.learn_patterns(learning_rule='ortho_hebb', nb=10, prob=0.5)
    
    jo = network(size=1000)
    jo.learn_patterns(learning_rule='hebb', nb=10)    
    
