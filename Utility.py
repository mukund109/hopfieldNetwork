from matplotlib import pyplot as plt
import os
import numpy as np
import PIL

def display_weights(pop, colourMap='nipy_spectral', save=False, 
                    directory=os.getcwd(), title='default'):
    fig, ax = plt.subplots()
    im = ax.imshow(pop.get_weights(), cmap=colourMap, 
                   interpolation='none', vmin=-1, vmax=1)
    fig.colorbar(im)
    ax.set_xlabel('x index')
    ax.set_ylabel('y index')
    ax.set_title(title)
    if save:
        plt.savefig(directory)
    plt.show()
    
def display_states(pop, pattern=None):
    import math as m
    side = m.sqrt(pop.size)
    iside = int(side)
    if side != m.floor(side) :
        raise NotImplementedError("Supports only proper squares")
    if pattern is None:
        array=pop.units
    else:
        array= pattern
    plt.imshow(array.reshape((iside,iside)), cmap='gray', 
               interpolation='none', vmin=-1, vmax=1) 
    plt.show()
    

def load_image(infilename) :
    img = PIL.Image.open(infilename)
    return np.array(img)

def random_vector(size, mean_positives):
    prob = mean_positives/size
    p = np.random.binomial(1, prob, size)
    p[p==0]=-1
    return p
      
def random_orthogonal(mean_positives, ortho_prob, size):
    op = ortho_prob
    prob = mean_positives/size
    pat_1 = np.random.binomial(1, prob, size) 
    pat_1[pat_1==0]=-1
    def rand(x):
        i_op = 1-op if (x==1) else op
        np.random.binomial(1, i_op)
        return 1 if x==-1 else -1
    o_func = np.vectorize(rand)
    pat_2 = np.fromfunction(o_func, pat_1.shape)
    return pat_1, pat_2
