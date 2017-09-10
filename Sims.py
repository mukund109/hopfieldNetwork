from Network import network
import numpy as np
try:
    import progressbar as pbar
except ImportError:
    pbar=False
pbar=False
verbose=True
v_print = print if verbose else lambda *a, **k: None    

def avg_retrieval(net):
    v_print("Starting retrieval process")
    nb_pat = len(net.learnt_patterns)
    data = np.ndarray((nb_pat,))
    if pbar:
        thisbar = pbar.ProgressBar(max_value=nb_pat)
    for i in range(nb_pat):
        net.apply_pattern(i)
        net.update_all()
        data[i]= net.correlation(pattern_index= i)
        if pbar:
            thisbar.update()
    return np.mean(data), np.var(data), data

if __name__=='__main__':
    pass
    