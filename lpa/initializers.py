import numpy as np
import scipy.linalg

def generate_W_init(m,p):
    """
    Generates an initial random guess for W based on the QR distribution formed by performing QR 
    factorization on a random normal variable
    
    args:
    -----
    m : (int) number of features
    p : (int) dimension of latent space

    returns:
    --------
    W : (numpy array (m,p)) random guess 
    """
    W = np.random.randn(m,p)
    [W,_] = scipy.linalg.qr(W)
    W = W[:,0:p]
    
    return W

