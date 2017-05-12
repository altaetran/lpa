import numpy as np
import sys
from optimizers import solve_lpa, solve_dag
from utility import extract_timeseries_XY

class LPA(object):
    def __init__(self, n_components, basis='causal', selection_strength=0, n_seed=5, max_iter=1000, tol=1e-3,
                 forward_backward=False,
                 verbose=False,
                 ):
        """        
        args:
        -----
        n_components : (int, float) Number of components to keep
        basis :
        selection_strength : (int, float) Strength of feature selection
        n_seed : (int) Number of restarts to try when optimizing
        max_iter : (int) Maximum number of iterations to use for optimizing each seed
        tol : (float) Minimum change in loss function to be considered converged
        forward_backward : (bool) If dataset should be fit both forward in time and backward
        verbose : (bool) Whether or not to print extra outputs 
        """

        self.p = n_components
        self.basis = basis
        self.mu = selection_strength
        self.lam = 0 # Depricated
        self.n_seed = n_seed
        self.max_iter = max_iter
        self.tol = tol
        self.forward_backward = forward_backward
        self.verbose = verbose
        self.is_fit = False

        # Assertion errors
        assert basis in ['causal', 'unstructured'], "Basis choice not supported. Currently only "
        "\'causal\' or \'unstructured\' are implemented."
        
        if basis == 'causal':
            self.basis_kwargs = {'n_trials': 10,
                                 'max_iter' : 10000,
                                 'kappa' : 0.0}
    
    def fit(self,X):
        """
        Fits a single time series to the latent process autoregressive model
        
        args:
        -----
        X : (numpy array (m,n)) of an m dimensional time series across n evenly spaced time points

        returns:
        --------
        None
        """
        
        Xr, Xp = extract_timeseries_XY(X,self.r,self.forward_backward)
        self.__fit(Xr,Xp)               
        self.__rotate_basis(self,self.basis)

    def __fit(self,X,Y):
        """
        Fits the regressors X with linear weights that predict Y using an ordered tensor ordinary least
        squares solver
        """
        m = X.shape[0]
        
        err, W, A, b, loss_hist = solve_lpa(X,Y,self.lam,self.mu,self.p,n_trials=self.n_seed,
                                            max_iter=self.max_iter, eps=self.tol,
                                            verbose=self.verbose)

        # Save model parameters
        self._W = W
        self._A = A
        self._b = b
        self.err = err

        self.is_fit = True

    def __rotate_basis(self,basis):
        # Rotate into appropriate basis
        if self.basis == 'unstructured':
            if self.verbose:
                sys.stdout.write('>>Leaving basis unstructured')
            self.W = self._W
            self.A = self._A
            self.b = self._b

        elif self.basis == 'causal':
            if self.verbose:
                sys.stdout.write('>>Rotating into causal basis')

            kappa = self.basis_kwargs['kappa']
            n_trials = self.basis_kwargs['n_trials']
            max_iter = self.basis_kwargs['max_iter']

            dag_weights = 1./(self.p-np.arange(self.p))  # Create dag weights
            err_cs, W_cs, A_cs, b_cs, loss_hist_cs = solve_dag(self._W,self._A, self._b,self.p,kappa,
                                                               n_trials=n_trials,max_iter=max_iter,verbose=False)
            
    def get_components(self):
        assert self.is_fit, 'Need to fit model before accessing components'
        return self.W

    def get_model_params(self):
        assert self.is_fit, 'Need to fit model before accessing model parameters'
        return self.A, self.b
