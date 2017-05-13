import numpy as np
import sys
from optimizers import solve_lpa, solve_dag
from utility import extract_timeseries_XY
from lpa_functions import predict_latent, get_A_b
from auxilary_functions import star_to_s

class VAR(object):
    """
    Standard vector autoregressive model with no dimensionality reduction or constraints
    """
    def __init__(self, lag_order=1, verbose=False):
        """            
        args:
        -----
        lag_order : (int) Number of time lags to be used for each step of prediction
        verbose : (boolean) Whether or not to print extra outputs
        """
        self.r = lag_order
        self.verbose = True
        self.is_fit = False

    def fit(self,X):
        """
        Fits a single time series to the vector autoregressive model
        
        args:
        -----
        X : (numpy array (m,n)) of an m dimensional time series across n evenly spaced time points

        returns:
        --------
        None
        """
        
        if self.verbose:
            sys.stdout.write('>>Extracting data from time series\n')

        X = np.copy(X)
        # Store means across time and subtract
        self.means = np.mean(X, axis=1)
        X -= self.means[:,np.newaxis]  # Subtract mean across time
        Xr, Xp = extract_timeseries_XY(X,self.r,self.forward_backward)

        if self.verbose:
            sys.stdout.write('>>Fitting vector autoregressive processe\n')

        m = X.shape[0]
        W = np.eye(m)  # Can solve for A,b assuming latent process autoregression with W=I

        A, b = get_A_b(W,Xr,Xp)

        self.W = W
        self.A = A
        self.b = b

        self.is_fit = True

    def predict_future_step(self,X):
        """
        For each time point where a prediction can be made, predicts the one time step future value for the
        input time series X. The returned array time dimension will be decreased by the lag_order 
        
        args:
        -----
        X : (numpy array (m,n)) of an m dimensional time series across n evenly spaced time points
        """

        # Subtract training means
        X -= self.means[:,np.newaxis]                


class LPA(object):
    """
    Latent process analysis/autoregression. This is similar to vector autoregression for multivariate
    signals, but performs dimensioanlity reduction on the space of features to generate 
    informative linear combinations of features that are maximally predictive of future values
    of the entire time series. 
    """
    def __init__(self, n_components, lag_order=1, basis='causal', selection_strength=0, n_seed=5, max_iter=1000, tol=1e-3,
                 forward_backward=False,store_unstructured=False,basis_opts={},
                 verbose=False,
                 ):
        """        
        args:
        -----
        n_components : (int, float) Number of components to keep
        lag_order : (int) Number of time lags to be used for each step of prediction
        basis : (string) Basis type for the underlying subspace
        selection_strength : (int, float) Strength of feature selection
        n_seed : (int) Number of restarts to try when optimizing
        max_iter : (int) Maximum number of iterations to use for optimizing each seed
        tol : (float) Minimum change in loss function to be considered converged
        forward_backward : (bool) If dataset should be fit both forward in time and backward
        store_unstructured : (bool) if unstructured basis/solution should be store regardless of basis
        basis_opts : (dictionary) providing options for viewing the basis
        verbose : (bool) Whether or not to print extra outputs 
        """

        self.p = n_components
        self.r = lag_order
        self.basis = basis
        self.mu = selection_strength
        self.lam = 0 # Depricated
        self.n_seed = n_seed
        self.max_iter = max_iter
        self.tol = tol
        self.forward_backward = forward_backward
        self.store_unstructured = store_unstructured
        self.verbose = verbose
        self.is_fit = False

        # Assertion errors
        assert basis in ['causal', 'unstructured'], "Basis choice not supported. Currently only "
        "\'causal\' or \'unstructured\' are implemented."
        
        self.basis_opts = {}

        if basis == 'causal':
            self.basis_opts['n_trials'] = int(basis_opts.get('n_trains', 10))
            self.basis_opts['max_iter'] = int(basis_opts.get('max_iter', 10000))
            self.basis_opts['kappa'] = float(basis_opts.get('kappa', 0.0))

    def fit_predict_latent(self,X):
        """
        Fits a single time series to the latent process autoregressive model, and returns the predicted
        latent space for the data
        args:
        -----
        X : (numpy array (m,n)) of an m dimensional time series across n evenly spaced time points

        returns:
        --------
        X_lat (numpy array (p,m)) of a p dimensional latent time series across n evenly spaced time points
        """
        
        self.fit(X)
        return self.predict_latent(X)
                                                  
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
        
        if self.verbose:
            sys.stdout.write('>>Extracting data from time series\n')

        X = np.copy(X)
        # Store means across time and subtract
        self.means = np.mean(X, axis=1)
        X -= self.means[:,np.newaxis]
        Xr, Xp = extract_timeseries_XY(X,self.r,self.forward_backward)

        if self.verbose:
            sys.stdout.write('>>Identifying latent processes\n')

        self.__fit(Xr,Xp)

        if self.verbose:
            if self.basis == 'unstructured':
                sys.stdout.write('>>Leaving basis unstructured\n')
            elif self.basis == 'causal':
                sys.stdout.write('>>Rotating into causal basis\n')

        self.__rotate_basis(self.basis)

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
        self.W = W
        self.A = A
        self.b = b

        if self.store_unstructured:
            self._W = np.copy(W)
            self._A = np.copy(A)
            self._b = np.copy(b)

        self.err = err

        self.is_fit = True

    def __rotate_basis(self,basis):
        # Rotate into appropriate basis
        if self.basis == 'unstructured':
            pass

        elif self.basis == 'causal':
            kappa = self.basis_opts['kappa']
            n_trials = self.basis_opts['n_trials']
            max_iter = self.basis_opts['max_iter']

            #dag_weights = 1./(self.p-np.arange(self.p))  # Create dag weights
            dag_weights = 1./(np.arange(self.p)+1)
            dag_weights = np.triu(np.tile(dag_weights[:,np.newaxis],[1,self.p]),1)

            # Solve for dag structure using the weights
            err_cs, W_cs, A_cs, b_cs, loss_hist_cs = solve_dag(self.W,self.A, self.b,self.p,
                                                               dag_weights,kappa=kappa,n_trials=n_trials,
                                                               max_iter=max_iter,verbose=self.verbose)

            self.W = W_cs
            self.A = A_cs
            self.b = b_cs

    def get_components(self):
        assert self.is_fit, 'Need to fit model before accessing components'
        return self.W

    def get_model_params(self):
        assert self.is_fit, 'Need to fit model before accessing model parameters'
        return self.W, star_to_s(self.A,self.r,self.p,self.p), self.b

    def predict_latent(self,X):
        assert self.is_fit, "Model must be fit before predicting latent trajectories"
        assert X.shape[0] == self.W.shape[0], "X must have the same number of features as the training data"
        X = np.copy(X)
        X -= self.means[:,np.newaxis]  # Subtract mean across time
        return predict_latent(X,self.W)
