import numpy as np
import sys
from optimizers import solve_lpa, solve_dag
from utility import extract_timeseries_XY
from lpa_functions import predict_latent, get_A_b, _predict_future_step, predict_traj
from auxilary_functions import star_to_s

class VAR(object):
    """
    Standard vector autoregressive model with no dimensionality reduction or constraints
    """
    def __init__(self, lag_order=1, verbose=False, forward_backward=False):
        """            
        args:
        -----
        lag_order : (int) Number of time lags to be used for each step of prediction
        verbose : (boolean) Whether or not to print extra outputs
        forward_backward : (bool) If dataset should be fit both forward in time and backward
        """
        self._r = lag_order
        self._verbose = True
        self._forward_backward = forward_backward
        self._is_fit = False
        self._max_rp = 10000

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
        
        if self._verbose:
            sys.stdout.write('>>Extracting data from time series\n')
                    
        m = X.shape[0]
        
        assert m*self._r < 10000, "The largest matrix inversion supported for the standard solver is 10000 x 10000."
        "Please reduce the number of lags, or the dimension of the input features if possible."
        "Contact for additional solutions, such as distributed options, or conjugate solvers if desired."
        
        X = np.copy(X)
        # Store means across time and subtract
        self._means = np.mean(X, axis=1)
        X -= self._means[:,np.newaxis]  # Subtract mean across time
        Xr, Xp = extract_timeseries_XY(X,self._r,self._forward_backward)

        if self._verbose:
            sys.stdout.write('>>Fitting vector autoregressive processe\n')

        W = np.eye(m)  # Can solve for A,b assuming latent process autoregression with W=I

        A, b = get_A_b(W,Xr,Xp)

        self._W = W
        self._A = A
        self._b = b

        self._is_fit = True

    def predict_future_step(self,X):
        """
        For each time point where a prediction can be made, predicts the one time step future value for the
        input time series X. The returned array time dimension will be decreased by the lag_order 
        
        args:
        -----
        X : (numpy array (m,n)) of an m dimensional time series across n evenly spaced time points

        returns:
        --------
        X_pred : (numpy array (m,n-r)) of the predicted values after each time point stating from t=r
        Xp     : (numpy array (m,n-r)) of the true values that are being predicted in X_pred
        """

        X = np.copy(X)

        X -= self._means[:,np.newaxis]
        Xr, Xp = extract_timeseries_XY(X,self._r,False)
    
        X_pred = _predict_future_step(Xr,self._W,self._A,self._b)

        X_pred += self._means[:,np.newaxis]
        Xp += self._means[:,np.newaxis]
        
        return X_pred, Xp

    def one_step_rmse(self,X):
        """
        Determines the prediction error of the single time step predictions made by the model on the 
        time series X. Uses predictions on the times from time step r onward. Returns the root mean
        squared error.

        args:
        -----
        X : (numpy array (m,n)) of an m dimensional time series across n evenly spaced time points

        returns:
        --------
        rmse : (float) root mean squared error of predictions
        """
        
        X_pred, Xp = self.predict_future_step(X)
        rmse =  np.sqrt(np.sum(np.square(X_pred-Xp))/X_pred.shape[1])
        return rmse

class LPA(object):
    """
    Latent process analysis/autoregression. This is similar to vector autoregression for multivariate
    signals, but performs dimensioanlity reduction on the space of features to generate 
    informative linear combinations of features that are maximally predictive of future values
    of the entire time series. 
    """
    def __init__(self, n_components, lag_order=1, basis='causal', selection_strength=0, n_seed=5, max_iter=1000, tol=1e-3,
                 forward_backward=False,store_unstructured=False,basis_opts={},approximate=False,
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
        approximate : (bool) Whether or not to use approximate gradients with risk of not converging
        basis_opts : (dictionary) providing options for viewing the basis
        verbose : (bool) Whether or not to print extra outputs 
        """

        self._p = n_components
        self._r = lag_order
        self._basis = basis
        self._mu = selection_strength
        self._lam = 0 # Depricated
        self._n_seed = n_seed
        self._max_iter = max_iter
        self._tol = tol
        self._forward_backward = forward_backward
        self._store_unstructured = store_unstructured
        self._approximate = approximate 
        self._verbose = verbose
        self._is_fit = False

        # Assertion errors
        assert basis in ['causal', 'unstructured'], "Basis choice not supported. Currently only "
        "\'causal\' or \'unstructured\' are implemented."
        
        self._basis_opts = {}

        if basis == 'causal':
            self._basis_opts['n_trials'] = int(basis_opts.get('n_trains', 10))
            self._basis_opts['max_iter'] = int(basis_opts.get('max_iter', 10000))
            self._basis_opts['kappa'] = float(basis_opts.get('kappa', 0.0))

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
        
        if self._verbose:
            sys.stdout.write('>>Extracting data from time series\n')

        assert self._p*self._r < 10000, "The largest matrix inversion supported for the standard solver is 10000 x 10000."
        "Please reduce the number of lags, or the number of components if possible"
        "Contact for additional solutions, such as distributed options, or conjugate solvers if desired."

        X = np.copy(X)
        # Store means across time and subtract
        self._means = np.mean(X, axis=1)
        X -= self._means[:,np.newaxis]
        Xr, Xp = extract_timeseries_XY(X,self._r,self._forward_backward)

        if self._verbose:
            sys.stdout.write('>>Identifying latent processes\n')

        self.__fit(Xr,Xp)

        if self._verbose:
            if self._basis == 'unstructured':
                sys.stdout.write('>>Leaving basis unstructured\n')
            elif self._basis == 'causal':
                sys.stdout.write('>>Rotating into causal basis\n')

        self.__rotate_basis(self._basis)

    def __fit(self,X,Y):
        """
        Fits the regressors X with linear weights that predict Y using an ordered tensor ordinary least
        squares solver
        """
        m = X.shape[0]
        
        err, W, A, b, loss_hist = solve_lpa(X,Y,self._lam,self._mu,self._p,n_trials=self._n_seed,
                                            max_iter=self._max_iter, eps=self._tol,
                                            verbose=self._verbose,approximate=self._approximate)

        # Save model parameters
        self._W_basis = W
        self._A_basis = A
        self._b_basis = b

        if self._store_unstructured:
            self._W = np.copy(W)
            self._A = np.copy(A)
            self._b = np.copy(b)

        self._err = err

        self._is_fit = True

    def __rotate_basis(self,basis):
        # Rotate into appropriate basis
        if self._basis == 'unstructured':
            pass

        elif self._basis == 'causal':
            kappa = self._basis_opts['kappa']
            n_trials = self._basis_opts['n_trials']
            max_iter = self._basis_opts['max_iter']

            #dag_weights = 1./(self._p-np.arange(self._p))  # Create dag weights
            dag_weights = 1./(np.arange(self._p)+1)
            dag_weights = np.triu(np.tile(dag_weights[:,np.newaxis],[1,self._p]),1)

            # Solve for dag structure using the weights
            err_cs, W_cs, A_cs, b_cs, loss_hist_cs = solve_dag(self._W_basis,self._A_basis, self._b_basis,
                                                               self._p,
                                                               dag_weights,kappa=kappa,n_trials=n_trials,
                                                               max_iter=max_iter,verbose=self._verbose)

            self._W_basis = W_cs
            self._A_basis = A_cs
            self._b_basis = b_cs

    def get_components(self):
        assert self._is_fit, 'Need to fit model before accessing components'
        return self._W_basis

    def get_model_params(self):
        assert self._is_fit, 'Need to fit model before accessing model parameters'
        return self._W_basis, star_to_s(self._A_basis,self._r,self._p,self._p), self._b_basis

    def predict_latent(self,X):
        assert self._is_fit, "Model must be fit before predicting latent trajectories"
        assert X.shape[0] == self._W_basis.shape[0], "X must have the same number of features as the training data"
        X = np.copy(X)
        X -= self._means[:,np.newaxis]  # Subtract mean across time
        return predict_latent(X,self._W_basis)

    def predict_future_step(self,X):
        """
        For each time point where a prediction can be made, predicts the one time step future value for the
        input time series X. The returned array time dimension will be decreased by the lag_order 
        
        args:
        -----
        X : (numpy array (m,n)) of an m dimensional time series across n evenly spaced time points

        returns:
        --------
        X_pred : (numpy array (m,n-r)) of the predicted values after each time point stating from t=r
        Xp     : (numpy array (m,n-r)) of the true values that are being predicted in X_pred
        """

        X = np.copy(X)

        X -= self._means[:,np.newaxis]
        Xr, Xp = extract_timeseries_XY(X,self._r,False)
        
        X_pred = _predict_future_step(Xr,self._W_basis,self._A_basis,self._b_basis)
        
        X_pred += self._means[:,np.newaxis]
        Xp += self._means[:,np.newaxis]
        
        return X_pred, Xp

    def one_step_rmse(self,X):
        """
        Determines the prediction error of the single time step predictions made by the model on the 
        time series X. Uses predictions on the times from time step r onward. Returns the root mean
        squared error.

        args:
        -----
        X : (numpy array (m,n)) of an m dimensional time series across n evenly spaced time points

        returns:
        --------
        rmse : (float) root mean squared error of predictions
        """
        
        X_pred, Xp = self.predict_future_step(X)
        rmse =  np.sqrt(np.sum(np.square(X_pred-Xp))/X_pred.shape[1])
        return rmse
