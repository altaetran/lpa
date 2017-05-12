import numpy as np

def extract_timeseries_XY(X,r,forward_backward):
    """
    Extracts the X and Y data for a time series with a specified number of lags for fitting autoregressive
    models. Can also extract the reverse direction data if needed

    args:
    -----
    X : (numpy array (m,n)) Time series to be extracted
    p : (int) number of lags to be used in the model
    forward_backward : (bool) Whether or not to extract the backward direction as well

    returns:
    Xr : (numpy array (r,m,n-r)) Time series predictors. Last dimension *2 if forward_backward
    Xp : (numpy array (m,n-r) Time series labels. Last dimension *2 if forward_backward
    """
    n = X.shape[1]
    n_train = n - r  # Compensate for lags

    Xrf = np.vstack([X[:,i:-r+i][np.newaxis,:,:n_train] for i in range(r)])
    Xpf = X[:,r:n_train+r]

    if not forward_backward:
        # Just use the forward direction
        Xr = Xrf
        Xp = Xpf

    else:
        # Get backward elements
        Xpr = X[:,0:n_train]
        Xrr = np.vstack([X[:,r-i:-i-1][np.newaxis,:,:n_train] for i in range(r)])
        Xr = np.dstack([Xrf, Xrr])
        Xp = np.hstack([Xpf, Xpr])

    return Xr, Xp
