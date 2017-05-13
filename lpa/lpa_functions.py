import numpy as np
import scipy, scipy.linalg

from auxilary_functions import *

# TODO: Improve residual calculation using properties of inner products and get rid of explicit 
# computation of E

def analytic_loss_grad(W,Mr,Mp,lam=0,mu=0):
    """
    Computes the loss function and analytic gradient of the objective function for latent process
    analysis. Can take in a regularization penatly mu that allows one to control the extent of entire 
    feature removal in the final latent subspace.
    
    Args:
    -----
    W    : (numpy array (m,p)) containing the orthonormal basis for the subspace
    Mr   : (numpy array (r,m,n)) containing the lagged data points in order Mr[0],..,Mr[r-1]
    Mp   : (numpy array (m,n)) containing the future points to be predicted
    lam  : (depricated)
    mu   : (flaot64) strength of feature selection
    
    Returns:
    --------
    f    : (numpy.float64) loss function
    G    : (numpy.float64 array (m,n)) gradient of loss with respect to W evaluated at W
    """
    n = Mp.shape[1]  # Number of data points
    m = Mp.shape[0]  # Number of features
    r = Mr.shape[0]  # Lag order of autoregression
    p = W.shape[1]  # Dimension of latent space (number of components)
    
    # Astrisk matricize the data and latent representation of the data used for prediction
    X_ast = np.reshape(Mr, [r*m, n])
    Z_ast = np.vstack([W.T.dot(Mr[i]) for i in range(r)])

    # Get the latent representation of the future values to be predicted
    Y = W.T.dot(Mp)
    
    # Generate the autocoviarnace matrix
    Gamma = I_m_Pi(Z_ast,Z_ast,n)

    # Get the inverse of the matrix, with a small addition for numerical stability
    Gamma_inv = scipy.linalg.inv(Gamma+np.eye(r*p)*1e-8)

    # Get the dynamics matricized tensor A_star
    V = I_m_Pi(Y,Z_ast,n)
    A = V.dot(Gamma_inv)

    # Multiply A_star by Z_ast and obtain error residuals 
    A_Z = A.dot(Z_ast)
    E = W.dot(right_I_m_proj(A_Z,n)+right_proj(Y,n)) - Mp
    
    # Compute the error in the latent space
    WT_E = W.T.dot(E) 
    
    # Get the partial deriviatve with respect to W (Term 1)
    T1_2 = Pi(E,Y,n)
    T1_1 = I_m_Pi(E,A_Z,n)
    T1_grad = T1_1 + T1_2
    
    # Get the partial derivative with respect to Z (Term 2)
    XEW = I_m_Pi(X_ast, WT_E, n)
    T2_grad = ast_to_star(XEW,r,m,p).dot(star_to_ast(A,r,p,p))  # Fast mul with star and ast transpose

    # Get the partial derivative with respect to Y (Term 3)
    T3_grad = Pi(Mp,WT_E,n)

    # Get the partial derivative with respect to A (Term 4)
    R = Y-A_Z

    WT_E_ZG = I_m_Pi(WT_E, Z_ast, n).dot(Gamma_inv)
    XR = I_m_Pi(X_ast, R, n)

    T4_1 = outer_sum(XR,WT_E_ZG,r,m,p,p)

    ZEW = I_m_Pi(Z_ast,WT_E,m)
    Gamma_invZEW = Gamma_inv.dot(ZEW)

    MpZ = I_m_Pi(Mp,Z_ast,n)
    XZ = I_m_Pi(X_ast,Z_ast,n)
    XZ_GammainvZEW = XZ.dot(Gamma_invZEW)

    T4_2 = MpZ.dot(Gamma_invZEW)
    T4_3 = outer_sum(XZ_GammainvZEW,A,r,m,p,p)

    T4_grad = T4_1 + T4_2 - T4_3
    
    # Get the partial derivative with respect to W of regularization if needed
    if mu != 0:
        norms = np.sqrt(np.sum(np.square(W), axis=1))    
        reg_grad = mu*W/norms[:,np.newaxis]
        reg_term = mu*np.sum(norms)
    else:
        reg_grad = 0
        reg_term = 0
    
    G = 2*(T1_grad+T2_grad+T3_grad+T4_grad)+reg_grad
    
    f = np.sum(np.square(E))/n+reg_term
    return f, G

def T_fun(W,Mr,Mp,lam,mu=0):
    """
    Get the loss function only without any gradient calculations    

    Args:
    -----
    W    : (numpy array (m,p)) containing the orthonormal basis for the subspace
    Mr   : (numpy array (r,m,n)) containing the lagged data points in order Mr[0],..,Mr[r-1]
    Mp   : (numpy array (m,n)) containing the future points to be predicted
    lam  : (depricated)
    mu   : (flaot64) strength of feature selection
    
    Returns:
    --------
    f    : (numpy.float64) loss function
    """
    n = Mp.shape[1]
    r = Mr.shape[0]
    p = W.shape[1]
    
    Z_ast_var = np.vstack([W.T.dot(Mr[i]) for i in range(r)])
    Y_var = W.T.dot(Mp)
    Gamma_var = I_m_Pi(Z_ast_var,Z_ast_var,n) + np.eye(r*p)*1e-6
    V_var = I_m_Pi(Y_var,Z_ast_var,n)
    A_var = np.linalg.solve(Gamma_var,V_var.T).T
    
    A_Z = A_var.dot(Z_ast_var)
    E_var = W.dot(right_I_m_proj(A_Z,n)+right_proj(Y_var,n)) - Mp
    norms = np.sqrt(np.sum(np.square(W), axis=1))
    return np.sum(np.square(E_var))/n+mu*np.sum(norms)

def T_err_fun(W,A,Mr,Mp):
    """
    Get the average variance of the latent model
    Args:
    -----
    W    : (numpy array (m,p)) containing the orthonormal basis for the subspace
    Mr   : (numpy array (r,m,n)) containing the lagged data points in order Mr[0],..,Mr[r-1]
    Mp   : (numpy array (m,n)) containing the future points to be predicted
    
    Returns:
    --------
    f    : (numpy.float64) loss function
    """
    n = Mp.shape[1]
    r = Mr.shape[0]
    p = W.shape[1]
    
    Z_ast_var = np.vstack([W.T.dot(Mr[i]) for i in range(r)])
    Y_var = W.T.dot(Mp)
    A_Z = A.dot(Z_ast_var)
    E_var = W.dot(right_I_m_proj(A_Z,n)+right_proj(Y_var,n)) - Mp
    return np.sum(np.square(E_var))/n

def get_A_b(W,Mr,Mp,lam=0.0):
    """
    Get the average variance of the latent model

    args:
    -----
    W    : (numpy array (m,p)) containing the orthonormal basis for the subspace
    Mr   : (numpy array (r,m,n)) containing the lagged data points in order Mr[0],..,Mr[r-1]
    Mp   : (numpy array (m,n)) containing the future points to be predicted
    lam  : (float) L2 regularization term for A to dampen signal uniformly
    
    returns:
    --------
    A    : (numpy array (p,r*p)) Autoregression coefficients
    b    : (numpy array (p)) Autoregression drift term
    """
    m = Mp.shape[1]
    n = Mr.shape[2]
    r = Mr.shape[0]
    p = W.shape[1]
    
    # Transform variables
    Z_ast_var = np.vstack([W.T.dot(Mr[i]) for i in range(r)])
    Y_var = W.T.dot(Mp)

    # Get autocovariance
    Gamma_var = I_m_Pi(Z_ast_var,Z_ast_var,n) + np.eye(r*p)*(lam+1e-6)
    V_var = I_m_Pi(Y_var,Z_ast_var,n)

    # Invert to get true values
    A = np.linalg.solve(Gamma_var,V_var.T).T    
    b = np.sum(Y_var-A.dot(Z_ast_var),axis=1)/n
    
    return A, b

def T_dag(O,W,A_s,C,kappa):
    p = O.shape[0]

    # Perform rotation in the space
    OA = np.matmul(O[np.newaxis,:,:],A_s)
    OAOT = np.matmul(OA, O.T[np.newaxis,:,:])
    
    # Norms across time
    K = np.sqrt(np.sum(np.square(OAOT), axis=0))
    err = np.sum(K*C)
    
    # Reverse transform basis
    WOT = W.dot(O.T)
    reg = kappa*np.sum(np.abs(WOT))
    
    loss = 1e4*(err+reg)  #Add 1e4 mult for stability
    
    return loss

def T_dag_err(O,W,A_s,C):
    p = O.shape[0]

    # Perform rotation in the space
    OA = np.matmul(O[np.newaxis,:,:],A_s)
    OAOT = np.matmul(OA, O.T[np.newaxis,:,:])
    
    # Norms across time
    K = np.sqrt(np.sum(np.square(OAOT), axis=0))
    err = np.sum(K*C)
    
    return err

def numerical_dag_loss_grad(O,W,A_s,C,kappa):
    f = T_dag(O,W,A_s,C,kappa)
    
    G = der(lambda O:T_dag(O,W,A_s,C,kappa), O, h=1e-8)
    return f, G

def analytic_dag_loss_grad(O,W,A_s,C,kappa):
    p = O.shape[0]

    # Perform rotation in the space
    OA = np.matmul(O[np.newaxis,:,:],A_s)
    OAOT = np.matmul(OA, O.T[np.newaxis,:,:])
    
    # Norms across time
    K = np.sqrt(np.sum(np.square(OAOT), axis=0))
    err = np.sum(K*C)
    
    # Reverse transform basis
    WOT = W.dot(O.T)
    reg = kappa*np.sum(np.abs(WOT))
    
    loss = 1e4*(err+reg)
    
    # Get gradient of error term with respect to O
    Ktilde = 1./(K+1e-6)
    OAT = np.matmul(O[np.newaxis,:,:],np.transpose(A_s,[0,2,1]))
    CKtOAOT = (C*Ktilde)[np.newaxis,:,:]*OAOT
    CKtOAOT_T = np.transpose(CKtOAOT,[0,2,1])
    
    G_err = np.sum(np.matmul(CKtOAOT,OAT) + np.matmul(CKtOAOT_T,OA), axis=0)
    
    # Get gradient of reg function
    S = ((WOT>=0)+0)-(WOT<0)+0  # Sign matrix
    G_reg = kappa*S.T.dot(W)
    
    G = G_err + G_reg
    G*= 1e4  # Add 1e4 mult for stability
    return loss, G

def predict_latent(X,W):
    return W.T.dot(X)

def predict_future_step(Xr,W,A,b):
    # Get transformed variables
    Z_ast = np.vstack([W.T.dot(Xr[i]) for i in range(r)])
    
    # Apply dynamics and transform back to original space
    X_pred = W.dot(A.dot(Z_ast)+b)
    return X_pred

def get_traj(W,A,b,r,M_init,T,T_start=-1,dt=1):
    """
    Predicts an error free trajectory using the latent process analaysis/autoregression model
    
    args:
    -----
    W    : (numpy array (m,p)) containing the orthonormal basis for the subspace
    Mr   : (numpy array (r,m,n)) containing the lagged data points in order Mr[0],..,Mr[r-1]
    Mp   : (numpy array (m,n)) containing the future points to be predicted
    r    : (int) lag order of the LPA model
    M_init : (numpy array (r,m,1)) Initial r consecutive states for trajectory determination
    T    : (float) Final time to integrate to
    T_start : (float) Starting time (the time of the last element in M_init
    dt   : (float) the step size of the integration
    
    returns:
    --------
    times : (numpy array) of the times of the trajectory
    Xp    : (numpy array) predicted trajectory over time
    X_lat : (numpy array) The latent states of the predicted trajectory
    
    if T_start == -1:
        T_start = r
        
    all_h = []
    all_obs = []
    times = np.arange(T_start,T+dt,dt)

    p = W.shape[1]
    h = np.reshape(np.matmul(W.T[np.newaxis,:,:],M_init), [r*p,1])
    all_h.append(h[-p:,:])
    all_obs.append(M_init[-1,:,:])
    for t in times[1:]:
        h_new = A.dot(h)+b[:,np.newaxis]
        obs = W.dot(h_new)
        h = np.vstack([h[p:,:], h_new[:,:]])
        
        all_h.append(h_new)
        all_obs.append(obs)
        
    return times, np.hstack(all_obs), np.hstack(all_h)


