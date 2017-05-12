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
    return nnp.sum(np.square(E_var))/n