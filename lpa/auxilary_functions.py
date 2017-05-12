import numpy as np
import numpy.matlib

# TODO: right_proj and left_proj when computing E could be sped up by using broadcasting instead of tile

def s_to_ast(B,r,a,b):
    """ 
    Reshapes a time indexed 3-tensor into an astrisk matricized 2-tensor
    """
    return np.reshape(B, [r*a,b])
    
def ast_to_s(B,r,a,b):
    """ 
    Reshapes an astrisk matricized 2-tensor into a time indexed 3-tensor 
    """
    return np.reshape(B, [r,a,b])

def s_to_star(B,r,a,b):
    """ 
    Reshapes a time indexed 3-tensor into a star matricized 2-tensor
    """
    return np.reshape(np.transpose(B, [0,2,1]), [r*b,a]).T

def star_to_s(B,r,a,b):
    """
    Reshapes a star matriczed 2-tensor into a time indexed 3-tensor
    """
    return np.reshape(B.T, [r,a,b]).transpose([0,2,1])

def star_to_ast(B,r,a,b):
    """
    Reshapes a star matricized 2-tensor into an astrisk matricized 2-tensor
    """
    return s_to_ast(star_to_s(B,r,a,b),r,a,b)

def ast_to_star(B,r,a,b):
    """
    Reshapes an ast matricized 2-tensor into a star matricized 2-tensor
    """
    return s_to_star(ast_to_s(B,r,a,b),r,a,b)

def outer_sum(B1,B2,r,a,b,c):
    """
    Efficiently computes the time summed matrix product of two 3-tensors using star and astrix
    matricization.

    Mathematically, computes:
    sum_s B1[s,:,:].dot(B2[s,:,:])
    """
    return ast_to_star(B1,r,a,b).dot(star_to_ast(B2,r,b,c))

def I_m_Pi(A,B,m):
    """
    Efficient computation of A(I-Pi)B.T
    """
    
    # Get eye term
    eye_term = A.dot(B.T)
    
    # Get proj term
    proj_i = np.sum(A,axis=1)
    proj_j = np.sum(B,axis=1)
    
    return eye_term - proj_i[:,np.newaxis]*proj_j[np.newaxis,:]/m

def Pi(A,B,m):
    """
    Efficient computation of A(Pi)B.T
    """
    
    # Get proj term
    proj_i = np.sum(A,axis=1)
    proj_j = np.sum(B,axis=1)
    
    return proj_i[:,np.newaxis]*proj_j[np.newaxis,:]/m
"""
def right_proj(A,m):
    return np.matlib.repmat(np.sum(A, axis=1, keepdims=True)/m,1,m)

def right_I_m_proj(A,m):
    return A - right_proj(A,m)

def left_proj(A,m):
    return np.matlib.repmat(np.sum(A, axis=0, keepdims=True)/m,m,1)

def left_I_m_proj(A,m):
    return A - left_proj(A,m)

"""
def right_proj(A,m):
    return np.tile(np.sum(A, axis=1, keepdims=True)/m,[1,m])

def right_I_m_proj(A,m):
    return A - right_proj(A,m)

def left_proj(A,m):
    return np.tile(np.sum(A, axis=0, keepdims=True)/m,[m,1])

def left_I_m_proj(A,m):
    return A - left_proj(A,m)

def der(func, W, h=1e-8):
    """
    Computes the numerical derivative of a function with respect to a matrix W.
    The input matrix specifying the value at which to compute the gradient wil not be changed.

    Inputs:
    func  : (python function handle) of the function to be differentiated numerically
    W     : (numpy matrix) of the variable argument of the function to be differentiated
    h     : (float64) containing the finite difference value to be used for finite differences
    """
    der_mat = np.zeros(W.shape) # Storage for matrix gradient
   
    # Iterate through all elements of the matrix
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            # Create finite differences additive vector (to avoid changing W)
            add = np.zeros(W.shape)
            add[i,j] = h            
            der_mat[i,j] = (func(W+add)-func(W))/h  # Finite differences
            
    return der_mat
