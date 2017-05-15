import numpy as np
import scipy.linalg
import lpa.lpa_functions
import lpa.utility

def predict_true_traj(model_params,times,X,T,dt=1):
    W, A_s, b = model_params
    A = A_s[0]
    r = A_s.shape[0]
    
    init_times, X_init = lpa.utility.extract_initial(times,X,r)
    T_start = times[-1]

    return lpa.lpa_functions._predict_traj(W,A,b,r,X_init,T,T_start)

def get_dataset(dag=True,np_seed=2,m=100,m_pos=20,p_true=10,t_train=900,t_val=50,t_test=50,r_true=1):
    np.random.seed(np_seed)
    n_to_avg = 100
    dt = 1
    t_total = t_train+t_val+t_test

    times = np.arange(0,t_total,dt)

    # Generate random dynamics
    diff = 0.5*np.random.randn(p_true,p_true)
    A_sys = diff+np.eye(p_true,p_true)

    # Create dag structure
    if dag:
        A_sys -= np.triu(A_sys,1)

    # Normalize by largest eigenvalue so it's not exploding 
    A_sys /= np.max(np.abs(scipy.linalg.eigvals(A_sys)))*1.05
    c_sys = 0.1*np.random.randn(p_true)

    m_n = m - m_pos

    [w,v] = np.linalg.eig(A_sys)

    # Generate random loading matrix
    [Z_part,_,_] = scipy.linalg.svd(np.random.randn(m_pos,p_true),full_matrices=False)

    if m_n > 0:
        [Z_part_n,_,_] = scipy.linalg.svd(np.random.randn(m_n,p_true),full_matrices=False)
        mix_v = 0.001
        Z = np.vstack([Z_part*(1-mix_v), Z_part_n*mix_v])
    else:
        Z = Z_part

    # Generate random trajectory
    s = np.random.randn(p_true)
    X = np.random.randn(m,n_to_avg)
    s_t = []
    X_t = []
    for i,t in enumerate(times):
        # Get observed signal
        X += np.random.randn(m,n_to_avg)
        X = (Z.dot(A_sys.dot(Z.T.dot(X))+c_sys[:,np.newaxis]))
        X_t.append(X)

    # Get the means of the X's
    M_t = [np.mean(X,axis=1) for X in X_t]
    M_all = np.vstack(M_t).T

    # Get the training validation and test trajectories
    T_train = times[:t_train]
    X_train = M_all[:,:t_train]
    T_val = times[t_train:t_train+t_val]
    X_val = M_all[:,t_train:t_train+t_val]
    T_test = times[t_train+t_val:t_total]
    X_test = M_all[:,t_train+t_val:t_total]

    A = A_sys[np.newaxis,:,:]
    model_params = (Z,A,c_sys)

    return T_train,X_train,T_val,X_val,T_test,X_test,model_params
    
