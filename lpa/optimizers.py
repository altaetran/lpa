import numpy as np

def Y_tau(W,tau,A):
    g = W.shape[0]
    
    inv = np.linalg.inv(np.eye(g)+tau/2*A)
    
    minus = np.eye(g)-tau/2*A
    
    Q = inv.dot(minus)
    return Q.dot(W)

def Y_tau_woodbury(W,tau,U,V): 
    p = W.shape[1]
    # Get right hand side 
    right = W-U.dot(V.dot(W))*tau/2

    # Now compute action of inverse using woodbury
    wood_right = V.dot(right)
    
    if tau > 1e-16:
        inner_mat = np.eye(2*p)*2/(tau+1e-16)+V.dot(U)
        # Solve the inverse multiplicaiton problem using linalg solver
        inv_mul = np.linalg.solve(inner_mat,wood_right)
    else:
        inv_mul = wood_right
    
    # Get the remainder of the term
    inv_term = U.dot(inv_mul)
    
    # Now add to get the final Y_tau
    Y_tau = right - inv_term
    
    return Y_tau

def armijo_wolfe_backtracking_line_search_woodbury(W,Mr,Mp,lam,mu,G,bool_woodbury=True):
    tau_init = 1
    rho_1 = 1e-3
    rho_2 = 0.9
    alpha = 0.7
    
    tau = tau_init
    
    bool_not_converged = True
    
    # Generate low rank matrices
    U = np.hstack([G,-W])
    V = np.vstack([W.T,G.T])
    
    def T_loss_path(tau):
        h = 1e-12
        if bool_woodbury:
            Yt = Y_W_tau_woodbury(W,tau,U,V)
            f = T_fun(Yt,Mr,Mp,lam,mu)
            d = (T_fun(Y_W_tau_woodbury(W,tau+h,U,V),Mr,Mp,lam,mu)-f)/h
        else:
            A = U.dot(V)
            Yt = Y_W_tau(W,tau,A)
            f = T_fun(Yt,Mr,Mp,lam,mu)
            d = (T_fun(Y_W_tau(W,tau+h,A),Mr,Mp,lam,mu)-f)/h            
        return f, d, Yt
    
    f, d, Yt = T_loss_path(0)
    
    i = 0
    max_iter = 100
    
    while bool_not_converged:
        # Get function value, derivative, and Yt at specific tau
        f_tau, d_tau, Yt = T_loss_path(tau)

        # Check wolfe conditions
        cond_a = f_tau < f + rho_1*tau*d
        cond_b = d_tau > rho_2*d
        
        if (cond_a and cond_b) or i >= max_iter:
            bool_not_converged = False
            
        tau *= alpha
        i += 1
    
    return f_tau, Yt, d_tau

def solve_one_numpy(W,Mr,Mp,lam,mu,max_iter=1000,bool_woodbury=True,bfgs=True,stochastic=True,analytic=True,verbose=True,eps=1e-3):
    bool_bfgs_inv = True
    if verbose:
        print('Numpy Version')
    batch_size = 20
    n = Mp.shape[1]
    m = Mp.shape[0]
    r = Mr.shape[0]
    p = W.shape[1]
    
    loss_hist = []
    
    f = T_fun(W,Mr,Mp,lam,mu)
    loss_hist.append(f)

    min_eps = eps

    if bfgs:
        if bfgs_inv:
            Binv = np.eye(m*p)
        else:
            B = np.eye(m*p)
    
    bool_not_converged = True
    i = 0

    if analytic:
        f, G = analytic_loss_grad(W,Mr,Mp,lam,mu)
    else:
        f, G = numerical_loss_grad(W,Mr,Mp,lam,mu)
        
    G_old = np.reshape(G, [m*p,1])
    while bool_not_converged:    
        if bfgs:
            #G = B.dot(G_old)
            if bfgs_inv:
                G = Binv.dot(G_old)
            else:
                G = np.linalg.solve(B, G_old)
            G = np.reshape(G, [m,p])
        
        if stochastic:
            c = np.random.choice(n,batch_size)
            f, Yt, d_tau = armijo_wolfe_backtracking_line_search_woodbury(W,Mr[:,:,c],Mp[:,c],lam,mu,G,bool_woodbury)
        else:
            f, Yt, d_tau = armijo_wolfe_backtracking_line_search_woodbury(W,Mr,Mp,lam,mu,G,bool_woodbury)

        loss_hist.append(f)
        W_old = W
        W = Yt

        # Get nabla f
        #DelF = G-W.dot(G.T.dot(W))
        
        loss_diff = loss_hist[-2] - loss_hist[-1]
        if (loss_diff < min_eps and loss_diff >= -1e-8) or i >= max_iter:
            bool_not_converged = False

        if analytic:
            f, G = analytic_loss_grad(W,Mr,Mp,lam,mu)
        else:
            f, G = numerical_loss_grad(W,Mr,Mp,lam,mu)
            
        if bfgs:
            G_new = np.reshape(G, [m*p,1])
            y = G_new - G_old
            s = np.reshape(W - W_old, [m*p,1])

            if bfgs_inv:
                Binvy = Binv.dot(y)
                sty = s.T.dot(y)+1e-6
                T1 = (sty+y.T.dot(Binvy))/np.square(sty)*(s.dot(s.T))
                Binvyst = Binvy.dot(s.T)
                T2 = (Binvyst + Binvyst.T)/sty
                Binv += T1 - T2
            else:
                Bs = B.dot(s)
                B += y.dot(y.T)/y.T.dot(s)-Bs.dot(Bs.T)/(s.T.dot(Bs))
            G_old = G_new
            
        if (i % 10 == 0 or not bool_not_converged) and verbose:
            print(i,loss_hist[-1])
        i += 1

    A, b = get_A_b(W,Mr,Mp,lam)

    err = T_err_fun(W,A,Mr,Mp)
    
    return err, W, A, b, loss_hist

def solve_lpa(Mr,Mp,lam,mu,p,n_trials=10,max_iter=100,bool_coordinate_wise=False,
              bool_woodbury=True,bool_numpy=False,bfgs=True,stochastic=True,analytic=True,
              verbose=True,eps=1e-3):
    g = Mp.shape[0]
    
    best_params = None
    best_loss = 1e64
    for i in range(n_trials):
        if verbose:
            print('>> Beginning seed: '+str(i))
        W = generate_W_init(g,p)
        if not bool_coordinate_wise:
            err, W, A, b, loss_hist = solve_one_numpy(W,Mr,Mp,lam,mu,max_iter,bool_woodbury,bfgs,stochastic,analytic,verbose=verbose,eps=eps)
        else:
            r = Mr.shape[0]
            A = np.random.randn(p,r*p)
            err, W, A, b, loss_hist = solve_one_A_numpy(W,A,Mr,Mp,lam,mu,max_iter,bool_woodbury,stochastic)
        
        if loss_hist[-1] < best_loss:
            best_loss = loss_hist[-1]
            best_params = (err, W, A, b, loss_hist)
    
    return best_params

def dag_linesearch(O,A_s,dag_weights,kappa,G,bool_woodbury=True):
    tau_init = 1
    rho_1 = 1e-3
    rho_2 = 0.9
    
    alpha = 0.7
    
    tau = tau_init
    
    bool_not_converged = True
    
    # Generate low rank matrices
    A = G.dot(O.T) - O.dot(G.T)
    def T_loss_path(tau):
        h = 1e-12
        Yt = Y_W_tau(O,tau,A)
        f = T_dag(Yt,A_s,dag_weights,kappa)
        d = (T_dag(Y_W_tau(O,tau+h,A),A_s,dag_weights,kappa)-f)/h
        return f, d, Yt
    
    f, d, Yt = T_loss_path(0)

    i = 0
    max_iter = 100
    while bool_not_converged:
        # Get function value, derivative, and Yt at specific tau
        f_tau, d_tau, Yt = T_loss_path(tau)

        # Check wolfe conditions
        cond_a = f_tau < f + rho_1*tau*d
        cond_b = d_tau > rho_2*d
        
        if (cond_a and cond_b) or i >= max_iter:
            bool_not_converged = False
            
        tau *= alpha
        i += 1

    return f_tau, Yt, d_tau

def solve_one_dag_numpy(O,W,A_s,b,dag_weights,kappa,max_iter=1000,stochastic=True,analytic=True,verbose=True):   
    loss_hist = []
    
    f = T_dag(O,A_s,dag_weights,kappa)
    loss_hist.append(f)

    min_eps = 1e-7
    
    bool_not_converged = True
    i = 0

    if analytic:
        f, G = analytic_dag_loss_grad(O,A_s,dag_weights,kappa)
    else:
        f, G = numerical_dag_loss_grad(O,A_s,dag_weights,kappa)
        
    while bool_not_converged:    
        f, Yt, d_tau = dag_linesearch(O,A_s,dag_weights,kappa,G,bool_woodbury=False)
        loss_hist.append(f/1e4)
        O_old = O
        O = Yt
        
        loss_diff = loss_hist[-2] - loss_hist[-1]
        if (loss_diff < min_eps and loss_diff >= -1e-8) or i >= max_iter:
            bool_not_converged = False

        if analytic:
            f, G = analytic_dag_loss_grad(O,A_s,dag_weights,kappa)
        else:
            f, G = numerical_dag_loss_grad(O,A_s,dag_weights,kappa)
            
        if (i % 100 == 0 or not bool_not_converged) and verbose:
            print(i,loss_hist[-1])
        i += 1

    W = W.dot(O.T)
    A_rot_s = np.matmul(np.matmul(O[np.newaxis,:,:],A_s), O.T[np.newaxis,:,:])
    A = np.reshape(np.transpose(A_rot_s, [0,2,1]), [r*p,p]).T
    b = O.dot(b)
    
    err = T_dag_err(O,A_s,dag_weights)
    
    return err, W, A, b, loss_hist

def solve_dag(W,A,b,p,dag_weights,kappa,n_trials=10,max_iter=10000,analytic=True,verbose=True):
    p = A.shape[0]
    r = A.shape[1]/p
    A_s = np.reshape(A.T, [r,p,p]).transpose([0,2,1])
    
    best_params = None
    best_loss = 9999999999999
    for i in range(n_trials):
        if verbose:
            print('>> Beginning seed: '+str(i))
        O = generate_W_init(p,p)
        err, W, A, b, loss_hist = solve_one_dag_numpy(O,W,A_s,b,dag_weights,kappa,max_iter=max_iter,analytic=analytic,verbose=verbose)

        if loss_hist[-1] < best_loss:
            best_loss = loss_hist[-1]
            best_params = (err, W, A, b, loss_hist)
    
    return best_params
