import numpy as np
from source.tensor_operations import kronecker_prod, multi
from source.decompositions.tt import ttsvd, ttrec

def divergence1d(n, delx, method='fd'):
    """
    n - Number of points
    delx - Distance between adjacent points
    method - Discretization method; 
        fd: forward difference
        bd: backward difference
        cd: central difference
    """
    if method == 'fd': res = np.diag([-1]*n) + np.diag([1]*(n-1), 1)
    elif method == 'bd': res = np.diag([1]*n) + np.diag([-1]*(n-1), -1)
    elif method == 'cd': res = 0.5*(np.diag([1]*(n-1), 1) + np.diag([-1]*(n-1), -1))
    res = res/delx

    # # Boundary conditions
    # res[0, 0] = 1; res[0, 1] = 0
    # res[-1, -2] = 0; res[-1, -1] = 1
    
    return res

# 1-D Laplace operator with dirichlets boundary conditions
def lap1d(n, delx):
    res = (np.diag([-2]*n) + np.diag([1]*(n-1), 1) + np.diag([1]*(n-1), -1))/delx**2
    
    # # Boundary conditions
    # res[0, 0] = 1; res[0, 1] = 0
    # res[-1, -2] = 0; res[-1, -1] = 1
    
    return res

def diff_operator(factors, delx, method, option):
    """
    factors - Tensor in its TT format
    option = 1 for divergence 
    option = 2 for Laplace
    """
    if option == 1: f = lambda x, y: divergence1d(x, y, method=method)
    elif option == 2: f = lap1d
    dim = len(factors)
    tmpfactors = factors.copy()
    res = 0
    for i in range(dim):
        lap = f(factors[i].shape[1], delx)
        tmpfactors[i] = np.einsum('ij,kjl->kil', lap, factors[i])
        res += ttrec(tmpfactors)
        tmpfactors = factors.copy()
    return res

# test
def sinfunc(X):
    X = np.array(X); d = len(X)
    s = np.sum(X, axis=0)
    return np.sin(s), d*np.cos(s)

def sinfunc_on_grid(n, d):
    x = np.linspace(0, 1, n)
    grid = np.meshgrid(*([x]*d), indexing='ij')
    return grid, sinfunc(grid)

def test_diff_opt(n, d, meth = 'cd', option = 2):
    dx = 1/(n-1)
    grid, fgrid = sinfunc_on_grid(n, d)
    func, grad = sinfunc(fgrid)
    functt = ttsvd(func, rank=[1] + [12]*(d-1) + [1])
    funcrec = ttrec(functt)
    print("TT reconstruction error: ", np.linalg.norm(func - funcrec))
    div = diff_operator(functt, dx, meth, option=option)

    mats = [np.eye(n)]*d
    div2opt = 0
    for i in range(d):
        tmp = mats.copy()
        if option == 1: tmp[i] = divergence1d(n, delx=dx, method=meth)
        elif option == 2: tmp[i] = lap1d(n, delx=dx)
        div2opt += multi(kronecker_prod, tmp)
    div2 = div2opt @ func.reshape((-1,))
    div2 = div2.reshape(func.shape)
    err1 = np.linalg.norm(div - div2)/(n-1)**2

    # Discrepancy between 2 approaches
    print("Discrepancy between 2 approaches: ", err1) 

    # error
    err21 = div - grad
    # L2 norm of error
    slices = tuple(slice(1, -1) for _ in range(d))
    err22 = np.linalg.norm(err21[slices])/(n-1)**2
    print("Discrepancy with true value (Discretization error): ", err22)

    return err1, err21, err22