import numpy as np
from source.decompositions import cp_to_tt
from source.decompositions.tt import tt_rounding


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

def diff_operator_tt(domain, n, d, order, method):
    tot = domain[1] - domain[0]
    dx = tot/(n-1)
    if order == 1: op1d = divergence1d(n-2, dx, method=method)
    elif order == 2: op1d = lap1d(n-2, dx)

    # Create CP factors
    Id = np.eye(n-2).flatten()
    cpmats = [np.zeros(((n-2)**2, d)) for _ in range(d)]
    for i in range(d):
        if i == 0: pass
        else: cpmats[i][:, :i] = np.array(list([Id])*i).T

        cpmats[i][:, i] = op1d.flatten()

        if i == d - 1: pass
        else: cpmats[i][:, i+1:] = np.array(list([Id])*(d-i-1)).T

    # Convert to TT-matrix format
    ttcores = cp_to_tt(lb=np.ones(d), factors=cpmats)
    ttcores = tt_rounding(ttcores, rank=[1]+[100]*(d-2)+[1])[0]
    ttcores = [s.reshape((s.shape[0], n-2, n-2, s.shape[-1])) for s in ttcores]

    return ttcores