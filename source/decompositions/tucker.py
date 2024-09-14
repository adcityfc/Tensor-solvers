import numpy as np 
from source.tensor_operations import unfold, n_mode_prod

def tuckerhosvd(X, ranks: list):
    assert len(ranks) == np.ndim(X), "Number of ranks should match the dimension of the tensor X"
    N = np.ndim(X)
    alist = []
    for i in range(N):
        u, s, v = np.linalg.svd(unfold(X, i+1))
        alist.append(u[:, :ranks[i]])
    G = X
    for i, A in enumerate(alist):
        G = n_mode_prod(G, A.T, i+1)
    
    return G, alist


def tuckerhooi(X, ranks: list, maxiters=100):
    assert len(ranks) == np.ndim(X), "Number of ranks should match the dimension of the tensor X"
    N = np.ndim(X)
    alist = []; modearray = np.arange(N)
    for i in range(N):
        u, s, v = np.linalg.svd(unfold(X, i+1))
        alist.append(u[:, :ranks[i]])
    k=0
    while k <= maxiters:
        for j in range(N):
            tmplist = alist.copy()
            tmplist.pop(j)
            Y = X
            for i, A in zip(list(modearray[:j])+list(modearray[j+1:]), tmplist):
                Y = n_mode_prod(Y, A.T, i+1)
            alist[j] = np.linalg.svd(unfold(Y, j+1))[0][:, :ranks[j]]
    
        k += 1    
    
    G = X
    for i, A in enumerate(alist):
        G = n_mode_prod(G, A.T, i+1)

    return G, alist

def tuckerrec(core, factors):
    for i, mat in enumerate(factors):
        core = n_mode_prod(core, mat, i+1)
    return core
