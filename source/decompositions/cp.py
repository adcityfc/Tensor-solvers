import numpy as np 
from source.tensor_operations import unfold, outerprod, multi, khatri_rao_prod

def initializecp(t, rank, init='random'):
    initvals = []
    dim = np.ndim(t)
    for i in range(dim):
        m = unfold(t, i+1)
        if init == 'random': u = np.random.randn(m.shape[0], rank)
        elif init == 'svd': u, s, v = np.linalg.svd(m); u = u[:, :rank]
        initvals.append(u)
    
    return initvals

def cpreconstruction(lb, cpmat):
    shape = tuple([mat.shape[0] for mat in cpmat])
    res = np.zeros(shape)
    for i in range(len(lb)):
        res += outerprod([lb[i]]+[mat[:, i] for mat in cpmat])
    return res 

def cpdecomposition(t, rank, init='random', maxiters=50, tol = 1e-8):
    cpmat = initializecp(t, rank, init)
    matlist = cpmat.copy()
    dim = np.ndim(t)
    i=0; check = tol+1
    while i <= maxiters and check >= tol:
        for n in range(dim):
            Xn = unfold(t, n+1)
            matlist.pop(n)
            vlist = [mat.T @ mat for mat in matlist]
            v = multi(lambda x, y: x*y, vlist)
            # print(Xn.shape, multi(khatri_rao_prod, matlist).shape, np.linalg.pinv(v).shape)
            An = Xn@multi(khatri_rao_prod, matlist[::-1])@np.linalg.pinv(v)
            lb = np.sqrt(np.sum(np.square(An), axis=0))
            An = np.divide(An, lb)
            cpmat[n] = An
            matlist = cpmat.copy()
        check = np.linalg.norm(t - cpreconstruction(lb, cpmat))/np.linalg.norm(t)
        i += 1
    return lb, cpmat