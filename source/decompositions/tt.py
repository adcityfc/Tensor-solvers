import numpy as np 
from source.tensor_operations import rq, eps_to_rank

def ttsvd(t, rank, eps=1e-14):
    norm = np.linalg.norm(t)
    tshape = t.shape
    dim = len(tshape)
    C = t
    
    factors = []
    assert len(rank) == dim + 1, "Rank is not compatible with tensor shape"
    assert rank[0] == 1 and rank[-1] == 1, "First and last entry of the rank should be equal to 1"

    for k in range(dim - 1):
        C = np.reshape(C, (int(rank[k] * tshape[k]), -1))
        nr, nc = C.shape
        U, S, V = np.linalg.svd(C)
        rkp1 = min(nr, nc, rank[k+1], eps_to_rank(S, eps*norm/(dim-1)**0.5))
        U = U[:, :rkp1]; S = S[:rkp1]; V = V[:rkp1, :]
        rank[k+1] = rkp1
        factors.append(np.reshape(U, (rank[k], tshape[k], rank[k+1])))
        C = np.reshape(S, (-1, 1)) * V

    # Final core
    r, d = C.shape
    factors.append(np.reshape(C, (r, d, 1)))
    
    return factors

def ttrec(factors):
    shape = [f.shape[1] for f in factors]
    res = factors[0].reshape((shape[0], -1))
    for i, f in enumerate(factors[1:]):
        rkm1, _, rkp1 = f.shape
        f = f.reshape((rkm1, -1))
        res = np.dot(res, f)
        res = res.reshape((-1, rkp1))
    return res.reshape(shape)

def tt_rounding(factors, rank, eps=1e-14):
    from source.decompositions.tt_operations import tt_dot
    norm = tt_dot(factors, factors)
    d = len(factors)
    for k in range(d-1, 0, -1):
        s = factors[k].shape
        r, g = rq((factors[k].reshape(s[0], -1)))
        factors[k] = g.reshape(g.shape[0], s[1], -1)
        factors[k-1] = np.einsum('ijk,kl->ijl', factors[k-1], r)
    for k in range(d-1):
        sh = factors[k].shape
        u, s, vt = np.linalg.svd(factors[k].reshape((-1, sh[-1])), full_matrices=False)
        r = min(sh[0]*sh[1], sh[2], rank[k+1], eps_to_rank(s, eps*norm/(d-1)**0.5))
        u = u[:, :r]; s = s[:r]; vt = vt[:r, :]
        factors[k] = u.reshape(sh[0], sh[1], r)
        factors[k+1] = np.einsum('jkl,ij->ikl', factors[k+1], np.diag(s).T @ vt)
    
    return factors