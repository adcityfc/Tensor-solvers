import numpy as np
from source.tensor_operations import kronecker_prod
from source.decompositions.tt import ttrec


def tt_zeros(size, rank):
    d = len(size)
    assert d + 1 == len(rank), "Rank and size are incompatible"
    factors = [None]*d
    for i in range(d):
        factors[i] = np.zeros(rank[i:i+1]+size[i:i+1]+rank[i+1:i+2])
    return factors

def rand_tt(size, rank):
    d = len(size)
    assert d + 1 == len(rank), "Rank and size are incompatible"
    factors = [None]*d
    for i in range(d):
        factors[i] = np.random.randn(rank[i], size[i], rank[i+1])
    return factors

def tt_add(factorsx, factorsy):
    dim = len(factorsx)
    factorsz = []
    for i, (matx, maty) in enumerate(zip(factorsx, factorsy)):
        sx = matx.shape; sy = maty.shape
        if i == 0:
            tmp = np.concatenate([matx, maty], axis=-1)
        elif i == dim-1:
            tmp = np.concatenate([matx, maty], axis=0)
        else:
            tmp1 = np.concatenate([matx, np.zeros(sx[:2]+sy[-1:])], axis=-1)
            tmp2 = np.concatenate([np.zeros(sy[:2]+sx[-1:]), maty], axis=-1)
            tmp = np.concatenate([tmp1, tmp2], axis=0)

        factorsz.append(tmp)
    return factorsz

def tt_multiply(factorsx, factorsy):
    factorsz = []
    for matx, maty in zip(factorsx, factorsy):
        sx = matx.shape; sy = maty.shape
        tmp = np.zeros((sx[0]*sy[0], sx[1], sx[2]*sy[2]))
        for j in range(sx[1]):
            tmp[:, j, :] = kronecker_prod(matx[:, j, :], maty[:, j, :])
        factorsz.append(tmp)

    return factorsz

def tt_dot(A, B):
    dim = len(A)
    C = A.copy()
    for i in range(dim-1):
        sb = B[i].shape
        scp1 = C[i+1].shape
        tmp = C[i][0, :, :].T @ B[i].reshape((sb[0]*sb[1], -1))
        C[i+1] = (tmp.T @ C[i+1].reshape((scp1[0], -1))).reshape((1, sb[-1]*scp1[-2], scp1[-1]))
        # C[i+1] = np.einsum('jk,jm,kno->mno', C[i][0, :, :], B[i].reshape((sb[0]*sb[1], -1)), C[i+1]).reshape((1, sb[-1]*scp1[-2], scp1[-1]))
    
    return np.dot(B[i+1][:, :, 0].flatten(), C[i+1][0, :, 0])

def vec_to_tt(x, sh, rank, eps=1e-14):
    from source.decompositions.tt import ttsvd
    s = x.shape; dim = len(sh)
    assert np.prod(np.array(list(sh))) == s[0], "Total number of elements do not match."
    assert len(rank) == dim + 1, "Ranks and dimension are incompatible."
    x = np.reshape(x, list(sh))
    
    factors = ttsvd(x, rank, eps)
    
    return factors

def mat_to_tt(A, rowsh, colsh, rank, eps=1e-14):
    from source.decompositions.tt import ttsvd
    s = A.shape; dim = len(rowsh)
    assert dim == len(colsh), "Number of elements in row shape should be equal to number of elements in column shape."
    assert np.prod(np.array(list(rowsh)+list(colsh))) == s[0]*s[1], "Total number of elements do not match."
    assert len(rank) == dim + 1, "Ranks and dimension are incompatible."
    A = np.reshape(A, list(rowsh) + list(colsh))
    nsh = np.arange(2*dim).reshape((2, -1)).flatten(order='F')
    A = np.transpose(A, tuple(nsh))
    A = np.reshape(A, [i[0]*i[1] for i in zip(rowsh, colsh)])
    
    factors = ttsvd(A, rank, eps)
    for i in range(dim):
        tmpsh = factors[i].shape
        factors[i] = factors[i].reshape((tmpsh[0], rowsh[i], colsh[i], tmpsh[-1]))

    return factors


def tt_matvec(A, x):
    # assert [s.shape[2] for s in A] == [s.shape[1] for s in x], "Incompatible shapes"
    dim = len(A)
    res = 1
    for i in range(dim):
        x[i] = np.einsum('ijkl,mkn->imjln', A[i], x[i])
        s = x[i].shape
        A[i] = None
        x[i] = np.reshape(x[i], (s[0]*s[1], s[2], s[-1]*s[-2]))
    return x

def tt_matmat(A, B):
    assert [s.shape[2] for s in A] == [s.shape[1] for s in B], "Incompatible shapes"
    dim = len(A)
    res = [None]*dim
    for i in range(dim):
        res[i] = np.einsum('ijkl,mkno->imjnlo', A[i], B[i])
        s = res[i].shape
        res[i] = np.reshape(res[i], (s[0]*s[1], s[2], s[3], s[-1]*s[-2]))
    return res

def tt_to_vecmat(factors):
    dim = len(factors); l = len(factors[0].shape)
    if l == 4:
        rshape = [s.shape[1] for s in factors]
        cshape = [s.shape[2] for s in factors]
        res = [None]*dim
        for i in range(dim):
            s = factors[i].shape
            res[i] = np.reshape(factors[i], (s[0], s[1]*s[2], -1))
        mat = ttrec(res).reshape([val for pair in zip(rshape, cshape) for val in pair])
        mat = np.transpose(mat, [2*i for i in range(dim)] + [2*i+1 for i in  range(dim)]).reshape((np.prod(np.array(rshape)), np.prod(np.array(cshape))))
        return mat

    elif l == 3: return ttrec(factors).flatten(order='F')

def get_tt_rank(tt):
    return [1] + [s.shape[-1] for s in tt]
