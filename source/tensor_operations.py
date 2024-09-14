import numpy as np 

# Matricization/Unfolding
def unfold(t, mode):
    return np.reshape(np.moveaxis(t, mode-1, 0), (t.shape[mode-1], -1))

def fold(mat, mode, shape):
    intshape = shape[mode-1:mode] + shape[:mode-1] + shape[mode:]
    t = np.reshape(mat, tuple(intshape))
    t = np.moveaxis(t, 0, mode-1)
    return t


# n-mode product
def n_mode_prod(t, mat, mode):
    newshape = list(t.shape); newshape[mode-1] = mat.shape[0]
    ttmp = unfold(t, mode)
    ttmp = np.dot(mat, ttmp)
    return fold(ttmp, mode, newshape)


# Matrix products - Kronecker, Khatri-Rao, Hadamard
def kronecker_prod(A, B):
    if np.ndim(A) == 1: A = A[:, np.newaxis]
    if np.ndim(B) == 1: B = B[:, np.newaxis]
    i, j = A.shape; k, l = B.shape 
    res = np.zeros((i*k, j*l))
    for s in range(i):
        for t in range(j):
            res[s*k:s*k+k, t*l:t*l+l] = A[s, t]*B
    return res

def khatri_rao_prod(A, B):
    if np.ndim(A) == 1: A = A[:, np.newaxis]
    if np.ndim(B) == 1: B = B[:, np.newaxis]
    i, j = A.shape; k, l = B.shape
    assert j == l, "Number of columns of A should be equal to number of columns in B"
    res = np.zeros((i*k, j))
    for s in range(j):
        res[:, s] = kronecker_prod(A[:, s], B[:, s])[:, 0]
    
    return res


# Tensor outer product
def outerprod(tensors):
    for i, tensor in enumerate(tensors):
        if i:
            shape = np.shape(tensor)
            s1 = len(shape)
            shape_1 = shape_res + (1,) * s1
            shape_2 = (1,) * sres + shape
            res = np.reshape(res, shape_1) * np.reshape(tensor, shape_2)
        else:
            res = tensor
        shape_res = np.shape(res)
        sres = len(shape_res)
    return res


# Performing operations iteratively on multiple inputs
# Original function should be of the form f(x, y)
def multi(f, listt):
    res = listt[0]
    for i in range(1, len(listt)):
        res = f(res, listt[i])
    return res

# RQ matrix decomposition
def rq(A):
    q1, r1 = np.linalg.qr(A[::-1, :].T)
    return (r1.T)[::-1, ::-1], (q1.T)[::-1, :]

def eps_to_rank(s, eps):
    l = (np.sqrt(np.cumsum(np.square(s[::-1])))[::-1]) <= eps
    res = np.argmax(l)
    if res == 0 and l[0] == False: return s.shape[0]
    else: return res + 1