import numpy as np

# CP to TT
def cp_to_tt(lb, factors):
    factors[0] = np.multiply(factors[0], lb)
    ndim = len(factors)
    for i in range(ndim):
        if i == 0:
            factors[i] = factors[i][np.newaxis, :, :]
        elif i == ndim-1:
            factors[i] = np.moveaxis(factors[i][:, :, np.newaxis], 0, 1)
        else:
            sh = factors[i].shape
            tmp = np.zeros((sh[1], sh[0], sh[1]))
            for j in range(sh[0]):
                tmp[:, j, :] = np.diag(factors[i][j, :])
            factors[i] = tmp
    
    return factors