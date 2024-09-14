import numpy as np
from source.decompositions.tt_operations import tt_dot, tt_matvec, tt_add
from source.decompositions.tt import tt_rounding


def gmres(A, b, x0, maxiter, tol=1e-6):
    n, m = A.shape
    assert b.shape[0] == x0.shape[0] == m, "Incompatible shapes"

    currsoln = x0
    r0 = b - A @ currsoln; beta = np.linalg.norm(r0)
    k = 0
    while beta > tol: 
        v1 = r0/beta
        H = np.zeros((maxiter+1, maxiter)); V = np.zeros((m, maxiter+1))
        V[:, 0] = v1

        for j in range(maxiter):
            tmpnext = A @ V[:, j]
            hij = np.einsum('i,ij->j', tmpnext, V[:, :j+1])
            H[:j+1, j] = hij
            vp1 = tmpnext - np.sum(V[:, :j+1] * hij, axis=1)
            hp1 = np.linalg.norm(vp1); H[j+1, j] = hp1
            vp1 = vp1/hp1; V[:, j+1] = vp1

        # Compute least squares solution
        betavec = np.zeros(maxiter+1); betavec[0] = beta
        y, residuals, rank, s = np.linalg.lstsq(H, betavec, rcond=None)

        currsoln = x0 + (V[:, :-1] @ y)
        
        r0 = b - A @ currsoln
        x0 = currsoln
        beta = np.linalg.norm(r0)
        k += 1
        # print(k, beta, currsoln)
        
    return currsoln, beta


def tt_gmres(A, b, x0, maxiter, outeriter=10, rank=[1, 1000, 1000, 1], eps=1e-14): 
    sh = [s.shape[1] for s in b]
    m = np.prod(np.array(sh))
    bnorm = np.sqrt(tt_dot(b, b))
    check = eps + 1
    k = 0; reshist = []
    while check > eps and k <= outeriter:
        V = []
        # Get initial residual
        tmp = tt_matvec(A, x0)
        tmp[0] = -tmp[0]
        r0 = tt_add(b, tmp)
        r0, beta = tt_rounding(r0, rank=rank, eps=eps)
        
        # Set current residual norm as norm of r0
        resi = beta.copy()
    
        # Initialize v1
        v1 = r0.copy()
        v1[0] = v1[0]/beta
        V.append(v1)
        H = np.zeros((maxiter+1, maxiter))
        for j in range(maxiter):
            # print(j)
            # Error in current iteration
            delta = eps/(resi/beta)
            
            # Next vector in Krylov subspace
            w = tt_matvec(A, V[j])
            w = tt_rounding(w, rank=rank, eps=delta)[0]
            
            # Orthogonalization (Obtain Vj using previous Vs and their projections)
            for i in range(j+1):
                hij = tt_dot(w, V[i])
                H[i, j] = hij
                tmp = V[i].copy()
                tmp[0] = -hij * tmp[0]
                w = tt_add(w, tmp)
            w, hp1 = tt_rounding(w, rank=rank, eps=delta)
            
            H[j+1, j] = hp1
            tmp = w.copy()
            tmp[0] = tmp[0]/hp1
            V.append(tmp)
            
            # Compute least squares solution
            betavec = np.zeros(j+2); betavec[0] = beta
            y, resi, _, _ = np.linalg.lstsq(H[:j+2, :j+1], betavec, rcond=None)
            if resi.size == 0: 
                resi = 0.0
            else: resi = resi[0]**0.5
            check = resi/bnorm
            reshist.append(check)
            if check <= eps: break
        
        # Update the solution
        currsoln = x0
        for i in range(j+1):
            tmp = V[i].copy()
            tmp[0] = y[i] * tmp[0]
            currsoln = tt_add(currsoln, tmp)
            
        currsoln = tt_rounding(currsoln, rank=rank, eps=eps)[0]

        # Set the next intial solution as the current solution
        x0 = currsoln.copy()
        print("Iteration : ", k, "\t\t\tResidual : ", check)
        k += 1
    
    return currsoln, reshist