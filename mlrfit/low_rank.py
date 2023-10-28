import numpy as np
from scipy.sparse.linalg import svds, eigsh
import numba as nb

from mlrfit.utils import *


def frob_low_rank(A, dim=None, symm=False, v0=None):
    """
    Return low rank approximation of A \approx B C^T and singular values
    in the decreasing order 
    """
    M =  min(A.shape[0], A.shape[1])
    if dim is None: dim = M
    dim = min(dim, min(A.shape[0], A.shape[1]))
    if dim < M:
        try:
            U, sigmas, Vt = svds(A, k=dim, which='LM', v0=v0)
        except:
            maxiter = min(A.shape) * 100
            try:
                print(f"svds fail: increase {maxiter=}")
                U, sigmas, Vt = svds(A, k=dim, which='LM', v0=v0, maxiter=maxiter)
            except:
                print(f"svds fail: decrease tol")
                U, sigmas, Vt = svds(A, k=dim, which='LM', v0=v0, tol=1e-2)
    else:
        U, sigmas, Vt = np.linalg.svd(A, full_matrices=False, hermitian=symm)
    # decreasing order of sigmas
    idx = np.argsort(sigmas)[::-1]
    sigmas = sigmas[idx]
    U = U[:, idx]
    Vt = Vt[idx, :]
    return U, Vt, sigmas


def frob_low_rank_psd(A, dim=None, v0=None):
    """
    Return low rank approximation of A \approx BSB^T and nonzero 
    positive eigenvalues in the decreasing order ord
    A is symmetric
    """
    if dim is None: dim = A.shape[0]
    dim = min(dim, A.shape[0])
    if dim < min(A.shape):
        lambdas, V = eigsh(A, k=dim, which='LA', v0=v0)
    else:
        lambdas, V = np.linalg.eigh(A)
    idx = np.argsort(lambdas)[::-1] # decreasing  order
    # (lambdas_1)_+ >= ... >= (lambdas_j)_+ >= 0
    lambdas = np.maximum(lambdas[idx], 0)  
    V = V[:, idx]
    zero_idices = np.where(lambdas==0)[0]
    if zero_idices.size >= 1:
        zero_idx = zero_idices.min()
        lambdas = lambdas[:zero_idx]
        V = V[:,:zero_idx]
    return V, lambdas


def svds_using_factoring(B:np.ndarray, C:np.ndarray): 
    """
    SVDs of A which factorization is given, ie, A=BC^T 
    Return
        B_lk, C_lk, b_lk, c_lk, sigma_rl, sigma_rlp1
    """
    r = B.shape[1]
    BtB = B.T @ B
    CtC = C.T @ C
    Vc, sq_sc = frob_low_rank_psd(CtC, dim = r)
    Uc = C @ (Vc * (sq_sc**(-1/2)))
    sc = sq_sc**(1/2)
    ZtZ = (Vc * sc).T @ BtB @ (Vc * sc)
    Vz, lbd_z = frob_low_rank_psd(ZtZ, dim = r)
    sa = np.sqrt(np.maximum(lbd_z, 0))
    zero_idx = np.where(sa == 0)[0]
    if zero_idx.size > 0:
        sa = sa[:zero_idx[0]]
    if sa.size == 0:
        b0 = np.zeros((B.shape[0], 1))
        c0 = np.zeros((C.shape[0], 1))
        return b0, c0, b0, c0, 0, 0
    Va = Uc @ Vz
    Ua = B @ (C.T @ (Va * (sa**(-1))))
    # assert np.allclose(B @ C.T, (Ua * sa) @ Va.T)
    if sa.size == r:
        srtq_s = sa[:-1]**(0.5)
        dB = Ua * srtq_s
        dC = Va * srtq_s
        b = Ua[:, -1:] * np.sqrt(sa[-1])
        c = Va[:, -1:] * np.sqrt(sa[-1])
        return dB, dC, b, c, sa[-2], sa[-1]
    else: # dont get score for increasing rank
        if sa.size == r-1: 
            sigma_rl = sa[-1]
        else:
            sigma_rl = 0
        srtq_s = sa**(0.5)
        dB = Ua * srtq_s
        dC = Va * srtq_s
        b = np.zeros((B.shape[0], 1))
        c = np.zeros((C.shape[0], 1))
        return dB, dC, b, c, sigma_rl, 0

 
def frob_low_rank_extended_to_compressed(eB:np.ndarray, eC:np.ndarray, inc_ranks:np.ndarray, hpart:HpartDict):
    """
    Get compressed version of extended eB, eC by decreasing r_l+1 to r_l
    using frobenius low rank
    """
    B = np.zeros((eB.shape[0], inc_ranks.sum()-inc_ranks.size))
    C = np.zeros((eC.shape[0], inc_ranks.sum()-inc_ranks.size))
    delta_rm1, delta_rp1 = np.zeros(inc_ranks.size), np.zeros(inc_ranks.size)
    b, c = np.zeros((eB.shape[0], inc_ranks.size)), np.zeros((eC.shape[0], inc_ranks.size))
    for level in range(inc_ranks.size):
        num_blocks = len(hpart['rows']['lk'][level])-1
        r_l = inc_ranks[:level].sum()
        r_lp1 = inc_ranks[:level+1].sum()
        if inc_ranks[level]-1 == 0: # cannot decrease rank further from 0
            delta_rm1[level] = np.inf
        for block in range(num_blocks):
            r1, r2 = hpart['rows']['lk'][level][block], hpart['rows']['lk'][level][block+1]
            c1, c2 = hpart['cols']['lk'][level][block], hpart['cols']['lk'][level][block+1]
            B_lk, C_lk, b_lk, c_lk, sigma_rl, sigma_rlp1 = \
                svds_using_factoring(eB[:,r_l:r_lp1][r1:r2], eC[:,r_l:r_lp1][c1:c2])
            B[r1:r2, r_l-level:r_l-level+B_lk.shape[1]] = B_lk
            C[c1:c2, r_l-level:r_l-level+C_lk.shape[1]] = C_lk
            b[r1:r2, level:level+1] = b_lk
            c[c1:c2, level:level+1] = c_lk
            delta_rm1[level] += sigma_rl**2
            delta_rp1[level] += sigma_rlp1**2
    return B, C, delta_rm1, delta_rp1, b, c 
