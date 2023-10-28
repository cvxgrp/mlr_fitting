import numpy as np
import numba as nb
from typing import List, Union

from mlrfit.utils import *
from mlrfit.low_rank import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from profilehooks import profile as tprofile
from memory_profiler import profile as mprofile


# ---------------------- Block coordinate descent (BCD) ----------------------

def single_level_factor_fit(R:np.ndarray, ranks:np.ndarray, hpart:HpartDict, level:int, \
                            svds_v0=False, symm=False, PSD=False, B_level=None, C_level=None):
    """
    Return compressed B_l, C_l from block diagonal approximation of R
    Input:
        R is already permuted residual according to hpartition
    B_level: list of size num_blocks at level
        list of left low rank features B
    C_level: list of size num_blocks at leve
        list of transposed right low rank features Ct
    delta_rm1[level]: scores for decreasing a rank by 1
    delta_rp1[level]: scores for increasing a rank by 1
    """
    if PSD:
        return single_level_factor_fit_psd(R, ranks, hpart, level, svds_v0=svds_v0, \
                                            B_level0=B_level)
    else:
        return single_level_factor_fit_general(R, ranks, hpart, level, svds_v0=svds_v0, symm=symm, \
                                                B_level0=B_level, C_level0=C_level)


@nb.jit(nopython=True)
def update_single_block_delta_psd_jit(B_level, b_level, delta_rp1, delta_rm1, V, lambdas, dim, r1, r2):
    max_rank_block = r2-r1
    if max_rank_block-1 >= dim >= 1:
        if lambdas.size == dim+1:
            # reward for increasing rank
            delta_rp1 += lambdas[-1]**2 # lambda_{r+1}^2
            # penalty for decreasing rank
            delta_rm1 += lambdas[-2]**2 # lambda_r^2
            sqrt_lambdas = np.sqrt(lambdas)
            B_level[r1:r2] = (V * sqrt_lambdas)[:, :-1]
            b_level[r1:r2, 0] = V[:, -1] * sqrt_lambdas[-1]
        else:
            # increasing rank won't improve approximation
            if lambdas.size == dim:
                # penalty for decreasing rank
                delta_rm1 += lambdas[-1]**2 # lambda_{r}^2
            sqrt_lambdas = np.sqrt(lambdas)
            B_level[r1:r2, :V.shape[1]] = V * sqrt_lambdas
    elif dim >= max_rank_block and lambdas.size >= 1:
        # increasing rank won't improve approximation
        if dim == max_rank_block and lambdas.size == dim:
            # penalty for decreasing rank
            delta_rm1 += lambdas[-1]**2 # lambda_r^2
        sqrt_lambdas = np.sqrt(lambdas)
        B_level[r1:r2, :V.shape[1]] = V * sqrt_lambdas
    else:
        if dim == 0 and lambdas.size == 1:
            # reward for increasing rank
            delta_rp1 += lambdas[-1]**2 # lambda_{r+1}^2
            b_level[r1:r2, 0] = V[:,-1]*np.sqrt(lambdas[-1])
    return (delta_rm1, delta_rp1)


def single_level_factor_fit_psd(R:np.ndarray, ranks:np.ndarray, hpart:HpartDict, level:int, \
                                svds_v0=False, B_level0=None):
    """
    Symmetric PSD hat_A
    Return list of BCt from block diagonal approximation of R
    where blocks are symmetric positive definite
    Input:
        R is already permuted residual according to hpartition
    B_level: np.ndarray(m, r_l)
            compressed matrix B_l
    delta_rm1[level]: scores for decreasing a rank by 1
    delta_rp1[level]: scores for increasing a rank by 1
    b_level: np.ndarray(m, 1) 
            b vectors for rank increase
    """
    dim = ranks[level]
    num_blocks = len(hpart['rows']['lk'][level]) - 1
    m = hpart['rows']['pi'].size
    B_level = np.zeros((m, dim))
    # error inrease/decrease when rank decreases/increases by 1
    delta_rm1, delta_rp1 = 0, 0
    if dim == 0: # cannot decrease rank further from 0
        delta_rm1 = np.inf
    v0 = None
    b_level = np.zeros((m,1))
    for block in range(num_blocks):
        r1, r2 = hpart['rows']['lk'][level][block], hpart['rows']['lk'][level][block+1]
        if svds_v0: # v0 has components for every direction of Krylov space
            v0 = B_level0[r1:r2].sum(axis=1)
            if np.linalg.norm(v0) < 1e-6: v0 = None
        V, lambdas = frob_low_rank_psd(R[r1:r2, r1:r2], dim = dim+1, v0=v0)
        # print(f"{level=}, {block=}, {dim=}, {lambdas.size=}, {delta_rm1=}, {delta_rp1=}")
        if lambdas.size == 0: continue
        delta_rm1, delta_rp1 = update_single_block_delta_psd_jit(B_level, b_level, delta_rp1, delta_rm1, \
                                np.ascontiguousarray(V), np.ascontiguousarray(lambdas), dim, r1, r2)
        del V, lambdas
    return B_level, B_level, delta_rm1, delta_rp1, b_level, b_level


@nb.jit(nopython=True)
def update_single_block_delta_general_jit(B_level, C_level, b_level, c_level, delta_rp1, delta_rm1, \
                                      U, V, sigmas, dim, r1, r2, c1, c2):
    max_rank_block = min(r2-r1, c2-c1)
    if max_rank_block-1 >= dim >= 1:
        if sigmas.size == dim+1:
            # reward for increasing rank
            delta_rp1 += sigmas[-1]**2 # sigma_{r+1}^2
            # penalty for decreasing rank
            delta_rm1 += sigmas[-2]**2 # sigma_r^2

            sqrt_sigmas = np.sqrt(np.maximum(sigmas, 0))
            B_level[r1:r2] = (U * sqrt_sigmas)[:, :-1] 
            C_level[c1:c2] = (V * sqrt_sigmas)[:, :-1] 
            b_level[r1:r2, 0] = U[:, -1] * sqrt_sigmas[-1]
            c_level[c1:c2, 0] = V[:, -1] * sqrt_sigmas[-1]
        else:
            # increasing rank won't improve approximation
            if sigmas.size == dim:
                # penalty for decreasing rank
                delta_rm1 += sigmas[-1]**2 # sigma_r^2
            sqrt_sigmas = np.sqrt(np.maximum(sigmas, 0))
            B_level[r1:r2, :U.shape[1]] = U * sqrt_sigmas
            C_level[c1:c2, :V.shape[1]] = V * sqrt_sigmas 
    elif dim >= max_rank_block:
        # increasing rank won't improve approximation
        if dim == max_rank_block == sigmas.size:
            # penalty for decreasing rank
            delta_rm1 += sigmas[-1]**2 # sigma_r^2
        sqrt_sigmas = np.sqrt(np.maximum(sigmas, 0))
        B_level[r1:r2, :U.shape[1]] = U * sqrt_sigmas
        C_level[c1:c2, :V.shape[1]] = V * sqrt_sigmas 
    else:
        if dim == 0 and sigmas.size == 1:
            # reward for increasing rank
            delta_rp1 += sigmas[-1]**2 # sigma_{r+1}^2
            b_level[r1:r2, 0] = U[:, -1] * np.sqrt(np.maximum(sigmas[-1], 0))
            c_level[c1:c2, 0] = V[:, -1] * np.sqrt(np.maximum(sigmas[-1], 0))
    return (delta_rm1, delta_rp1)


def single_level_factor_fit_general(R:np.ndarray, ranks:np.ndarray, hpart:HpartDict, level:int,\
                                    svds_v0=False, symm=False, B_level0=None, C_level0=None):
    """
    Return list of BCt from block diagonal approximation of R
    Input:
        R is already permuted residual according to hpartition
    B_level: np.ndarray(m, r_l)
            compressed matrix B_l
    C_level: np.ndarray(n, r_l)
            compressed matrix C_l
    delta_rm1[level]: scores for decreasing a rank by 1
    delta_rp1[level]: scores for increasing a rank by 1
    b_level: np.ndarray(m, 1) 
            b vectors for rank increase
    c_level: np.ndarray(n,1) 
            c vectors for rank increase
    """
    dim = ranks[level]
    num_blocks = len(hpart['rows']['lk'][level]) - 1
    m, n = hpart['rows']['pi'].size, hpart['cols']['pi'].size
    B_level = np.zeros((m, dim))
    C_level = np.zeros((n, dim))
    # error inrease/decrease when rank decreases/increases by 1
    delta_rm1, delta_rp1 = 0, 0
    if dim == 0: # cannot decrease rank further from 0
        delta_rm1 = np.inf
    v0 = None
    b_level, c_level = np.zeros((m,1)), np.zeros((n,1))
    for block in range(num_blocks):
        r1, r2 = hpart['rows']['lk'][level][block], hpart['rows']['lk'][level][block+1]
        c1, c2 = hpart['cols']['lk'][level][block], hpart['cols']['lk'][level][block+1]
        if svds_v0: # v0 contains components for every direction of Krylov space
            if r2-r1 < c2-c1:
                v0 = B_level0[r1:r2].sum(axis=1)
            else:
                v0 = C_level0[c1:c2].sum(axis=1)
            if np.linalg.norm(v0) < 1e-5: v0 = None
        U, Vt, sigmas = frob_low_rank(R[r1:r2, c1:c2], dim = dim+1, symm=symm, v0=v0)
        
        delta_rm1, delta_rp1 = update_single_block_delta_general_jit(B_level, C_level, b_level, \
                                      c_level, delta_rp1, delta_rm1, \
                                      np.ascontiguousarray(U), np.ascontiguousarray(Vt.T), \
                                      np.ascontiguousarray(sigmas), dim, r1, r2, c1, c2)
        del U, Vt, sigmas
    
    return B_level, C_level, delta_rm1, delta_rp1, b_level, c_level


# ---------------------- Alternating least squares (ALS) using Pytorch ----------------------

def torch_cat_blkdiag_tilde_M(M:torch.Tensor, hp_entry:EntryHpartDict, ranks:np.ndarray) -> torch.Tensor:
    num_levels = ranks.size
    tilde_M = []
    for level in range(num_levels):
        blocks = []
        num_blocks = len(hp_entry['lk'][level])-1
        M_level = M[:,ranks[:level].sum():ranks[:level+1].sum()]
        for block in range(num_blocks):
            r1, r2 = hp_entry['lk'][level][block], hp_entry['lk'][level][block+1]
            blocks += [ M_level[r1:r2] ]
        tilde_M += [torch.block_diag(*blocks)]
    tilde_M = torch.cat(tilde_M, dim=1)
    return tilde_M

def torch_variable(M:Union[np.ndarray,torch.Tensor], variable=False) -> torch.Tensor:
    if torch.is_tensor(M): 
        var_M = M
        var_M.requires_grad = variable
    else:
        var_M = Variable(torch.from_numpy(M).to(get_device(0)), requires_grad=variable)
    return var_M

def torch_to_numpy(M:torch.Tensor) -> np.ndarray:
    return M.data.cpu().numpy()

def torch_compressed_hat_A(var_B:torch.Tensor, var_C:torch.Tensor, hpart:HpartDict, ranks:np.ndarray) -> torch.Tensor:
    """
    Create hat_A matrix from compressed format B, C
    """
    num_levels = ranks.size
    hat_A = torch.zeros((var_B.shape[0], var_C.shape[0]), dtype=var_B.dtype).to(get_device(0))
    for level in range(num_levels):
        A_level = []
        num_blocks = len(hpart['rows']['lk'][level])-1
        B_level = var_B[:,ranks[:level].sum():ranks[:level+1].sum()]
        C_level = var_C[:,ranks[:level].sum():ranks[:level+1].sum()]
        for block in range(num_blocks):
            r1, r2 = hpart['rows']['lk'][level][block], hpart['rows']['lk'][level][block+1]
            c1, c2 = hpart['cols']['lk'][level][block], hpart['cols']['lk'][level][block+1]
            A_level += [ B_level[r1:r2] @ C_level[c1:c2].T ]
        hat_A = hat_A + torch.block_diag(*A_level)
    return hat_A

def torch_zero_grad(var:torch.Tensor):
    var.grad.zero_()

def torch_grad_step(var:torch.Tensor, t:float):
    var.data -= t*var.grad

def torch_frob_loss(A, hat_A):
    return 0.5 * torch.square(A - hat_A).sum()

def torch_cg_step(A:torch.Tensor, hat_A:torch.Tensor, var_B:torch.Tensor, var_C:torch.Tensor, \
                        d_current:torch.Tensor, hpart:HpartDict, ranks:np.ndarray, f_0:float, update:int):
    """
    Find optimal step size t along the direction of the d_k, since loss is
    quadratic in t
    f(t) = loss(var - t*d_k) = a*t**2 - b*t + c
    """
    if update == 0: # update B
        hat_Ap1 = torch_compressed_hat_A(var_B - d_current, var_C, hpart, ranks)
        f_prime_0 = torch.trace((A - hat_A).T @ torch_compressed_hat_A(d_current, var_C, hpart, ranks)).item()
    else: # update C
        hat_Ap1 = torch_compressed_hat_A(var_B, var_C - d_current, hpart, ranks)
        f_prime_0 = torch.trace((A - hat_A).T @ torch_compressed_hat_A(var_B, d_current, hpart, ranks)).item()
    f_p1 = torch_frob_loss(A, hat_Ap1).item()
    c = f_0
    b = - f_prime_0
    a = f_p1 + b - c
    if a == 0:
        t = np.sign(b)
    else:
        t = b / (2*a)
    # find optimal step size
    if update==0:
        var_B.data -= t*d_current
    elif update==1:
        var_C.data -= t*d_current
        

def als_factor_fit_alt_cg(A, B, C, hpart:HpartDict, ranks:np.ndarray, epoch_len=1, update=0, normalization=1):
    """
    Alternating Least Squares (ALS) for finding B and C
    using CG (The Fletcher-Reeves Method)
    Minimize over B is update=0 else over C
    """
    var_B = torch_variable(B, variable=(update==0))
    var_C = torch_variable(C, variable=(update==1))

    hat_A = torch_compressed_hat_A(var_B, var_C, hpart, ranks)
    # (1/2)*\|A - hat_A \|_F^2
    loss = torch_frob_loss(A, hat_A)
    loss.backward()
    if update==0:
        d_prev = -var_B.grad
    else:
        d_prev = -var_C.grad

    norm2_grad_prev = torch.square(d_prev).sum()
    for _ in range(epoch_len):
        hat_A = torch_compressed_hat_A(var_B, var_C, hpart, ranks)
        # (1/2)*\|A - hat_A \|_F^2
        loss = torch_frob_loss(A, hat_A)
        loss.backward()
        if update==0:
            grad = var_B.grad
        else:
            grad = var_C.grad
        norm2_grad = torch.square(grad).sum()
        d_current = -grad + d_prev*(norm2_grad / norm2_grad_prev)
        torch_cg_step(A, hat_A, var_B, var_C, d_current, hpart, ranks, loss.item(), update)
        if update == 0:
            torch_zero_grad(var_B)
        else:
            torch_zero_grad(var_C)
        d_prev = d_current
        norm2_grad_prev = norm2_grad

    torch_hat_A = hat_A
    return var_B, var_C, [ (2*loss.item())**0.5 / normalization], torch_hat_A
