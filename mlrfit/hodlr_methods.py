import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import copy, time, sys
import random
import pickle
from tqdm import tqdm 

from mlrfit.low_rank import *


def low_rank_approx_tol(A, tol):
    U, Vt, sigmas = frob_low_rank(A)
    # normalization = np.linalg.norm(A, ord='fro')
    # assert np.allclose(normalization, ((sigmas**2).sum())**0.5)
    sigmas2 = (sigmas**2)[::-1]
    frob_losses = np.concatenate([(np.cumsum(sigmas2)**0.5)[::-1], np.array([0])])
    # frob_losses_norm = frob_losses / normalization
    r = np.where(frob_losses <= tol)[0][0] 
    sqrt_sigmas = np.sqrt(np.maximum(0, sigmas[:r]))
    U = U[:, :r]
    V = Vt.T[:, :r]
    B = U * sqrt_sigmas
    C = V * sqrt_sigmas
    return r, frob_losses, B, C


def build_hodlr(hpart, m, n, A, tol):
    perm_A = A[hpart['rows']['pi'], :][:, hpart['cols']['pi']]
    perm_A_HODLR, A_HODLR_fillin, bl_sizes, count =  build_hodlr_contiguous(hpart, m, n, perm_A, tol)
    pi_inv_rows, pi_inv_cols = inv_permutation(hpart['rows']['pi'], hpart['cols']['pi'])
    A_HODLR = perm_A_HODLR[pi_inv_rows, :][:, pi_inv_cols]
    assert np.allclose(rel_diff(perm_A_HODLR, den=perm_A), rel_diff(A_HODLR, den=A))
    return A_HODLR, A_HODLR_fillin, bl_sizes, count


def build_hodlr_contiguous(hpart, m, n, A, tol):
    num_levels = len(hpart['rows']['lk'])
    sizes = []
    A_HODLR = np.zeros((m, n))
    A_HODLR_fillin = np.zeros((m, n))
    count = 0
    for level in tqdm(range(num_levels)):
        if level == num_levels-1:
            # number of blocks on leaf level
            num_blocks = len(hpart['rows']['lk'][level]) - 1
            for block in range(num_blocks):
                r1, r2 = hpart['rows']['lk'][level][block], hpart['rows']['lk'][level][block+1]
                c1, c2 = hpart['cols']['lk'][level][block], hpart['cols']['lk'][level][block+1]
                rank_b = min(r2-r1, c2-c1) # full rank
                sizes += [[rank_b, r2-r1, c2-c1]]
                A_HODLR[r1 : r2, c1 : c2] = A[r1 : r2, c1 : c2]
                count += (r2-r1)*(c2-c1) 
                A_HODLR_fillin[r1 : r2, c1 : c2] += np.ones((r2-r1, c2-c1))
        else:
            # number of blocks on parent level
            par_num_blocks = len(hpart['rows']['lk'][level]) - 1
            for par_block in range(par_num_blocks):
                # print(f"{level=}, {par_block=}")
                parent_r1, parent_r2 = hpart['rows']['lk'][level][par_block], hpart['rows']['lk'][level][par_block+1]
                parent_c1, parent_c2 = hpart['cols']['lk'][level][par_block], hpart['cols']['lk'][level][par_block+1]
                num_blocks = len(hpart['rows']['lk'][level+1]) - 1
                for block in range(num_blocks - 1):
                    r1, r2 = hpart['rows']['lk'][level+1][block], hpart['rows']['lk'][level+1][block+1]
                    c1, c2 = hpart['cols']['lk'][level+1][block], hpart['cols']['lk'][level+1][block+1]
                    # loop on blocks that are refinement of parent block
                    if r1 < parent_r1 or c1 < parent_c1: continue 
                    if r2 >= parent_r2 or c2 >= parent_c2: break

                    if r2 - r1 == 1:
                        A_b = A[r1 : r2, c2 : parent_c2]
                        rank_b, frob_losses, B_b, C_b = low_rank_approx_tol(A_b, tol=tol)
                        if rank_b >= 1:
                            A_HODLR[r1 : r2, c2 : parent_c2] = B_b.dot(C_b.T)
                            if B_b.shape[0] >= 2 and C_b.shape[0] >= 2:
                                count += min(B_b.size + C_b.size, B_b.shape[0] * C_b.shape[0])
                            else:
                                count += max(B_b.size, C_b.size)
                        A_HODLR_fillin[r1 : r2, c2 : parent_c2] += np.ones((r2-r1, parent_c2-c2))
                        sizes += [[rank_b, r2-r1, parent_c2-c2]]
                        assert np.allclose(frob_losses[rank_b], np.linalg.norm(B_b.dot(C_b.T)-A_b, ord='fro'))
                    else:
                        for bl in range(block+1, num_blocks):
                            # print(level, par_block, block, bl)
                            b_c1, b_c2 = hpart['cols']['lk'][level+1][bl], hpart['cols']['lk'][level+1][bl+1]
                            if b_c2 > parent_c2: break
                            A_b = A[r1 : r2, b_c1 : b_c2]
                            rank_b, frob_losses, B_b, C_b = low_rank_approx_tol(A_b, tol=tol)
                            if rank_b >= 1:
                                A_HODLR[r1 : r2, b_c1 : b_c2] = B_b.dot(C_b.T)
                                if B_b.shape[0] >= 2 and C_b.shape[0] >= 2:
                                    count += min(B_b.size + C_b.size, B_b.shape[0] * C_b.shape[0])
                                else:
                                    count += max(B_b.size, C_b.size)
                            A_HODLR_fillin[r1 : r2, b_c1 : b_c2] += np.ones((r2-r1, b_c2-b_c1))
                            sizes += [[rank_b, r2-r1, b_c2-b_c1]]
                            assert np.allclose(frob_losses[rank_b], np.linalg.norm(B_b.dot(C_b.T)-A_b, ord='fro'))
                        
                    if c2 - c1 == 1:
                        A_b = A[r2 : parent_r2, c1 : c2]
                        rank_b, frob_losses, B_b, C_b = low_rank_approx_tol(A_b, tol=tol)
                        if rank_b >= 1:
                            A_HODLR[r2 : parent_r2, c1 : c2] = B_b.dot(C_b.T)
                            if B_b.shape[0] >= 2 and C_b.shape[0] >= 2:
                                count += min(B_b.size + C_b.size, B_b.shape[0] * C_b.shape[0])
                            else:
                                count += max(B_b.size, C_b.size)
                        A_HODLR_fillin[r2 : parent_r2, c1 : c2] += np.ones((parent_r2-r2, c2-c1))
                        sizes += [[rank_b, parent_r2-r2, c2-c1]]
                        assert np.allclose(frob_losses[rank_b], np.linalg.norm(B_b.dot(C_b.T)-A_b, ord='fro'))
                    else:    
                        for bl in range(block+1, num_blocks):
                            b_r1, b_r2 = hpart['rows']['lk'][level+1][bl], hpart['rows']['lk'][level+1][bl+1]
                            if b_r2 > parent_r2: break
                            A_b = A[b_r1 : b_r2, c1 : c2]
                            rank_b, frob_losses, B_b, C_b = low_rank_approx_tol(A_b, tol=tol)
                            if rank_b >= 1:
                                A_HODLR[b_r1 : b_r2, c1 : c2] = B_b.dot(C_b.T)
                                if B_b.shape[0] >= 2 and C_b.shape[0] >= 2:
                                    count += min(B_b.size + C_b.size, B_b.shape[0] * C_b.shape[0])
                                else:
                                    count += max(B_b.size, C_b.size)
                            A_HODLR_fillin[b_r1 : b_r2, c1 : c2] += np.ones((b_r2-b_r1, c2-c1))
                            sizes += [[rank_b, b_r2-b_r1, c2-c1]]
                            assert np.allclose(frob_losses[rank_b], np.linalg.norm(B_b.dot(C_b.T)-A_b, ord='fro'))

    assert (A_HODLR_fillin == np.ones((m, n))).all()
    assert count <=  m*n
    A_HODLR_fillin.sum() == m*n
    bl_sizes = np.array(sizes)
    return A_HODLR, A_HODLR_fillin, bl_sizes, count


def get_hodlr_storage(bl_sizes, count, m, n):
    total_shape = np.multiply(bl_sizes[:, 1], bl_sizes[:, 2]).sum()
    hodlr_storage = 0
    for i in range(bl_sizes.shape[0]):
        if bl_sizes[i, 1] > 1 and bl_sizes[i, 2] > 1:
            hodlr_storage += min(bl_sizes[i, 0] * (bl_sizes[i, 1] + bl_sizes[i, 2]), bl_sizes[i, 1] * bl_sizes[i, 2])
        else:
            hodlr_storage += bl_sizes[i, 0] * max((bl_sizes[i, 1], bl_sizes[i, 2]))
    assert count == hodlr_storage, print(count, hodlr_storage)
    assert hodlr_storage <= m * n, print(m*n - hodlr_storage)
    assert m * n == total_shape, print(m * n, total_shape)
    return hodlr_storage
