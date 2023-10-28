import numpy as np
import copy
import time
import numba as nb
from numba import prange
from scipy.sparse.linalg import svds, eigsh
from typing import List, Tuple, Callable, Dict, List, Set, Optional, Union

from mlrfit.utils import *
from mlrfit.low_rank import *

from profilehooks import profile as tprofile
from memory_profiler import profile as mprofile



def random_hpartition(m, n, num_levels=None, level_list:Optional[List[int]]=None, \
                                    symm=False, perm=True) -> HpartDict:
    """
    Return
    hpartition: dict
            {'rows':{'pi':np.ndarray(m), 'lk':List[List[int]]},
             'cols':{'pi':np.ndarray(m), 'lk':List[List[int]]},}
            rows/cols hierarchical partitioning containing
                block segments for every level
    """
    if perm:
        pi_rows, pi_cols = np.random.permutation(m), np.random.permutation(n)
    else:
        pi_rows, pi_cols = np.arange(m), np.arange(n)
    if num_levels is None: 
        num_levels = int(np.ceil(np.log2(min(m,n))) + 1) #int(np.log2(min(m,n))+2) if 2**int(np.log2(min(m,n)))!=min(m,n) else int(np.log2(min(m,n))+1)
    if level_list is None: 
        level_list = list(range(num_levels))

    hpart = {'rows':{'pi':pi_rows, 'lk':[]}, 'cols':{'pi':pi_cols, 'lk':[]}} 
    for level in level_list:
        if 2**level+1 <= m and 2**level+1 <= n: 
            hpart['rows']['lk'] += [ np.linspace(0, m, 2**level+1, endpoint=True, dtype=int)]
            hpart['cols']['lk'] += [ np.linspace(0, n, 2**level+1, endpoint=True, dtype=int)]
        else:
            hpart['rows']['lk'] += [ np.linspace(0, m, min(m,n)+1, endpoint=True, dtype=int)]
            hpart['cols']['lk'] += [ np.linspace(0, n, min(m,n)+1, endpoint=True, dtype=int)]
    if symm:
        hpart['cols'] = hpart['rows']
    return hpart


def hpart_from_gsubind(gsubind, pi_rows):
    # create hpart from GICS
    m = pi_rows.size
    hpart = {'rows':{'pi':pi_rows, 'lk':[]}, 'cols':{'pi':pi_rows, 'lk':[]}}
    num_levels = 6
    for level in range(num_levels):
        if level == num_levels-1:
            # atoms as leaf clusters
            hpart['rows']['lk'] += [np.linspace(0, m, m+1, endpoint=True, dtype=int)]
        elif level == 0:
            hpart['rows']['lk'] += [np.array([0, m])]
        else:
            s = 0
            prev = gsubind[pi_rows][0][:2*level]
            blocks =  [0]
            while s < gsubind.size:
                count = 0
                while s+count<m and prev==gsubind[pi_rows][s+count][:2*level]:
                    count += 1
                blocks += [s+count]
                s += count
                if s < m:
                    prev = gsubind[pi_rows][s][:2*level]
            hpart['rows']['lk'] += [np.array(blocks)]
            assert s == m
    hpart['cols'] = hpart['rows']
    test_hpartition(hpart, m, m)
    return hpart


def update_htree_B_C_perm_leaf(htree:List[EntryHtree], pi2_rows:np.ndarray, pi2_cols:np.ndarray, \
                               ranks:Optional[np.ndarray]=None, B:Optional[np.ndarray]=None, \
                               C:Optional[np.ndarray]=None, debug=False) -> Tuple[List[EntryHtree], np.ndarray, np.ndarray]:
    """
    We have  
            \hat A_contiguous \approx \pi_1(A) = A[pi1, :][:, pi1]
    Here we will apply permutation rho s.t.
            rho(\hat A_contiguous) \approx \pi_2(A) = A[pi2, :][:, pi2]
    Update hier. tree to respect order in pi2_rows and pi2_cols
    Update Bs, Cts using relative mapping rho from old pi1 to new permutation pi2
            pi2 = rho(pi1)
    """
    num_levels = len(htree)
    pi1_rows = htree[0]['rows'][0]
    pi1_cols = htree[0]['cols'][0]
    # rho relative mapping to go from pi1 to pi2
    m, n = pi1_rows.size, pi1_cols.size
    rho_rows = np.zeros(m).astype(int)
    rho_cols = np.zeros(n).astype(int)
    for idx, row in enumerate(pi2_rows): # pi1_rows[rho_rows[idx]] = pi2_rows[idx]
        rho_rows[idx] = np.argwhere(pi1_rows == row)[0][0]
    for idx, col in enumerate(pi2_cols):
        rho_cols[idx] = np.argwhere(pi1_cols == col)[0][0]

    if debug:
        assert (pi1_rows[rho_rows] == pi2_rows).all()  
        assert (pi1_cols[rho_cols] == pi2_cols).all()
        Mtrx = np.arange(m*n).reshape(m, n)
        assert np.allclose((Mtrx[pi1_rows,:][:,pi1_cols])[rho_rows,:][:,rho_cols], Mtrx[pi2_rows,:][:,pi2_cols])

    # update htree according to current permutations: perm_rows, perm_cols
    for level in range(num_levels):
        num_blocks = len(htree[level]['rows'])
        r1, c1 = 0, 0
        # print(f"{level=}, {num_blocks=}, {htree[level]['rows']=}")
        for block in range(num_blocks):
            rows, cols = htree[level]['rows'][block], htree[level]['cols'][block]
            # print(f"{level=}, {block=}, {num_blocks=}, {rows=}")
            r2 = r1 + htree[level]['rows'][block].size
            c2 = c1 + htree[level]['cols'][block].size  
            htree[level]['rows'][block] = pi2_rows[r1:r2]
            htree[level]['cols'][block] = pi2_cols[c1:c2]
            # permute all but leaf level blocks of B and C
            if level <= num_levels-2:
                if ranks is not None:
                    r_l, r_lp1 = ranks[:level].sum(), ranks[:level+1].sum()
                if B is not None and C is not None:
                    B[:,r_l:r_lp1][r1:r2] = B[:,r_l:r_lp1][r1:r2][rho_rows[r1:r2]-r1, :]
                    C[:,r_l:r_lp1][c1:c2] = C[:,r_l:r_lp1][c1:c2][rho_cols[c1:c2]-c1, :]

            if debug:
                assert (np.sort(rows) == np.sort(pi2_rows[r1:r2])).all()
                assert (np.sort(cols) == np.sort(pi2_cols[c1:c2])).all()
                if level <= num_levels-2:
                    assert (rho_rows[r1:r2]-r1 >=0 ).all() and (rho_cols[c1:c2]-c1 >= 0).all()
            r1, c1 = r2, c2
        if debug:
            assert [el for bl in htree[level]['rows'] for el in bl] == list(pi2_rows)
            assert [el for bl in htree[level]['cols'] for el in bl] == list(pi2_cols)
    if debug:
        m, n = htree[0]['rows'][0].size, htree[0]['cols'][0].size 
        assert r1 == m and c1 == n
    return (htree, B, C)


def swap_two_elements(i1:int, i2:int, pi_rows0:np.ndarray, pi_inv_rows0:np.ndarray, \
                            inplace=True) -> Tuple[np.ndarray,np.ndarray]:
    if inplace:
        pi_rows, pi_inv_rows = pi_rows0, pi_inv_rows0
    else:
        pi_rows, pi_inv_rows = copy.deepcopy(pi_rows0), copy.deepcopy(pi_inv_rows0)
    idx_i1 = pi_inv_rows[i1]
    idx_i2 = pi_inv_rows[i2]
    pi_rows[idx_i1] = i2
    pi_rows[idx_i2] = i1
    pi_inv_rows[i2] = idx_i1
    pi_inv_rows[i1] = idx_i2
    return (pi_rows, pi_inv_rows)


@nb.jit(nopython=True, parallel=True)
def delta_obj_part_colswap_sum(R:np.ndarray, col_sums:np.ndarray, pi1:np.ndarray, pi2:np.ndarray, r:int, c:int):
    """
    Minimum objective change
    """
    n = r*c
    deltas = np.empty((n,), dtype=nb.double)
    for k in prange(n):
        j1 = pi1[k // c]
        j2 = pi2[k % c]
        deltas[k] = delta_obj_partition_sum_col_jit(R, col_sums, j1, j2)
    min_delta_obj = deltas.min()
    k = np.argmin(deltas)
    j1 = pi1[k // c]
    j2 = pi2[k % c]
    return min_delta_obj, j1, j2


@nb.jit(nopython=True, parallel=True)
def delta_obj_part_rowswap_sum(R:np.ndarray, row_sums:np.ndarray, pi1:np.ndarray, pi2:np.ndarray, r:int, c:int):
    """
    Minimum objective change
    """
    n = r*c
    deltas = np.empty((n,), dtype=nb.double)
    for k in prange(n):
        i1 = pi1[k // c]
        i2 = pi2[k % c]
        deltas[k] = delta_obj_partition_sum_row_jit(R, row_sums, i1, i2)
    min_delta_obj = deltas.min()
    k = np.argmin(deltas)
    i1 = pi1[k // c]
    i2 = pi2[k % c]
    return min_delta_obj, i1, i2


@nb.jit(nopython=True, parallel=True)
def delta_obj_part_sum(R:np.ndarray, row_sums:np.ndarray, pi1:np.ndarray, pi2:np.ndarray, r:int, c:int):
    """
    Minimum objective change
    """
    n = r*c
    deltas = np.empty((n,), dtype=nb.double)
    for k in prange(n):
        i1 = pi1[k // c]
        i2 = pi2[k % c]
        deltas[k] = delta_obj_partition_sum_jit(R, row_sums, row_sums, i1, i2, i1, i2)
    min_delta_obj = deltas.min()
    k = np.argmin(deltas)
    i1 = pi1[k // c]
    i2 = pi2[k % c]
    return min_delta_obj, i1, i2


@nb.jit
def delta_obj_partition_sum_jit(R:np.ndarray, row_sums:np.ndarray, col_sums:np.ndarray, \
                            i1:int, i2:int, j1:int, j2:int):
    """
    Objective change u^TSv if we swap rows i1, i2 and cols j1, j2 
                    old_obj - new_obj
    """
    delta_obj = 2*( row_sums[0,i1] -row_sums[0,i2] + \
                    row_sums[1,i2] -row_sums[1,i1] + \
                    col_sums[0,j1] -col_sums[0,j2] + \
                    col_sums[1,j2] -col_sums[1,j1] + \
                   2*(R[i1,j2] + R[i2,j1] - R[i1,j1] - R[i2,j2]) ) 
    return delta_obj


def delta_obj_partition_sum(R:np.ndarray, row_sums:Dict, col_sums:Dict, \
                            i1:int, i2:int, j1:int, j2:int):
    """
    Objective change u^TSv if we swap rows i1, i2 and cols j1, j2 
                    old_obj - new_obj
    """
    delta_obj = 2*( row_sums[1][i1] -row_sums[1][i2] + \
                   row_sums[-1][i2]-row_sums[-1][i1] + \
                    col_sums[1][j1] -col_sums[1][j2] + \
                   col_sums[-1][j2]-col_sums[-1][j1] + \
                   2*(R[i1,j2] + R[i2,j1] - R[i1,j1] - R[i2,j2]) ) 
    return delta_obj


@nb.jit
def delta_obj_partition_sum_row_jit(R:np.ndarray, row_sums:np.ndarray, i1, i2):
    """
    Objective change u^TSv if we swap rows i1, i2
    """
    delta_obj = 2*( row_sums[0,i1]-row_sums[0,i2] + \
                    row_sums[1,i2]-row_sums[1,i1]) 
    return delta_obj


@nb.jit
def delta_obj_partition_sum_col_jit(R:np.ndarray, col_sums:Dict, j1, j2):
    """
    Objective change u^TSv if we swap cols j1, j2 
    """
    delta_obj = 2*( col_sums[0,j1]-col_sums[0,j2] + \
                    col_sums[1,j2]-col_sums[1,j1] ) 
    return delta_obj

                            
def obj_partition_sum_full(R:np.ndarray, pi_rows:np.ndarray, pi_cols:np.ndarray, sep_r:List[int], \
                           sep_c:List[int]):
    """
    Objective value for given partition: x^T R x
    """
    # perm_R = R[pi_rows,:][:,pi_cols]
    blck_diag = 0
    for i in range(2):
        blck_diag += (R[pi_rows,:][:,pi_cols][sep_r[i]:sep_r[i+1], :][:, sep_c[i]:sep_c[i+1]]).sum()
    return 2*blck_diag - R.sum()


def update_rowcol_sum_swap(R, row_sum, j1, j2, transpose=False):
    """
    Update half row/col sums of first and second parition
    cols_parts = [pi_cols[:sep_c[1]], pi_cols[sep_c[1]:sep_c[2]]]
    row_sums = np.stack([(R[:,cols_parts[0]]).sum(axis=1), (R[:,cols_parts[1]]).sum(axis=1)], axis=0)
    """
    if transpose:
        row_sum[0] += -R[j1, :] + R[j2, :]
        row_sum[1] += -R[j2, :] + R[j1, :]
    else:
        row_sum[0] += -R[:, j1] + R[:, j2]
        row_sum[1] += -R[:, j2] + R[:, j1]


def greedy_heur_refinement(R:np.ndarray, pi_rows:np.ndarray, sep_r:List[int], pi_cols:np.ndarray, \
                           sep_c:List[int], symm=False, max_iters=1000, debug=False):
    """
    Greedy heristic refinement of rows and columns partitinoning of 
    matrix R into 2 parts, to increase further the sum of values on diagonal blocks
    """
    itr = 0 
    m, n = R.shape
    obj_all = [obj_partition_sum_full(R, pi_rows, pi_cols, sep_r, sep_c)]
    pi_inv_rows, pi_inv_cols = inv_permutation(pi_rows, pi_cols)
    # Rt = R.T
    # permute rows and cols: put partitions on the diagonal
    rows_parts = [pi_rows[:sep_r[1]], pi_rows[sep_r[1]:sep_r[2]]]
    cols_parts = [pi_cols[:sep_c[1]], pi_cols[sep_c[1]:sep_c[2]]]
    # half row/col sums of first and second parition
    row_sums = np.stack([(R[:,cols_parts[0]]).sum(axis=1), (R[:,cols_parts[1]]).sum(axis=1)], axis=0)
    col_sums = np.stack([(R[rows_parts[0],:]).sum(axis=0), (R[rows_parts[1],:]).sum(axis=0)], axis=0)
    while itr < max_iters:
        ## preprocessing: compute half row/cols sums
        # half row/col sums of first and second parition
        if itr > 0:
            if symm:
                (i1, i2) = best_tuple
                update_rowcol_sum_swap(R, row_sums, i1, i2)
                col_sums = row_sums
            elif (itr-1) % 2 == 0: # swapped rows in previous iter
                (i1, i2) = best_tuple
                # update_rowcol_sum_swap(Rt, col_sums, i1, i2)
                update_rowcol_sum_swap(R, col_sums, i1, i2, transpose=True)
            elif (itr-1) % 2 == 1: # swapped cols in previous iter
                (j1, j2) = best_tuple
                update_rowcol_sum_swap(R, row_sums, j1, j2)
        
        # find \min delta_obj = old_obj - new_obj, and best pair
        if symm:
            min_delta_obj, i1, i2 = delta_obj_part_sum(R, row_sums, pi_rows[sep_r[0]:sep_r[1]], \
                                                  pi_rows[sep_r[1]:sep_r[2]], sep_r[1], sep_r[2]-sep_r[1])
            best_tuple = (i1, i2)
        else:
            # Alternate between row and column swap
            if itr % 2 == 0: # swap rows
                min_delta_obj, i1, i2 = delta_obj_part_rowswap_sum(R, row_sums, pi_rows[sep_r[0]:sep_r[1]], \
                                                  pi_rows[sep_r[1]:sep_r[2]], sep_r[1], sep_r[2]-sep_r[1])
                best_tuple = (i1, i2)
            else: # swap columns
                min_delta_obj, j1, j2 = delta_obj_part_colswap_sum(R, col_sums, pi_cols[sep_c[0]:sep_c[1]], \
                                                  pi_cols[sep_c[1]:sep_c[2]], sep_c[1], sep_c[2]-sep_c[1])
                best_tuple = (j1, j2)
        ## exchange best tuple
        if min_delta_obj >= 0: break
        if symm:
            (i1, i2) = best_tuple
            swap_two_elements(i1, i2, pi_rows, pi_inv_rows, inplace=True)
        elif itr%2 == 0: # swap rows
            (i1, i2) = best_tuple
            swap_two_elements(i1, i2, pi_rows, pi_inv_rows, inplace=True)
        elif itr%2 == 1: # swap cols
            (j1, j2) = best_tuple
            swap_two_elements(j1, j2, pi_cols, pi_inv_cols, inplace=True)
        cur_obj = obj_all[-1] - min_delta_obj
        if symm: 
            pi_cols = pi_rows
            pi_inv_cols = pi_inv_rows
        if debug:
            assert cur_obj + 1e-9 >= obj_all[-1] # current objective increases previous objective
            assert np.allclose(cur_obj, obj_partition_sum_full(R, pi_rows, pi_cols, sep_r, sep_c)) 
            assert (np.sort(pi_rows)==np.arange(m)).all() and (np.sort(pi_cols)==np.arange(n)).all()
            assert np.allclose(R.sum(), row_sums.sum()) and \
                   np.allclose(R.sum(), col_sums.sum())
        itr += 1
        obj_all += [cur_obj]
    return pi_rows, pi_cols, obj_all


def bipartite_spectral_partition(symm=False, refined=True, max_iters=100, debug=False):
    def spectral_partition_helper(R:np.ndarray, k=2):
        """
        Spectral partitioning of rows and columns of residual matrix R
        """
        m, n = R.shape
        if np.allclose(R.sum(), 0):
            # divide arbitrarily 
            u = np.arange(m)
            v = np.arange(n)
        elif symm:
            R = R + 1e-5*np.ones(R.shape)
            d = R.sum(axis=1)
            tilde_R = np.diag(d**(-0.5)) @ R @ np.diag(d**(-0.5)) 
            M = d.max() + 1e-8
            # eigenvectors for second largest eigenvalue
            U, sigmas, Vh = svds(M*np.eye(n) + tilde_R, k=2) # shift normalized adjacency
            idx = np.argsort(sigmas)[::-1] # dec order M+lambda_n>=M+lambda_{n-1}
            sigmas = sigmas[idx]
            U = U[:, idx]
            u = v = np.diag(d**(-0.5)) @ U[:, 1] 
            if  debug:
                assert np.allclose(R.sum(), 0) or np.allclose(tilde_R, 0) or \
                    np.allclose(d@u, 0), print(f"{M=}, {sigmas=}, {d@u=}")
        else:
            R = R + 1e-5*np.ones(R.shape)
            d1 = R.sum(axis=1)
            d2 = R.sum(axis=0)
            tilde_R = np.diag(d1**(-0.5)) @ R @ np.diag(d2**(-0.5)) 
            U, sigmas, Vh = svds(tilde_R, k=2)
            idx = np.argsort(sigmas)[::-1] # decreasing order
            sigmas = sigmas[idx]
            U = U[:, idx]
            Vh = Vh[idx, :]
            # left and right singluar vectors for second largest singular value
            u = np.diag(d1**(-0.5)) @ U[:, 1]
            v = np.diag(d2**(-0.5)) @ Vh[1, :].T
            if  debug:
                assert (d1>= -1e-9).all() and (d2 >=-1e-9).all()
                assert np.allclose(R.sum(), 0) or np.allclose(tilde_R, 0) or \
                    np.allclose(d1@u + d2@v, 0), print(f"{sigmas=}, {d1@u[:,0]+d2@v[:,0]=}")
        rows = np.argsort(u)
        cols = np.argsort(v)
        sep_r = [0, m//2, m]; sep_c = [0, n//2, n]
        if refined:
            rows, cols, _ = greedy_heur_refinement(R, rows, sep_r, cols, sep_c, symm=symm, \
                                                    max_iters=max_iters, debug=debug)
        return rows, sep_r, cols, sep_c
    return spectral_partition_helper

# @tprofile
# @mprofile
def spectral_partition(symm=False, refined=False, max_iters=100, debug=False):
    def spectral_partition_helper(R:np.ndarray, k=2):
        """
        Spectral partitioning of rows and columns of matrix R
        elementwise nonnegative
        value_type = similarity
            residual matrix R stores "similarity" between i, j
            the objective is to **maximize** the sum of pairwise similarities within clusters
        """
        m, n = R.shape
        if np.allclose(R.sum(), 0):
            # divide arbitrarily 
            u = np.arange(m)
            v = np.arange(n)
        elif symm:
            if R.size == 1:
                return [0], [0,1], [0], [0,1]
            # Laplacian from |residual matrix|**2 for a "connected" R
            R = np.maximum(R, 0) + np.ones((n,n))*1e-5
            d = R.sum(axis=1)
            L = np.diag(d) - R # is psd matrix
            d_max = d.max()
            L_shift = 2*d_max*np.eye(n) - L
            if not debug:
                del L
            try:
                sh_lambdas, U = eigsh(L_shift, k=2, which='LM')
            except: # ARPACK failed
                sh_lambdas, U = eigsh(L_shift, k=2, which='LM', tol=1e-2)
            lambdas = 2*d_max - sh_lambdas
            idx = np.argsort(lambdas)
            lambdas = lambdas[idx] # increasing  order
            U = U[:, idx]
            u = v = U[:, 1] # second smallest eigenvector
            # eigenvector for (second if connected graph) smallest nonzero eigenvalue
            if  debug:
                l_true, V_true = np.linalg.eigh(L)
                idx = np.argsort(l_true) # increasing order
                l_true = l_true[idx]
                V_true = V_true[:, idx]
                v_true  = V_true[:, 1] # lambda_2
                assert (R >= -1e-9).all() and np.allclose(L, V_true@np.diag(l_true)@V_true.T)
                assert np.allclose(lambdas, l_true[:2])
                assert m == n and (np.diff(lambdas) >= -1e-9).all() and np.allclose(L.sum(axis=1), 0)
                assert np.abs(v.sum())<= 1e-7, print(f"{lambdas=}, {v.sum()=}")
            del L_shift
        else:
            R_sum = R.sum()
            t = (1./m) * R_sum / 2 
            s = (1./n) * R_sum / 2 
            a = (1./n) * (R.sum(axis=1, keepdims=True) - t * np.ones((m, 1)))
            b = (1./m) * ((R.T).sum(axis=1, keepdims=True) - s * np.ones((n, 1)))
            tilde_R = R - a @ np.ones((1, n)) - np.ones((m, 1)) @ b.T
            U, sigmas, Vh = svds(tilde_R, k=1)
            # left and right singluar vectors for maximum singular value
            u = U[:, 0]
            v = Vh[0, :].T
            if  debug:
                assert np.linalg.norm(tilde_R.sum(axis=0))<1e-8 and np.linalg.norm(tilde_R.sum(axis=1))<1e-8
                assert np.allclose(R_sum, 0) or np.allclose(tilde_R, 0) or np.allclose(u.sum(), 0) and \
                                                np.allclose(v.sum(), 0), print(f"{sigmas=}, {v.sum()=}")
            del tilde_R
        rows = np.argsort(u)
        cols = np.argsort(v)
        sep_r = [0, m//2, m]; sep_c = [0, n//2, n]
        if refined:
            rows, cols, _ = greedy_heur_refinement(R, rows, sep_r, cols, sep_c, symm=symm, \
                                                    max_iters=max_iters, debug=debug)
        return rows, sep_r, cols, sep_c
    return spectral_partition_helper


def spectral_partition_minimize(symm=False, refined=False, max_iters=100, debug=False):
    def spectral_partition_helper(D:np.ndarray, k=2):
        """
        Spectral partitioning of rows and columns of matrix D elementwise nonnegative
        value_type = distance
            distance matrix D stores "distance" between i, j
            the objective is to **minimize** the sum of pairwise distances within clusters
        """
        m, n = D.shape
        if np.allclose(D.sum(), 0):
            # divide arbitrarily 
            u = np.arange(m)
            v = np.arange(n)
        elif symm:
            if D.size == 1:
                return [0], [0,1], [0], [0,1]
            # Laplacian from distance matrix D
            D = np.maximum(D, 0) 
            d = D.sum(axis=1)
            L = np.diag(d) - D # is psd matrix
            try:
                lambdas, U = eigsh(L, k=1, which='LM')
            except: # ARPACK failed
                lambdas, U = eigsh(L, k=1, which='LM', tol=1e-2)
            u = v = U[:, 0] # largest eigenvector
            # eigenvector for (second if connected graph) smallest nonzero eigenvalue
            if  debug:
                l_true, V_true = np.linalg.eigh(L)
                idx = np.argsort(l_true) # increasing order
                l_true = l_true[idx]
                V_true = V_true[:, idx]
                v_true  = V_true[:, -1] # lambda_max
                assert (D >= -1e-9).all() and np.allclose(L, V_true@np.diag(l_true)@V_true.T)
                assert np.allclose(lambdas, l_true[-1])
                assert m == n and np.allclose(L.sum(axis=1), 0)
            del L
        else:
            D_sum = D.sum()
            t = (1./m) * D_sum / 2 
            s = (1./n) * D_sum / 2 
            a = (1./n) * (D.sum(axis=1, keepdims=True) - t * np.ones((m, 1)))
            b = (1./m) * ((D.T).sum(axis=1, keepdims=True) - s * np.ones((n, 1)))
            neg_tilde_D = -(D - a @ np.ones((1, n)) - np.ones((m, 1)) @ b.T)
            U, sigmas, Vh = svds(neg_tilde_D, k=1)
            # left and right singluar vectors for maximum singular value of - \tilde D
            u = U[:, 0]
            v = Vh[0, :].T
            if  debug:
                assert np.linalg.norm(neg_tilde_D.sum(axis=0))<1e-8 and np.linalg.norm(neg_tilde_D.sum(axis=1))<1e-8
                assert np.allclose(D_sum, 0) or np.allclose(neg_tilde_D, 0) or np.allclose(u.sum(), 0) and \
                                                np.allclose(v.sum(), 0), print(f"{sigmas=}, {v.sum()=}")
            del neg_tilde_D
        rows = np.argsort(u)
        cols = np.argsort(v)
        sep_r = [0, m//2, m]; sep_c = [0, n//2, n]
        if refined: # refine with negative sign, because refinement increases the sum of diagonal values
            rows, cols, _ = greedy_heur_refinement(-D, rows, sep_r, cols, sep_c, symm=symm, \
                                                    max_iters=max_iters, debug=debug)
        return rows, sep_r, cols, sep_c
    return spectral_partition_helper


def hpart_dist_clustering(D:np.ndarray, num_levels:int, ks:Optional[np.ndarray]=None, \
                            func_partition:Optional[Callable]=None, \
                            symm=False, debug=False,\
                            grref_max_iters=5000):
    """
    Reursive hier. partitioning of distance matrix
    For every level update:
        htree: list of dictionaries contatining rows and cols partitioning
            for every level
            [{ 'rows': [row partitioning], 
                'cols': [column partitioning]}] * num_levels
            stores full permutations
    Return 
        hpart: dict
            {'rows':{'pi':np.ndarray(m), 'lk':List[List[int]]},
            'cols':{'pi':np.ndarray(m), 'lk':List[List[int]]},}
            rows/cols hierarchical partitioning containing
                block segments for every level
    """
    m, n = D.shape
    if ks is None:
        ks = (num_levels-2)*[2] + [np.inf]
    if func_partition is None:
        func_partition = spectral_partition_minimize(symm=symm, refined=True, max_iters=grref_max_iters)

    htree = [{'rows':[np.arange(m)], 'cols':[np.arange(n)]}]
    # store absolute permutation of rows/cols that makes block diagonal
    perm_rows = np.arange(m)
    perm_cols = np.arange(n)
    hpart = full_htree_to_hpart(htree)

    for level in range(num_levels):
        if level == num_levels-1: 
            break
    
        k = ks[level]
        if k == np.inf:
            # add atoms on the leaf level
            num_blocks = len(hpart['rows']['lk'][-1]) - 1
            rows = []; cols = []
            for block in range(num_blocks):
                r1, r2 = hpart['rows']['lk'][-1][block], hpart['rows']['lk'][-1][block+1]
                c1, c2 = hpart['cols']['lk'][-1][block], hpart['cols']['lk'][-1][block+1]
                rows += [np.linspace(r1, r2, min(r2-r1, c2-c1), endpoint=False, dtype=int)]
                cols += [np.linspace(c1, c2, min(r2-r1, c2-c1), endpoint=False, dtype=int)]
            rows = np.concatenate(rows + [np.array([m])])
            cols = np.concatenate(cols + [np.array([n])])
            assert rows.size == min(m, n) + 1 and cols.size == min(m, n) + 1
            assert len(set(rows)) == min(m, n) + 1 and len(set(cols)) == min(m, n) + 1
            hpart['rows']['lk'] += [ rows ]
            hpart['cols']['lk'] += [ cols ]
        else:
            # permute distance in same order as to make D contiguous 
            # perm_D = D[perm_rows, :][:, perm_cols]
            num_blocks = len(htree[level]['rows'])
            r1, c1 = 0, 0
            htree += [{'rows':[], 'cols':[]}]
            new_perm_rows = np.zeros(m).astype(int)
            new_perm_cols = np.zeros(n).astype(int)
            # find permutations of each block using given partition function
            for block in range(num_blocks):
                r2 = r1 + htree[level]['rows'][block].size
                c2 = c1 + htree[level]['cols'][block].size 
                # permute rows and columns according to given partition function for a given block 
                # of distance matrix
                pi_rows_bl, sep_r_bl, pi_cols_bl, sep_c_bl = func_partition(D[perm_rows, :][:, perm_cols][r1:r2,c1:c2], k=k)
                # del perm_D
                num_partitions = len(sep_r_bl)-1
                # add a level to a hierarchy
                for si in range(num_partitions):
                    # permute rows and cols: put partitions on the diagonal
                    new_perm_rows[r1:r2][sep_r_bl[si]:sep_r_bl[si+1]] = perm_rows[r1:r2][pi_rows_bl[sep_r_bl[si]:sep_r_bl[si+1]]]
                    new_perm_cols[c1:c2][sep_c_bl[si]:sep_c_bl[si+1]] = perm_cols[c1:c2][pi_cols_bl[sep_c_bl[si]:sep_c_bl[si+1]]]
                    htree[level+1]['rows'] += [new_perm_rows[r1:r2][sep_r_bl[si]:sep_r_bl[si+1]]]
                    htree[level+1]['cols'] += [new_perm_cols[c1:c2][sep_c_bl[si]:sep_c_bl[si+1]]]
                r1, c1 = r2, c2
            perm_rows = new_perm_rows
            perm_cols = new_perm_cols
            # permute the htree to respect the perm_rows/cols on the new leaf level
            htree = update_htree_B_C_perm_leaf(htree, perm_rows, perm_cols, debug=debug)[0]
            hpart = full_htree_to_hpart(htree)

    return  hpart


# -------------------- TOP DOWN HIERARCHICAL PARTITIONING with rank fitting -----------------------------


def hpartition_topdown(R:np.ndarray, ranks:np.ndarray, ks:np.ndarray, func_partition:Callable, \
                       symm=False, PSD=False, debug=False) -> Tuple[HpartDict, np.ndarray]:
    """
    Return 
    hpartition: dict
            {'rows':{'pi':np.ndarray(m), 'lk':List[List[int]]},
             'cols':{'pi':np.ndarray(m), 'lk':List[List[int]]},}
            rows/cols hierarchical partitioning containing
                block segments for every level
    """
    assert ks.size == ranks.size - 1
    m, n = R.shape
    num_levels = ranks.size
    unord_htree = [{'rows':[], 'cols':[]} for _ in range(num_levels)]
    unord_htree[0] = {'rows':[np.arange(m)], 'cols':[np.arange(n)]}
    # store children indices on each level
    trees = {'rows':[[] for _ in range(len(ks))], 'cols':[[] for _ in range(len(ks))]}
    rows = np.array(range(m))
    cols = np.array(range(n))
    
    # find hpartitioning and tree
    H = hpartition_topdown_rec(R, rows, cols, 0, ranks, ks, unord_htree, trees, \
                                func_partition, symm=symm, PSD=PSD, debug=debug)
    
    # merge partitions bottom up to preserve the leaf ordering of the rows and cols
    htree = [{'rows':[], 'cols':[]} for _ in range(num_levels)]
    htree[-1] = unord_htree[-1]
    for level in reversed(range(num_levels-1)):
        num_partitions = len(trees['rows'][level])
        for part_i in range(num_partitions):
            r1, r2 = trees['rows'][level][part_i]
            merged_rows = np.array([el for part in htree[level+1]['rows'][r1:r2] for el in part])
            c1, c2 = trees['cols'][level][part_i]
            merged_cols = np.array([el for part in htree[level+1]['cols'][c1:c2] for el in part])
            htree[level]['rows'] += [merged_rows]
            htree[level]['cols'] += [merged_cols]

    hpart = full_htree_to_hpart(htree, debug=debug)

    if debug:
        test_hpartition(hpart, m, n)
    return (hpart, H)


def hpartition_topdown_rec(R, rows:np.ndarray, cols:np.ndarray, level:int, ranks:np.ndarray, \
                            ks:np.ndarray, htree:List[EntryHtree], trees, func_partition:Callable, \
                            symm=False, PSD=False, debug=False):
    """
    Input:
        ks: branching / number of partitions on each level
        rows: np.array() current row partitioning 
        cols: np.array() current column partitioning 
        R: np.array().shape = (m, n) current residual matrix, in the row order
            of rows and column order of cols
    Invariant: rows.size == A.shape[0] and cols.size == A.shape[1]
    Output:
        H : np.array
            Hierarchihcal approximation of A
    Modify:
        htree: 
            list of dictionaries contatining rows and cols partitioning
        trees: {'rows': list of  lists, 'cols': list of lists}
            for every level, entry part_i contains index range on children on (level+1)
    """
    dim = ranks[level]
    if debug:
        assert rows.size == R.shape[0] and cols.size == R.shape[1]
        
    # low rank approximation of residual R
    if PSD:
        U, sigmas = frob_low_rank_psd(R, dim=dim)
        Vt  = U.T
    else:
        U, Vt, sigmas = frob_low_rank(R, dim=dim, symm=symm)
    BC = np.dot(U * sigmas, Vt)
    
    if debug: 
        assert np.allclose(U @ np.diag(sigmas) @ Vt, np.dot(U * sigmas, Vt))
    # return low rank approximation if at the leaf level
    if level == len(ks):
        return BC
    # new residual matrix on current level
    R_new = R - BC
    block_diag = np.zeros(R_new.shape)
    k = ks[level]
    # permutation of all rows and columns according to given partition function
    pi_rows, sep_r, pi_cols, sep_c = func_partition(np.square(R_new), k=k)
    pi_inv_rows, pi_inv_cols = inv_permutation(pi_rows, pi_cols)
    # approximate permuted R_new with block diagonal
    perm_R_new = R_new[pi_rows,:][:,pi_cols]
    
    # block diagonal approximation of permuted R_new 
    r1, c1 = 0, 0
    num_partitions = len(sep_r)-1
    for si in range(num_partitions):
        # permute rows and cols: put partitions on the diagonal
        rows_i = rows[pi_rows[sep_r[si]:sep_r[si+1]]]
        cols_i = cols[pi_cols[sep_c[si]:sep_c[si+1]]]
        # add a new partition on the next level
        htree[level+1]['rows'] += [rows_i]
        htree[level+1]['cols'] += [cols_i]
        r2 = r1 + rows_i.size
        c2 = c1 + cols_i.size
        perm_R_new_i = perm_R_new[r1:r2, c1:c2]
        H_i = hpartition_topdown_rec(perm_R_new_i, rows_i, cols_i, level+1, ranks, ks, \
                                     htree, trees, func_partition, PSD=PSD, \
                                     symm=symm, debug=debug)
        block_diag[r1:r2, c1:c2] = H_i
        r1, c1 = r2, c2
    
    if level < len(ks):
        # indices for the children nodes of rows and cols
        s_idx = len(htree[level+1]['rows']) - (len(sep_r)-1)
        trees['rows'][level] += [(s_idx, s_idx + len(sep_r)-1)]
        s_idx = len(htree[level+1]['cols']) - (len(sep_c)-1)
        trees['cols'][level] += [(s_idx, s_idx + len(sep_c)-1)]
            
    H = BC + block_diag[pi_inv_rows,:][:,pi_inv_cols]
    
    if debug:
        # print(f"{level=}")
        res1 = np.linalg.norm(BC - R, 'fro')
        res2 = np.linalg.norm(H - R, 'fro')
        assert (r1, c1) == R.shape and (res1 + 1e-9 >= res2)
        
    return H
