import numpy as np

from typing import List, Tuple, Callable, TypedDict, List, Set, Optional, Union 
import numba as nb
import numba as nb
from numba import prange

import torch
import copy

from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.csgraph import shortest_path

import linecache
import os
import tracemalloc


def display_top(snapshot, key_type='lineno', limit=10):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print("#%s: %s:%s: %.1f KiB"
              % (index, frame.filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


class EntryHpartDict(TypedDict):
    pi: np.ndarray 
    lk: List[Union[List[int], np.ndarray]] 


class HpartDict(TypedDict):
    rows: EntryHpartDict 
    cols: EntryHpartDict
 

class EntryHtree(TypedDict):
    rows: List[np.ndarray] 
    cols: List[np.ndarray]    


def get_device(gpu_no=0):
    if torch.cuda.is_available():
        return torch.device('cuda', gpu_no)
    else:
        return torch.device('cpu')


def rel_diff(a, b=None, den=None):
    """
    Relative difference
    """
    if torch.is_tensor(a):
        order = 'fro'
        if den is None:
            d = min(torch.linalg.norm(a, ord=order), torch.linalg.norm(b, ord=order))
        else:
            b = den
            d = torch.linalg.norm(den, ord=order)
        return torch.linalg.norm(a-den, ord=order) / d
    else:
        if not type(a).__module__ == np.__name__:
            a = np.array([a])
            b = np.array([b])  
        if a.ndim == 0 or a.shape[0] == a.size or a.ndim>1 and a.shape[1] == a.size:
            order = None
        else:
            order = 'fro'
        if den is None:
            d = min(np.linalg.norm(a, ord=order), np.linalg.norm(b, ord=order))
        else:
            b = den
            d = np.linalg.norm(den, ord=order)
        return np.linalg.norm(a-b, ord=order) / d


def full_htree_to_hpart(htree:List[EntryHtree], debug=False) -> HpartDict:
    hpart = {'rows':{'pi':htree[0]['rows'][0], 'lk':[]}, \
             'cols':{'pi':htree[0]['cols'][0], 'lk':[]}} 
    num_levels = len(htree)
    for level in range(num_levels):
        r1, c1 = 0, 0
        r_blocks, c_blocks = [0], [0]
        for block in range(len(htree[level]['rows'])):
            r2 = r1 + htree[level]['rows'][block].size
            c2 = c1 + htree[level]['cols'][block].size
            r_blocks += [r2]
            c_blocks += [c2]
            r1, c1 = r2, c2
        hpart['rows']['lk'] += [ np.array(r_blocks) ]
        hpart['cols']['lk'] += [ np.array(c_blocks) ]
        if debug:
            assert len(hpart['rows']['lk'][level])-1 == len(htree[level]['rows'])
            assert len(hpart['cols']['lk'][level])-1 == len(htree[level]['cols'])
    return hpart


def row_cols_permutations(hpart:HpartDict):
    # get permutations pi and pi_inv from given hpartition
    pi_rows = hpart['rows']['pi']
    pi_cols = hpart['cols']['pi']
    pi_inv_rows, pi_inv_cols = inv_permutation(pi_rows, pi_cols)
    return pi_rows, pi_inv_rows, pi_cols, pi_inv_cols


def inv_permutation(pi_rows, pi_cols):
    pi_inv_rows = np.zeros(pi_rows.shape).astype(int)
    pi_inv_cols = np.zeros(pi_cols.shape).astype(int)
    for idx, row in enumerate(pi_rows):
        pi_inv_rows[row] = idx
    for idx, col in enumerate(pi_cols):
        pi_inv_cols[col] = idx
    return pi_inv_rows, pi_inv_cols


def test_hpartition(hpart, m, n):
    num_levels = len(hpart['rows']['lk'])
    assert set(hpart['rows']['pi']) == set(range(m))
    assert set(hpart['cols']['pi']) == set(range(n))
    for level in range(num_levels):
        assert (np.diff(np.array(hpart['rows']['lk'][level])) > 0).all()
        assert (np.diff(np.array(hpart['cols']['lk'][level])) > 0).all()
        assert hpart['rows']['lk'][level][0] == 0 and hpart['rows']['lk'][level][-1] == m
        assert hpart['cols']['lk'][level][0] == 0 and hpart['cols']['lk'][level][-1] == n
        assert len(hpart['rows']['lk'][level]) == len(hpart['cols']['lk'][level])


def test_eigsh_svds(U1, l1, Vt1, dim, A, mode='svds'):
    if mode == 'svds':
        U, sigmas, Vt = np.linalg.svd(A, full_matrices=False)
        # decreasing order of singular values
        idx = np.argsort(sigmas)[::-1]
        sigmas = sigmas[idx]
        U = U[:, idx]
        Vt = Vt[idx, :]
        assert np.allclose(sigmas[:dim], l1) and \
               np.allclose(np.dot(U1*l1, Vt1), np.dot(U[:,:dim]*sigmas[:dim], Vt[:dim])),\
                print(sigmas[:dim][:10], l1[:10], rel_diff(sigmas[:dim], l1), \
               np.allclose(np.dot(U1*l1, Vt1), np.dot(U[:,:dim]*sigmas[:dim], Vt[:dim])),\
                rel_diff(np.dot(U1*l1, Vt1), np.dot(U[:,:dim]*sigmas[:dim], Vt[:dim])))
        assert np.allclose(np.dot(U*sigmas, Vt), A)
    else:
        # decreasing order of eigenvalues
        lambdas, V = np.linalg.eigh(A)
        idx = np.argsort(lambdas)[::-1]
        lambdas = np.maximum(lambdas[idx], 0)  # (lambdas_1)_+ >= ... >= (lambdas_n)_+
        V = V[:, idx]
        zero_idices = np.where(lambdas==0)[0]
        if zero_idices.size >= 1:
            zero_idx = min(zero_idices.min(), dim)
        else:
            zero_idx = dim
        assert np.allclose(lambdas[:zero_idx], l1) and \
               np.allclose(np.dot(U1*l1, Vt1), np.dot(V[:,:dim]*lambdas[:dim], V[:,:dim].T))


def graph_distance_matrix(A=None, sparse=False, printing=False, directed=False) -> np.ndarray:
    if sparse:
        A_srs = A
    else:
        A_srs = csr_matrix(A)
        assert directed == (not np.allclose(A, A.T))
    Dist = shortest_path(csgraph=A_srs, directed=directed)
    if printing:
        print(f"|E|={(A>0).sum()/2}")
        print(f"deg_max = {A.sum(axis=1).max()}, deg_mean = {A.sum(axis=1).mean()}, deg_min = {A.sum(axis=1).min()}")
        print(f"{Dist.max()=}")
    return Dist


def cap_distance_max_element(Dist:np.ndarray):
    """
    Cap disconnected distance matrix by value of the maximum element
    """
    if Dist.max() < np.inf: return Dist
    # set distance to maximum between disconnected components
    M = Dist[Dist < np.inf].max()
    mask = (Dist==np.inf)
    Dist[mask]=M
    return Dist


def demean_clip(C, factor_std=3):
    # demean and clip
    mu = C.mean(axis=1, keepdims=True)
    std = C.std(axis=1,)
    Z = np.zeros(C.shape)
    for i in range(C.shape[0]):
        Z[i,:] = np.clip(C[i,:]-mu[i], -factor_std*std[i], factor_std*std[i])
    assert (std + 1e-6 >= Z.std(axis=1)).all()
    return Z


def convert_compressed_to_sparse(B:np.ndarray, hp_entry:EntryHpartDict, \
                        ranks:np.ndarray, mtype='csc'):
    data, i_idx, j_idx = [], [], []
    col_count = 0
    num_levels = len(hp_entry['lk'])
    for level in range(num_levels):
        # B_level = B[:,ranks[:level].sum():ranks[:level+1].sum()]
        num_blocks = len(hp_entry['lk'][level])-1
        for block in range(num_blocks):
            r1, r2 = hp_entry['lk'][level][block], hp_entry['lk'][level][block+1]
            data += [B[:,ranks[:level].sum():ranks[:level+1].sum()][r1:r2].flatten(order='C')]
            i_idx += [np.tile(np.arange(r1, r2), [ranks[level],1]).flatten(order='F')]
            j_idx += [np.tile(np.arange(col_count, col_count+ranks[level]), [r2-r1])]
            col_count += ranks[level]
    data = np.concatenate(data, axis=0)
    i_idx = np.concatenate(i_idx, axis=0)
    j_idx = np.concatenate(j_idx, axis=0)

    s = sum([(len(hp_entry['lk'][level])-1)*ranks[level] for level in range(num_levels)])
    tilde_B = coo_matrix((data, (i_idx, j_idx)), shape=(B.shape[0], s))
    if mtype == 'csc':
        tilde_B = tilde_B.tocsc()
    elif mtype == 'csr':
        tilde_B = tilde_B.tocsr()
    return tilde_B


def cap_ranks_block_size(hpart, ranks):
    """
    Cap rank values by minimum block size for each level 
    """
    res = copy.deepcopy(ranks)
    m = hpart['rows']['pi'].size
    num_levels = len(hpart['rows']['lk'])
    min_sizes = np.zeros(ranks.size)
    capped_rank = 0; num_capped_rank = 0
    for level in range(num_levels):
        min_size = m
        num_blocks = hpart['rows']['lk'][level].size-1
        for block in range(num_blocks):
            r1, r2 = hpart['rows']['lk'][level][block], hpart['rows']['lk'][level][block+1]
            c1, c2 = hpart['cols']['lk'][level][block], hpart['cols']['lk'][level][block+1]
            min_size = min(min(min_size, r2-r1), c2-c1)
        min_sizes[level] = min_size
        if res[level] > min_size:
            res[0] += res[level] - min_size
            res[level] = min_size
            capped_rank += min_size
            num_capped_rank += 1
    avg_rank = (res.sum()-capped_rank)//(res.size - num_capped_rank)
    if res[0] > avg_rank:
        indices = np.where(min_sizes > res)[0][1:]
        for level in indices:
            shift = np.clip(avg_rank, 0, min_sizes[level]) - res[level]
            res[0] -= shift
            res[level] += shift
    # assert res.sum() == ranks.sum()
    return res


def uniform_ranks(rank, num_levels):
    ranks = np.ones(num_levels, dtype=int) * (rank // num_levels)
    residual = rank - num_levels * (rank // num_levels)
    for l in range(residual):
        ranks[l] += 1
    assert ranks.sum() == rank
    return ranks


def uniform_capped_ranks(rank, hpart):
    num_levels = len(hpart['rows']['lk'])
    ranks = uniform_ranks(rank, num_levels)
    ranks = cap_ranks_block_size(hpart, ranks)
    return ranks


def hpart_info_print(hpart):
    for level in range(len(hpart['cols']['lk'])):
        avg_rows = np.diff(hpart['rows']['lk'][level]).mean()
        avg_cols = np.diff(hpart['cols']['lk'][level]).mean()
        num_blocks = hpart['rows']['lk'][level].size-1
        print(f"{level=},  {num_blocks}")
        print(f"    avg_row_bl_size={avg_rows:.1f}, avg_col_bl_size={avg_cols:.1f}")

 
def compressed_to_padded(B, ranks):
    """
    Pad matrix B with 0 by increasing r_l to (r_l+1)
    """
    eB = []
    if torch.is_tensor(B):
        zeros = torch.zeros((B.shape[0], 1), dtype=int)
    else:
        zeros = np.zeros((B.shape[0], 1), dtype=int) 
    for level in range(ranks.size):
        eB += [B[:, ranks[:level].sum():ranks[:level+1].sum()], zeros]
    if torch.is_tensor(B):
        return torch.cat(eB, dim=1)
    else:
        return np.concatenate(eB, axis=1)


def extended_to_compressed(eB:Union[np.ndarray,torch.Tensor], inc_ranks:np.ndarray):
    """
    Get compressed version of extended eB by decreasing r_l+1 to r_l
    by tossing away last vector in each level
    """
    B = []
    for level in range(inc_ranks.size):
        B += [eB[:, inc_ranks[:level].sum():inc_ranks[:level+1].sum()-1]]
    
    if torch.is_tensor(eB):
        B = torch.cat(B, dim=1)
    else:
        B = np.concatenate(B, axis=1)
    assert B.shape[1] == eB.shape[1]-inc_ranks.size
    return B


@nb.jit(nopython=True, parallel=True)
def ra_delta_obj(delta_rp1:np.ndarray, delta_rm1:np.ndarray, num_levels:int, top_k:int):
    """
     Find (i,j): level i to increase rank and level j to decrease rank
     that minimizes the new loss:
        delta = loss_new - loss_old
    """
    deltas = np.empty((num_levels**2,), dtype=nb.double)
    for k in prange(num_levels**2):
        ti = k // num_levels
        tj = k % num_levels
        if ti == tj: 
            deltas[k] = np.inf
        else:
            deltas[k] = delta_rm1[tj] - delta_rp1[ti]
    
    idx = np.argsort(deltas)[:top_k]
    tis = idx // num_levels
    tjs = idx % num_levels
    # print(deltas)
    return deltas[idx], tis, tjs


@nb.jit(nopython=True)
def update_B_rank_alloc_ij(B:np.ndarray, b:np.ndarray, i_plus:int, j_minus:int, ranks:np.ndarray):
    # assert i_plus < j_minus
    newB = np.empty_like(B)
    newB[:,:ranks[:i_plus+1].sum()] = B[:,:ranks[:i_plus+1].sum()]
    newB[:,ranks[:i_plus+1].sum()] = b[:,i_plus]
    newB[:,ranks[:i_plus+1].sum()+1:ranks[:j_minus+1].sum()] = B[:,ranks[:i_plus+1].sum():ranks[:j_minus+1].sum()-1]
    newB[:,ranks[:j_minus+1].sum():] = B[:,ranks[:j_minus+1].sum():]
    return newB


@nb.jit(nopython=True)
def update_B_rank_alloc_ji(B:np.ndarray, b:np.ndarray, i_plus:int, j_minus:int, ranks:np.ndarray):
    # assert j_minus < i_plus
    newB = np.empty_like(B)
    newB[:,:ranks[:j_minus+1].sum()-1] = B[:,:ranks[:j_minus+1].sum()-1]
    newB[:,ranks[:j_minus+1].sum()-1:ranks[:i_plus+1].sum()-1] = B[:,ranks[:j_minus+1].sum():ranks[:i_plus+1].sum()]
    newB[:,ranks[:i_plus+1].sum()-1] = b[:,i_plus]
    newB[:,ranks[:i_plus+1].sum():] = B[:,ranks[:i_plus+1].sum():]
    return newB


@nb.jit(nopython=True)
def compute_perm_hat_A_jit(B:np.ndarray, C:np.ndarray, rows_lk,  cols_lk, ranks:np.ndarray):
    """
    Compute permuted hat_A with each A_level being block diagonal matrix 
    """
    num_levels = ranks.size
    hat_A = np.zeros((B.shape[0], C.shape[0]))
    for level in range(num_levels):
        B_level = B[:,ranks[:level].sum():ranks[:level+1].sum()]
        C_level = C[:,ranks[:level].sum():ranks[:level+1].sum()]
        num_blocks = len(rows_lk[level])-1
        for block in range(num_blocks):
            r1, r2 = rows_lk[level][block], rows_lk[level][block+1]
            c1, c2 = cols_lk[level][block], cols_lk[level][block+1]
            hat_A[r1:r2, c1:c2] += np.dot(B_level[r1:r2], C_level[c1:c2].T) 
    return hat_A


@nb.jit(nopython=True)
def compute_perm_residual_jit(perm_A:np.ndarray, B:np.ndarray, C:np.ndarray, cur_level:int, \
                              rows_lk,  cols_lk, ranks:np.ndarray):
    """
    Compute permuted residual for a given level cur_level
    """
    hat_A_except_level = np.zeros((B.shape[0], C.shape[0]))
    num_levels = ranks.size 
    list_levels = np.empty((num_levels-1,), dtype=nb.types.uint32)
    list_levels[:cur_level] = np.arange(cur_level)
    list_levels[cur_level:] = np.arange(cur_level+1, num_levels)
    for level in list_levels:
        B_level = B[:,ranks[:level].sum():ranks[:level+1].sum()]
        C_level = C[:,ranks[:level].sum():ranks[:level+1].sum()]
        num_blocks = len(rows_lk[level])-1
        for block in range(num_blocks):
            r1, r2 = rows_lk[level][block], rows_lk[level][block+1]
            c1, c2 = cols_lk[level][block], cols_lk[level][block+1]
            hat_A_except_level[r1:r2, c1:c2] += np.dot(B_level[r1:r2], C_level[c1:c2].T)
    return perm_A - hat_A_except_level

@nb.jit(nopython=True, parallel=True)
def compute_perm_residual_jit_parallel(perm_A:np.ndarray, B:np.ndarray, C:np.ndarray, cur_level:int, \
                              rows_lk,  cols_lk, ranks:np.ndarray):
    """
    Compute permuted residual for a given level cur_level
    """
    hat_A_except_level = np.zeros((B.shape[0], C.shape[0]))
    num_levels = ranks.size 
    list_levels = np.empty((num_levels-1,), dtype=nb.types.uint32)
    list_levels[:cur_level] = np.arange(cur_level)
    list_levels[cur_level:] = np.arange(cur_level+1, num_levels)
    for l in nb.prange(num_levels-1):
        level = list_levels[l]
        B_level = B[:,ranks[:level].sum():ranks[:level+1].sum()]
        C_level = C[:,ranks[:level].sum():ranks[:level+1].sum()]
        num_blocks = len(rows_lk[level])-1
        for block in range(num_blocks):
            r1, r2 = rows_lk[level][block], rows_lk[level][block+1]
            c1, c2 = cols_lk[level][block], cols_lk[level][block+1]
            hat_A_except_level[r1:r2, c1:c2] += np.dot(B_level[r1:r2], C_level[c1:c2].T)
    return perm_A - hat_A_except_level


@torch.jit.ignore
def blockdiag_matvec_nbmm(nested_B_level:torch.Tensor, nested_Ct_level:torch.Tensor, parts:torch.Tensor, x:torch.Tensor, level:int):
    nested_x = torch.nested.nested_tensor(list(torch.tensor_split(x, parts[level][1:-1], dim=0)))
    nested_z = torch.bmm(nested_Ct_level, nested_x)
    nested_y = torch.bmm(nested_B_level, nested_z)
    return torch.cat(nested_y.unbind(), axis=0)


@torch.jit.script
def jit_parallel_matvec_nbmm(nested_B:List[torch.Tensor], nested_Ct:List[torch.Tensor], parts:torch.Tensor, x:torch.Tensor):
    futures : List[torch.jit.Future[torch.Tensor]] = []
    for level in range(parts.size(0)):
        futures.append(torch.jit.fork(blockdiag_matvec_nbmm, nested_B[level], nested_Ct[level], parts, x, level))
    results = []
    for future in futures:
        results.append(torch.jit.wait(future))
    return torch.sum(torch.stack(results, dim=0), dim=0)



def blockdiag_matvec_sp(tilde_B_level:torch.Tensor, tilde_Ct_level:torch.Tensor, x:torch.Tensor):
    z_l = torch.mv(tilde_Ct_level, x)
    y_l = torch.mv(tilde_B_level, z_l)
    return y_l


@torch.jit.script
def jit_parallel_matvec_sp(tilde_B_level:List[torch.Tensor], tilde_Ct_level:List[torch.Tensor], x:torch.Tensor, num_levels:int):
    futures : List[torch.jit.Future[torch.Tensor]] = []
    for level in range(num_levels):
        futures.append(torch.jit.fork(blockdiag_matvec_sp, tilde_B_level[level], tilde_Ct_level[level], x))
    results = []
    for future in futures:
        results.append(torch.jit.wait(future))
    return torch.sum(torch.stack(results, dim=0), dim=0)
    

def nojit_blockdiag_matvec_nbmm(nested_B_level:torch.Tensor, nested_Ct_level:torch.Tensor, parts:torch.Tensor, x:torch.Tensor, level:int):
    nested_x = torch.nested.nested_tensor(list(torch.tensor_split(x, parts[level][1:-1], dim=0)))
    nested_z = torch.bmm(nested_Ct_level, nested_x)
    nested_y = torch.bmm(nested_B_level, nested_z)
    return torch.cat(nested_y.unbind(), axis=0)


def nojit_parallel_matvec_nbmm(nested_B:List[torch.Tensor], nested_Ct:List[torch.Tensor], parts:torch.Tensor, x:torch.Tensor):
    results = []
    for level in range(parts.size(0)):
        y_level = nojit_blockdiag_matvec_nbmm(nested_B[level], nested_Ct[level], parts, x, level)
        results.append(y_level)
    return torch.sum(torch.stack(results, dim=0), dim=0)


def nj_parallel_matvec_nbmm(hat_A, x):
    perm_y = nojit_parallel_matvec_nbmm(hat_A.nested_B, hat_A.nested_Ct, hat_A.nested_hpart, x[hat_A.pi_cols])
    return perm_y[hat_A.pi_inv_rows]


def torch_sparse_blockdiag(M:torch.Tensor, hp_entry:EntryHpartDict, ranks:np.ndarray, level:int, transpose=False) -> torch.Tensor:
    blocks = []
    num_blocks = len(hp_entry['lk'][level])-1
    if transpose:
        M_level = M[ranks[:level].sum():ranks[:level+1].sum(), :]
    else:
        M_level = M[:, ranks[:level].sum():ranks[:level+1].sum()]
    for block in range(num_blocks):
        r1, r2 = hp_entry['lk'][level][block], hp_entry['lk'][level][block+1]
        if transpose:
            blocks += [ M_level[:, r1:r2] ]
        else:
            blocks += [ M_level[r1:r2] ]
    return torch.block_diag(*blocks).to_sparse_csr().to(get_device(0))
