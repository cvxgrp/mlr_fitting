import copy
from typing import List, Tuple, Callable, TypedDict, List, Set, Optional 

import numpy as np
import numba as nb
from scipy.linalg import block_diag
from scipy.sparse import bmat
from scipy.sparse.linalg import splu, LinearOperator, lsmr

import torch

from mlrfit.utils import *
from mlrfit.fit import *
from mlrfit.hpartitioning import *


import scipy
import pypardiso



"""
Multilevel Low Rank matrix class
"""
class MLRMatrix:
    def __init__(self, hpart:Optional[HpartDict]=None, ranks:Optional[np.ndarray]=None, \
                       B:Optional[np.ndarray]=None, C:Optional[np.ndarray]=None, debug=False, tracemem=False):
        """
        Use provided A_levels and BC_levels as a warm start
        hpartition: dict
            {'rows':{'pi':np.ndarray(m), 'lk':List[List[int]]},
             'cols':{'pi':np.ndarray(m), 'lk':List[List[int]]},}
            rows/cols hierarchical partitioning containing
                block segments for every level
        B: np.ndarray(m, r) 
            compressed format of B_lk
        C: np.ndarray(n, r) 
            compressed format of C_lk
        """
        self.B = B
        self.C = C
        self.ranks = ranks
        self.debug = debug
        self.tracemem = tracemem
        if hpart is not None:
            self._update_hpart(hpart)


    def _update_hpart(self, hpart:HpartDict):
        self.hpart = hpart
        self.pi_rows = hpart['rows']['pi']
        self.pi_cols = hpart['cols']['pi']
        self.pi_inv_rows, self.pi_inv_cols = inv_permutation(self.pi_rows, self.pi_cols)


    def matrix(self):
        perm_hat_A = self._compute_perm_hat_A(self.B, self.C, self.hpart, self.ranks)
        # pi_inv to permute \hat A_l from block diagonal in order approximating A
        hat_A = perm_hat_A[self.pi_inv_rows, :][:, self.pi_inv_cols]
        return hat_A
    

    def shape(self):
        return (self.B.shape[0], self.C.shape[0])


    def _compute_perm_hat_A(self, B:np.ndarray, C:np.ndarray, hpart:HpartDict, ranks:np.ndarray):
        """
        Compute permuted hat_A with each A_level being block diagonal matrix 
        """
        num_levels = ranks.size
        if torch.is_tensor(B):
            perm_hat_A = torch.zeros((B.shape[0], C.shape[0])).to(get_device(0))
        else:
            perm_hat_A = np.zeros((B.shape[0], C.shape[0]))
        for level in range(num_levels):
            perm_hat_A += self._block_diag_BCt(level, hpart, B[:,ranks[:level].sum():ranks[:level+1].sum()], \
                                                        C[:,ranks[:level].sum():ranks[:level+1].sum()])
        return perm_hat_A

    
    def _block_diag_BCt(self, level:int, hpart:HpartDict, B_level:np.ndarray, C_level:np.ndarray):
        A_level = []
        num_blocks = len(hpart['rows']['lk'][level])-1
        for block in range(num_blocks):
            r1, r2 = hpart['rows']['lk'][level][block], hpart['rows']['lk'][level][block+1]
            c1, c2 = hpart['cols']['lk'][level][block], hpart['cols']['lk'][level][block+1]
            A_level += [ B_level[r1:r2] @ C_level[c1:c2].T ]
        if torch.is_tensor(B_level):
            return torch.block_diag(*A_level)
        else:
            return block_diag(*A_level)


    def _compute_perm_residual(self, perm_A:np.ndarray, B:np.ndarray, C:np.ndarray, cur_level:int, \
                                    hpart:HpartDict, ranks:np.ndarray):
        """
        Compute permuted residual for a given level cur_level
        """
        hat_A_except_level = 0
        num_levels = ranks.size
        for level in range(num_levels):
            if level == cur_level: continue
            hat_A_except_level += self._block_diag_BCt(level, hpart, B[:,ranks[:level].sum():ranks[:level+1].sum()], \
                                                        C[:,ranks[:level].sum():ranks[:level+1].sum()])
        R = perm_A - hat_A_except_level
        return R


    def init_B_C(self, ranks:np.ndarray, hpart:HpartDict, init_type='bcd', perm_A=None, params=False):
        """
        Initialize B and C given ranks and hpartition
        """
        m, n = hpart['rows']['pi'].size, hpart['cols']['pi'].size
        r = ranks.sum()
        if init_type == 'zeros':
            B = np.zeros((m, r))
            C = np.zeros((n, r))
        elif init_type == 'random':
            B = np.random.randn(m, r)
            C = np.random.randn(n, r)
        elif init_type == 'bcd':
            B = np.zeros((m, r))
            C = np.zeros((n, r))
            B, C = self._single_epoch_bcd(perm_A, B, C, ranks, np.arange(ranks.size), params, return_delta=False)
        return B, C


    def factor_fit(self, A:np.ndarray, ranks:np.ndarray, hpart:HpartDict, method='bcd', order='tbu', \
                        eps_ff=0.01, symm=False, PSD=False, svds_v0=False, max_iters_ff=100, \
                        freq=1000, warm_start=False, printing=False, epoch_len=10, init_type='bcd'):
        """
        Factor fitting using Block Coordinate Descent (BCD) or Alternating Least
        Squares (ALS)
        """
        if warm_start:
            B0, C0 = self.B, self.C
        else:
            B0, C0 = None, None
        self.ranks = ranks
        self._update_hpart(hpart)
        # permute rows and cols to place partitions on the diagonal
        perm_A = A[self.pi_rows, :][:, self.pi_cols]
        if 'bcd' in method:
            B, C, losses = self._factor_fit_bcd(perm_A, eps_ff=eps_ff, order=order, \
                                                 B=B0, C=C0, svds_v0=svds_v0, PSD=PSD, symm=symm, \
                                                 printing=printing, freq=freq, init_type=init_type, \
                                                 max_iters_ff=max_iters_ff, epoch_len=epoch_len)[:3]
        elif 'als' in method:
            B, C, losses  = self._factor_fit_als(perm_A, eps_ff=eps_ff, B=B0, C=C0, PSD=PSD, symm=symm, \
                                                 printing=printing, freq=freq, init_type=init_type,\
                                                 max_iters_ff=max_iters_ff, epoch_len=epoch_len)[:3]
        self.B, self.C = B, C
        del perm_A
        return losses


    def _single_epoch_bcd(self, perm_A, B, C, ranks, levels, params, return_delta):
        symm, PSD, svds_v0 = params
        num_levels = ranks.size
        if return_delta:
            delta_rm1, delta_rp1 = np.zeros(num_levels), np.zeros(num_levels)
            b, c = np.zeros((perm_A.shape[0], num_levels)), np.zeros((perm_A.shape[1], num_levels))
        # sweep over the hierarchy in the given order
        for level in levels:
            # approximate block diagonals on level
            R = compute_perm_residual_jit(perm_A, B, C, level, nb.typed.List(self.hpart['rows']['lk']),\
                                                        nb.typed.List(self.hpart['cols']['lk']), ranks)
            B_level, C_level, delta_rm1_lev, delta_rp1_lev, b_level, c_level = \
                    single_level_factor_fit(R, ranks, self.hpart, level, svds_v0=svds_v0, \
                                            symm=symm, PSD=PSD, \
                                            B_level=B[:,ranks[:level].sum():ranks[:level+1].sum()], \
                                            C_level=C[:,ranks[:level].sum():ranks[:level+1].sum()])
            B[:,ranks[:level].sum():ranks[:level+1].sum()] = B_level
            C[:,ranks[:level].sum():ranks[:level+1].sum()] = C_level
            if return_delta:
                b[:,level:level+1] = b_level
                c[:,level:level+1] = c_level
                delta_rm1[level] = delta_rm1_lev
                delta_rp1[level] = delta_rp1_lev
            del R
        if return_delta:
            return B, C, delta_rm1, delta_rp1, b, c
        else:
            return B,  C
        

    def _factor_fit_bcd(self, perm_A:np.ndarray, ranks:Optional[np.ndarray]=None, eps_ff=0.01, \
                    B=None, C=None, PSD=False, svds_v0=False, symm=False, freq=1000, \
                    max_iters_ff=10**2, printing=False, epoch_len=10, order='tbu', init_type='bcd'):
        """
        * REQUIRES SELF.HPARTITION set *
        Fit MLR matrix to a given rank allocation and hierarchical partitioning
        using block coordinate descent
        hpartition: HpartDict
        Returns the scores for rank update in each level:
            delta_rm1: 
                scores for decreasing a rank by 1 
            delta_rp1: 
                scores for increasig a rank by 1
        the loss delta for increasing i-th level rank and decreasing j-th level
        is                  delta = delta_rm1[j] - delta_rp1[i]
        the loss is improving when delta <= 0
        """
        if ranks is None:
            ranks = self.ranks
        num_levels = len(ranks)
        params = symm, PSD, svds_v0
        # fit A with warm start As by sweeping over the hierarhy tree in the given order
        if order == 'tbu':   levels = np.concatenate([np.arange(num_levels), np.arange(num_levels-1)[::-1]], axis=0)
        elif order == 'td':  levels = np.arange(num_levels)
        elif order == 'bu':  levels = np.arange(num_levels)[::-1]

        itr = 0; losses_ff = []
        rows_lk = nb.typed.List(self.hpart['rows']['lk'])
        cols_lk = nb.typed.List(self.hpart['cols']['lk'])
        if B is None: 
            B, C = self.init_B_C(ranks, self.hpart, init_type=init_type, perm_A=perm_A, params=params)
            # compute initial loss 
            perm_hat_A = compute_perm_hat_A_jit(B, C, rows_lk, cols_lk, ranks)
            losses_ff = [rel_diff(perm_hat_A, den=perm_A)]
            if itr % freq == 0 and printing:
                print(f"{itr=}, {losses_ff[-1]}, {ranks}")
            itr += 1
            del perm_hat_A
        else: 
            B, C = copy.deepcopy(B), copy.deepcopy(C) 

        if itr % freq == 0 and printing:
            print(f"{itr=}, {losses_ff[-1]}, {ranks}")

        while itr < max_iters_ff:
            start_time = time.time()
            # single V epoch of BCD
            B, C = self._single_epoch_bcd(perm_A, B, C, ranks, levels, params, return_delta=False)
            time_v_epoch = time.time() - start_time
            start_time = time.time()
            # compute loss at the end of every epoch
            perm_hat_A = compute_perm_hat_A_jit(B, C, rows_lk, cols_lk, ranks)
            losses_ff += [rel_diff(perm_hat_A, den=perm_A)]
            del perm_hat_A
            time_loss = time.time() - start_time

            if itr % freq == 0 and printing:
                print(f"{itr=}, {losses_ff[-1]}, {ranks}, {time_v_epoch=}, {time_loss=}")
            if  itr >= 1 or num_levels==1:
                if self.debug: 
                    assert num_levels==1 or losses_ff[-2] - losses_ff[-1] >= -1e-9
                if num_levels==1 or losses_ff[-2]-losses_ff[-1] < eps_ff*losses_ff[-2]: 
                    break
            itr += 1 
        B, C, delta_rm1, delta_rp1, b, c = self._single_epoch_bcd(perm_A, B, C, ranks, np.arange(num_levels), \
                                                                  params, return_delta=True)
        perm_hat_A = compute_perm_hat_A_jit(B, C, rows_lk, cols_lk, ranks)
        losses_ff += [rel_diff(perm_hat_A, den=perm_A)]
        del perm_hat_A
        return B, C, losses_ff, delta_rm1, delta_rp1, b, c
    

    def _factor_fit_als(self, perm_A:np.ndarray, ranks:Optional[np.ndarray]=None, eps_ff=0.01,
                    B=None, C=None, PSD=False, symm=False, freq=1000, svds_v0=False, \
                    max_iters_ff=10**2, printing=False, epoch_len=10, init_type='bcd'):
        """
        * REQUIRES SELF.HPARTITION set *
        Fit MLR matrix to a given rank allocation and hierarchical partitioning
        using alternating least squares
        hpartition: list of dictionaries contatining rows and cols partitioning
                    for every level
                    [{ 'rows': [row partitioning], 
                       'cols': [column partitioning]}] * num_levels
        """
        if ranks is None:
            ranks = self.ranks
        torch_perm_A = torch.from_numpy(perm_A).to(get_device(0))
        normalization = np.linalg.norm(perm_A, ord='fro')
        params = symm, PSD, svds_v0
        itr = 0; losses_ff = []
        rows_lk = nb.typed.List(self.hpart['rows']['lk'])
        cols_lk = nb.typed.List(self.hpart['cols']['lk'])
        if B is None: 
            B, C = self.init_B_C(ranks, self.hpart, init_type=init_type, perm_A=perm_A, params=params)
            # compute initial loss 
            perm_hat_A = compute_perm_hat_A_jit(B, C, rows_lk, cols_lk, ranks)
            losses_ff = [rel_diff(perm_hat_A, den=perm_A)]
            if itr % freq == 0 and printing:
                print(f"{itr=}, {losses_ff[-1]}, {ranks}")
            itr += 1
            del perm_hat_A
        else: 
            B, C = copy.deepcopy(B), copy.deepcopy(C) 
        
        while itr < max_iters_ff:
            start_time = time.time()
            # alternate to minimization over B and then over C
            for update in [0, 1]: 
                B, C, losses, torch_perm_hat_A = als_factor_fit_alt_cg(torch_perm_A, B, C, self.hpart, ranks, \
                                                        epoch_len=epoch_len, update=update, normalization=normalization)
                if not self.debug:
                    del torch_perm_hat_A
            losses_ff += losses 
            time_epoch = time.time() - start_time
            epoch_len = len(losses) if len(losses)>=2 else 2
            if itr % freq == 0 and printing:
                print(f"{itr=}, {losses_ff[-1]}, {ranks}, {time_epoch=}")
            if  itr >= 1:
                if self.debug: 
                    assert np.allclose(rel_diff(torch_perm_hat_A, den=torch_perm_A).item(), losses[-1])
                    assert len(losses_ff)<epoch_len+1 or len(losses_ff)>=epoch_len+1 \
                            and losses_ff[-1-epoch_len] - losses_ff[-1] >= -1e-9
                if len(losses_ff)>=epoch_len+1 and \
                    0 <= losses_ff[-1-epoch_len] - losses_ff[-1] < eps_ff*losses_ff[-1-epoch_len]: 
                    break
            itr += 1 
        B, C = torch_to_numpy(B), torch_to_numpy(C)
        B, C, delta_rm1, delta_rp1, b, c = self._single_epoch_bcd(perm_A, B, C, ranks, np.arange(ranks.size), \
                                                                                    params, return_delta=True)
        perm_hat_A = compute_perm_hat_A_jit(B, C, rows_lk, cols_lk, ranks)
        losses_ff += [rel_diff(perm_hat_A, den=perm_A)]
        del perm_hat_A
        return B, C, losses_ff, delta_rm1, delta_rp1, b, c

    
    def _update_B_C_rank_alloc(self, B:np.ndarray, C:np.ndarray, b:np.ndarray, \
                                        c:np.ndarray, i_plus:int, j_minus:int, ranks:np.ndarray):
        """
        Update B and C with vectors b, c by increasing rank on level i_plus
        and by decreasing rank on level j_minus 
            add rank to level i_plus
            (m x r_i) --> (m x (r_i+1))
            remove rank from level j_minus
            (m x r_j) --> (m x (r_j-1))
        """
        # organized in decreasing order of singular values 
        # sigma_1 >= ... >= sigma_r
        if j_minus < i_plus:
            newB = update_B_rank_alloc_ji(B, b, i_plus, j_minus, ranks)
            newC = update_B_rank_alloc_ji(C, c, i_plus, j_minus, ranks)
        elif i_plus < j_minus:
            newB = update_B_rank_alloc_ij(B, b, i_plus, j_minus, ranks)
            newC = update_B_rank_alloc_ij(C, c, i_plus, j_minus, ranks)
        return newB, newC


    def rank_alloc(self, A:np.ndarray, ranks:np.ndarray, hpart:HpartDict, printing=True, \
                        method='bcd', eps=0.001, eps_ff=0.01, max_iters_ff=2, max_iters=10**3, \
                        symm=False, PSD=False, warm_start=False, freq=1000, top_k=3):
        """
        Fit MLR matrix for given hierarchical partitioning using block
        coordinate descent and a rank allocation
            hpart: dict
                {'rows':{'pi':np.ndarray(m), 'lk':List[List[int]]},
                'cols':{'pi':np.ndarray(m), 'lk':List[List[int]]},}
                rows/cols hierarchical partitioning containing
                    block segments for every level
        Returns the scores for rank update in each level:
            delta_rm1: 
                    scores for decreasing a rank by 1 
            delta_rp1: 
                    scores for increasig a rank by 1
            the loss delta for increasing i-th level rank and decreasing j-th level
            is                  delta = delta_rm1[j] - delta_rp1[i]
            the loss is improving when delta <= 0
        """
        num_levels = ranks.size
        self.ranks = ranks
        self._update_hpart(hpart)
        # permute rows and cols to place partitions on the diagonal
        perm_A = A[self.pi_rows, :][:, self.pi_cols]
        losses = []
        if warm_start: 
            B0, C0 = self.B, self.C
            # compute initial loss 
            perm_hat_A = compute_perm_hat_A_jit(B0, C0, nb.typed.List(self.hpart['rows']['lk']), \
                                                nb.typed.List(self.hpart['cols']['lk']), ranks)
            losses = [rel_diff(perm_hat_A, den=perm_A)]
            del perm_hat_A
        else: B0, C0 = None, None

        if 'bcd' in method:
            optim_func = self._factor_fit_bcd
        elif 'als' in method:
            optim_func = self._factor_fit_als
        B, C, losses0, delta_rm1, delta_rp1, b, c = optim_func(perm_A, eps_ff=eps_ff, B=B0, C=C0, \
                                                              PSD=PSD, max_iters_ff=max_iters_ff, symm=symm)
        losses += losses0
        itr = 0; epochs = [0, len(losses)]
        if printing: print(f"{itr=}, t={epochs[-1]}, {losses[0]=}, {losses[-1]=}, {ranks}")
        itr += 1
        ranks_history = [ranks]
        if self.tracemem:
            snapshot1 = tracemalloc.take_snapshot()
            if printing: print("RA: Before while loop")
            display_top(snapshot1, limit=20)
        top_k = min(top_k, num_levels * (num_levels - 1))
        while itr < max_iters:
            improved_rank = False
            k = 0
            min_deltas, is_plus, js_minus = ra_delta_obj(delta_rp1, delta_rm1, num_levels, top_k)
            infs = np.where(min_deltas == np.inf)[0]
            if infs.size != 0:
                min_deltas = min_deltas[:infs[0]]
                assert np.inf not in min_deltas
                is_plus = is_plus[:infs[0]]
                js_minus = js_minus[:infs[0]]
                top_k = min(top_k, min_deltas.size)
            while k < top_k and not improved_rank:
                i_plus = is_plus[k]
                j_minus = js_minus[k]
                if ranks[j_minus] > 0:
                    # try new rank arrangement
                    ranks_new = copy.deepcopy(ranks)
                    ranks_new[i_plus] += 1; ranks_new[j_minus] -= 1
                    B_new_rank0, C_new_rank0 = self._update_B_C_rank_alloc(B, C, b, c, i_plus, j_minus, ranks)
                    # start_time = time.time()
                    B_new_rank, C_new_rank, losses_new_rank, delta_rm1, delta_rp1, b, c = \
                                    optim_func(perm_A, ranks_new, B=B_new_rank0, \
                                                    C=C_new_rank0, PSD=PSD, 
                                                    eps_ff=eps_ff, max_iters_ff=max_iters_ff, symm=symm)
                    # print(f"factor fit total time={time.time()-start_time}")
                    improved_rank = losses_new_rank[-1] < losses[-1]
                k += 1
            ## accept new rank alloc if there is improvement in the new rank allocation
            if improved_rank: 
                ranks = ranks_new
                ranks_history += [ranks]
                B = B_new_rank
                C = C_new_rank
                losses += losses_new_rank
                epochs += [len(losses)]
            else: # stop if there is no progress in the new rank allocation
                if printing: print("quit: new rank allocation is worse")
                break
            if itr % freq == 0:
                if printing: print(f"{itr=}, t={epochs[-1]}, {losses[-1]}, {ranks}")
            itr += 1
            if itr >= 2: # compare the error difference between two epochs
                epoch_len = epochs[-1] - epochs[-2]
                if losses[-1-epoch_len] - losses[-1] < eps*losses[-1-epoch_len]: 
                    if printing: print(f"quit: {losses[-1-epoch_len] - losses[-1]=}, {eps*losses[-1-epoch_len]=}")
                    break 
        ## extra factor fit on finalized rank allocation
        B, C, losses_ff, _, _, _, _ = optim_func(perm_A, ranks, \
                                                    eps_ff=eps_ff, B=B, C=C, PSD=PSD, \
                                                    max_iters_ff=max_iters, symm=symm)
        if self.tracemem:
            snapshot1 = tracemalloc.take_snapshot()
            print("RA: After while loop")
            display_top(snapshot1, limit=20)

        if self.debug: assert losses_ff[-1] - 1e-9 <= losses[-1]
        losses += losses_ff
        epochs += [len(losses)]
        if printing: print(f"{itr=}, t={epochs[-1]}, {losses[-1]}, {ranks}")
        self.B, self.C = B, C
        self.ranks = ranks
        del perm_A
        return np.array(losses), np.array(epochs), ranks_history


    def hpartition_topdown(self, A:np.ndarray, ranks:np.ndarray, ks:Optional[np.ndarray]=None, \
                            func_partition:Optional[Callable]=None, printing=True, \
                            method='bcd', eps_ff=1e-2, symm=False, balanced=True, \
                            PSD=False, max_iters_ff=1, grref_max_iters=5000):
        """
        Top-down hier. partitioning that uses factor fit
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
        m, n = A.shape
        tracemalloc.start()
        num_levels = ranks.size
        if ks is None:
            ks = (num_levels-2)*[2] + [np.inf]
        else:
            assert len(ks) == len(ranks) - 1
        if func_partition is None:
            func_partition = spectral_partition(symm=symm, refined=True, balanced=balanced, max_iters=grref_max_iters)

        htree = [{'rows':[np.arange(m)], 'cols':[np.arange(n)]}]
        # store absolute permutation of rows/cols that makes block diagonal
        perm_rows = np.array(range(m))
        perm_cols = np.array(range(n))
        hpart = full_htree_to_hpart(htree)
        self.B, self.C = self.init_B_C(ranks[:1], hpart, init_type='zeros')
        prev_loss = np.inf
        all_losses = []; epochs = []

        if self.tracemem:
            snapshot1 = tracemalloc.take_snapshot()
            top_stats = snapshot1.statistics('lineno')
            print("Before for loop")
            display_top(snapshot1, limit=20)

        for level in range(num_levels):
            # factor fit
            losses = self.factor_fit(A, ranks[:level+1], hpart, method=method, \
                                    eps_ff=eps_ff, warm_start=True, PSD=PSD, \
                                    symm=symm, max_iters_ff=max_iters_ff)
            all_losses += losses 
            epochs += [len(all_losses)]
            # current residual
            R = A - self.matrix()
            if printing: print(f"* {level=}, {losses[0]=:.3f}, {losses[-1]=:.3f}, {len(losses)=}, {ranks[:level+1]}")
            if self.debug:
                # sub group permutations implemented correctly -- reconstruct loss
                assert (prev_loss + 1e-9 >= losses[-1]) # and (level==0 or np.allclose(prev_loss, losses[0]))
                # the same because leaf level B_l, C_l are initialized as 0
                assert np.allclose(rel_diff(self.matrix(), den=A), losses[-1])

            prev_loss = losses[-1]
            if level == num_levels-1: 
                if printing: print(f"{level=}, loss={all_losses[-1]}")
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
                # print(f"{num_blocks=}, {rows.size=}, {cols.size=}")
                assert not balanced or rows.size == min(m, n) + 1 and cols.size == min(m, n) + 1, \
                    print(rows.size, min(m, n) + 1, cols.size, min(m, n) + 1, set(range(m)).difference(rows))
                assert not balanced or len(set(rows)) == min(m, n) + 1 and len(set(cols)) == min(m, n) + 1
                hpart['rows']['lk'] += [ rows ]
                hpart['cols']['lk'] += [ cols ]
            else:
                # permute residual in same order as to make \hat A contiguous MLR
                # perm_R = R[perm_rows, :][:, perm_cols]
                # del R
                num_blocks = len(htree[level]['rows'])
                r1, c1 = 0, 0
                htree += [{'rows':[], 'cols':[]}]
                new_perm_rows = np.zeros(m).astype(int)
                new_perm_cols = np.zeros(n).astype(int)
                count1 = 0; count2 = 0
                # find permutations of each block using given partition function
                for block in range(num_blocks):
                    r2 = r1 + htree[level]['rows'][block].size
                    c2 = c1 + htree[level]['cols'][block].size 
                    # permute rows and columns according to given partition function for a given block 
                    # of squared residual matrix
                    pi_rows_bl, sep_r_bl, pi_cols_bl, sep_c_bl = func_partition(np.square(R[perm_rows, :][:, perm_cols][r1:r2,c1:c2]), k=k)
                    # del perm_R
                    num_partitions = len(sep_r_bl)-1
                    # add a level to a hierarchy
                    for si in range(num_partitions):
                        # permute rows and cols: put partitions on the diagonal
                        new_perm_rows[r1:r2][sep_r_bl[si]:sep_r_bl[si+1]] = perm_rows[r1:r2][pi_rows_bl[sep_r_bl[si]:sep_r_bl[si+1]]]
                        new_perm_cols[c1:c2][sep_c_bl[si]:sep_c_bl[si+1]] = perm_cols[c1:c2][pi_cols_bl[sep_c_bl[si]:sep_c_bl[si+1]]]
                        htree[level+1]['rows'] += [new_perm_rows[r1:r2][sep_r_bl[si]:sep_r_bl[si+1]]]
                        htree[level+1]['cols'] += [new_perm_cols[c1:c2][sep_c_bl[si]:sep_c_bl[si+1]]]
                    count1 += sep_r_bl[-1]
                    count2 += sep_c_bl[-1]
                    r1, c1 = r2, c2
                assert count1 == m and count2 == n, print(f"{m=}, {count1=}, {n=}, {count2=}")
                perm_rows = new_perm_rows
                perm_cols = new_perm_cols
                # permute the htree, B and C to respect the perm_rows/cols on the leaf level
                htree, B, C = update_htree_B_C_perm_leaf(htree, perm_rows, perm_cols, ranks[:level+1], self.B, self.C, self.debug)
                self.B = B; self.C = C
                hpart = full_htree_to_hpart(htree)
            self._update_hpart(hpart)
            # add new level to B and C
            self.B = np.concatenate([self.B, np.zeros((m, ranks[level+1]))], axis=1)
            self.C = np.concatenate([self.C, np.zeros((n, ranks[level+1]))], axis=1)

            if self.tracemem:
                snapshot2 = tracemalloc.take_snapshot()
                top_stats = snapshot2.compare_to(snapshot1, 'lineno')
                print("[ Top 10 differences ]")
                for stat in top_stats[:10]:
                    print(stat)
                snapshot1 = snapshot2
        if self.tracemem:
            snapshot1 = tracemalloc.take_snapshot()
            top_stats = snapshot1.statistics('lineno')
            if printing: print("After for loop")
            display_top(snapshot1, limit=20)

        self.hpart = hpart
        self.ranks = ranks
        return  np.array(all_losses), np.array(epochs)
    

    def construct_torch_format(self):
        if self.B is None:
            self.B, self.C = self.init_B_C(self.ranks, self.hpart, init_type='random')
        self.nested_B = []
        self.nested_Ct = []
        self.list_torch_Bs = []
        self.list_torch_Cts = []
        self.torch_tilde_B_level = []
        self.torch_tilde_Ct_level = []
        torch_B = torch.from_numpy(self.B).to(get_device(0))
        torch_Ct = torch.from_numpy(self.C.T).to(get_device(0))
        for level in range(len(self.hpart["rows"]["lk"])):
            B_level =  torch_B[:, self.ranks[:level].sum():self.ranks[:level+1].sum()]
            Ct_level = torch_Ct[self.ranks[:level].sum():self.ranks[:level+1].sum(), :]
            # self.nested_B += [torch.nested.nested_tensor(list(torch.tensor_split(B_level, \
            #                                             list(self.hpart["rows"]["lk"][level][1: -1]), dim=0)))]
            # self.nested_Ct += [torch.nested.nested_tensor(list(torch.tensor_split(Ct_level, \
            #                                             list(self.hpart["cols"]["lk"][level][1: -1]), dim=1)))]
            self.list_torch_Bs += list(torch.tensor_split(B_level, \
                                                        list(self.hpart["rows"]["lk"][level][1: -1]), dim=0))
            self.list_torch_Cts += list(torch.tensor_split(Ct_level, \
                                                        list(self.hpart["cols"]["lk"][level][1: -1]), dim=1))
            
            self.torch_tilde_B_level += [torch_sparse_blockdiag(torch_B, self.hpart["rows"], self.ranks, level, transpose=False)]
            self.torch_tilde_Ct_level += [torch_sparse_blockdiag(torch_Ct, self.hpart["cols"], self.ranks, level, transpose=True)]
            del B_level, Ct_level
        # self.nested_hpart = torch.nested.nested_tensor([torch.from_numpy(el) for el in self.hpart["cols"]["lk"]])


    def parallel_matvec_nbmm(self, x):
        perm_y = jit_parallel_matvec_nbmm(self.nested_B, self.nested_Ct, self.nested_hpart, x[self.pi_cols]) 
        return perm_y[self.pi_inv_rows]
    

    def parallel_matvec_sp(self, x):
        perm_y = jit_parallel_matvec_sp(self.torch_tilde_B_level, self.torch_tilde_Ct_level, x[self.pi_cols].view(-1), len(self.hpart["rows"]["lk"]))
        return perm_y[self.pi_inv_rows]


    def construct_sparse_format(self):
        if self.B is None:
            self.B, self.C = self.init_B_C(self.ranks, self.hpart, init_type='random')
        self.tilde_B = convert_compressed_to_sparse(self.B, self.hpart['rows'], self.ranks)
        self.tilde_Bt = (self.tilde_B.T).tocsc()
        self.tilde_C = convert_compressed_to_sparse(self.C, self.hpart['cols'], self.ranks)
        self.tilde_Ct = (self.tilde_C.T).tocsc()
        self.s = self.tilde_B.shape[1]

        # matrices for linear system solve
        if self.B.shape[0] == self.C.shape[0]:
            self.zeros_s = np.zeros((self.s, 1))
            self.E_solve = scipy.sparse.bmat([[self.tilde_B, None], \
                                   [-scipy.sparse.eye(self.s), self.tilde_Ct]], format='csr')
            
        # matrices for least squares solve, A^T A must be invertible
        if self.B.shape[0] >= self.C.shape[0]:
            self.zeros_sms = np.zeros((2 * self.s + self.B.shape[0], 1))
            self.E_lstsq = scipy.sparse.bmat([[self.tilde_Ct, -scipy.sparse.eye(self.s), None, None], \
                                   [None, self.tilde_B, -scipy.sparse.eye(self.B.shape[0]), None],\
                                   [None, None, self.tilde_Bt, -scipy.sparse.eye(self.s)],\
                                   [None, None, None, self.tilde_C]\
                                    ], format='csr')
 
        # linear operator for least squares
        self.linop = LinearOperator((self.B.shape[0], self.C.shape[0]), \
                                    matvec=self.matvec, rmatvec=self.rmatvec)


    def matvec(self, X):
        # A @ X
        return self.tilde_B.dot(self.tilde_Ct.dot(X[self.pi_cols]))[self.pi_inv_rows]


    def rmatvec(self, X):
        # A.T @ X
        return self.tilde_C.dot(self.tilde_Bt.dot(X[self.pi_rows]))[self.pi_inv_cols]

    
    def solve(self, b):
        eb = np.concatenate([b[self.pi_rows], np.zeros((self.s, 1))], axis=0)
        # yx = splu(self.E).solve(eb)
        yx = pypardiso.spsolve(self.E_solve, eb)
        return yx[self.s:][self.pi_inv_cols]
    
    def lstsq(self, b):
        eb = np.concatenate([self.zeros_sms, self.tilde_C.dot(self.tilde_Bt.dot(b[self.pi_rows]))], axis=0)
        yx = pypardiso.spsolve(self.E_lstsq, eb)
        # yx = scipy.sparse.linalg.splu(self.E_lstsq).solve(eb)
        return yx[:self.C.shape[0]][self.pi_inv_cols]
    