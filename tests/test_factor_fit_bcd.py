import numpy as np
from tqdm import tqdm

import mlrfit as mf



def test_single_level_factor_fit(R, ranks, hpart, level, symm=False):
    """
    Return updated block diagonal, where each block is BCt
    delta_rm1[level]: scores for decreasing a rank by 1
    delta_rp1[level]: scores for increasing a rank by 1
    """
    dim = ranks[level]
    num_blocks = len(hpart['rows']['lk'][level])-1
    r1, c1 = 0, 0
    A_level = np.zeros(R.shape)
    for block in range(num_blocks):
        r1, r2 = hpart['rows']['lk'][level][block], hpart['rows']['lk'][level][block+1]
        c1, c2 = hpart['cols']['lk'][level][block], hpart['cols']['lk'][level][block+1]
        U, Vt, sigmas = mf.frob_low_rank(R[r1:r2, c1:c2], dim = dim+1, symm=symm)
        max_rank_block = min(r2-r1, c2-c1)
        if max_rank_block-1 >= dim >= 1:
            A_level[r1:r2, c1:c2] = U[:, :-1] @ np.diag(sigmas[:-1]) @ Vt[:-1, :] 
        elif dim >= max_rank_block:
            A_level[r1:r2, c1:c2] = U @ np.diag(sigmas) @ Vt
        r1 = r2; c1 = c2 
    return A_level



def main():
    M = 50
    m, n  = 100, 75
    dim = 10

    # Test low rank implementation
    for _ in tqdm(range(M)):
        A = np.random.randn(m, n)*10
        U2, Vt2, sigmas2 = mf.frob_low_rank(A, dim=min(m,n))
        mf.test_eigsh_svds(U2, sigmas2, Vt2, min(m,n), A, mode='svds')
        assert (np.diff(sigmas2) <= 1e-9).all() and (sigmas2 == np.sort(sigmas2)[::-1]).all()
        assert np.allclose(U2 @ np.diag(sigmas2) @ Vt2, A)

    for _ in tqdm(range(M)):
        A = np.random.randn(m, n)*10
        U2, Vt2, sigmas2 = mf.frob_low_rank(A, dim=dim)
        mf.test_eigsh_svds(U2, sigmas2, Vt2, dim, A, mode='svds')
        assert (np.diff(sigmas2) <= 1e-9).all() and (sigmas2 == np.sort(sigmas2)[::-1]).all()

    print("PASSED low rank implementation tests")


    # Test low rank and single level fit implementation
    hpart = mf.random_hpartition(m,  n)
    mf.test_hpartition(hpart, m, n)
    num_levels = len(hpart)
    hat_A = mf.MLRMatrix(hpart=hpart, debug=True)

    for _ in tqdm(range(M)):
        ranks = np.random.randint(1, min(m,n), num_levels)
        R = np.random.randn(m,n)
        for level in range(num_levels):
            B_level, C_level, delta_rm1, delta_rp1, b_level, c_level = mf.single_level_factor_fit(R, ranks, hpart, \
                                                            level)
            A_level = test_single_level_factor_fit(R, ranks, hpart, level)
            assert (delta_rm1 + 1e-9 >= delta_rp1)
            assert np.allclose(A_level, hat_A._block_diag_BCt(level, hpart, B_level, C_level))
    print("PASSED single_level_factor_fit implementation tests")

    # Test low rank and block coordinate descent implementation
    hpart = mf.random_hpartition(m,  n)
    mf.test_hpartition(hpart, m, n)
    eps = 1e-2

    for _ in tqdm(range(M//2)):
        ranks = np.random.randint(1, min(m,n), num_levels)
        A = np.random.randn(m,n)
        
        cycle_size = 1 
        losses = hat_A.factor_fit(A, ranks, hpart, eps_ff=eps, method='bcd',\
                                    max_iters_ff=10**3, symm=False, warm_start=False)
        for i in range(1, len(losses)-cycle_size):
            assert losses[-i-cycle_size] - losses[-i] >= -1e-9, \
                print(f"{i = }, {losses[-i-cycle_size] - losses[-i]}", \
                    "loss is not decreasing with epochs")
        assert eps >= losses[-1-cycle_size] - losses[-1]
        assert np.allclose(mf.rel_diff(hat_A.matrix(), den=A), losses[-1]),\
            print(mf.rel_diff(hat_A.matrix(), den=A), losses[-1])

    print("PASSED factor_fit implementation tests")



if __name__ == '__main__':
    main()