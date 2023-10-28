import numpy as np

import mlrfit as mf



def main():
    M = 20
    m, n  = 100, 75
    dim = 4

    # Test spectral_partition implementation
    fracs = []
    for _ in range(M):
        Ai = np.random.randn(m, n)*10
        for A in [Ai, np.square(Ai)]:
            rows, sep_r, cols, sep_c = mf.spectral_partition(symm=False, debug=True)(A)
            sum_diag = 0
            perm_sum_diag = 0
            perm_A = A[rows, :][:, cols]
            for i in range(len(sep_r)-1):
                block_r = np.arange(sep_r[i], sep_r[i+1])
                block_c = np.arange(sep_c[i], sep_c[i+1])
                sum_diag += (A[block_r, :][:, block_c]).sum()
                perm_sum_diag += (perm_A[block_r, :][:, block_c]).sum()
            # check if permutation increases the sum on the block diag
            fracs += [perm_sum_diag + 1e-9 >= sum_diag]
    ratio = sum(fracs)*1.0 / len(fracs)
    assert ratio > 0.9
    print(f"Diagonal improv. freq = {ratio}")
    fracs = []
    for _ in range(M):
        Ai = np.random.randn(n, n)*10
        A = np.square(Ai + Ai.T)
        perm_sum_diags = {}
        # check if permutation symm has smaller sum than not symmetric
        for symm in [True, False]:
            rows, sep_r, cols, sep_c = mf.spectral_partition(symm=symm, debug=True)(A)
            perm_A = A[rows, :][:, cols]
            perm_sum_diag = 0
            sum_diag = 0
            for i in range(len(sep_r)-1):
                block_r = np.arange(sep_r[i], sep_r[i+1])
                block_c = np.arange(sep_c[i], sep_c[i+1])
                sum_diag += (A[block_r, :][:, block_c]).sum()
                perm_sum_diag += (perm_A[block_r, :][:, block_c]).sum()
            perm_sum_diags[symm] = perm_sum_diag
            # check if permutation increases the sum on the block diag
            fracs += [perm_sum_diag + 1e-9 >= sum_diag]
            
    ratio = sum(fracs)*1.0 / len(fracs)
    assert ratio > 0.9
    print(f"Diagonal improv. freq = {ratio}")
    print("PASSED spectral_partition implementation tests")


    # Test topdown partitioning implementation
    func_partition =  mf.spectral_partition(symm=False, debug=True)
    num_levels = min(int(np.log2(m)), int(np.log2(n)))-1
    ks = np.array([2]*(num_levels-1))
    for _ in range(M):
        ranks = np.random.randint(1, dim, num_levels)
        A = np.random.randn(m,n)
        hpartition, H = mf.hpartition_topdown(A, ranks, ks, func_partition, debug=True)
        mf.test_hpartition(hpartition, m, n)
    print("PASSED hpartition_topdown implementation tests")


    # Test rank alloc implementation
    hpart = mf.random_hpartition(m,  n)
    mf.test_hpartition(hpart, m, n)
    num_levels = len(hpart['rows']['lk'])
    eps = 1e-3
    hat_A = mf.MLRMatrix(hpart=hpart, debug=True)
    for _ in range(M):
        ranks0 = np.random.randint(1, dim, num_levels)
        A = np.random.randn(m,n)
        print(f"{ranks = }")
        for method in ['bcd', 'als']:
            max_iters_ff=2 if method=='bcd' else 4
            losses, epochs, ranks_history = hat_A.rank_alloc(A, ranks0, hpart, method=method, eps=eps,\
                            max_iters=10**3, max_iters_ff=max_iters_ff, symm=False, warm_start=False)
            print(f"{method} tuned: {ranks_history[-1] = }")
            for i in range(1, len(epochs)-1):
                assert losses[-i-1] - losses[-i] >= -1e-6, print("loss is not decreasing with epochs")
            assert eps >= losses[-2] - losses[-1], print(eps, losses[-2] - losses[-1], losses[-5:])
            assert losses[0] > losses[-1]
            assert np.allclose(mf.rel_diff(hat_A.matrix(), den=A), losses[-1]), \
            print(mf.rel_diff(hat_A.matrix(), den=A), losses[-10:])
    print("PASSED rank_alloc implementation tests")



if __name__ == '__main__':
    main()