import numpy as np
import random

import mlrfit as mf


def main():
    np.random.seed(1001)
    random.seed(1001)

    M = 20
    m, n  = 200, 175
    dim = 10

    # Test rank alloc implementation
    num_levels = min(int(np.log2(m)), int(np.log2(n)))
    eps_ff = 1e-3
    ks = np.array([2]*(num_levels-1))
    func_partition1 = mf.spectral_partition(symm=False, refined=True, max_iters=200)
    func_partition2 = mf.spectral_partition(symm=False, refined=False)
    hat_A = mf.MLRMatrix(debug=True)
    for _ in range(M):
        ranks = np.random.randint(1, dim, num_levels)
        A = np.random.randn(m,n)
        # top-down hier. partitioning + factor fitting
        hat_A.hpartition_topdown(A, ranks, ks, func_partition1, eps_ff=eps_ff, symm=False, \
                                    PSD=False, max_iters_ff=1000)
        loss_ff = mf.rel_diff(hat_A.matrix(), den=A)
        # top-down hier. partitioning + rank fitting
        hpartition, H = mf.hpartition_topdown(A, ranks, ks, func_partition1, debug=True)
        loss_rf = mf.rel_diff(H, den=A)
        print(f"refined   {loss_ff=}, {loss_rf=}")
        assert loss_ff - 1e-9 <=  loss_rf
        # top-down hier. partitioning + factor fitting
        hat_A.hpartition_topdown(A, ranks, ks, func_partition2, eps_ff=eps_ff, symm=False, \
                                    PSD=False, max_iters_ff=1000)
        loss_ff = mf.rel_diff(hat_A.matrix(), den=A)
        # top-down hier. partitioning + rank fitting
        hpartition, H = mf.hpartition_topdown(A, ranks, ks, func_partition2, debug=True)
        loss_rf = mf.rel_diff(H, den=A)
        print(f"unrefined {loss_ff=}, {loss_rf=}")
        
    print("PASSED hat_A.hpartition_topdown+ff implementation tests")



if __name__ == '__main__':
    main()