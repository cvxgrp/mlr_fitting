import numpy as np

import mlrfit as mf


def main():

    M = 5

    # Test greedy heur. refinement
    fracs = {'pi':[], 'ref_pi':[], 'orig':[]}
    for (m,n) in [(200,300), (200,200)]:
        symm = (m==n)
        for _ in range(M):
            Ai = np.random.randn(m, n)
            if symm: Ai = Ai + Ai.T # for Laplacian to be defined in symmetric partitioning
            for A in [np.abs(Ai), np.square(Ai)]:
                rows, sep_r, cols, sep_c = mf.spectral_partition(symm=symm, debug=True)(A)
                sum_diag = 0
                perm_sum_diag = 0
                ref_perm_sum_diag = 0
                perm_A = A[rows, :][:, cols]
                pi_rows2, pi_cols2, obj_all = mf.greedy_heur_refinement(A, rows, sep_r, cols, sep_c, \
                                                                        symm=symm, max_iters=200, debug=True)
                assert np.allclose(obj_all[-1], mf.obj_partition_sum_full(A, pi_rows2, pi_cols2, sep_r, sep_c))
                print(f"itr={len(obj_all)}, {symm=}, % inc. {100*(obj_all[-1]-obj_all[0])/np.abs(obj_all[0]):.2f}")
                ref_perm_A = A[pi_rows2, :][:, pi_cols2]
                for i in range(len(sep_r)-1):
                    sum_diag += (A[sep_r[i]:sep_r[i+1], :][:, sep_c[i]:sep_c[i+1]]).sum()
                    perm_sum_diag += (perm_A[sep_r[i]:sep_r[i+1], :][:, sep_c[i]:sep_c[i+1]]).sum()
                    ref_perm_sum_diag += (ref_perm_A[sep_r[i]:sep_r[i+1], :][:, sep_c[i]:sep_c[i+1]]).sum()
                # check if permutation increases the sum on the block diag
                fracs['pi'] += [perm_sum_diag]
                fracs['orig'] += [sum_diag]
                fracs['ref_pi'] += [ref_perm_sum_diag]
                assert ref_perm_sum_diag+1e-9 >= perm_sum_diag
    print("PASSED greedy heur. refinement implementation test")

    M = 5
    m, n  = 300, 200
    dim = 8
    # Test random partitioning vs spectral paritioning
    num_levels = min(int(np.log2(m)), int(np.log2(n)))+1
    ks = np.array([2]*(num_levels-1))
    eps = 1e-2
    eps_ff = 1e-2
    rand_loss, spec_loss, spec_loss_ff = [], [], []
    func_partition =  mf.spectral_partition(symm=False, debug=True)
    print("### hat_A.factor fit ###")
    for t in range(M):
        ranks = np.random.randint(1, dim, num_levels)
        A = np.random.randn(m,n)

        # random parititioning binary tree
        hpart = mf.random_hpartition(m,  n)
        mf.test_hpartition(hpart, m, n)
        hat_A = mf.MLRMatrix(hpart=hpart, debug=True)
        losses = hat_A.factor_fit(A, ranks, hpart,  \
                                    eps_ff=eps_ff, max_iters_ff=10**3, symm=False, warm_start=False)
        rand_loss += [losses[-1]]
        print(f"{t}  rand: {rand_loss[-1] = }")
        assert np.allclose(mf.rel_diff(hat_A.matrix(), den=A), losses[-1]), \
            print(mf.rel_diff(hat_A.matrix(), den=A), losses[-10:])

        # spectral partitioning + top down rank fitting
        hpart, H = mf.hpartition_topdown(A, ranks, ks, func_partition, debug=True)
        mf.test_hpartition(hpart, m, n)
        hat_A = mf.MLRMatrix(hpart=hpart, debug=True)
        losses = hat_A.factor_fit(A, ranks, hpart, \
                                    eps_ff=eps_ff, max_iters_ff=10**3, symm=False, warm_start=False)
        spec_loss += [losses[-1]]
        print(f"{t} spec rf: {spec_loss[-1] = }, {mf.rel_diff(H, A)}")
        assert np.allclose(mf.rel_diff(hat_A.matrix(), den=A), losses[-1]), \
            print(mf.rel_diff(hat_A.matrix(), den=A), losses[-10:])

        # spectral partitioning + top down factor fitting
        hat_A = mf.MLRMatrix(debug=True)
        hat_A.hpartition_topdown(A, ranks, ks, func_partition, eps_ff=eps_ff, symm=False, \
                                    PSD=False, max_iters_ff=10)
        losses = hat_A.factor_fit(A, ranks, hat_A.hpart, \
                                    eps_ff=eps_ff, max_iters_ff=10**3, symm=False, warm_start=False)
        spec_loss_ff += [losses[-1]]
        print(f"{t} spec ff: {spec_loss_ff[-1] = }")
        assert np.allclose(mf.rel_diff(hat_A.matrix(), den=A), losses[-1]), \
            print(mf.rel_diff(hat_A.matrix(), den=A), losses[-10:])

        print(f"*** rand={rand_loss[-1]:.5f},  rf={spec_loss[-1]:.5f}, ff={spec_loss_ff[-1]:.5f}")
    print("### hat_A.rank_alloc ###")
    for t in range(M):
        ranks = np.random.randint(1, dim, num_levels)
        A = np.random.randn(m,n)

        # random parititioning binary tree
        hpart = mf.random_hpartition(m,  n)
        mf.test_hpartition(hpart, m, n)
        hat_A = mf.MLRMatrix(hpart=hpart, debug=True)
        losses, epochs, ranks_new = hat_A.rank_alloc(A, ranks, hpart,  eps=eps,\
                                    eps_ff=eps_ff, max_iters=10**3, symm=False, warm_start=False)
        rand_loss += [losses[-1]]
        print(f"{t}  rand: {rand_loss[-1] = }")
        for i in range(1, len(epochs)):
            a, b = max(0,epochs[i-1]-1), epochs[i]-1
            assert losses[a] - losses[b] >= -1e-9, print("loss is not decreasing with epochs")
        assert eps >= losses[-1-num_levels] - losses[-1]
        assert np.allclose(mf.rel_diff(hat_A.matrix(), den=A), losses[-1]), \
            print(mf.rel_diff(hat_A.matrix(), den=A), losses[-10:])

        # spectral partitioning + top down rank fitting
        hpart, H = mf.hpartition_topdown(A, ranks, ks, func_partition, debug=True)
        mf.test_hpartition(hpart, m, n)
        hat_A = mf.MLRMatrix(hpart=hpart, debug=True)
        losses, epochs, ranks_new = hat_A.rank_alloc(A, ranks, hpart, eps=eps,\
                                    eps_ff=eps_ff, max_iters=10**3, symm=False, warm_start=False)
        spec_loss += [losses[-1]]
        print(f"{t} spec rf: {spec_loss[-1] = }, {mf.rel_diff(H, A)}")
        for i in range(1, len(epochs)):
            a, b = max(0,epochs[i-1]-1), epochs[i]-1
            assert losses[a] - losses[b] >= -1e-9, print("loss is not decreasing with epochs")
        assert eps >= losses[-1-num_levels] - losses[-1]
        assert np.allclose(mf.rel_diff(hat_A.matrix(), den=A), losses[-1]), \
            print(mf.rel_diff(hat_A.matrix(), den=A), losses[-10:])

        # spectral partitioning + top down factor fitting
        hat_A = mf.MLRMatrix(debug=True)
        hat_A.hpartition_topdown(A, ranks, ks, func_partition, eps_ff=eps_ff, symm=False, \
                                    PSD=False, max_iters_ff=10)
        losses, epochs, ranks_new = hat_A.rank_alloc(A, ranks, hat_A.hpart,  eps=eps,\
                                    eps_ff=eps_ff, max_iters=10**3, symm=False, warm_start=False)
        spec_loss_ff += [losses[-1]]
        print(f"{t} spec ff: {spec_loss_ff[-1] = }")
        print(f"*** losses:  rand={rand_loss[-1]:.5f},  rf={spec_loss[-1]:.5f}, ff={spec_loss_ff[-1]:.5f}")
        for i in range(1, len(epochs)):
            a, b = max(0,epochs[i-1]-1), epochs[i]-1
            assert losses[a] - losses[b] >= -1e-9, print("loss is not decreasing with epochs")
        assert eps >= losses[-1-num_levels] - losses[-1]
        assert np.allclose(mf.rel_diff(hat_A.matrix(), den=A), losses[-1]), \
            print(mf.rel_diff(hat_A.matrix(), den=A), losses[-10:])
        


if __name__ == '__main__':
    main()