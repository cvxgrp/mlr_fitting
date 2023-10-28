import numpy as np
import random

import mlrfit as mf


def main():
    np.random.seed(1001)
    random.seed(1001)

    M = 20
    m, n  = 200, 175
    dim = 10

    # Test spectral_partition_minimize implementation
    for (m,n) in [(200,300), (200,200)]:
        symm = (m==n)
        fracs = {'pi':[], 'ref_pi':[], 'orig':[]}
        for _ in range(M):
            Ai = np.random.randn(m, n)
            if symm: Ai = Ai + Ai.T # for Laplacian to be defined in symmetric partitioning
            for A in [np.abs(Ai), np.square(Ai)]:
                rows, sep_r, cols, sep_c = mf.spectral_partition_minimize(symm=symm, debug=True)(A)
                sum_diag = 0
                perm_sum_diag = 0
                ref_perm_sum_diag = 0
                perm_A = A[rows, :][:, cols]
                pi_rows2, pi_cols2, obj_all = mf.greedy_heur_refinement(-A, rows, sep_r, cols, sep_c, \
                                                                        symm=symm, max_iters=200, debug=True)
                obj_all = -np.array(obj_all)
                assert np.allclose(obj_all[-1], mf.obj_partition_sum_full(A, pi_rows2, pi_cols2, sep_r, sep_c))
                print(f"itr={len(obj_all)}, {symm=}, % dec. {100*(obj_all[0]-obj_all[-1])/np.abs(obj_all[0]):.2f}")
                ref_perm_A = A[pi_rows2, :][:, pi_cols2]
                for i in range(len(sep_r)-1):
                    sum_diag += (A[sep_r[i]:sep_r[i+1], :][:, sep_c[i]:sep_c[i+1]]).sum()
                    perm_sum_diag += (perm_A[sep_r[i]:sep_r[i+1], :][:, sep_c[i]:sep_c[i+1]]).sum()
                    ref_perm_sum_diag += (ref_perm_A[sep_r[i]:sep_r[i+1], :][:, sep_c[i]:sep_c[i+1]]).sum()
                # check if permutation increases the sum on the block diag
                fracs['pi'] += [perm_sum_diag]
                fracs['orig'] += [sum_diag]
                fracs['ref_pi'] += [ref_perm_sum_diag]
                assert ref_perm_sum_diag - 1e-9 <= perm_sum_diag
        fracs['pi'] = np.array(fracs['pi'])
        fracs['orig'] = np.array(fracs['orig'])
        fracs['ref_pi'] = np.array(fracs['ref_pi'])
        print(f"{symm=}, {fracs['orig'].mean()=}, {fracs['pi'].mean()=}, {fracs['ref_pi'].mean()=}")
    print("PASSED greedy heur. refinement implementation test")



if __name__ == '__main__':
    main()