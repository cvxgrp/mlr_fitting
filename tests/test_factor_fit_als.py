import copy

import numpy as np
from tqdm import tqdm

import torch

import mlrfit as mf


def test_equivalence_funcs(A, hat_A, loss, hpart, ranks):
    # test implementation of compressed format and factor format of A
    var_B = mf.torch_variable(hat_A.B, variable=False)
    var_C = mf.torch_variable(hat_A.C, variable=False)

    val_hat_A = hat_A.matrix()
    hat_A_val1 = mf.torch_compressed_hat_A(var_B, var_C, hpart, ranks)[hat_A.pi_inv_rows, :][:, hat_A.pi_inv_cols].data.numpy()

    tilde_B = mf.torch_cat_blkdiag_tilde_M(var_B, hpart['rows'], ranks)
    tilde_C = mf.torch_cat_blkdiag_tilde_M(var_C, hpart['cols'], ranks)
    hat_A_val2 = torch.matmul(tilde_B, tilde_C.T)[hat_A.pi_inv_rows, :][:, hat_A.pi_inv_cols].data.numpy()

    assert np.allclose(hat_A_val2, hat_A_val1) and np.allclose(val_hat_A, hat_A_val1) \
           and np.allclose(loss, mf.rel_diff(hat_A.matrix(), den=A)), \
           print(loss, mf.rel_diff(hat_A.matrix(), den=A), np.allclose(hat_A_val2, hat_A_val1), np.allclose(val_hat_A, hat_A_val1))


def main():
    M = 10
    m, n = 80, 60
    dim = 8
    method = 'als'

    # Test als_factor_fit implementation
    ranks = np.array([dim])
    hpart = mf.random_hpartition(m,  n, num_levels=1)
    num_levels = len(ranks)
    hat_A = mf.MLRMatrix(hpart=hpart, debug=True)
    # for single level ALS converges to low rank fit
    for _ in tqdm(range(M)):
        A = np.random.randn(m,n)
        torch_A = torch.from_numpy(A)
        U, Vt, sigmas = mf.frob_low_rank(A, dim)
        BC_true = np.dot(U * sigmas, Vt)

        B, C = hat_A.init_B_C(ranks, hpart, init_type='bcd', perm_A=A, params=(False, False, False))
        C0 = copy.deepcopy(C)
        B, C, losses, hat_A_val = mf.als_factor_fit_alt_cg(torch_A, B, C, hpart, ranks, update=0)
        B, C = mf.torch_to_numpy(B), mf.torch_to_numpy(C)
        assert np.allclose(C, C0)
        B0 = copy.deepcopy(B)
        B, C, losses, hat_A_val = mf.als_factor_fit_alt_cg(torch_A, B, C, hpart, ranks, update=1)
        B, C = mf.torch_to_numpy(B), mf.torch_to_numpy(C)
        assert np.allclose(B, B0)
        
        losses = hat_A.factor_fit(A, ranks, hpart, method=method, eps_ff=1e-9, freq=1, printing=False,
                                            max_iters_ff=2*10**3, symm=False, warm_start=False)
        test_equivalence_funcs(A, hat_A, losses[-1], hpart, ranks)
        assert np.abs(mf.rel_diff(hat_A.matrix(), den=A)-mf.rel_diff(BC_true, den=A)) <= 1e-2, \
                        print(mf.rel_diff(hat_A.matrix(), den=A), mf.rel_diff(BC_true, den=A), \
                            np.abs(mf.rel_diff(hat_A.matrix(), den=A)-mf.rel_diff(BC_true, den=A)))
    print("PASSED als_factor_fit implementation tests")

    # Test low rank and als implementation
    hpart = mf.random_hpartition(m,  n)
    mf.test_hpartition(hpart, m, n)
    eps = 1e-2
    num_levels = len(hpart['rows']['lk'])
    hat_A = mf.MLRMatrix(hpart=hpart, debug=True)
    for _ in tqdm(range(M)):
        ranks = np.random.randint(1, min(m,n), num_levels)
        A = np.random.randn(m,n)
        
        losses = hat_A.factor_fit(A, ranks, hpart, method=method, eps_ff=eps, init_type='bcd', \
                                max_iters_ff=10**3, symm=False, warm_start=False)
        test_equivalence_funcs(A, hat_A, losses[-1], hpart, ranks)
        for i in range(1, len(losses)-1):
            assert losses[-i-1] - losses[-i] >= -1e-6
        assert losses[0] > losses[-1]
        assert np.allclose(mf.rel_diff(hat_A.matrix(), den=A), losses[-1])

    print("PASSED factor_fit with als implementation tests")



if __name__ == '__main__':
    main()