import numpy as np
import random
from tqdm import tqdm
from scipy.sparse.linalg import lsmr
import torch

import mlrfit as mf


def main():
    np.random.seed(1001)
    random.seed(1001)

    M = 20
    m, n  = 300, 300
    rank = 40

    for _ in tqdm(range(M)):
        hpart = mf.random_hpartition(m,  n)
        ranks = mf.uniform_capped_ranks(rank, hpart)
        hat_A = mf.MLRMatrix(hpart=hpart, ranks=ranks, debug=True)
        hat_A.construct_sparse_format()

        # tilde_B and tilde_C equivalence to B and C
        spB = hat_A.tilde_B.todense()
        tilde_B2 = mf.torch_cat_blkdiag_tilde_M(torch.from_numpy(hat_A.B), \
                                            hpart['rows'], ranks).data.numpy()
        assert np.allclose(spB, tilde_B2) and mf.rel_diff(spB, tilde_B2) < 1e-8

        spC = hat_A.tilde_C.todense()
        tilde_C2 = mf.torch_cat_blkdiag_tilde_M(torch.from_numpy(hat_A.C), \
                                            hpart['cols'], ranks).data.numpy()
        assert np.allclose(spC, tilde_C2) and mf.rel_diff(spC, tilde_C2) < 1e-8

        hat_A_val = hat_A.matrix()
        assert np.allclose(hat_A_val, \
            (hat_A.tilde_B.dot(hat_A.tilde_Ct)).todense()[hat_A.pi_inv_rows, :][:, hat_A.pi_inv_cols])

        # matvec operation
        x = np.random.randn(n, 1)
        assert np.allclose(hat_A.matvec(x), hat_A_val @ x)
        assert np.allclose(hat_A.rmatvec(x), hat_A_val.T @ x)

        # linear system solve
        true_x = np.random.randn(n, 1)
        b = hat_A_val @ true_x
        x1 = np.linalg.solve(hat_A_val, b)
        x2 = hat_A.solve(b)
        assert np.allclose(b.flatten(), (hat_A_val @ np.linalg.solve(hat_A_val, b)).flatten() )
        assert np.allclose((hat_A_val @ hat_A.solve(b)).flatten(), b.flatten())
        # assert np.allclose(x1, x2) and np.allclose(x2, true_x), print(np.linalg.norm(x2 - true_x))
        
    print("PASSED hat_A.tilde_B and hat_A.tilde_C implementation tests")
    print("PASSED hat_A.solve and hat_A.matvec implementation tests")

    M = 20
    m, n  = 300, 400
    rank = 40

    for _ in tqdm(range(M)):
        hpart = mf.random_hpartition(m,  n)
        ranks = mf.uniform_capped_ranks(rank, hpart)
        hat_A = mf.MLRMatrix(hpart=hpart, ranks=ranks, debug=True)
        hat_A.construct_sparse_format()
        hat_A_val = hat_A.matrix()

        # least squares
        b = np.random.randn(m, 1)
        
        x1 = hat_A.lsmr(b, tol=1e-8)
        x2 = lsmr(hat_A_val, b, atol=1e-08, btol=1e-08)[0].reshape(-1,1)
        assert mf.rel_diff(mf.rel_diff(hat_A.matvec(x1), den=b), \
               mf.rel_diff(hat_A_val @ x2, den=b)) < 0.2
        
    print("PASSED hat_A.lsmr implementation tests")



if __name__ == '__main__':
    main()