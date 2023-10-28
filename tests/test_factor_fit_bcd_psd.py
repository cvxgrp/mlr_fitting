import numpy as np
from tqdm import tqdm

import mlrfit as mf

def test_hat_A_psd(hat_A, num_levels, ranks, hpart):
    for level in range(num_levels):
        if ranks[level] == 0: continue
        num_blocks = len(hpart['rows']['lk'][level])-1
        B_level = hat_A.B[:,ranks[:level].sum():ranks[:level+1].sum()]
        C_level = hat_A.C[:,ranks[:level].sum():ranks[:level+1].sum()]
        for block in range(num_blocks):
            r1, r2 = hpart['rows']['lk'][level][block], hpart['rows']['lk'][level][block+1]
            c1, c2 = hpart['cols']['lk'][level][block], hpart['cols']['lk'][level][block+1]
            BC_lk =  B_level[r1:r2] @ C_level[c1:c2].T
            lambdas, V = np.linalg.eigh(BC_lk)
            assert lambdas.min() > -1e-9, print(f"{lambdas.min() = }")

def direct_bcd_covar_diag(A, orders, n, dim):
    """
    Direct computation of BCD for A = FF^T + D
    """
    ff_losses = {}
    for order in orders:
        hat_F = np.zeros((n, dim-1))
        hat_d = np.zeros(n)
        losses = [mf.rel_diff(hat_F @ hat_F.T + np.diag(hat_d), den=A)]
        for t in range(20):
            if order=='bu' and t%2==0 or order=='td' and t%2==1:
                R = A - hat_F @ hat_F.T
                V, lambdas = mf.frob_low_rank_psd(np.diag(np.diag((R))), dim=n)
                hat_d = np.diag(V @ np.diag(lambdas) @ V.T)
            elif order=='bu' and t%2==1 or order=='td' and t%2==0:
                R = A - np.diag(hat_d)
                V, lambdas = mf.frob_low_rank_psd(R, dim=hat_F.shape[1])
                hat_F = V @ np.diag(np.sqrt(lambdas))
            if t%2 == 1:
                losses += [mf.rel_diff(hat_F @ hat_F.T + np.diag(hat_d), den=A)]
        ff_losses[order] = losses
    return ff_losses

def main():
    M = 100
    m = n = 100
    dim = 6

    # Test low rank psd implementation
    for _ in tqdm(range(M)):
        F = np.random.randn(m, n)*10
        A = (F + F.T)/2
        lambdas_all, V_all = np.linalg.eigh(A)
        idx = np.argsort(lambdas_all)
        lambdas_all = lambdas_all[idx]
        V_all = V_all[:, idx]
        V2, lambdas2 = mf.frob_low_rank_psd(A, dim=dim)
        mf.test_eigsh_svds(V2, lambdas2, V2.T, dim, A, mode='psd')
        assert (np.diff(lambdas2) <= 1e-9).all() and (lambdas2 == np.sort(lambdas2)[::-1]).all()

        F = np.random.randn(m, n)
        A = F @ F.T
        lambdas_all, V_all = np.linalg.eigh(A)
        idx = np.argsort(lambdas_all)
        lambdas_all = lambdas_all[idx]
        V_all = V_all[:, idx]
        V2, lambdas2 = mf.frob_low_rank_psd(A, dim=min(m,n))
        mf.test_eigsh_svds(V2, lambdas2, V2.T, min(m,n), A, mode='psd')
        assert np.allclose(V2 @ np.diag(lambdas2) @ V2.T, A)
        assert (np.diff(lambdas2) <= 1e-9).all() and (lambdas2 == np.sort(lambdas2)[::-1]).all()

    print("PASSED low rank psd implementation tests")

    # Test PSD block coordinate descent implementation
    m = n = 200
    hpart = mf.random_hpartition(m,  n, symm=True)
    num_levels = len(hpart)
    mf.test_hpartition(hpart, m, n)
    eps = 1e-4

    hat_A = mf.MLRMatrix(hpart=hpart, debug=True)
    cycle_size = 1
    # A is symmetric PSD
    for _ in tqdm(range(M//2)):
        ranks = np.random.randint(1, dim, num_levels)
        F = np.random.randn(m, n)
        A = (F @ F.T)/2
        losses = hat_A.factor_fit(A, ranks, hpart, eps_ff=eps, symm=True, \
                                    max_iters_ff=10**3, PSD=True, warm_start=False)
        assert np.allclose(mf.rel_diff(hat_A.matrix(), den=A), losses[-1])
        for i in range(1, len(losses)-cycle_size):
            assert losses[-i-cycle_size] +1e-9 >= losses[-i],\
                print(f"{i = }, {losses[-i-cycle_size] - losses[-i]}, {len(losses)=}", \
                    losses, "loss is not decreasing with epochs")
        test_hat_A_psd(hat_A, num_levels, ranks, hpart)
    
    print("PASSED factor_fit PSD implementation tests")

    # Direct BCD and hat_A.factor_fit(..., PSD=True) comparison
    orders = ['bu', 'td']
    symm = True
    PSD = True
    num_levels = 2
    ranks = np.array([dim-1, 1])
    # true hier. partition.
    hpart = mf.random_hpartition(m,  n, level_list=[0, int(np.log2(m))+1], symm=symm, perm=False)
    hat_A = mf.MLRMatrix(hpart=hpart, debug=True)
    for _ in tqdm(range(M//2)):
        # A = FF^T + diag(d)
        F = np.random.randn(n, dim-1)
        D = np.diag(np.abs(np.random.randn(n)))
        A = F @ F.T + D
        ff_losses_MLR = {}
        for order in['bu', 'td']:
            losses = hat_A.factor_fit(A, ranks, hpart, order=order, init_type='zeros', \
                        PSD=PSD, symm=symm, freq=10, eps_ff=1e-4, max_iters_ff=10**3)
            ff_losses_MLR[order] = losses
        ff_losses_bcd = direct_bcd_covar_diag(A, orders, n, dim)
        for order in orders:
            min_len = min(len(ff_losses_bcd[order]), len(ff_losses_MLR[order])) - 1
            for t in range(min_len):
                assert np.allclose(ff_losses_bcd[order][t], ff_losses_MLR[order][t]),\
                    print("direct BCD and factor_fit PSD don't match")

    print("PASSED covariance matrix FF^T+diag recovery tests")



if __name__ == '__main__':
    main()