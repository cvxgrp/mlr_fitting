import pickle
import mlrfit as hf
import numpy as np
from scipy.sparse.linalg import lsmr

from time import perf_counter

def timer(f,*args):   
    start = perf_counter()
    f(*args)
    return (perf_counter() - start)


m = 2000; n = 2010; rank = 50

hpart = hf.random_hpartition(m,  n)
num_levels = len(hpart['rows']['lk'])
ranks = hf.uniform_capped_ranks(rank, hpart)
hat_A = hf.MLRMatrix(hpart=hpart, ranks=ranks, debug=True)
hat_A.construct_sparse_format()
b = np.random.randn(m, 1)
hat_A_val = hat_A.matrix()
hat_At_val = hat_A_val.T

x1 = hat_A.lsmr(b)
x2 = lsmr(hat_A_val, b)[0].reshape(-1,1)
print(np.linalg.norm(hat_A.matvec(x1) - b)**2, np.linalg.norm(hat_A_val@x2 - b)**2)

print("mlr", np.mean([timer(hat_A.lsmr, b) for _ in range(7)])*1000)
print("np", np.mean([timer(lsmr, hat_A_val, b) for _ in range(7)])*1000)