

# Factor Fitting, Rank Allocation, and Partitioning in Multilevel Low Rank Matrices
 
This repository accompanies the [manuscript](https://stanford.edu/~boyd/papers/pdf/mlr_fitting.pdf).

A multilevel low rank (MLR) matrix is a row and column permutation of
sum of matrices, each one a block diagonal refinement of the previous one, 
with all blocks low rank given in factored form.
MLR matrices extend low rank matrices
but share many of their properties, such as the total storage required and
complexity of matrix-vector multiplication.

In this repository we provide solutions to three problems that arise in fitting a given matrix by an 
MLR matrix in the Frobenius norm 
1. factor fitting,
where we adjust the factors of the MLR matrix
1. rank allocation, where we choose the ranks of the blocks 
in each level, subject to the total rank having a given value
1. hierarchical partitioning of rows and 
columns, along with the ranks and factors.


## Installation
To install `mlrfit` 1) activate virtual environment, 2) clone the repo, 3) from inside the directory run 
```python3
python setup.py install
```
Requirements
* python >= 3.9
* numpy >= 1.22.2, scipy >= 1.10.0, pandas >= 1.5.2
* PyTorch >= 1.13
* numba >= 0.55.1
* osmnx >= 1.3.0
* networkx >= 2.7.1


## Getting started
Given data matrix $A$, we fit the data using MLR matrix model.

1. Define MLR matrix object $\hat A$ using `hat_A = mlrfit.MLRMatrix()`

2. Define hierarchical partitioning of type `HpartDict`
   * if there is an existing hierarchy, specify it in `hpart` 
   * if there is no existing hierarchy, create new `hpart` using spectral partitioning with initial rank allocation `ranks`
       * by calling `mlrfit.MLRMatrix.hpartition_topdown(A, ranks)`

3. Fit $\hat A$ to given $A$
    * if rank allocation is given, specify it in `ranks` and run factor fit
      * by calling `mlrfit.MLRMatrix.factor_fit(A, ranks, hpart)`
    * if rank allocation is not given, use rank allocation method with initial rank allocation `ranks`
        * by calling `mlrfit.MLRMatrix.rank_alloc(A, ranks, hpart)`

Once the $\hat A$ model has been fitted, we can use it for fast linear algebra:
1. Matrix-vector multiplication $\hat A x$ by calling
```python3
b = hat_A.matvec(x)
```
2. Linear system solve $\hat A x = b$ (when $m=n$ and $\hat A$ is invertible)
```python3
x = hat_A.solve(b)
``` 
3. Least squares $\| \hat A x - b\|_2^2$
```python3
x = hat_A.lstsq(b)
```


### Hello world
We provide a guideline on how to use our methods using the 
[hello world example](https://github.com/cvxgrp/mlr_fitting/tree/main/examples/hello_world.ipynb). 


## Example notebooks
We have [example notebooks](https://github.com/cvxgrp/mlr_fitting/tree/main/examples) 
that show how to use our method on a number of different problems
* asset covariance matrix, see [notebook](https://github.com/cvxgrp/mlr_fitting/blob/main/examples/gics_fit_cov.ipynb) 
* distance matrix, see [notebook](https://github.com/cvxgrp/mlr_fitting/blob/main/examples/full_fit_dist_pacifica.ipynb)  
* DGT matrix, see [notebook](https://github.com/cvxgrp/mlr_fitting/blob/main/examples/full_fit_dgt.ipynb)           

Please consult our [manuscript](XXX) for the details of mentioned problems. 

## Linear algebra
We implement fast linear algebra operations that use sparse form of MLR model, see 
[notebook](https://github.com/cvxgrp/mlr_fitting/blob/main/examples/sparse_linop.ipynb)
 * matrix vector multiplication
 * linear system solve: comparison of storage efficiency vs time efficiency between dense and sparse
   representations of $\hat A$

| $(m \times n) / (mr+nr)$   | $t_{\mathrm{dense}} / t_{\mathrm{sparse}}$ |
| :------------ | :----------- | 
10       |     11.5      |
20       |     39.3      |
50      |     200.8      |
80      |     462.9      |
100     |     701.1      |
160     |     1623.2      |

 * least squares: comparison of storage efficiency vs time efficiency between dense and sparse
   representations
of $\hat A$

| $(m \times n) / (mr+nr)$   | $t_{\mathrm{dense}} / t_{\mathrm{sparse}}$ |
| :------------ | :----------- | 
12.0       |     94.5      |
22.2       |     391.1      |
52.4       |     2435.2      |
72.4       |     5899.3      |


Linear system solve and least squares are based on the parallel direct sparse
solver (PARDISO) implemented with `pypardiso`.
This results in a speed improvement factor ranging from **10x to 1000x** when 
compared to solving problems of equivalent size using dense matrices.


### Extra
We also provide additional experiments 
* ALS vs BCD factor fit comparison, see [notebook](https://github.com/cvxgrp/mlr_fitting/blob/main/examples/factor_fit_als_bcd.ipynb)
* ALS vs BCD rank allocation comparison, see
  [notebook](https://github.com/cvxgrp/mlr_fitting/blob/main/examples/rank_alloc_als_bcd.ipynb)
