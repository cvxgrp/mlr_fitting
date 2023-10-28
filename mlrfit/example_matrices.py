import  numpy as np
import networkx as nx
import osmnx as ox

import scipy
from scipy.linalg import block_diag, toeplitz

from mlrfit.utils import *



def dgt_matrix(m, n, d=3, k=5):
    delta = 1./k**2
    s = np.random.rand(d*n).reshape(n, d)
    t = np.random.rand(d*m).reshape(m, d)
    ts = np.dot(t, s.T)
    s2 = (s**2).sum(axis=1).reshape(n, 1)
    t2 = (t**2).sum(axis=1).reshape(m, 1)
    Dist2 = s2.T - 2*ts + t2
    A = np.exp(- Dist2 / delta)
    return A


def dist_matrix_osmnx(place):
    G = ox.graph_from_place(place, network_type="drive")
    Adj_spr = nx.adjacency_matrix(G)
    Adj0 = np.array(Adj_spr.todense())
    Adj = (Adj0 + Adj0.T)*0.5
    Dist = graph_distance_matrix(Adj, directed=False)
    # assert (Dist==np.inf).sum() == 0
    k = dict(G.degree())
    print("degrees:", {n: list(k.values()).count(n) for n in range(max(k.values()) + 1)})
    return G, Adj, Dist


def dct_matrix(n):
    A = np.zeros((n,n))
    A[0,:] = np.ones(n)*1./np.sqrt(n)
    for i in range(1, n):
        for j in range(n):
            A[i,j] = np.sqrt(2./n) * np.cos(np.pi*(2*j+1)*i/(2*n))
    return A


def toeplitz_matrix(m, n, symm=False):
    r = np.random.uniform(low=0, high=1, size=m)
    c = np.random.uniform(low=0, high=1, size=n)
    if symm: c = r
    A = toeplitz(c, r=r)
    return A.T


def sample_on_unit_sphere(n, d):
    # generate points on a sphere
    t = np.random.randn(n*d).reshape(n, d)
    t = np.divide(t, np.linalg.norm(t, axis=1, keepdims=True))
    assert np.allclose(np.linalg.norm(t, axis=1), np.ones(n))
    return t


def radial_kernel_matrix(d, sigma, n, La, kern_func):
    t = sample_on_unit_sphere(n, d)
    s = sample_on_unit_sphere(n, d)

    A = np.zeros((n, n))
    Dist = np.diag(t.dot(t.T)).reshape(-1,1).dot(np.ones((1, n))) - 2 * t.dot(s.T) + np.ones((n, 1)).dot(np.diag(s.dot(s.T)).reshape(1, -1))
    Dist = np.sqrt(Dist)
    print(F"{np.histogram(Dist.reshape(-1), 7)}")
    for l in range(La): 
        Kl = kern_func(Dist / (sigma / 2**l))
        A += Kl
    return A, Dist


def hadamard_matrix(n):
    return scipy.linalg.hadamard(n)*1.


def hilbert_matrix(n):
    return scipy.linalg.hilbert(n)


def covariance_lr_diag(n, dim):
    F = np.random.randn(n, dim)
    D = np.diag(np.abs(np.random.randn(n)))
    A = F @ F.T + D
    return A


def normal_matrix(m, n, symm=False):
    A = np.random.randn(m, n)
    if symm: A = (A+A.T)/2
    return A


def sum_rand_normal_block_diag(num_blocks_list, m, n, symm=False, PSD=False):
    A = 0
    for num_blocks in num_blocks_list:
        A += rand_normal_block_diag(num_blocks, m, n, symm=symm, PSD=PSD)
    return A


def rand_normal_block_diag(num_blocks, m, n, symm=False, PSD=False):
    size1 = m//num_blocks
    size2 = n//num_blocks
    blks = []
    for _ in range(num_blocks-1):
        F = np.random.randn(size1, size2)
        if PSD:
            blks += [F@F.T]
        else:
            blks += [F]
    blks += [np.random.randn(m-size1*(num_blocks-1), n-size2*(num_blocks-1))]
    A = block_diag(*blks)
    assert A.shape == (m,n)
    if symm: A = (A + A.T)/2
    return A


def dist_geom_rand_graph(n, dim=4, r=0.35):
    points = np.random.rand(n, dim)
    Adj = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1, n):
            w  = np.linalg.norm(points[i] - points[j])
            if w <= r:
                Adj[i,j] = w; Adj[j,i] = w
    Dist = graph_distance_matrix(Adj)
    # set distance to maximum between disconnected components
    return cap_distance_max_element(Dist)


def dist_rand_tree(n):
    G = nx.random_tree(n=n)
    A_srs = nx.adjacency_matrix(G)
    Dist = graph_distance_matrix(A_srs, sparse=True)
    # set distance to maximum between disconnected components
    return cap_distance_max_element(Dist)


def dist_rand_regular_graph(n, deg):
    G = nx.random_regular_graph(deg, n)
    A_srs = nx.adjacency_matrix(G)
    Dist = graph_distance_matrix(A_srs, sparse=True)
    # set distance to maximum between disconnected components
    return cap_distance_max_element(Dist)


def dist_small_world_graph(n, k=10, p=0.1):
    G = nx.watts_strogatz_graph(n=n, k=k, p=p)
    A_srs = nx.adjacency_matrix(G)
    Dist = graph_distance_matrix(A_srs, sparse=True)
    # set distance to maximum between disconnected components
    return cap_distance_max_element(Dist)

    
def dist_expander_graph(n):
    G = nx.margulis_gabber_galil_graph(int(np.sqrt(n)+1))
    G.to_undirected()
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    A_srs = nx.adjacency_matrix(G)
    Dist = graph_distance_matrix(A_srs, sparse=True)
    # set distance to maximum between disconnected components
    return cap_distance_max_element(Dist)


def dist_grid_graph(n):
    G = nx.grid_graph(dim=(int(n**(1./3)+1), int(n**(1./3)+1), int(n**(1./3)+1)))
    A_srs = nx.adjacency_matrix(G)
    Dist = graph_distance_matrix(A_srs, sparse=True)
    # set distance to maximum between disconnected components
    return cap_distance_max_element(Dist)