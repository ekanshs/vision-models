
import jax
from jax import random
import math
import jax.numpy as jnp
import optax
from scipy.linalg import eigh_tridiagonal

from utils import (add_trees, 
                   tree_inner_prod, 
                   tree_zeros_like, 
                   tree_add, tree_subtract,
                   tree_scalar_multiply,
                   tree_norm,
                   normal_tree_like, 
                   rademacher_tree_like, 
                   normalize_tree, 
                   orthnormal)

TINY = 1e-6
MAX_ITER=100



def compute_eigenvalues(rng, hvp_fn, params, eigenvalues=[], eigenvectors=[], max_iter=MAX_ITER, tol=1e-3, top_n=1):
    """
    Compute the top_n eigenvalues using power iteration method
    """
    assert len(eigenvalues) == len(eigenvectors)
    rngs = random.split(rng, top_n)
    curr_n = len(eigenvalues)
    while curr_n < top_n:            
        eigenvalue = None
        v =  normalize_tree(normal_tree_like(rngs[curr_n], params)) # generate random vector
        for _ in range(max_iter):
            v = orthnormal(v, eigenvectors)
            Hv = hvp_fn(theta=params, v=v)
            t_eigen = tree_inner_prod(v, Hv)
            v = normalize_tree(Hv)
            if eigenvalue == None:
                eigenvalue = t_eigen
            else:
                if abs(eigenvalue - t_eigen) / (abs(eigenvalue) + TINY) < tol:
                    break
                else:
                    eigenvalue = t_eigen
        
        eigenvalues.append(eigenvalue)
        eigenvectors.append(v)
        curr_n += 1

    return eigenvalues, eigenvectors


def compute_trace(rng, hvp_fn, params, max_iter=MAX_ITER, tol=TINY):
    """
    compute the trace of hessian using Hutchinson's method
    maxIter: maximum iterations used to compute trace
    tol: the relative tolerance
    """
    trace_vhv = []
    trace = 0.
    rngs = random.split(rng, max_iter)
    for i in range(max_iter):
        rad_v = rademacher_tree_like(rngs[i], params) # generate Rademacher random variables 
        Hv =  hvp_fn(theta=params,v=rad_v)
        trace_vhv += [tree_inner_prod(rad_v, Hv)]
        if abs(jnp.mean(trace_vhv) - trace) / (abs(trace) + TINY) < tol:
            return trace_vhv
        else:
            trace = jnp.mean(trace_vhv)
    return trace_vhv


def compute_density(rng, hvp_fn, params, n_slq=1, n_eigs=100):
    """
    compute estimated eigenvalue density using stochastic lanczos algorithm (SLQ)
    iter: number of iterations used to compute trace
    n_slq: number of SLQ runs
    """
    eigen_list_full = []
    weight_list_full = []
    slq_rngs = random.split(rng, n_slq)
    for slq_rng in slq_rngs:
        ## Lanczos algorithm
        rngs = random.split(slq_rng, n_eigs)
        alphas = []
        betas = []
        vs = [normalize_tree(normal_tree_like(rngs[0], params))]
        w_ = hvp_fn(theta=params, v=vs[0])
        alphas += [tree_inner_prod(w_, vs[0])]
        w = add_trees([w_, vs[0]], [1., - alphas[0]])
        for i in range(1, n_eigs):
            beta_i = tree_norm(w)
            betas += [beta_i]
            if beta_i != 0.:
                vs += [tree_scalar_multiply(1./beta_i, w)]
            else:
                # generate a new vector
                vs += [orthnormal(normal_tree_like(rngs[i], params), vs)]
            w_ = hvp_fn(theta=params, v=vs[i])
            alphas += [tree_inner_prod(w_, vs[i])]
            w = add_trees([w_, vs[i], vs[i-1]] , [1., -alphas[i], -betas[i]])
        eigenvalues, eigenvectors = eigh_tridiagonal(alphas, betas)

        eigen_list_full += [eigenvalues]
        weight_list_full += [jnp.square(eigenvectors[0,:])]
    return eigen_list_full, weight_list_full

