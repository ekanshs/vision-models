import jax.numpy as jnp
from jax import random
from scipy.optimize import linear_sum_assignment

from permutations.permutation_spec import PermutationSpec
from permutations.util import get_permuted_param, flatten_params

def weight_matching(rng, ps: PermutationSpec, params_a, params_b, max_iter=100, init_perm=None):
  """Find a permutation of `params_b` to make them match `params_a`."""
  flat_params_a = flatten_params(params_a)
  flat_params_b = flatten_params(params_b)
  perm_sizes = {p: flat_params_a[axes[0][0]].shape[axes[0][1]] for p, axes in ps.perm_to_axes.items()}

  perm = {p: jnp.arange(n) for p, n in perm_sizes.items()} if init_perm is None else init_perm
  perm_names = list(perm.keys())
  rngs = random.split(rng, max_iter)
  
  for iteration in range(max_iter):
    progress = False
    for p_ix in random.permutation(rngs[iteration], len(perm_names)):
      p = perm_names[p_ix]
      n = perm_sizes[p]
      A = jnp.zeros((n, n))
      for wk, axis in ps.perm_to_axes[p]:
        w_a = flat_params_a[wk]
        w_b = get_permuted_param(ps, perm, wk, flat_params_b, except_axis=axis)
        w_a = jnp.moveaxis(w_a, axis, 0).reshape((n, -1))
        w_b = jnp.moveaxis(w_b, axis, 0).reshape((n, -1))
        A += w_a @ w_b.T

      ri, ci = linear_sum_assignment(A, maximize=True)
      assert (ri == jnp.arange(len(ri))).all()

      oldL = jnp.vdot(A, jnp.eye(n)[perm[p]])
      newL = jnp.vdot(A, jnp.eye(n)[ci, :])
      print(f"{iteration}/{p}: {newL - oldL}")
      progress = progress or newL > oldL + 1e-12

      perm[p] = jnp.array(ci)

    if not progress:
      break
  return perm
