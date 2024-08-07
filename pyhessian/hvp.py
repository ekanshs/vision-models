import jax
from utils import tree_inner_prod
from jax import lax

# def hessian_vector_product(f, x, v):
#     """Compute HVP: d^2f(x)\dx^2 * v  
#     Assumes x, v have tree structure.
#     """
#     return jax.grad(lambda x: tree_inner_prod(jax.grad(f)(x), v))(x)


def hvp(f, x, v):
    """Computes Hessian-vector-product:
    Source: https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#hessian-vector-products-using-both-forward-and-reverse-mode
    """
    return jax.jvp(jax.grad(f), (x, ), (v, ))[1]

def vmap_hvp(f, x, vs):
    """Compute HVP: d^2f(x)\dx^2 * v  
    Assumes x, vs have tree structure.
    """
    _hvp = lambda v : hvp(f, x, v)
    return jax.vmap(_hvp)(vs)


def compute_batch_hvp(loss_fn, batch, params, v, axis_name=None):  
    Hv = hvp(lambda params: loss_fn(batch, params), params, v)
    if axis_name is not None:
        Hv = lax.pmean(Hv, axis_name=axis_name)
    return Hv

def compute_batch_vmap_hvp(loss_fn, batch, params, vs, axis_name=None):  
    Hv = vmap_hvp(lambda params: loss_fn(batch, params), params, vs)
    if axis_name is not None:
        Hv = lax.pmean(Hv, axis_name=axis_name)
    return Hv