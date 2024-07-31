import jax
from utils import tree_inner_prod
from jax import lax

def hessian_vector_product(f, x, v):
    """Compute HVP: d^2f(x)\dx^2 * v  
    Assumes x, v have tree structure.
    """
    return jax.grad(lambda x: tree_inner_prod(jax.grad(f)(x), v))(x)


def compute_hessian_vector_product_for_batch(loss_fn, batch, params, v, axis_name=None):  
    Hv = hessian_vector_product(lambda params: loss_fn(batch, params), params, v)
    if axis_name is not None:
        Hv = lax.pmean(Hv, axis_name=axis_name)
    return Hv

