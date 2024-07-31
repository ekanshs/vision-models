import jax.numpy as jnp
from jax.tree_util import tree_map

from flax import traverse_util

from pruning import keep_top_k
import utils
from min_norm_solver import find_min_norm_element_FW



def compute_ties_vector(task_vectors, top_k=1.0, **kwargs):
    #TRIM
    task_vectors = [keep_top_k(vector, top_k) for vector in task_vectors]

    #SIGN-ELECT: 
    sign_vector = utils.tree_sign(utils.add_trees(task_vectors))

    #MERGE
    elected_sum_vector = utils.add_trees([tree_map(lambda v, sign: jnp.where(v*sign >= 0, v, jnp.zeros(v.shape)), t, sign_vector) for t in task_vectors])
    elected_count_vector = utils.add_trees([tree_map(lambda v, sign: jnp.where(v*sign >= 0, jnp.ones(v.shape), jnp.zeros(v.shape)), t, sign_vector) for t in task_vectors])
    ties_vector = utils.tree_divide(elected_sum_vector, elected_count_vector)
  
    return ties_vector

def compute_task_arithmetic_vector(task_vectors, lam=0.4, **kwargs):
    return utils.add_trees(task_vectors, lam * jnp.ones((len(task_vectors,))))

def compute_mgda_vector(task_vectors, scale=1.0, **kwargs):
  alpha = find_min_norm_element_FW(task_vectors)[0]
  return utils.add_trees(task_vectors, scale*alpha)

def compute_normalized_mgda_vector(task_vectors, scale=1.0, **kwargs):
  normalized_task_vectors = [utils.normalize_tree(t)[0] for t in task_vectors]
  
  alpha = find_min_norm_element_FW(normalized_task_vectors)[0]
  return utils.add_trees(normalized_task_vectors, scale*alpha)

