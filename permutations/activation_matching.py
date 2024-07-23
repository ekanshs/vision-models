import jax.numpy as jnp
from jax import random

import tqdm

from permutations.util import flatten_params
from permutations.online_stats import OnlineMean, OnlineCovariance

from einops import rearrange
from scipy.optimize import linear_sum_assignment
import flax.linen as nn

def activation_matching(ps, model, params_a, params_b, data_iter, nsteps = 100):
  p_layers = ps.perm_to_axes.keys()
  act_layer = lambda x : x.removeprefix('P/')+'/__call__'
  
  def get_flat_intermediates(params, batch):
    _, state = model.apply({"params": params}, 
                                    batch['image'],
                                    capture_intermediates=True,
                                    mutable=['intermediates']
                                    )
    return flatten_params(state['intermediates'])
  
  def extract_act(intermediates, layer):
    if "Conv" in layer:
      act = intermediates[layer][0]
      act= nn.relu(act)
      act = rearrange(act, "batch w h c -> (batch w h) c")
      return act
    if "Dense" in layer:
      act = intermediates[layer][0]
      act= nn.relu(act)
      return act
    
  
  # Calculate mean activations
  def _calc_means():
    def one(params):
      means = {p: OnlineMean.init(flatten_params(params)[axes[0][0]].shape[axes[0][1]]) for p, axes in ps.perm_to_axes.items()}
      for i in range(nsteps):
        batch = next(data_iter)
        flat_intermediates = get_flat_intermediates(params,batch)
        means = {p_layer: means[p_layer].update(extract_act(flat_intermediates, act_layer(p_layer))) for p_layer in p_layers}
      return means

    return one(params_a), one(params_b)

  a_means, b_means = _calc_means()
  # Calculate the Pearson correlation between activations of the two models on
  # each layer.
  def _calc_corr():  
    stats = {
        p_layer: OnlineCovariance.init(a_means[p_layer].mean(), b_means[p_layer].mean())
        for p_layer in p_layers
    }
    for i in range(nsteps):
      batch = next(data_iter)
      flat_intermediates_a = get_flat_intermediates(params_a, batch)
      
      flat_intermediates_b = get_flat_intermediates(params_b, batch)
      
      stats = {p_layer: stats[p_layer].update(extract_act(flat_intermediates_a, act_layer(p_layer)), 
                                              extract_act(flat_intermediates_b, act_layer(p_layer))) 
              for p_layer in p_layers}
    return stats
  
  cov_stats = _calc_corr()  
  def find_permutation(corr):
    ri, ci = linear_sum_assignment(corr, maximize=True)
    assert (ri == jnp.arange(len(ri))).all()
    return ci

  return {f"{p_layer}": find_permutation(cov_stats[p_layer].pearson_correlation()) for p_layer in p_layers}