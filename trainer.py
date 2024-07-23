
from absl import logging

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

import optax
from utils import tree_norm

def zero_one_loss(logits, labels):
  accuracy = jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(labels, -1))
  return 1-accuracy

def cross_entropy_loss(logits, labels):
  xentropy = optax.softmax_cross_entropy(logits=logits, labels=labels)
  return jnp.mean(xentropy)


def compute_metrics(logits, labels):
  loss = cross_entropy_loss(logits, labels)
  accuracy = 1 - zero_one_loss(logits, labels)
  metrics = {
      'loss': loss,
      'accuracy': accuracy,
  }
  return metrics


def train_step(state, batch, axis_name=None):
  """Perform a single training step."""
  def loss_fn(params):
    """loss function used for training."""
    logits = state.apply_fn(
        {'params': params},
        batch['image'])
    loss = cross_entropy_loss(logits, batch['label'])
    return loss, logits

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, logits), grads = grad_fn(state.params)
  # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
  if axis_name is not None:
    grads = lax.pmean(grads, axis_name=axis_name)
  
  is_fin = jnp.array(True)
  for g in jax.tree_util.tree_leaves(grads):
    is_fin &= jnp.all(lax.is_finite(g))

  metrics = compute_metrics(logits, batch['label'])
  if axis_name is not None:
    metrics = lax.pmean(metrics, axis_name=axis_name)
  
  new_state = state.apply_gradients(
      grads=grads
  )
  new_state = new_state.replace(
    opt_state=jax.tree_util.tree_map(
        partial(jnp.where, is_fin),
        new_state.opt_state,
        state.opt_state,
    ),
    params=jax.tree_util.tree_map(
            partial(jnp.where, is_fin), new_state.params, state.params
        )
  )
  metrics.update({'g_norm': tree_norm(grads)})
  return new_state, metrics

def eval_step(state, batch, axis_name=None):
  variables = {'params': state.params}
  logits = state.apply_fn(variables, batch['image'])
  metrics = compute_metrics(logits, batch['label'])
  
  if axis_name is not None:
    metrics = lax.pmean(metrics, axis_name=axis_name)
  return metrics

