"""
This script trains a model on a given dataset
The data is loaded using tensorflow_datasets.
"""

import os
import pickle
from absl import logging

import jax
from jax import lax, random
import jax.numpy as jnp

from flax import jax_utils
from flax.training import common_utils

import ml_collections

from data import input_pipeline
import models

from trainer import compute_metrics, cross_entropy_loss

from tqdm import tqdm
import wandb

import numpy as np
from plotting import get_esd_plot


from utils import tree_add, tree_scalar_multiply, tree_zeros_like, save_checkpoint, restore_checkpoint
from pyhessian import compute_batch_hvp, compute_eigenvalues, compute_density, compute_trace
def get_metrics(metrics, axis_name):
  if axis_name is not None:
    return common_utils.get_metrics(metrics)
  else:
    return common_utils.stack_forest(metrics)


def eval_step(apply_fn, params, batch, axis_name=None):
  variables = {'params': params}
  logits = apply_fn(variables, batch['image'])
  metrics = compute_metrics(logits, batch['label'])
  
  if axis_name is not None:
    metrics = lax.pmean(metrics, axis_name=axis_name)
  return metrics


def compute_hessian_stats(
    config: ml_collections.ConfigDict, workdir: str
) :
  """Execute model training and evaluation loop.

  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.

  Returns:
    Final TrainState.
  """
  ## Setup wandb
  wandb_run = wandb.init(
    project="train-vision-model",
    entity="ekanshs",
    tags=[config.dataset, config.model, 'hessian'],
    mode="online",
    job_type='compute-hessian-stats',
    config=config
  )
  
  rng = random.PRNGKey(config.seed)
  
  axis_name = None
  if jax.device_count() > 1:
    axis_name = 'batch'
    p_eval_step = jax.pmap(eval_step, axis_name=axis_name, static_broadcasted_argnums=(0, 3,))
  else:
    p_eval_step = jax.jit(eval_step, static_argnums=(0, 3,))

  def compute_loss_and_accuracy(apply_fn, params, dataset, nbatches=None):
    eval_iter = input_pipeline.prefetch(dataset, config.prefetch, axis_name)
    eval_metrics = []
    ix = 0
    for eval_batch in eval_iter:
        metrics = p_eval_step(apply_fn, params, eval_batch, axis_name)
        eval_metrics.append(metrics)
        ix+=1
        if nbatches is not None:
            if ix >= nbatches:
                break
    eval_metrics = get_metrics(eval_metrics, axis_name)
    summary = {
            f'eval_{k}': v
            for k, v in jax.tree_util.tree_map(
                lambda x: x.mean(), eval_metrics
            ).items()
        }
    return summary

  # Setup input pipeline
  dataset = config.dataset
  pp = config[dataset].pp
  dataset_info = input_pipeline.get_dataset_info(dataset, pp['train'])
  num_classes = dataset_info['num_classes']
  num_train_examples = dataset_info['num_examples']
  
  ds_train, ds_test = input_pipeline.get_datasets(config, dataset)
  
  logging.info(ds_train)
  logging.info(ds_test)
  
  # Setup model and train state
  model = models.create_model(
    model_cls= getattr(models, config.model), 
    num_classes=num_classes, 
    width_multiplier=config.width_multiplier,
    projection_dim=512, 
    half_precision=config.half_precision)  

  params = restore_checkpoint(workdir)['params']  
  summary = compute_loss_and_accuracy(model.apply, params, ds_test, nbatches=None)

  logging.info(f"Eval accuracy = {summary['eval_accuracy']}.")
  wandb_run.log(summary)
  

  if config.full_ds_estimate == True:
     nbatches = None
  else:
     nbatches = config.nbatches

  def hvp_fn(params, v):
    def loss_fn(batch, params):
      """Loss function used for the hessian."""
      logits = model.apply(
          {'params': params},
          batch['image'])
      loss = cross_entropy_loss(logits, batch['label'])
      return loss
    eval_iter =  input_pipeline.prefetch(ds_test, config.prefetch, axis_name)
    if axis_name is not None: 
        hvp_batch = jax.pmap(compute_batch_hvp, static_argnums=(0, 4))
    else:
        hvp_batch = jax.jit(compute_batch_hvp, static_argnums=(0, 4))
    N = 0
    Hv = tree_zeros_like(v)
    if axis_name is not None:
        N_fn = lambda batch: batch['label'].shape[0] * batch['label'].shape[1]
    else:
        N_fn = lambda batch: batch['label'].shape[0] 
    ix = 0
    for batch in eval_iter:
        batch_n = N_fn(batch)
        N += batch_n
        _Hv = hvp_batch(loss_fn, batch, params, v, axis_name)
        Hv = tree_add(Hv, tree_scalar_multiply(batch_n, _Hv))
        ix+=1
        if nbatches is not None:
          if ix >= nbatches:
              break
    return tree_scalar_multiply(1./N, Hv)


  if config.compute_trace:
    trace = compute_trace(rng, hvp_fn, params)
    logging.info(f"Trace estimate = {jnp.mean(jnp.stack(trace))}")
    wandb_run.log({
      'trace_estimate': jnp.mean(jnp.stack(trace))
    })
    wandb_run.log({
      'trace_vhv': trace
    })

  if config.compute_density:  
    (eigenvalues_ls, eigevectors_ls), weights_ls = compute_density(rng, hvp_fn, params, n_eigs=config.n_eigs)
    wandb_run.log({
      'lanczos_eig': (eigenvalues_ls, eigevectors_ls)
    })

    fig = get_esd_plot(eigenvalues_ls, weights_ls)
    wandb_run.log({"density_plot": fig})

  if config.compute_top_n_eig:
    eig_fname = f"{workdir}/eigs.pkl"
    if os.path.isfile(eig_fname):
      with open(eig_fname, 'rb') as f: 
        curr_eig_vals, curr_eig_vectors = pickle.load(f)
    else:
      curr_eig_vals, curr_eig_vectors = [], []

    eig_vals, eig_vectors = compute_eigenvalues(rng, hvp_fn, params, eigenvalues=curr_eig_vals, eigenvectors=curr_eig_vectors, top_n=config.top_n_eigs)
    logging.info(f"Top {config.top_n_eigs} = {eig_vals}")
    
    wandb_run.log({
      'top_n_eigs': eig_vals
    })

    with open(eig_fname, "wb") as f:
      pickle.dump((eig_vals, eig_vectors), f)

  jax.random.normal(jax.random.key(0), ()).block_until_ready()

  wandb_run.finish()
  return 

