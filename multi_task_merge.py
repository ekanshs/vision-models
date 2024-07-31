"""
This script trains a model on a given dataset
The data is loaded using tensorflow_datasets.
"""

import functools
from absl import logging

import jax
from jax import random
import jax.numpy as jnp

from flax import jax_utils
from flax.training import common_utils

import ml_collections

from data import input_pipeline
import models

import wandb

import numpy as np
import utils

from configs.eval_merge import get_config

import seaborn as sns
from matplotlib import cm
import matplotlib.pyplot as plt
from utils import restore_checkpoint

import merging
import models


from trainer import compute_metrics 

def get_metrics(metrics):
  return common_utils.stack_forest(metrics)

  
@functools.partial(jax.jit, static_argnums=(0,))
def eval_step(apply_fn, params, batch):
    variables = {'params': params}
    logits = apply_fn(variables, batch['image'])
    metrics = compute_metrics(logits, batch['label'])  
    return metrics


def compute_loss_and_accuracy(params, apply_fn, dataset, nbatches=None):
  eval_iter = input_pipeline.prefetch(dataset, 10, None)
  eval_metrics = []
  ix = 0
  for eval_batch in eval_iter:
      metrics = eval_step(apply_fn, params, eval_batch)
      eval_metrics.append(metrics)
      ix+=1
      if nbatches is not None:
          if ix >= nbatches:
              break
  
  eval_metrics = get_metrics(eval_metrics)
  summary = {
          f'eval_{k}': v
          for k, v in jax.tree_util.tree_map(
              lambda x: x.mean(), eval_metrics
          ).items()
      }

  return summary['eval_loss'], summary['eval_accuracy']

def evaluate(config: ml_collections.ConfigDict):
  """Execute model training and evaluation loop.

  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.

  Returns:
    Final TrainState.
  """
  ## Setup wandb
  wandb_run = wandb.init(
    project="non-local-model-merging",
    entity="ekanshs",
    tags=[functools.reduce(lambda x,y: x+"_"+y, config.datasets), config.model, config.merging_method.name],
    mode="online",
    job_type='multi-task-merging',
    config=config
  )
  
  # Setup input pipeline
  datasets = config.datasets
  dataset_info_ls = [input_pipeline.get_dataset_info(dataset, config[dataset].pp['train']) for dataset in config.datasets]
  num_classes = [dataset_info['num_classes'] for dataset_info in dataset_info_ls] 
  num_train_examples = [dataset_info['num_examples'] for dataset_info in dataset_info_ls]
  
  _, ds_test_ls = input_pipeline.get_datasets_for_mtl(config, datasets)
  
  model_ls = []
  for nclass in num_classes:
    model_ls += [models.create_model(
                  model_cls= getattr(models, config.model), 
                  num_classes=nclass, 
                  width_multiplier=config.width_multiplier,
                  projection_dim=512, 
                  half_precision=config.half_precision) ]

  ## Get init params
  init_params_ls = []
  for dataset in datasets:
    init_params_ls += [restore_checkpoint(config[dataset].init_model_dir)['params']]  
  
  ## Get expert models
  expert_params_ls = []
  for dataset in datasets:
    expert_params_ls += [restore_checkpoint(config[dataset].model_dir)['params']]
  
  ## Compute task vectors
  task_vectors = [utils.tree_subtract({
      'encoder' : params['encoder'],
      'visual_projection' : params['visual_projection']
  }, {
      'encoder' : init_params['encoder'],
      'visual_projection' : init_params['visual_projection']
  }) for params, init_params  in zip(expert_params_ls, init_params_ls)]  
  
  ## Produce and save plots with 



  if config.merging_method.name == "task-arithmetic":
    merging_method = functools.partial(merging.compute_task_arithmetic_vector, )
  elif config.merging_method.name == "ties-merging":
    merging_method = merging.compute_ties_vector
  elif config.merging_method.name == "mgda-merging":
    merging_method = merging.compute_mgda_vector
  elif config.merging_method.name == "normalized-mgda-merging":
    merging_method = merging.compute_normalized_mgda_vector
  elif config.merging_method.name == "average-merging":
    merging_method = merging.compute_task_arithmetic_vector
  
  lams = np.linspace(config.merging_method.min, config.merging_method.max, config.merging_method.n) 
  
  perf_gain = lambda acc, init_acc: (acc - init_acc) / init_acc
  perf_drop = lambda acc, expert_acc: (expert_acc - acc) / expert_acc

  summary = {dataset: {lam: {} for lam in lams} for dataset in datasets}

  for lam in lams:
    merged_task_vector = merging_method(task_vectors, lam)
    for dataset, ds_test, model, init_params,  params in zip(datasets, ds_test_ls, model_ls, init_params_ls, expert_params_ls):
      merged_params = {
            'encoder': utils.tree_add(init_params['encoder'], merged_task_vector['encoder']),
            'visual_projection': utils.tree_add(init_params['visual_projection'], merged_task_vector['visual_projection']),
            'logit_scale': params['logit_scale'], 
            'classifier': params['classifier']
            }
      merged_loss, merged_accuracy = compute_loss_and_accuracy(merged_params, model.apply, ds_test)
      expert_loss, expert_accuracy = compute_loss_and_accuracy(params, model.apply, ds_test)
      init_loss, init_accuracy = compute_loss_and_accuracy(init_params, model.apply, ds_test)
      acc_gain = perf_gain(merged_accuracy, init_accuracy)
      acc_drop = perf_drop(merged_accuracy, expert_accuracy)
      
      summary[dataset][lam]['merged_loss'] = merged_loss
      summary[dataset][lam]['merged_accuracy'] = merged_accuracy
      summary[dataset][lam]['expert_loss'] = expert_loss
      summary[dataset][lam]['expert_accuracy'] = expert_accuracy
      summary[dataset][lam]['init_loss'] = init_loss
      summary[dataset][lam]['init_accuracy'] = init_accuracy
      summary[dataset][lam]['acc_gain'] = acc_gain
      summary[dataset][lam]['acc_drop'] = acc_drop

      
      dataset_summary = {
            f'{dataset}_{k}': v
            for k, v in summary[dataset][lam].items()
        }

      dataset_summary['lam'] = lam
      wandb_run.log(dataset_summary)

    
  wandb_run.log(summary)
  jax.random.normal(jax.random.key(0), ()).block_until_ready()

  wandb_run.finish()
  return None


if __name__ == '__main__':
  config = get_config(f'ViTB32,cifar10.sun397.food101.eurosat.svhn_cropped,task-arithmetic')
  config.cifar10.model_dir = 'experiments/ViT-B-32/cifar10/sgd_LR_1e-3_WD_1e-4/cosine/40_epochs_1_warmup/seed_0'
  config.cifar10.init_model_dir = 'experiments/ViT-B-32/cifar10/sgd_LR_1e-3_WD_1e-4/cosine/40_epochs_1_warmup/seed_0/init'
  config.sun397.model_dir = 'experiments/ViT-B-32/sun397/sgd_LR_1e-3_WD_1e-4/cosine/40_epochs_1_warmup/seed_0'
  config.sun397.init_model_dir = 'experiments/ViT-B-32/sun397/sgd_LR_1e-3_WD_1e-4/cosine/40_epochs_1_warmup/seed_0/init'
  config.food101.model_dir = 'experiments/ViT-B-32/food101/sgd_LR_1e-3_WD_1e-4/cosine/40_epochs_1_warmup/seed_0'
  config.food101.init_model_dir = 'experiments/ViT-B-32/food101/sgd_LR_1e-3_WD_1e-4/cosine/40_epochs_1_warmup/seed_0/init'
  config.eurosat.model_dir = 'experiments/ViT-B-32/eurosat/sgd_LR_1e-3_WD_1e-4/cosine/40_epochs_1_warmup/seed_0'
  config.eurosat.init_model_dir = 'experiments/ViT-B-32/eurosat/sgd_LR_1e-3_WD_1e-4/cosine/40_epochs_1_warmup/seed_0/init'
  config.svhn_cropped.model_dir = 'experiments/ViT-B-32/svhn_cropped/sgd_LR_1e-3_WD_1e-4/cosine/40_epochs_1_warmup/seed_0'
  config.svhn_cropped.init_model_dir = 'experiments/ViT-B-32/svhn_cropped/sgd_LR_1e-3_WD_1e-4/cosine/40_epochs_1_warmup/seed_0/init'
  evaluate(config)