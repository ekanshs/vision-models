"""
This script trains a model on a given dataset
The data is loaded using tensorflow_datasets.
"""

import os
from absl import logging

import jax
from jax import random
import jax.numpy as jnp

from flax import jax_utils
from flax.training import common_utils, checkpoints

import ml_collections

from data import input_pipeline
import models

from flax.training.train_state import TrainState
import trainer

from tqdm import tqdm
import wandb

from optax_utils import create_learning_rate_fn, create_path_aware_tx

import numpy as np


def restore_checkpoint(workdir, target=None):
  return checkpoints.restore_checkpoint(workdir, target=target)

def save_checkpoint(workdir, state, keep=1, keep_every_n_steps=None):
  step = int(state.step)
  logging.info('Saving checkpoint step %d.', step)
  checkpoints.save_checkpoint(workdir, state, step, keep=keep, keep_every_n_steps=keep_every_n_steps) 


def get_metrics(metrics, axis_name):
  if axis_name is not None:
    return common_utils.get_metrics(metrics)
  else:
    return common_utils.stack_forest(metrics)


def train_and_evaluate(
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
    tags=[config.dataset, config.model, f'{config.optimizer.name}_lr_{config.optimizer.learning_rate}', f'{config.training_schedule.num_epochs}'],
    mode="online",
    job_type='train-and-evaluate',
    config=config
  )
  
  platform = jax.local_devices()[0].platform
  if config.half_precision:
    if platform == 'tpu':
      model_dtype = jnp.bfloat16
    else:
      model_dtype = jnp.float16
  else:
    model_dtype = jnp.float32

  rng = random.PRNGKey(config.seed)
  
  axis_name = None
  if jax.device_count() > 1:
    axis_name = 'batch'
    train_step = jax.pmap(trainer.train_step, axis_name=axis_name, donate_argnums=(0,), static_broadcasted_argnums=(2,))
    eval_step = jax.pmap(trainer.eval_step, axis_name=axis_name, static_broadcasted_argnums=(2,))
  else:
    train_step = jax.jit(trainer.train_step, donate_argnums=(0,), static_argnums=(2,))
    eval_step = jax.jit(trainer.eval_step, static_argnums=(2,))

  def compute_loss_and_accuracy(state, dataset, nbatches=None):
    eval_iter = input_pipeline.prefetch(dataset, config.prefetch, axis_name)
    eval_metrics = []
    ix = 0
    for eval_batch in eval_iter:
        metrics = eval_step(state, eval_batch, axis_name)
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
  
  # Training arguments
  num_epochs = int(config.training_schedule.num_epochs)
  train_batch_size = int(config.training_schedule.per_device_train_batch_size) * jax.device_count()
  per_device_eval_batch_size = int(config.training_schedule.per_device_eval_batch_size)

  steps_per_epoch = num_train_examples // train_batch_size
  total_train_steps = steps_per_epoch * num_epochs
  total_warmup_steps = int(steps_per_epoch * config.training_schedule.warmup_epochs)
  
  learning_rate_fn = create_learning_rate_fn(
        config.training_schedule.decay_schedule, 
        total_train_steps, total_warmup_steps, 
        learning_rate=config.optimizer.learning_rate * np.sqrt(jax.device_count())
    )
  
  # Setup model and train state
  model = models.create_model(
    model_cls= getattr(models, config.model), 
    num_classes=num_classes, 
    dtype=model_dtype)  
  
  def train_classifier(params, at_init=True):
    # train classifier
    total_classifier_train_steps = steps_per_epoch * int(config.training_schedule.classifier.num_epochs)
    total_classifier_warmup_steps = int(steps_per_epoch * config.training_schedule.classifier.warmup_epochs)

    classifier_lr_fn = create_learning_rate_fn(
      config.training_schedule.decay_schedule, 
      total_classifier_train_steps, total_classifier_warmup_steps, 
      learning_rate=config.optimizer.classifier.learning_rate * np.sqrt(jax.device_count())
    )
    logging.info("Updating the classifier layer.")
    
    tx = create_path_aware_tx(config.optimizer.classifier, classifier_lr_fn, params, ['classifier', 'logit_scale'])
    ## create classifier train state:
    state = TrainState.create(
      apply_fn=model.apply,
      params=params,
      tx=tx
    )  
    train_iter = input_pipeline.prefetch(ds_train, config.prefetch, axis_name)
    if at_init:
      state = restore_checkpoint(os.path.join(workdir, 'init'), state)
      step_offset = int(state.step)
    
    if axis_name is not None:
      state = jax_utils.replicate(state)

    progress_bar = tqdm(range(step_offset+1, total_classifier_train_steps + 1))
    for step in progress_bar:
      batch = next(train_iter)
      progress_bar.set_description(f"Classifier: {step} / {total_classifier_train_steps}")
      state, _ = train_step(state, batch, axis_name)
      
      if at_init and step==total_classifier_train_steps:
        if axis_name is not None:
          save_checkpoint(os.path.join(workdir, 'init'), jax_utils.unreplicate(state))
        else:
          save_checkpoint(os.path.join(workdir, 'init'), state)
    
    logging.info(f"Classifier layer trained.") 
    if axis_name is not None:
      state = jax_utils.unreplicate(state)

    return state.params
  
  
  def initialize(key, model, batch_shape):
    @jax.jit
    def init(*args):
      return model.init(*args)
    variables = init(key, jnp.ones(batch_shape))
    return variables['params']
  
  init_params = initialize(rng, model, batch_shape=(1,config[dataset].pp.crop, config[dataset].pp.crop, 3))
  
  if config.from_pretrained:
    if config.pretrained_dir is not None:
      pretrained_params = restore_checkpoint(config.pretrained_dir)['params']
      init_params = {'encoder' : pretrained_params['params']['encoder'],
          'visual_projection' : pretrained_params['params']['visual_projection'],
          'logit_scale' : pretrained_params['params']['logit_scale'],
          'classifier' : init_params['classifier']
          }
    else:
      init_params = models.get_zero_shot_params(config.model, dataset)

    if config.train_classifier_at_init:
      init_params = train_classifier(init_params, at_init=True)
  
  keywords = []
  if config.train_projection:
    keywords += ['visual_projection']
  if config.train_encoder:
    keywords += ['encoder']
  if config.train_classifier:
    keywords += ['classifier']
  if config.train_logit_scale:
    keywords += ['logit_scale']


  tx = create_path_aware_tx(config.optimizer, learning_rate_fn, init_params, keywords)
  state = TrainState.create(
    apply_fn=model.apply,
    params=init_params,
    tx=tx
  )

  if axis_name is not None:
    state = jax_utils.replicate(state)
  
  init_summary = compute_loss_and_accuracy(state, ds_test, nbatches=None)
  init_summary.update({"step": 0})
  logging.info(f"Init eval accuracy = {init_summary['eval_accuracy']}.")
  wandb_run.log(init_summary)
  if axis_name is not None:
    state = jax_utils.unreplicate(state)
  
  state = restore_checkpoint(workdir, state)
  step_offset = int(state.step)
  
  if axis_name is not None:
    state = jax_utils.replicate(state)
  
  train_iter = input_pipeline.prefetch(ds_train, config.prefetch, axis_name)
  train_metrics = []
  progress_bar = tqdm(range(step_offset + 1, total_train_steps + 1))
  for step in progress_bar:
    epoch = step // steps_per_epoch
    progress_bar.set_description(f"Epoch: {epoch} / {num_epochs}")
    batch = next(train_iter)
    state, metrics = train_step(state, batch, axis_name)
    
    train_metrics.append(metrics)
    if (step + 1) % config.progress_every == 0:
      train_metrics = get_metrics(train_metrics, axis_name)
      summary = {
            k: float(v)
            for k, v in jax.tree_util.tree_map(
                lambda x: x.mean(), train_metrics
            ).items()
        }
      summary.update({"step": step})
      summary.update({"learning_rate": learning_rate_fn(step)})
      wandb_run.log(summary)
      train_metrics = []
    
    if step == step_offset + 1:
      logging.info("Initial compilation complete")
    
    if step % config.checkpoint_every == 0 or step == total_train_steps:
      if axis_name is not None:
        save_checkpoint(workdir, jax_utils.unreplicate(state))
      else:
        save_checkpoint(workdir, state)
    
    if step % config.eval_every == 0 or step == total_train_steps:
      summary = compute_loss_and_accuracy(state, ds_test)
      summary.update({"step": step})
      wandb_run.log(summary)
      progress_bar.set_postfix_str(f"Eval loss: {summary['eval_loss']:0.4f}\t Eval accuracy: {summary['eval_accuracy'] * 100 :0.2f}%")
    
    
  jax.random.normal(jax.random.key(0), ()).block_until_ready()

  wandb_run.finish()
  return 

