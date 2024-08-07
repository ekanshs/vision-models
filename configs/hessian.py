from configs import common
from configs.common import TRAINING_SCHEDULE
import ml_collections

def get_config(config_str):
  """Returns default parameters for training/finetuning a `model` on `dataset`."""
  model, dataset = config_str.split(',')
  config = ml_collections.ConfigDict(common.get_config().to_dict())
  config.model = model
  config.dataset = dataset
  for dataset in common.DATASET_PRESETS.keys():
    config[dataset] = common.DATASET_PRESETS[dataset]
  config.training_schedule = TRAINING_SCHEDULE['cosine']
  config.compute_trace = True
  config.compute_density = True
  config.compute_top_n_eig = True
  config.full_ds_estimate = True
  config.nbatches = 20
  config.n_eigs = 50
  config.top_n_eigs = 10
  return config
