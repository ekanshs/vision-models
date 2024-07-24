r"""Train/Finetune a model.

Example for training CLIP on CIFAR10:

python main.py --job_type=train --config=configs/default.py:ResNet18;cifar10;sgd --expdir=./experiments/ 

"""

from configs.common import TRAIN_OPTIMIZER_PRESETS, DATASET_PRESETS, TRAINING_SCHEDULE
from configs import common
import ml_collections

def get_config(model_dataset_optimizer):
  """Returns default parameters for training/finetuning a `model` on `dataset`."""
  model, dataset, opt, decay_schedule = model_dataset_optimizer.split(',')
  config = with_model_dataset_opt(common.get_config(), model, dataset, opt, decay_schedule)  
  return config

def with_model_dataset_opt(config: ml_collections.ConfigDict, 
                            model: str, 
                            dataset: str, 
                            opt:str,
                            decay_schedule: str
                          ):
  config = ml_collections.ConfigDict(config.to_dict())
  config.model = model
  config.dataset = dataset
  
  config[dataset] = DATASET_PRESETS[dataset]

  config.training_schedule = TRAINING_SCHEDULE[decay_schedule]
  config.training_schedule.num_epochs = NUM_EPOCHS_PER_DATASET[dataset]
  config.training_schedule.warmup_epochs = 1
  config.optimizer = TRAIN_OPTIMIZER_PRESETS[opt]
  config.from_pretrained = False
  config.train_classifier_at_init = False
  return config.lock()


NUM_EPOCHS_PER_DATASET = {
  'cifar10': 180,
  'cifar100': 180,
  'cars': 180,
  'dtd': 180,
  'eurosat': 180,
  'gtsrb': 180,
  'mnist': 180,
  'resisc45': 180,
  'sun397': 180,
  'svhn_cropped': 180,
  'food101': 180, 
  'imagenet2012' : 120
}