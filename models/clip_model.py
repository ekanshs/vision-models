from functools import partial
from typing import Any, Callable, Sequence, Tuple

from flax import linen as nn
import jax.numpy as jnp
import jax

from transformers import (AutoConfig, 
                          FlaxCLIPModel,
                          FlaxCLIPVisionModel
                          )

from data.templates import get_templates
from data.input_pipeline import get_dataset_info
import tqdm

ModuleDef=Any

vision_module = FlaxCLIPVisionModel.module_class

class CLIPModelwithClassifier(nn.Module):
  config: Any
  vision_module : ModuleDef
  classifier_module : ModuleDef
  dtype: jnp.dtype = jnp.float32  

  @nn.compact
  def __call__(self, pixel_values):
    encoder_outputs = self.vision_module(self.config.vision_config, dtype=self.dtype, name='encoder')(
            pixel_values=pixel_values
        )
    pooled_outputs = encoder_outputs[1]
    image_features = nn.Dense(
        self.config.projection_dim,
        dtype=self.dtype,
        kernel_init=jax.nn.initializers.normal(0.02),
        use_bias=False,
        name='visual_projection'
    )(pooled_outputs)
    
    # normalize features
    image_features /= jnp.linalg.norm(image_features, axis=-1, keepdims=True)
    
    logits = self.classifier_module(logit_scale = self.param("logit_scale", lambda _, shape: jnp.ones(shape) * self.config.logit_scale_init_value, []),
                                    dtype=self.dtype, 
                                    name='classifier'
                                    )(image_features) 
    return logits
  
  def permutation_spec(self, skip_classifier=False):
    ## TBD:
    pass

class MultiheadClassifier(nn.Module):
  """Classifier Layer."""
  num_classes : Sequence[int]
  logit_scale: Any
  dtype: Any = jnp.float32
  
  @nn.compact
  def __call__(self, x):
    logits = []
    for classes in self.num_classes:
      logits += [jnp.exp(self.logit_scale) * jnp.asarray(nn.Dense(classes, dtype=self.dtype, use_bias=False)(x), self.dtype)]
    return logits

class Classifier(nn.Module):
  """Classifier Layer."""
  num_classes : int
  logit_scale: Any
  dtype: Any = jnp.float32
  
  @nn.compact
  def __call__(self, x):
    logits = jnp.exp(self.logit_scale) * jnp.asarray(nn.Dense(self.num_classes, dtype=self.dtype, use_bias=False)(x), self.dtype)
    return logits

def get_clip_model_with_classifier(*, num_classes, **kwargs):
  try:
    nheads = len(num_classes)
    classifier = partial(MultiheadClassifier, num_classes=num_classes)
  except:
    classifier = partial(Classifier, num_classes=num_classes)
  return CLIPModelwithClassifier(classifier_module=classifier, **kwargs)


def ViTB16(*, num_classes, dtype, **kwargs):
  model_config = AutoConfig.from_pretrained('openai/clip-vit-base-patch16')
  return get_clip_model_with_classifier(num_classes=num_classes, 
                                        vision_module=vision_module, 
                                        config=model_config,
                                        dtype=dtype)

def ViTB32(*, num_classes, dtype, **kwargs):
  model_config = AutoConfig.from_pretrained('openai/clip-vit-base-patch32')
  return get_clip_model_with_classifier(num_classes=num_classes, 
                                        vision_module=vision_module, 
                                        config=model_config,
                                        dtype=dtype)

def ViTL14(*, num_classes, dtype, **kwargs):
  model_config = AutoConfig.from_pretrained('openai/clip-vit-large-patch14')
  return get_clip_model_with_classifier(num_classes=num_classes, 
                                        vision_module=vision_module, 
                                        config=model_config,
                                        dtype=dtype)

def get_zero_shot_params(model_name, dataset=None, datasets=None):
  assert (dataset is not None) or (datasets is not None)
  model = FlaxCLIPModel.from_pretrained(model_name)
  tokenizer = FlaxCLIPModel.from_pretrained(model_name)
  if dataset is not None:
    return _get_zero_shot_params(model, tokenizer, dataset)
  return _get_multihead_zero_shot_params(model, tokenizer, datasets)


def _get_zero_shot_params(model, tokenizer, dataset_name):
  classifier_params = _build_zero_shot_classification_params(model, tokenizer, dataset_name)
  params = {
    'encoder': {'vision_model': model.params['vision_model']}, 
    'visual_projection': model.params['visual_projection'],
    'classifier': {'Dense_0': {'kernel': classifier_params}}, 
    'logit_scale': model.params['logit_scale'],
  }
  
  return params

def _get_multihead_zero_shot_params(model, tokenizer, dataset_name_ls):
  classifier_params_ls = []
  classifier_params_ls = [_build_zero_shot_classification_params(model, tokenizer, dataset_name) for dataset_name in dataset_name_ls]

  params = {
    'encoder': {'vision_model': model.params['vision_model']}, 
    'visual_projection': model.params['visual_projection'],
    'classifier': {f'Dense_{i}': {'kernel': classifier_params_ls[i]} for i in range(len(dataset_name_ls))} , 
    'logit_scale': model.params['logit_scale']
  }
  return params

def _build_zero_shot_classification_params(model, tokenizer, dataset_name):
  template = get_templates(dataset_name)

  d_info = get_dataset_info(dataset_name, 'train')
  
  print('Building classification head.')
  zeroshot_classifier_params = []
  for class_id in tqdm(range(d_info['num_classes'])):
      texts = []
      for t in template:
          texts.append(t(d_info['int2str'](class_id)))
      
      inputs = tokenizer(text=texts, return_tensors="np", padding=True)
      embeddings = model.get_text_features(**inputs) # embed with text encoder
      
      embeddings /= jnp.linalg.norm(embeddings, axis=-1, keepdims=True)
      embeddings = embeddings.mean(axis=0)
      embeddings /= jnp.linalg.norm(embeddings, axis=-1, keepdims=True)
      zeroshot_classifier_params.append(embeddings)

  zeroshot_classifier_params = jnp.stack(zeroshot_classifier_params, axis=1)
  return zeroshot_classifier_params

