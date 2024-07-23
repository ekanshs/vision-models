# Adapted from https://github.com/google/flax/blob/main/examples/imagenet/models.py
# See issue #620.
# pytype: disable=wrong-arg-count

"""Flax implementation of ResNet with Layernorm and multi-head classifier.
Added utility for generating permutation spec to apply permutations
"""
from functools import partial
from typing import Any, Callable, Sequence, Tuple

from flax import linen as nn
import jax.numpy as jnp
import jax

from einops import rearrange

from permutations import PermutationSpec, permutation_spec_from_axes_to_perm, conv_axes_to_perm, dense_axes_to_perm, norm_axes_to_perm


ModuleDef = Any

class ResNetBlock(nn.Module):
  """ResNet block."""
  filters: int
  conv: ModuleDef
  norm: ModuleDef
  act: Callable
  strides: Tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(self, x,):
    residual = x
    y = self.conv(self.filters, (3, 3), self.strides)(x)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters, (3, 3))(y)
    y = self.norm(scale_init=nn.initializers.zeros_init())(y)

    if residual.shape != y.shape:
      residual = self.conv(self.filters, (1, 1),
                          self.strides, name='conv_proj')(residual)
      residual = self.norm(name='norm_proj')(residual)
    return self.act(residual + y)


class BottleneckResNetBlock(nn.Module):
  """Bottleneck ResNet block."""
  filters: int
  conv: ModuleDef
  norm: ModuleDef
  act: Callable
  strides: Tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(self, x):
    residual = x
    y = self.conv(self.filters, (1, 1))(x)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters, (3, 3), self.strides)(y)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters * 4, (1, 1))(y)
    y = self.norm(scale_init=nn.initializers.zeros_init())(y)
    
    if residual.shape != y.shape:
      residual = self.conv(self.filters * 4, (1, 1),
                          self.strides, name='conv_proj')(residual)
      residual = self.norm(name='norm_proj')(residual)

    return self.act(residual + y)


class ResNetEncoder(nn.Module):
  """ResNetEncoder."""
  stage_sizes: Sequence[int]
  block_cls: ModuleDef
  num_filters: int = 64
  dtype: Any = jnp.float32
  act: Callable = nn.relu
  conv: ModuleDef = nn.Conv
  norm: ModuleDef = nn.LayerNorm
  
  @nn.compact
  def __call__(self, x):
    conv = partial(self.conv, use_bias=False, dtype=self.dtype)
    norm = self.norm
    
    x = conv(self.num_filters, (7, 7), (2, 2),
            padding=[(3, 3), (3, 3)],
            name='conv_init')(x)
    x = norm(name='ln_init')(x)
    x = nn.relu(x)
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
    for i, block_size in enumerate(self.stage_sizes):
      for j in range(block_size):
        strides = (2, 2) if i > 0 and j == 0 else (1, 1)
        x = self.block_cls(self.num_filters * 2 ** i,
                          strides=strides,
                          conv=conv,
                          norm=norm,
                          act=self.act
                          )(x)
    x = jnp.mean(x, axis=(1, 2))
    return x

class MultiheadClassifier(nn.Module):
  """Classifier Layer."""
  num_classes : Sequence[int]
  logit_scale : Any
  dtype: Any = jnp.float32
  
  @nn.compact
  def __call__(self, x):
    logits = []
    for classes in self.num_classes:
      logits += [jnp.exp(self.logit_scale) * jnp.asarray(nn.Dense(classes, dtype=self.dtype, use_bias=False)(x), self.dtype)]
    return logits

class Classifier(nn.Module):
  """Classifier Layer."""
  num_classes : Sequence[int]
  logit_scale : Any
  dtype: Any = jnp.float32
  
  @nn.compact
  def __call__(self, x):
    logits = jnp.exp(self.logit_scale) * jnp.asarray(nn.Dense(self.num_classes, dtype=self.dtype, use_bias=False)(x), self.dtype)
    return logits


class ResNet(nn.Module):
  """ResNet with Layer Normalization"""
  stage_sizes: Sequence[int]
  block_cls: ModuleDef
  classifier: ModuleDef
  projection_dim: int
  logit_scale_init_value: float = 2.6592 # from openai/clip-vit-base-patch32
  num_filters: int = 64
  width_multiplier: int = 1
  dtype: Any = jnp.float32
  act: Callable = nn.relu
  conv: ModuleDef = nn.Conv
  norm: ModuleDef = nn.LayerNorm
  
  @nn.compact
  def __call__(self, x):
    outputs = ResNetEncoder(self.stage_sizes, 
                    self.block_cls, 
                    self.num_filters * self.width_multiplier, 
                    self.dtype, 
                    self.act, 
                    self.conv, 
                    self.norm, 
                    name="encoder")(x)
    
    image_features = nn.Dense(
        self.projection_dim,
        dtype=self.dtype,
        kernel_init=jax.nn.initializers.normal(0.02),
        use_bias=False,
        name='visual_projection'
    )(outputs)
    
    # normalize features
    image_features /= jnp.linalg.norm(image_features, axis=-1, keepdims=True)
    
    logits = self.classifier(logit_scale=self.param("logit_scale", lambda _, shape: jnp.ones(shape) * self.logit_scale_init_value, []),
                            dtype=self.dtype, 
                            name='classifier'
                            )(image_features) 
    
    return logits

  def permutation_spec(self, skip_classifier=False) -> PermutationSpec:
    p_in = None
    p_out = "P/encoder/ln_init"
    axes_to_perm = {
      **conv_axes_to_perm("encoder/conv_init", p_in, p_out),
      **norm_axes_to_perm("encoder/ln_init", p_out),
    }
    block_id = 0
    block_name = f"encoder/{self.block_cls.__name__}"
    if self.block_cls.__name__ == 'BottleneckResNetBlock':
      residual_block_axes_to_perm = residual_bottleneck_resnet_block_axes_to_perm
      projection_block_axes_to_perm = projection_bottleneck_resnet_block_axes_to_perm
      stage_sizes = self.stage_sizes
    else:
      residual_block_axes_to_perm = residual_resnet_block_axes_to_perm
      projection_block_axes_to_perm = projection_resnet_block_axes_to_perm
      for _ in range(self.stage_sizes[0]):
        axes_to_perm.update(residual_block_axes_to_perm(f"{block_name}_{block_id}",p_out))
        block_id+=1
      stage_sizes = self.stage_sizes[1:]
    
    for stage in stage_sizes:
        p_in = p_out
        p_out = f"P/{block_name}_{block_id}"
        axes_to_perm.update(projection_block_axes_to_perm(f"{block_name}_{block_id}",p_in, p_out))
        block_id+=1
        for _ in range(stage - 1):
          axes_to_perm.update(residual_block_axes_to_perm(f"{block_name}_{block_id}",p_out))
          block_id+=1
    
    p_in = p_out 
    p_out = None
    
    if not skip_classifier:
      try:
        nheads = len(self.num_classes)
      except:
        nheads = 1
      
      for i in range(nheads):
        axes_to_perm.update(dense_axes_to_perm(f"classifier/Dense_{i}", p_in, p_out))
    return permutation_spec_from_axes_to_perm(axes_to_perm)


residual_resnet_block_axes_to_perm = lambda name, p: {
      **conv_axes_to_perm(f"{name}/Conv_0", p, f"P/{name}/LayerNorm_0"),
      **norm_axes_to_perm(f"{name}/LayerNorm_0",  f"P/{name}/LayerNorm_0"),
      **conv_axes_to_perm(f"{name}/Conv_1", f"P/{name}/LayerNorm_0", p),
      **norm_axes_to_perm(f"{name}/LayerNorm_1", p),
  }
projection_resnet_block_axes_to_perm = lambda name, p_in, p_out: {
      **conv_axes_to_perm(f"{name}/Conv_0", p_in, f"P/{name}/LayerNorm_0"),
      **norm_axes_to_perm(f"{name}/LayerNorm_0",  f"P/{name}/LayerNorm_0"),
      **conv_axes_to_perm(f"{name}/Conv_1", f"P/{name}/LayerNorm_0", p_out),
      **norm_axes_to_perm(f"{name}/LayerNorm_1", p_out),
      **conv_axes_to_perm(f"{name}/conv_proj", p_in, p_out),
      **norm_axes_to_perm(f"{name}/norm_proj", p_out),
  }

residual_bottleneck_resnet_block_axes_to_perm = lambda name, p: {
      **conv_axes_to_perm(f"{name}/Conv_0", p, f"P/{name}/LayerNorm_0"),
      **norm_axes_to_perm(f"{name}/LayerNorm_0",  f"P/{name}/LayerNorm_0"),
      **conv_axes_to_perm(f"{name}/Conv_1", f"P/{name}/LayerNorm_0", f"P/{name}/LayerNorm_1"),
      **norm_axes_to_perm(f"{name}/LayerNorm_1", f"P/{name}/LayerNorm_1"),
      **conv_axes_to_perm(f"{name}/Conv_2", f"P/{name}/LayerNorm_1", p),
      **norm_axes_to_perm(f"{name}/LayerNorm_2", p),
  }
projection_bottleneck_resnet_block_axes_to_perm = lambda name, p_in, p_out: {
      **conv_axes_to_perm(f"{name}/Conv_0", p_in, f"P/{name}/LayerNorm_0"),
      **norm_axes_to_perm(f"{name}/LayerNorm_0",  f"P/{name}/LayerNorm_0"),
      **conv_axes_to_perm(f"{name}/Conv_1", f"P/{name}/LayerNorm_0", f"P/{name}/LayerNorm_1"),
      **norm_axes_to_perm(f"{name}/LayerNorm_1", f"P/{name}/LayerNorm_1"),
      **conv_axes_to_perm(f"{name}/Conv_2", f"P/{name}/LayerNorm_1", p_out),
      **norm_axes_to_perm(f"{name}/LayerNorm_2", p_out),
      **conv_axes_to_perm(f"{name}/conv_proj", p_in, p_out),
      **norm_axes_to_perm(f"{name}/norm_proj", p_out),
  }

def get_resnet(*, num_classes, **kwargs):
  try:
    nheads = len(num_classes)
    classifier = partial(MultiheadClassifier, num_classes=num_classes)
  except:
    classifier = partial(Classifier, num_classes=num_classes)
  return ResNet(classifier=classifier, **kwargs)


ResNet18 = partial(get_resnet, stage_sizes=[2, 2, 2, 2],
                    block_cls=ResNetBlock)

ResNet20 = partial(get_resnet, stage_sizes=[3, 3, 3],
                    block_cls=ResNetBlock)

ResNet34 = partial(get_resnet, stage_sizes=[3, 4, 6, 3],
                    block_cls=ResNetBlock)

ResNet50 = partial(get_resnet, stage_sizes=[3, 4, 6, 3],
                    block_cls=BottleneckResNetBlock)

ResNet101 = partial(get_resnet, stage_sizes=[3, 4, 23, 3],
                    block_cls=BottleneckResNetBlock)

ResNet152 = partial(get_resnet, stage_sizes=[3, 8, 36, 3],
                    block_cls=BottleneckResNetBlock)

ResNet200 = partial(get_resnet, stage_sizes=[3, 24, 36, 3],
                    block_cls=BottleneckResNetBlock)

ResNet18Local = partial(get_resnet, stage_sizes=[2, 2, 2, 2],
                        block_cls=ResNetBlock, conv=nn.ConvLocal)


# Used for testing only.
_ResNet1 = partial(ResNet, stage_sizes=[1], block_cls=ResNetBlock)
_ResNet1Local = partial(ResNet, stage_sizes=[1], block_cls=ResNetBlock,
                        conv=nn.ConvLocal)


def test_resnet():
  model = ResNet18(num_classes=[10,100, 397, 101], width_multiplier=4, projection_dim=512)
  batch_x = jnp.ones([128, 224, 224, 3])  # (N, H, W, C) format
  rng = jax.random.PRNGKey(0)
  params = model.init(rng, batch_x)['params']
  logits = model.apply({'params':params}, batch_x)
  print([logit.shape for logit in logits])
  model = ResNet18(num_classes=10, width_multiplier=4, projection_dim=512)
  batch_x = jnp.ones([128, 224, 224, 3])  # (N, H, W, C) format
  rng = jax.random.PRNGKey(0)
  params = model.init(rng, batch_x)['params']
  logits = model.apply({'params':params}, batch_x)
  print(logits.shape)

if __name__=='__main__':
  test_resnet()
