
import jax
import jax.numpy as jnp
from .resnet import (ResNet18 as ResNet18, 
                    ResNet20 as ResNet20, 
                    ResNet34 as ResNet34, 
                    ResNet50 as ResNet50,
                    ResNet101 as ResNet101)

from .vgg import (VGG16 as VGG16, 
                  VGG19 as VGG19)

from .clip_model import (ViTB16 as ViTB16, 
                         ViTB32 as ViTB32, 
                         ViTL14 as ViTL14, 
                         get_zero_shot_params as get_zero_shot_params)

def create_model(*, model_cls, num_classes, half_precision, **kwargs):
  platform = jax.local_devices()[0].platform
  if half_precision:
    if platform == 'tpu':
      model_dtype = jnp.bfloat16
    else:
      model_dtype = jnp.float16
  else:
    model_dtype = jnp.float32

  return model_cls(num_classes=num_classes, dtype=model_dtype, **kwargs)

