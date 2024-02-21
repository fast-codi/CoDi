import jax
import numpy as np
import jax.numpy as jnp
from flax.jax_utils import replicate
from flax.training import checkpoints
from flax.training.common_utils import shard
from diffusers.utils import make_image_grid
from diffusers import FlaxControlNetModel, FlaxUNet2DConditionModel
from codi.controlnet_flax import FlaxControlNetModel
from codi.pipeline_flax_controlnet import FlaxStableDiffusionControlNetPipeline

rng = jax.random.PRNGKey(0)

MODEL_NAME = "CompVis/stable-diffusion-v1-4"

unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
    MODEL_NAME,
    subfolder="unet",
    revision="flax",
    dtype=jnp.float32,
)
controlnet = FlaxControlNetModel(
    in_channels=unet.config.in_channels,
    down_block_types=unet.config.down_block_types,
    only_cross_attention=unet.config.only_cross_attention,
    block_out_channels=unet.config.block_out_channels,
    layers_per_block=unet.config.layers_per_block,
    attention_head_dim=unet.config.attention_head_dim,
    cross_attention_dim=unet.config.cross_attention_dim,
    use_linear_projection=unet.config.use_linear_projection,
    flip_sin_to_cos=unet.config.flip_sin_to_cos,
    freq_shift=unet.config.freq_shift,
)
pipeline, pipeline_params = FlaxStableDiffusionControlNetPipeline.from_pretrained(
    MODEL_NAME,
    controlnet=controlnet,
    dtype=jnp.float32,
    safety_checker=None,
    revision="flax",
)
controlnet_params = checkpoints.restore_checkpoint("experiments/checkpoint_72001", target=None)

pipeline_params["controlnet"] = controlnet_params['ema_params']['image_a']['params']
pipeline_params["unet"] = unet_params

num_samples = jax.device_count()
rng = jax.random.split(rng, jax.device_count())

prompts = "oranges"
negative_prompts = "monochrome, lowres, bad anatomy, worst quality, low quality"

prompt_ids = pipeline.prepare_text_inputs([prompts] * num_samples)
negative_prompt_ids = pipeline.prepare_text_inputs([negative_prompts] * num_samples)

output = pipeline(
    prompt_ids=prompt_ids,
    image=None,
    params=pipeline_params,
    prng_seed=rng,
    num_inference_steps=4,
    guidance_scale=8.5,
    neg_prompt_ids=negative_prompt_ids,
    jit=False,
).images

output_images = pipeline.numpy_to_pil(np.asarray(output.reshape((num_samples,) + output.shape[-3:])))
output_images[0].save("figs/generated_image.png")
