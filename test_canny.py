import jax
import numpy as np
import jax.numpy as jnp
from flax.training import checkpoints
from diffusers import FlaxUNet2DConditionModel, FlaxAutoencoderKL, FlaxDDIMScheduler
from codi.controlnet_flax import FlaxControlNetModel
from codi.pipeline_flax_controlnet import FlaxStableDiffusionControlNetPipeline
from transformers import CLIPTokenizer, FlaxCLIPTextModel
from flax.training.common_utils import shard
from flax.jax_utils import replicate
from PIL import Image

rng = jax.random.PRNGKey(0)

MODEL_NAME = "stabilityai/stable-diffusion-2-1"

unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
    MODEL_NAME,
    subfolder="unet",
    revision="flax",
    dtype=jnp.float32,
)
vae, vae_params = FlaxAutoencoderKL.from_pretrained(
    MODEL_NAME,
    subfolder="vae",
    revision="flax",
    dtype=jnp.float32,
)
text_encoder = FlaxCLIPTextModel.from_pretrained(
    MODEL_NAME,
    subfolder="text_encoder",
    revision="flax",
    dtype=jnp.float32,
)
tokenizer = CLIPTokenizer.from_pretrained(
    MODEL_NAME,
    subfolder="tokenizer",
    revision="flax",
    dtype=jnp.float32,
)

controlnet, controlnet_params = FlaxControlNetModel.from_pretrained(
    'experiments/99000',
    dtype=jnp.float32,
)

scheduler = FlaxDDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    trained_betas=None,
    set_alpha_to_one=True,
    steps_offset=0,
)
scheduler_state = scheduler.create_state()

pipeline = FlaxStableDiffusionControlNetPipeline(
    vae,
    text_encoder,
    tokenizer,
    unet,
    controlnet,
    scheduler,
    None,
    None,
    onestepode_sample_eps='vprediction',
    dtype=jnp.float32,
)

pipeline_params = {
    "vae": vae_params,
    "unet": unet_params,
    "text_encoder": text_encoder.params,
    "scheduler": scheduler_state,
    "controlnet": controlnet_params,
}

num_samples = jax.device_count()
rng = jax.random.split(rng, jax.device_count())

prompts = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
negative_prompts = "monochrome, lowres, bad anatomy, worst quality, low quality"

prompt_ids = pipeline.prepare_text_inputs([prompts] * num_samples)
negative_prompt_ids = pipeline.prepare_text_inputs([negative_prompts] * num_samples)

pipeline_params = replicate(pipeline_params)
prompt_ids = shard(prompt_ids)
negative_prompt_ids = shard(negative_prompt_ids)

validation_image = Image.open("figs/control_bird_canny.png").convert("RGB")
processed_image = pipeline.prepare_image_inputs(
    num_samples * [validation_image]
)
processed_image = shard(processed_image)

output = pipeline(
    prompt_ids=prompt_ids,
    image=processed_image,
    params=pipeline_params,
    prng_seed=rng,
    num_inference_steps=4,
    guidance_scale=4.5,
    neg_prompt_ids=negative_prompt_ids,
    jit=True,
).images

output_images = pipeline.numpy_to_pil(np.asarray(output.reshape((num_samples,) + output.shape[-3:])))
output_images[0].save("experiments/generated_image.png")
