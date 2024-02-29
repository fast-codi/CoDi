import gradio as gr
import jax
import numpy as np
import jax.numpy as jnp
from diffusers import FlaxControlNetModel, FlaxUNet2DConditionModel, FlaxAutoencoderKL, FlaxDDIMScheduler
from codi.controlnet_flax import FlaxControlNetModel
from codi.pipeline_flax_controlnet import FlaxStableDiffusionControlNetPipeline
from transformers import CLIPTokenizer, FlaxCLIPTextModel
from flax.training.common_utils import shard
from flax.jax_utils import replicate
from PIL import Image


MODEL_NAME = "stabilityai/stable-diffusion-2-1"

unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
    MODEL_NAME,
    subfolder="unet",
    revision="main",
    from_pt=True,
    dtype=jnp.bfloat16,
)
vae, vae_params = FlaxAutoencoderKL.from_pretrained(
    MODEL_NAME,
    subfolder="vae",
    revision="main",
    from_pt=True,
    dtype=jnp.bfloat16,
)
text_encoder = FlaxCLIPTextModel.from_pretrained(
    MODEL_NAME,
    subfolder="text_encoder",
    revision="main",
    from_pt=True,
    dtype=jnp.bfloat16,
)
tokenizer = CLIPTokenizer.from_pretrained(
    MODEL_NAME,
    subfolder="tokenizer",
    revision="main",
    dtype=jnp.bfloat16,
)

controlnet, controlnet_params = FlaxControlNetModel.from_pretrained(
    'experiments/canny_99000',
    dtype=jnp.bfloat16,
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
    dtype=jnp.float32,
)

pipeline_params = {
    "vae": vae_params,
    "unet": unet_params,
    "text_encoder": text_encoder.params,
    "scheduler": scheduler_state,
    "controlnet": controlnet_params,
}
pipeline_params = replicate(pipeline_params)

def infer(seed, image, prompt, negative_prompt, steps, cfgr):
    rng = jax.random.PRNGKey(int(seed))

    num_samples = jax.device_count()
    rng = jax.random.split(rng, num_samples)

    prompt_ids = pipeline.prepare_text_inputs([prompt] * num_samples)
    negative_prompt_ids = pipeline.prepare_text_inputs([negative_prompt] * num_samples)

    prompt_ids = shard(prompt_ids)
    negative_prompt_ids = shard(negative_prompt_ids)

    processed_image = pipeline.prepare_image_inputs(
        num_samples * [Image.fromarray(image)]
    )
    processed_image = shard(processed_image)

    output = pipeline(
        prompt_ids=prompt_ids,
        image=processed_image,
        params=pipeline_params,
        prng_seed=rng,
        num_inference_steps=int(steps),
        guidance_scale=float(cfgr),
        neg_prompt_ids=negative_prompt_ids,
        jit=True,
    ).images

    output_images = pipeline.numpy_to_pil(np.asarray(output.reshape((num_samples,) + output.shape[-3:])))
    return output_images

with gr.Blocks(theme='gradio/soft') as demo:
    gr.Markdown("## CoDi: Conditional Diffusion Distillation for Higher-Fidelity and Faster Image Generation")
    gr.Markdown("[\[Paper\]](https://arxiv.org/abs/2310.01407) [\[Project Page\]](https://fast-codi.github.io) [\[Code\]](https://github.com/fast-codi/CoDi)")

    with gr.Tab("CoDi on Canny-to-Image"):

        with gr.Row():
            with gr.Column():
                gr.Radio(["stabilityai/stable-diffusion-2-1"], value="stabilityai/stable-diffusion-2-1", label="baseline model", info="Choose the undistilled baseline model")
            with gr.Column():
                gr.Radio(["CoDi/canny-to-image-v0-1"], value="CoDi/canny-to-image-v0-1", label="distilled codi", info="Choose the distilled conditional model")

        with gr.Row():
            with gr.Column():
                canny_input = gr.Image(label="Input Canny Image")
                prompt_input = gr.Textbox(label="Prompt")
                negative_prompt = gr.Textbox(label="Negative Prompt", value="monochrome, lowres, bad anatomy, worst quality, low quality")
                seed = gr.Number(label="Seed", value=0)
            output = gr.Gallery(label="Output Images")

        with gr.Row():
            num_inference_steps = gr.Slider(2, 50, value=4, step=1, label="Steps")
            guidance_scale = gr.Slider(2.0, 14.0, value=7.5, step=0.5, label='Guidance Scale')
        submit_btn = gr.Button(value = "Submit")
        inputs = [
            seed,
            canny_input,
            prompt_input,
            negative_prompt,
            num_inference_steps,
            guidance_scale
        ]
        submit_btn.click(fn=infer, inputs=inputs, outputs=[output])

        with gr.Row():
            gr.Examples(
                examples=[["birds", "figs/control_bird_canny.png"]],
                inputs=[prompt_input, canny_input],
                fn=infer
            )

demo.launch(max_threads=1, share=True)
