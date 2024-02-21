import gradio as gr
from flax.training import checkpoints
from flax import jax_utils
from diffusers import CLIPTokenizer, FlaxCLIPTextModel, FlaxAutoencoderKL, FlaxUNet2DConditionModel, FlaxStableDiffusionControlNetPipeline
from codi.controlnet_flax import FlaxControlNetModel

MODEL_NAME = "CompVis/stable-diffusion-v1-4"

tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer")
text_encoder = FlaxCLIPTextModel.from_pretrained(
    MODEL_NAME,
    subfolder="text_encoder",
    dtype="float16",
    from_pt=True,
)

vae, vae_params = FlaxAutoencoderKL.from_pretrained(
    MODEL_NAME,
    subfolder="vae",
    dtype="float16",
    from_pt=True,
)
unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
    MODEL_NAME,
    subfolder="unet",
    dtype="float16",
    from_pt=True,
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
    tokenizer=tokenizer,
    controlnet=controlnet,
    safety_checker=None,
    dtype="float16",
    from_pt=True,
)
controlnet_params = checkpoints.restore_checkpoint("experiments/checkpoint_72001", target=None)

pipeline_params = jax_utils.replicate(pipeline_params)


def infer(prompt, negative_prompt, steps, cfgr):
    # your inference function for canny control 
    return image

with gr.Blocks(theme='gradio/soft') as demo:
    gr.Markdown("## Conditional Distillation (CoDi) with Different Controls")
    gr.Markdown("[\[Paper\]](https://arxiv.org/abs/2310.01407) [\[Project Page\]](https://fast-codi.github.io)")

    with gr.Tab("CoDi on Text-to-Image"):
        
        with gr.Row():
            with gr.Column():
                prompt_input_canny = gr.Textbox(label="Prompt")
                negative_prompt_canny = gr.Textbox(label="Negative Prompt")
            output = gr.Image(label="Output Image")

        with gr.Row():
            num_inference_steps = gr.Slider(2, 8, value=4, step=1, label="Steps")
            guidance_scale = gr.Slider(2.0, 14.0, value=7.5, step=0.5, label='Guidance Scale')
        submit_btn = gr.Button(value = "Submit")
        inputs = [
            prompt_input_canny,
            negative_prompt_canny,
            num_inference_steps,
            guidance_scale
        ]
        submit_btn.click(fn=infer, inputs=inputs, outputs=[output])

demo.launch()