import gradio as gr

def infer_canny(prompt, negative_prompt, image, steps, cfgr):
    # your inference function for canny control 
    return image

with gr.Blocks(theme='gradio/soft') as demo:
    gr.Markdown("## Conditional Distillation (CoDi) with Different Controls")
    gr.Markdown("[\[Paper\]](https://arxiv.org/abs/2310.01407) [\[Project Page\]](https://fast-codi.github.io)")

    with gr.Tab("CoDi on Canny Filter "):
        prompt_input_canny = gr.Textbox(label="Prompt")
        negative_prompt_canny = gr.Textbox(label="Negative Prompt")
        with gr.Row():
            canny_input = gr.Image(label="Input Image")
            canny_output = gr.Image(label="Output Image")

        with gr.Row():
            num_inference_steps = gr.Slider(2, 8, value=4, step=1, label="Steps")
            guidance_scale = gr.Slider(2.0, 14.0, value=7.0, step=0.5, label='Guidance Scale')
        submit_btn = gr.Button(value = "Submit")
        canny_inputs = [
            prompt_input_canny,
            negative_prompt_canny,
            canny_input,
            num_inference_steps,
            guidance_scale
        ]
        submit_btn.click(fn=infer_canny, inputs=canny_inputs, outputs=[canny_output])

demo.launch()