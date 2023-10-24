import torch
import requests
import rembg
import random
import gradio as gr
import numpy

from PIL import Image
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

# Load the pipeline
pipeline = DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.1", custom_pipeline="sudo-ai/zero123plus-pipeline",
    torch_dtype=torch.float16
)

# Feel free to tune the scheduler!
# `timestep_spacing` parameter is not supported in older versions of `diffusers`
# so there may be performance degradations
# We recommend using `diffusers==0.20.2`
pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipeline.scheduler.config, timestep_spacing='trailing'
)
pipeline.to('cuda:0')


def inference(input_img, num_inference_steps, guidance_scale, seed ):
    # Download an example image.
    cond = Image.open(input_img)
    if seed==0:
        seed = random.randint(1, 1000000)
        
    # Run the pipeline!
    #result = pipeline(cond, num_inference_steps=75).images[0]
    result = pipeline(cond, num_inference_steps=num_inference_steps, 
                  guidance_scale=guidance_scale, 
                  generator=torch.Generator(pipeline.device).manual_seed(int(seed))).images[0]

    # for general real and synthetic images of general objects
    # usually it is enough to have around 28 inference steps
    # for images with delicate details like faces (real or anime)
    # you may need 75-100 steps for the details to construct
    
    #result.show()
    #result.save("output.png")
    return result

def remove_background(result):
    print(type(result))
    # Check if the variable is a PIL Image
    if isinstance(result, Image.Image):
        result = rembg.remove(result)
    # Check if the variable is a str filepath
    elif isinstance(result, str):
        result = Image.open(result)
        result = rembg.remove(result)
    elif isinstance(result, numpy.ndarray):
      print('here ELIF 2')
      # Convert the NumPy array to a PIL Image
      result = Image.fromarray(result)
      result = rembg.remove(result)
    return result


abstract = '''Zero123++ is an image-conditioned diffusion model for generating 3D-consistent multi-view images from a single input view. To take full advantage of pretrained 2D generative priors, authors have developed various conditioning and training schemes to minimize the effort of finetuning from off-the-shelf image diffusion models such as Stable Diffusion. Zero123++ excels in producing high-quality, consistent multi-view images from a single image, overcoming common issues like texture degradation and geometric misalignment. Furthermore, authors showcase the feasibility of training a ControlNet on Zero123++ for enhanced control over the generation process. 
'''
# Create a Gradio interface for the Zero123++ model
with gr.Blocks() as demo:
# Display a title
    gr.HTML("<h1><center> Interactive WebUI : Zero123++ </center></h1>")
    with gr.Row():
      with gr.Column(scale=1):
        gr.HTML('''<img src='https://huggingface.co/spaces/ysharma/Zero123PlusDemo/resolve/main/teaser-low.jpg'>''')
      with gr.Column(scale=5):
        gr.HTML("<h2>A Single Image to Consistent Multi-view Diffusion Base Model</h2>")
        gr.HTML('''<a href='https://arxiv.org/abs/2310.15110' target='_blank'>ArXiv</a> - <a href='https://github.com/SUDO-AI-3D/zero123plus/tree/main' target='_blank'>Code</a>''')
        gr.HTML(f'<b>Abstract:</b> {abstract}')
    with gr.Row():
      # Input section: Allow users to upload an image
      with gr.Column():
          input_img = gr.Image(label='Input Image', type='filepath')
      
      # Output section: Display the Zero123++ output image
      with gr.Column():
          output_img = gr.Image(label='Zero123++ Output')
    
    # Submit button to initiate the inference
    btn = gr.Button('Submit')

    # Advanced options section with accordion for hiding/showing
    with gr.Accordion("Advanced options:", open=False):
        rm_in_bkg = gr.Checkbox(label='Remove Input Background', info='Select this checkbox to run an extra background removal pass like rembg to remove background in Input image ')
        rm_out_bkg = gr.Checkbox(label='Remove Output Background', info='Select this checkbox to run an extra background removal pass like rembg to remove the gray background for Output image')
        num_inference_steps = gr.Slider(label="Number of Inference Steps", minimum=15, maximum=100, step=1, value=75, interactive=True)
        guidance_scale = gr.Slider(label="Classifier Free Guidance Scale", minimum=1.00, maximum=10.00, step=0.1, value=4.0, interactive=True)
        seed = gr.Number(0, label='Seed', info='A random seed value will be used if seed is set to 0')
    

    btn.click(inference, [input_img, num_inference_steps, guidance_scale, seed ], output_img)
    rm_in_bkg.input(remove_background, input_img, input_img)
    rm_out_bkg.input(remove_background, output_img, output_img)
    


demo.launch(debug=False)
