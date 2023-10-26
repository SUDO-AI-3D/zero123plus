# Zero123++: a Single Image to Consistent Multi-view Diffusion Base Model

![Teaser](resources/teaser-low.jpg)

[Report](https://arxiv.org/abs/2310.15110) [Official Demo](https://huggingface.co/spaces/sudo-ai/zero123plus-demo-space) [Demo by @yvrjsharma](https://huggingface.co/spaces/ysharma/Zero123PlusDemo) [Google Colab](https://colab.research.google.com/drive/1_5ECnTOosRuAsm2tUp0zvBG0DppL-F3V?usp=sharing)

## Get Started

To generate multi-view images from a single input image using the Zero123++ model, follow these steps:

1. Install the required dependencies:

   - `torch` (recommended version 2.0 or higher)
   - `diffusers` (recommended version 0.20.2)
   - `transformers`
   
   If you are using `torch` version 1.x, it is recommended to install `xformers` for efficient attention computation. The code also runs on older versions of `diffusers`, but you may see a decrease in model performance.

2. You are all set! The provided code includes a custom pipeline for `diffusers`, so no extra code is required.

3. To generate multi-view images from a single input image, use the following Python code (also see [examples/img_to_mv.py](examples/img_to_mv.py)):

```python
import torch
import requests
from PIL import Image
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

# Load the pipeline
pipeline = DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.1", custom_pipeline="sudo-ai/zero123plus-pipeline",
    torch_dtype=torch.float16
)

# Feel free to tune the scheduler!
# `timestep_spacing` parameter is not supported in older versions of `diffusers`,
# so there may be performance degradations.
# We recommend using `diffusers==0.20.2`.
pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipeline.scheduler.config, timestep_spacing='trailing'
)
pipeline.to('cuda:0')

# Download an example image.
cond = Image.open(requests.get("https://d.skis.ltd/nrp/sample-data/lysol.png", stream=True).raw)

# Run the pipeline!
result = pipeline(cond, num_inference_steps=75).images[0]
# for general real and synthetic images of general objects,
# usually it is enough to have around 28 inference steps.
# for images with delicate details like faces (real or anime),
# you may need 75-100 steps for the details to construct.

result.show()
result.save("output.png")

This example requires ~5.7GB VRAM to operate.

## Models

The models are available at [https://huggingface.co/sudo-ai](https://huggingface.co/sudo-ai):

+ `sudo-ai/zero123plus-v1.1`, base Zero123++ model release (v1.1).
+ `sudo-ai/controlnet-zp11-depth-v1` depth ControlNet checkpoint release (v1) for Zero123++ (v1.1).

The source code for diffusers custom pipeline is available in the [diffusers-support](diffusers-support) directory.

## Camera Poses

Output views are a fixed set of camera poses relative to the input view:

+ Azimuth: `30, 90, 150, 210, 270, 330`.
+ Elevation: `30, -20, 30, -20, 30, -20`.

## Running Demo Locally

You will need to install extra dependencies:
```
pip install -r requirements.txt
```

Then run `streamlit run app.py`.

For Gradio Demo you can run `python gradio_app.py`.

## Citation

If you found Zero123++ helpful, please cite our report:
```bibtex
@misc{shi2023zero123plus,
      title={Zero123++: a Single Image to Consistent Multi-view Diffusion Base Model}, 
      author={Ruoxi Shi and Hansheng Chen and Zhuoyang Zhang and Minghua Liu and Chao Xu and Xinyue Wei and Linghao Chen and Chong Zeng and Hao Su},
      year={2023},
      eprint={2310.15110},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
