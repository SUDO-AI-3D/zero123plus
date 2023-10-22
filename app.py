import sys
import torch
import rembg
from PIL import Image
import streamlit as st


@st.cache_data
def check_dependencies():
    reqs = []
    try:
        import diffusers
    except ImportError:
        import traceback
        traceback.print_exc()
        print("Error: `diffusers` not found.", file=sys.stderr)
        reqs.append("diffusers==0.20.2")
    else:
        if not diffusers.__version__.startswith("0.20"):
            print(
                f"Warning: You are using an unsupported version of diffusers ({diffusers.__version__}), which may lead to performance issues.",
                file=sys.stderr
            )
            print("Recommended version is `diffusers==0.20.2`.", file=sys.stderr)
    try:
        import transformers
    except ImportError:
        import traceback
        traceback.print_exc()
        print("Error: `transformers` not found.", file=sys.stderr)
        reqs.append("transformers==4.29.2")
    if torch.__version__ < '2.0':
        try:
            import xformers
        except ImportError:
            print("Warning: You are using PyTorch 1.x without a working `xformers` installation.", file=sys.stderr)
            print("You may see a significant memory overhead when running the model.", file=sys.stderr)
    if len(reqs):
        print(f"Info: Fix all dependency errors with `pip install {' '.join(reqs)}`.")


@st.cache_resource
def load_zero123plus_pipeline():
    from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
    pipeline = DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.1", custom_pipeline="sudo-ai/zero123plus-pipeline",
    torch_dtype=torch.float16
    )
    # Feel free to tune the scheduler
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipeline.scheduler.config, timestep_spacing='trailing'
    )
    if torch.cuda.is_available():
        pipeline.to('cuda:0')
    return pipeline


check_dependencies()
pipeline = load_zero123plus_pipeline()
torch.set_grad_enabled(False)

st.title("Zero123++ Demo")
# st.caption("For faster inference without waiting in queue, you may clone the space and run it yourself.")
prog = st.progress(0.0, "Idle")
with st.form("imgform"):
    pic = st.file_uploader("Upload an Image", key='imageinput')
    if st.form_submit_button("Run!"):
        img = Image.open(pic)
        st.image(img)
        prog.progress(0.1, "Preparing Inputs")
        pipeline.set_progress_bar_config(disable=True)
        result = pipeline(
            img,
            num_inference_steps=75,
            callback=lambda i, t, latents: prog.progress(0.1 + 0.8 * i / 75, "Diffusion Step %d" % i)
        ).images[0]
        prog.progress(0.9, "Post Processing")
        result = rembg.remove(result)
        st.image(result)
        prog.progress(1.0, "Idle")
