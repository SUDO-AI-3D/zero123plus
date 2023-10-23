import sys
import torch
import rembg
from PIL import Image
import streamlit as st


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


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
    left, right = st.columns(2)
    with left:
        rem_input_bg = st.checkbox("Remove Input Background")
    with right:
        rem_output_bg = st.checkbox("Remove Output Background")
    num_inference_steps = st.slider("Number of Inference Steps", 15, 100, 28)
    st.caption("Diffusion Steps. For general real or synthetic objects, around 28 is enough. For objects with delicate details such as faces (either realistic or illustration), you may need 75 or more steps.")
    seed = st.text_input("Seed", "1337")
    if st.form_submit_button("Run!"):
        seed = int(seed)
        torch.manual_seed(seed)
        img = Image.open(pic)
        left, right = st.columns(2)
        with left:
            st.image(img)
            st.caption("Input Image")
            img = expand2square(img, (127, 127, 127, 0))
        if rem_input_bg:
            with right:
                img = rembg.remove(img, bgcolor=[127, 127, 127, 255])
                st.image(img)
                st.caption("Input (Background Removed)")
        prog.progress(0.1, "Preparing Inputs")
        pipeline.set_progress_bar_config(disable=True)
        result = pipeline(
            img,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator(pipeline.device).manual_seed(seed),
            callback=lambda i, t, latents: prog.progress(0.1 + 0.8 * i / num_inference_steps, "Diffusion Step %d" % i)
        ).images[0]
        prog.progress(0.9, "Post Processing")
        left, right = st.columns(2)
        with left:
            st.image(result)
            st.caption("Result")
        if rem_output_bg:
            result = rembg.remove(result)
            with right:
                st.image(result)
                st.caption("Result (Background Removed)")
        prog.progress(1.0, "Idle")
