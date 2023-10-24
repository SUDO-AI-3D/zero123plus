import os
import sys
import numpy
import torch
import rembg
import threading
import urllib.request
from PIL import Image
import streamlit as st
import huggingface_hub


img_example_counter = 0
iret_base = 'resources/examples'
iret = [
    dict(rimageinput=os.path.join(iret_base, x), dispi=os.path.join(iret_base, x))
    for x in sorted(os.listdir(iret_base))
]


class SAMAPI:
    predictor = None

    @staticmethod
    @st.cache_resource
    def get_instance(sam_checkpoint=None):
        if SAMAPI.predictor is None:
            if sam_checkpoint is None:
                sam_checkpoint = "tmp/sam_vit_h_4b8939.pth"
            if not os.path.exists(sam_checkpoint):
                os.makedirs('tmp', exist_ok=True)
                urllib.request.urlretrieve(
                    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                    sam_checkpoint
                )
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            model_type = "default"

            from segment_anything import sam_model_registry, SamPredictor

            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device=device)

            predictor = SamPredictor(sam)
            SAMAPI.predictor = predictor
        return SAMAPI.predictor

    @staticmethod
    def segment_api(rgb, mask=None, bbox=None, sam_checkpoint=None):
        """

        Parameters
        ----------
        rgb : np.ndarray h,w,3 uint8
        mask: np.ndarray h,w bool

        Returns
        -------

        """
        np = numpy
        predictor = SAMAPI.get_instance(sam_checkpoint)
        predictor.set_image(rgb)
        if mask is None and bbox is None:
            box_input = None
        else:
            # mask to bbox
            if bbox is None:
                y1, y2, x1, x2 = np.nonzero(mask)[0].min(), np.nonzero(mask)[0].max(), np.nonzero(mask)[1].min(), \
                                 np.nonzero(mask)[1].max()
            else:
                x1, y1, x2, y2 = bbox
            box_input = np.array([[x1, y1, x2, y2]])
        masks, scores, logits = predictor.predict(
            box=box_input,
            multimask_output=True,
            return_logits=False,
        )
        mask = masks[-1]
        return mask


def image_examples(samples, ncols, return_key=None, example_text="Examples"):
    global img_example_counter
    trigger = False
    with st.expander(example_text, True):
        for i in range(len(samples) // ncols):
            cols = st.columns(ncols)
            for j in range(ncols):
                idx = i * ncols + j
                if idx >= len(samples):
                    continue
                entry = samples[idx]
                with cols[j]:
                    st.image(entry['dispi'])
                    img_example_counter += 1
                    with st.columns(5)[2]:
                        this_trigger = st.button('\+', key='imgexuse%d' % img_example_counter)
                    trigger = trigger or this_trigger
                    if this_trigger:
                        trigger = entry[return_key]
    return trigger


def segment_img(img: Image):
    output = rembg.remove(img)
    mask = numpy.array(output)[:, :, 3] > 0
    sam_mask = SAMAPI.segment_api(numpy.array(img)[:, :, :3], mask)
    segmented_img = Image.new("RGBA", img.size, (0, 0, 0, 0))
    segmented_img.paste(img, mask=Image.fromarray(sam_mask))
    return segmented_img


def segment_6imgs(zero123pp_imgs):
    imgs = [zero123pp_imgs.crop([0, 0, 320, 320]),
            zero123pp_imgs.crop([320, 0, 640, 320]),
            zero123pp_imgs.crop([0, 320, 320, 640]),
            zero123pp_imgs.crop([320, 320, 640, 640]),
            zero123pp_imgs.crop([0, 640, 320, 960]),
            zero123pp_imgs.crop([320, 640, 640, 960])]
    segmented_imgs = []
    for i, img in enumerate(imgs):
        output = rembg.remove(img)
        mask = numpy.array(output)[:, :, 3]
        mask = SAMAPI.segment_api(numpy.array(img)[:, :, :3], mask)
        data = numpy.array(img)[:,:,:3]
        data[mask == 0] = [255, 255, 255]
        segmented_imgs.append(data)
    result = numpy.concatenate([
        numpy.concatenate([segmented_imgs[0], segmented_imgs[1]], axis=1),
        numpy.concatenate([segmented_imgs[2], segmented_imgs[3]], axis=1),
        numpy.concatenate([segmented_imgs[4], segmented_imgs[5]], axis=1)
    ])
    return Image.fromarray(result)


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
    if 'HF_TOKEN' in os.environ:
        huggingface_hub.login(os.environ['HF_TOKEN'])
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
    sys.main_lock = threading.Lock()
    return pipeline


check_dependencies()
pipeline = load_zero123plus_pipeline()
SAMAPI.get_instance()
torch.set_grad_enabled(False)

st.title("Zero123++ Demo")
# st.caption("For faster inference without waiting in queue, you may clone the space and run it yourself.")
prog = st.progress(0.0, "Idle")
pic = st.file_uploader("Upload an Image", key='imageinput', type=['png', 'jpg', 'webp'])
left, right = st.columns(2)
with left:
    rem_input_bg = st.checkbox("Remove Input Background")
with right:
    rem_output_bg = st.checkbox("Remove Output Background")
num_inference_steps = st.slider("Number of Inference Steps", 15, 100, 75)
st.caption("Diffusion Steps. For general real or synthetic objects, around 28 is enough. For objects with delicate details such as faces (either realistic or illustration), you may need 75 or more steps.")
cfg_scale = st.slider("Classifier Free Guidance Scale", 1.0, 10.0, 4.0)
seed = st.text_input("Seed", "42")
submit = False
if st.button("Submit"):
    submit = True
results_container = st.container()
sample_got = image_examples(iret, 4, 'rimageinput')
if sample_got:
    pic = sample_got
with results_container:
    if sample_got or submit:
        prog.progress(0.03, "Waiting in Queue...")
        with sys.main_lock:
            seed = int(seed)
            torch.manual_seed(seed)
            img = Image.open(pic)
            if max(img.size) > 1280:
                w, h = img.size
                w = round(1280 / max(img.size) * w)
                h = round(1280 / max(img.size) * h)
                img = img.resize((w, h))
            left, right = st.columns(2)
            with left:
                st.image(img)
                st.caption("Input Image")
            prog.progress(0.1, "Preparing Inputs")
            if rem_input_bg:
                with right:
                    img = segment_img(img)
                    st.image(img)
                    st.caption("Input (Background Removed)")
            img = expand2square(img, (127, 127, 127, 0))
            pipeline.set_progress_bar_config(disable=True)
            result = pipeline(
                img,
                num_inference_steps=num_inference_steps,
                guidance_scale=cfg_scale,
                generator=torch.Generator(pipeline.device).manual_seed(seed),
                callback=lambda i, t, latents: prog.progress(0.1 + 0.8 * i / num_inference_steps, "Diffusion Step %d" % i)
            ).images[0]
            prog.progress(0.9, "Post Processing")
            left, right = st.columns(2)
            with left:
                st.image(result)
                st.caption("Result")
            if rem_output_bg:
                result = segment_6imgs(result)
                with right:
                    st.image(result)
                    st.caption("Result (Background Removed)")
            prog.progress(1.0, "Idle")
