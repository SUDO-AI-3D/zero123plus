import cv2
import copy
import numpy
import torch
import requests
from PIL import Image
from diffusers import DiffusionPipeline, ControlNetModel
from matting_postprocess import postprocess


def rescale(single_res, input_image, ratio=0.95):
    # Rescale and recenter
    image_arr = numpy.array(input_image)
    ret, mask = cv2.threshold(numpy.array(input_image.split()[-1]), 0, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(mask)
    max_size = max(w, h)
    side_len = int(max_size / ratio)
    padded_image = numpy.zeros((side_len, side_len, 4), dtype=numpy.uint8)
    center = side_len//2
    padded_image[center-h//2:center-h//2+h, center-w//2:center-w//2+w] = image_arr[y:y+h, x:x+w]
    rgba = Image.fromarray(padded_image).resize((single_res, single_res), Image.LANCZOS)
    return rgba


# Load the pipeline
pipeline: DiffusionPipeline = DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.2", custom_pipeline="sudo-ai/zero123plus-pipeline",
    torch_dtype=torch.float16, local_files_only=True
)
normal_pipeline = copy.copy(pipeline)
normal_pipeline.add_controlnet(ControlNetModel.from_pretrained(
    "sudo-ai/controlnet-zp12-normal-gen-v1", torch_dtype=torch.float16, local_files_only=True
), conditioning_scale=1.0)
pipeline.to("cuda:0", torch.float16)
normal_pipeline.to("cuda:0", torch.float16)
# Run the pipeline
cond = Image.open(requests.get("https://d.skis.ltd/nrp/sample-data/10_cond.png", stream=True).raw)
# Optional: rescale input image if it occupies only a small region in input
# cond = rescale(512, cond)
# Generate 6 images
genimg = pipeline(
    cond,
    prompt='', guidance_scale=4, num_inference_steps=75, width=640, height=960
).images[0]
# Generate normal image
# We observe that a higher CFG scale (4) is more robust
# but with CFG = 1 it is faster and is usually good enough for normal image
# You can adjust to your needs
normalimg = normal_pipeline(
    cond, depth_image=genimg,
    prompt='', guidance_scale=4, num_inference_steps=75, width=640, height=960
).images[0]
genimg, normalimg = postprocess(genimg, normalimg)
genimg.save("colors.png")
normalimg.save("normals.png")
