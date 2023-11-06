import os

import torch
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from PIL import Image

WEIGHTS_CACHE = "./weights/zero123plusplus"

if not os.path(WEIGHTS_CACHE).exists():
    print("Loading diffusion pipeline from huggingface...")
    # Load the pipeline
    pipeline = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.1", custom_pipeline="sudo-ai/zero123plus-pipeline",
        torch_dtype=torch.float16
    )

    # save it to file
    pipeline.save_pretrained(WEIGHTS_CACHE, safe_serialization=True)
