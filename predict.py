# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import subprocess
from typing import List

import rembg
import torch
from cog import BasePredictor, Input, Path
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from PIL import Image

WEIGHTS_CACHE = "/src/weights/zero123plusplus"
CHECKPOINT_URLS = [
    ("https://weights.replicate.delivery/default/zero123plusplus/zero123plusplus.tar", WEIGHTS_CACHE),
]

def download_model(url, dest):
    print("Downloading weights...")
    try:
        output = subprocess.check_output(["pget", "-x", url, "/src/tmp"])
    except subprocess.CalledProcessError as e:
        # If download fails, clean up and re-raise exception
        print(e.output)
        raise e
    
    os.rename("/src/tmp/zero123plusplus", dest)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
            
        if not os.path.exists("weights"):
            os.mkdir("weights")
       
        for (CKPT_URL, target_folder) in CHECKPOINT_URLS:
            if not os.path.exists(target_folder):
                download_model(CKPT_URL, target_folder)

        print("Setting up pipeline...")
        
        self.pipeline = DiffusionPipeline.from_pretrained(
            "./weights/zero123plusplus", 
            custom_pipeline="./diffusers-support/",
            torch_dtype=torch.float16, 
            local_files_only=True
        )
        self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipeline.scheduler.config, timestep_spacing='trailing'
        )
        self.pipeline.to('cuda:0')

    def predict(
        self,
        image: Path = Input(description="Input image. Aspect ratio should be 1:1. Recommended resolution is >= 320x320 pixels."),
        remove_background: bool = Input(description="Remove the background of the input image", default=False),
        return_intermediate_images: bool = Input(description="Return the intermediate images together with the output images", default=False),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        outputs = []
       
        cond = Image.open(str(image))
        image_filename = "original" + image.suffix

        # optional background removal step
        if remove_background:
            rembg_session = rembg.new_session()
            cond = rembg.remove(cond, session=rembg_session)
            # image should be a png after background removal
            image_filename += ".png"
        
        if return_intermediate_images:
            temp_original = f"/tmp/{image_filename}"
            cond.save(temp_original)
            outputs.append(temp_original)

        all_results = self.pipeline(cond, num_inference_steps=75)
        for i, output_img in enumerate(all_results.images):
            filename = f"/tmp/output{i}.jpg"
            output_img.save(filename)
            outputs.append(filename)
        
        return([Path(output) for output in outputs])






