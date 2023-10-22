from diffusers import StableDiffusionXLPipeline
import torch
import rembg

# text-to-image with SDXL for text-to-image-to-3d
pipeline = StableDiffusionXLPipeline.from_single_file(
    "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0.safetensors",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to("cuda")
pipeline.enable_model_cpu_offload()

num_images_per_prompt = 1
res = 1024
text = input("Prompt > ")
bkgd_color = "white"
prompt = f"a ((full-body:2)) shot of a ((single:2)) {text}, isolated on {bkgd_color} background, 4k, highly detailed"
images = pipeline(prompt=prompt, num_images_per_prompt=num_images_per_prompt, height=res, width=res).images
image = images[0]
image.show()
image.save("output.png")
image = rembg.remove(image)
image.save("output-rembg.png")
