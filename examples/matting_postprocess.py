import numpy
from PIL import Image
from pymatting.alpha.estimate_alpha_cf import estimate_alpha_cf
from pymatting.foreground.estimate_foreground_ml import estimate_foreground_ml
from pymatting.util.util import stack_images
from scipy.ndimage import binary_erosion
from typing import Tuple


def postprocess(rgb_img: Image.Image, normal_img: Image.Image) -> Tuple[Image.Image, Image.Image]:
    normal_vecs_pred = numpy.array(normal_img, dtype=numpy.float64) / 255.0 * 2 - 1
    alpha_pred = numpy.linalg.norm(normal_vecs_pred, axis=-1)

    is_foreground = alpha_pred > 0.6
    is_background = alpha_pred < 0.2
    structure = numpy.ones(
        (4, 4), dtype=numpy.uint8
    )

    is_foreground = binary_erosion(is_foreground, structure=structure)
    is_background = binary_erosion(is_background, structure=structure, border_value=1)

    trimap = numpy.full(alpha_pred.shape, dtype=numpy.uint8, fill_value=128)
    trimap[is_foreground] = 255
    trimap[is_background] = 0

    img_normalized = numpy.array(rgb_img, dtype=numpy.float64) / 255.0
    trimap_normalized = trimap.astype(numpy.float64) / 255.0

    alpha = estimate_alpha_cf(img_normalized, trimap_normalized)
    foreground = estimate_foreground_ml(img_normalized, alpha)
    cutout = stack_images(foreground, alpha)

    cutout = numpy.clip(cutout * 255, 0, 255).astype(numpy.uint8)
    cutout = Image.fromarray(cutout)

    normal_vecs_pred = normal_vecs_pred / (numpy.linalg.norm(normal_vecs_pred, axis=-1, keepdims=True) + 1e-8)
    normal_vecs_pred = normal_vecs_pred * 0.5 + 0.5
    normal_vecs_pred = normal_vecs_pred * alpha[..., None] + 0.5 * (1 - alpha[..., None])
    normal_image_normalized = numpy.clip(normal_vecs_pred * 255, 0, 255).astype(numpy.uint8)

    return cutout, Image.fromarray(normal_image_normalized)
