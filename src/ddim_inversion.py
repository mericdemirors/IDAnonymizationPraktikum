from .ddim_inversion_section.ddim_image2noise import *
from .ddim_inversion_section.utils import *


# function to call ddim inversion merhod
def ddim_invert_image(
    image_path: str,
    pipeline_bundle,
    prompt: str = "",
    num_inference_steps: int = 150,
    guidance_scale: float = 7.5,
):
    """
    Takes an image path and inverts it back into latents
    if an optional prompt is given, process is done conditionally
    """
    # prepare the image
    image_tensor = read_and_prepare_image(
        image_path=image_path, device=pipeline_bundle["device"]
    )
    image_tensor = image_tensor.to(dtype=pipeline_bundle["dtype"])
    latents = embed_image_latent(
        pipeline_bundle=pipeline_bundle, image_tensor=image_tensor
    )
    latents = latents.to(dtype=pipeline_bundle["dtype"])

    # inverse it with or without condition
    if prompt:
        latent = ddim_cond_inversion(
            pipeline_bundle=pipeline_bundle,
            latents=latents,
            num_inference_steps=num_inference_steps,
            prompt=prompt,
            guidance_scale=guidance_scale,
        )
    else:
        latent = ddim_nocond_inversion(
            pipeline_bundle=pipeline_bundle,
            latents=latents,
            num_inference_steps=num_inference_steps,
        )
    return latent
