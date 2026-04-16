from ddim_inversion_section.ddim_image2noise import *
from ddim_inversion_section.utils import *


# function to call ddim inversion merhod
def ddim_invert_image(
    image_path: str,
    device: str,
    model_version: str = "runwayml/stable-diffusion-v1-5",
    num_inference_steps: int = 50,
    prompt: str = "",
    guidance_scale: float = 7.5,
):
    """
    Takes an image path and inverts it back into latents
    if an optional prompt is given, process is done conditionally
    """
    # prepare the pipe and image
    pipe, scheduler = prepare_pipe_and_scheduler(
        model_version, device, num_inference_steps
    )
    image_tensor = read_and_prepare_image(image_path, device)
    latents = embed_image_latent(pipe, image_tensor)

    # inverse it with or without condition
    if prompt:
        latent = ddim_cond_inversion(
            pipe,
            scheduler,
            latents,
            device,
            num_inference_steps,
            prompt,
            guidance_scale,
        )
    else:
        latent = ddim_nocond_inversion(
            pipe,
            scheduler,
            latents,
            device,
            num_inference_steps,
        )
    return latent
