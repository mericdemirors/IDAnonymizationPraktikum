from ddim_inversion_section.ddim_image2noise import (
    ddim_nocond_inversion,
    ddim_cond_inversion,
)


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
    if prompt:
        latent = ddim_cond_inversion(
            image_path,
            device,
            model_version,
            num_inference_steps,
            prompt,
            guidance_scale,
        )
    else:
        latent = ddim_nocond_inversion(
            image_path,
            device,
            model_version,
            num_inference_steps,
        )
    return latent
