from .ddim_inversion_section.ddim_image2noise import *
from .ddim_inversion_section.utils import *


# function to call ddim inversion merhod
def ddim_invert_image(
    generation_method: str,
    image_path: str,
    pipeline_bundle,
    ip_adapter_model,
    prompt: str = "",
    num_inference_steps: int = 150,
    guidance_scale: float = 7.5,
):

    if "text" in generation_method:  # this block uses the generic pipeline
        image_tensor = read_and_prepare_image(
            image_path=image_path, device=pipeline_bundle["device"]
        )
        image_tensor = image_tensor.to(dtype=pipeline_bundle["dtype"])
        latents = embed_image_latent(
            pipeline_bundle_or_ipadapter=pipeline_bundle, image_tensor=image_tensor
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
    elif (
        "id" in generation_method
    ):  # this pipeline uses the IP adapter way of DDIM inverting
        image_tensor = read_and_prepare_image(
            image_path=image_path, device=pipeline_bundle["device"]
        )
        image_tensor = image_tensor.to(dtype=pipeline_bundle["dtype"])
        latents = embed_image_latent(
            pipeline_bundle_or_ipadapter=ip_adapter_model, image_tensor=image_tensor
        )
        latents = latents.to(dtype=pipeline_bundle["dtype"])

        # inversion with or without text condition
        if prompt:
            latent = ddim_cond_inversion_ip(
                ip_adapter_model=ip_adapter_model,
                latents=latents,
                faceid_embeds=torch.zeros(
                    (1, 512),
                    dtype=pipeline_bundle["dtype"],
                    device=pipeline_bundle["device"],
                ),
                num_inference_steps=num_inference_steps,
                prompt=prompt,
                guidance_scale=guidance_scale,
            )
        else:
            latent = ddim_nocond_inversion_ip(
                pipeline_bundle=pipeline_bundle,
                ip_adapter_model=ip_adapter_model,
                latents=latents,
                faceid_embeds=torch.zeros(
                    (1, 512),
                    dtype=pipeline_bundle["dtype"],
                    device=pipeline_bundle["device"],
                ),
                num_inference_steps=num_inference_steps,
            )
        return latent
