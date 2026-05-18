from tqdm import tqdm
from .diffusion_section.diffusion_single_step import *


# function to generate image from noise by positive and negative parallel diffusions
def parallel_generate(
    pipeline_bundle,
    latent,
    negative_condition,
    positive_condition,
    n_coeff,
    p_coeff,
    n_coeff_update_fn,
    p_coeff_update_fn,
    num_inference_steps=150,
    guidance_scale=7.5,
):
    """
    does positive and negative parallel diffusions
    """

    pipeline_bundle["scheduler"].set_timesteps(num_inference_steps)

    for t in tqdm(pipeline_bundle["scheduler"].timesteps):
        neg_noise_pred = take_diffusion_step(
            pipeline_bundle=pipeline_bundle,
            latent=latent,
            timestep=t,
            conditioning={"cond": negative_condition, "uncond": ""},
            guidance_scale=guidance_scale,
        )

        pos_noise_pred = take_diffusion_step(
            pipeline_bundle=pipeline_bundle,
            latent=latent,
            timestep=t,
            conditioning={"cond": positive_condition, "uncond": ""},
            guidance_scale=guidance_scale,
        )

        noise_pred = (n_coeff * neg_noise_pred) + (p_coeff * pos_noise_pred)
        latent = pipeline_bundle["scheduler"].step(noise_pred, t, latent).prev_sample

        n_coeff = n_coeff_update_fn(n_coeff, t)
        p_coeff = p_coeff_update_fn(p_coeff, t)

    latent = latent.to(dtype=pipeline_bundle["dtype"])
    return latent


def single_generate(
    pipeline_bundle,
    latent,
    positive_condition,
    num_inference_steps=150,
    guidance_scale=7.5,
):
    pipeline_bundle["scheduler"].set_timesteps(num_inference_steps)

    for t in tqdm(pipeline_bundle["scheduler"].timesteps):
        noise_pred = take_diffusion_step(
            pipeline_bundle=pipeline_bundle,
            latent=latent,
            timestep=t,
            conditioning={"cond": positive_condition, "uncond": ""},
            guidance_scale=guidance_scale,
        )

        latent = pipeline_bundle["scheduler"].step(noise_pred, t, latent).prev_sample

    latent = latent.to(dtype=pipeline_bundle["dtype"])
    return latent
