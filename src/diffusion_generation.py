import torch
from tqdm import tqdm

from diffusion_section.diffusion_single_step import *


# function to generate image from noise by positive and negative parallel diffusions
def parallel_generate(
    negative_model,
    positive_model,
    scheduler,
    negative_condition,
    positive_condition,
    n_coeff,
    p_coeff,
    n_coeff_update_fn,
    p_coeff_update_fn,
    num_inference_steps=50,
    guidance_scale=7.5,
    latent_shape=(1, 4, 64, 64),
    device="cuda",
):
    """
    does positive and negative parallel diffusions
    """
    latent = torch.randn(latent_shape, device=device)

    scheduler.set_timesteps(num_inference_steps)

    for t in tqdm(scheduler.timesteps):
        neg_latent_step = take_diffusion_step(
            model=negative_model,
            latent=latent,
            timestep=t,
            scheduler=scheduler,
            conditioning=negative_condition,
            guidance_scale=guidance_scale,
        )

        pos_latent_step = take_diffusion_step(
            model=positive_model,
            latent=latent,
            timestep=t,
            scheduler=scheduler,
            conditioning=positive_condition,
            guidance_scale=guidance_scale,
        )

        latent = (n_coeff * neg_latent_step) + (p_coeff * pos_latent_step)
        n_coeff = n_coeff_update_fn(n_coeff, t)
        p_coeff = p_coeff_update_fn(p_coeff, t)

    return latent
