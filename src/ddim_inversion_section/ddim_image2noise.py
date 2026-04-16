import torch
from .utils import *


# function to reverse image into ddim noise with no condition
def ddim_nocond_inversion(
    pipe,
    scheduler,
    latents,
    device: str,
    num_inference_steps: int,
):
    # embed image, embed uncond prompt
    text_embeddings = embed_prompt(pipe, "", device)

    for i, t in enumerate(reversed(scheduler.timesteps)):
        t = t.to(device)  # get time step

        # pred noise
        with torch.no_grad():
            noise_pred = pipe.unet(
                latents, t, encoder_hidden_states=text_embeddings
            ).sample

        # take one step
        latents = inversion_step(
            latents, scheduler, t, num_inference_steps, i, device, noise_pred
        )

    return latents


# function to reverse image into DDIM noise with condition
def ddim_cond_inversion(
    pipe,
    scheduler,
    latents,
    device: str,
    num_inference_steps: str,
    prompt: int,
    guidance_scale: float,
):
    # embed image, embed uncond and cond prompt, merge text embeds
    cond_embeddings = embed_prompt(pipe, prompt, device)
    uncond_embeddings = embed_prompt(pipe, "", device)
    text_embeddings = torch.cat([uncond_embeddings, cond_embeddings], dim=0)

    for i, t in enumerate(reversed(scheduler.timesteps)):
        t = t.to(device)  # get time step

        # we need both uncond and cond noise pred so double the batch
        latent_model_input = torch.cat([latents] * 2)

        # pred noise
        with torch.no_grad():
            noise_pred = pipe.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            ).sample

        # split batch into uncond and cond, calculate CFG
        noise_uncond, noise_cond = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

        latents = inversion_step(
            latents, scheduler, t, num_inference_steps, i, device, noise_pred
        )

    return latents
