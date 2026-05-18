import torch
from .utils import *
from src.helper_functions import embed_prompt


# function to reverse image into ddim noise with no condition
def ddim_nocond_inversion(pipeline_bundle, latents, num_inference_steps: int):
    pipeline_bundle["scheduler"].set_timesteps(num_inference_steps)

    # embed image, embed uncond prompt
    text_embeddings = embed_prompt(pipeline_bundle, "")

    latents = latents.to(dtype=pipeline_bundle["dtype"])
    text_embeddings = text_embeddings.to(dtype=pipeline_bundle["dtype"])

    for i, t in enumerate(reversed(pipeline_bundle["scheduler"].timesteps)):
        t = t.to(pipeline_bundle["device"])  # get time step

        # pred noise
        noise_pred = pipeline_bundle["unet"](
            latents, t, encoder_hidden_states=text_embeddings
        ).sample

        # take one step
        latents = inversion_step(
            pipeline_bundle=pipeline_bundle,
            latents=latents,
            t=t,
            num_inference_steps=num_inference_steps,
            i=i,
            noise_pred=noise_pred,
        )

    return latents


# function to reverse image into DDIM noise with condition
def ddim_cond_inversion(
    pipeline_bundle,
    latents,
    num_inference_steps: str,
    prompt: int,
    guidance_scale: float,
):
    pipeline_bundle["scheduler"].set_timesteps(num_inference_steps)

    # embed image, embed uncond and cond prompt, merge text embeds
    cond_embeddings = embed_prompt(pipeline_bundle, prompt)
    uncond_embeddings = embed_prompt(pipeline_bundle, "")
    text_embeddings = torch.cat([uncond_embeddings, cond_embeddings], dim=0)

    latents = latents.to(dtype=pipeline_bundle["dtype"])
    text_embeddings = text_embeddings.to(dtype=pipeline_bundle["dtype"])

    for i, t in enumerate(reversed(pipeline_bundle["scheduler"].timesteps)):
        t = t.to(pipeline_bundle["device"])  # get time step

        # we need both uncond and cond noise pred so double the batch
        latent_model_input = torch.cat([latents] * 2)

        # pred noise
        noise_pred = pipeline_bundle["unet"](
            latent_model_input, t, encoder_hidden_states=text_embeddings
        ).sample

        # split batch into uncond and cond, calculate CFG
        noise_uncond, noise_cond = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

        latents = inversion_step(
            pipeline_bundle=pipeline_bundle,
            latents=latents,
            t=t,
            num_inference_steps=num_inference_steps,
            i=i,
            noise_pred=noise_pred,
        )

    return latents
