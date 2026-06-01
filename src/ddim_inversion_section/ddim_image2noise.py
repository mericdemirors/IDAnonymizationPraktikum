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
            scheduler=pipeline_bundle["scheduler"],
            latents=latents,
            t=t,
            num_inference_steps=num_inference_steps,
            i=i,
            noise_pred=noise_pred,
        )
        latents = latents.to(dtype=pipeline_bundle["dtype"])

    return latents


# function to reverse image into DDIM noise with condition
def ddim_cond_inversion(
    pipeline_bundle, latents, num_inference_steps: str, prompt, guidance_scale: float
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
            scheduler=pipeline_bundle["scheduler"],
            latents=latents,
            t=t,
            num_inference_steps=num_inference_steps,
            i=i,
            noise_pred=noise_pred,
        )
        latents = latents.to(dtype=pipeline_bundle["dtype"])

    return latents


def ddim_nocond_inversion_ip(
    ip_adapter_model, latents, faceid_embeds, num_inference_steps: int
):
    device = ip_adapter_model.device
    dtype = latents.dtype
    pipe = ip_adapter_model.pipe
    scheduler = pipe.scheduler
    unet = pipe.unet

    scheduler.set_timesteps(num_inference_steps)
    image_prompt_embeds, _ = ip_adapter_model.get_image_embeds(faceid_embeds)
    image_prompt_embeds = image_prompt_embeds.to(device=device, dtype=dtype)

    text_embeddings = embed_prompt(ip_adapter_model, "")

    text_embeddings = text_embeddings.to(device=device, dtype=dtype)
    encoder_hidden_states = torch.cat([text_embeddings, image_prompt_embeds], dim=1)
    latents = latents.to(device=device, dtype=dtype)

    for i, t in enumerate(reversed(scheduler.timesteps)):
        t = t.to(device)
        noise_pred = unet(
            latents, t, encoder_hidden_states=encoder_hidden_states
        ).sample

        latents = inversion_step(
            scheduler=scheduler,
            latents=latents,
            t=t,
            num_inference_steps=num_inference_steps,
            i=i,
            noise_pred=noise_pred,
        )
        latents = latents.to(dtype=dtype)

    return latents


def ddim_cond_inversion_ip(
    ip_adapter_model,
    latents,
    faceid_embeds,
    num_inference_steps: int,
    prompt,
    guidance_scale: float,
):

    device = ip_adapter_model.device
    dtype = latents.dtype
    pipe = ip_adapter_model.pipe
    scheduler = pipe.scheduler
    unet = pipe.unet

    scheduler.set_timesteps(num_inference_steps)

    image_prompt_embeds, uncond_image_prompt_embeds = ip_adapter_model.get_image_embeds(
        faceid_embeds
    )
    image_prompt_embeds = image_prompt_embeds.to(device=device, dtype=dtype)
    uncond_image_prompt_embeds = uncond_image_prompt_embeds.to(
        device=device, dtype=dtype
    )

    prompt_embeds = embed_prompt(ip_adapter_model, prompt)
    negative_prompt_embeds = embed_prompt(ip_adapter_model, "")
    prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
    negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=dtype)
    cond_embeddings = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
    uncond_embeddings = torch.cat(
        [negative_prompt_embeds, uncond_image_prompt_embeds], dim=1
    )

    encoder_hidden_states = torch.cat([uncond_embeddings, cond_embeddings], dim=0)
    latents = latents.to(device=device, dtype=dtype)

    for i, t in enumerate(reversed(scheduler.timesteps)):
        t = t.to(device)

        latent_model_input = torch.cat([latents] * 2)
        noise_pred = unet(
            latent_model_input, t, encoder_hidden_states=encoder_hidden_states
        ).sample

        noise_uncond, noise_cond = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

        latents = inversion_step(
            scheduler=scheduler,
            latents=latents,
            t=t,
            num_inference_steps=num_inference_steps,
            i=i,
            noise_pred=noise_pred,
        )
        latents = latents.to(dtype=dtype)

    return latents
