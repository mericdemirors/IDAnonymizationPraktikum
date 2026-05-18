import torch
from src.helper_functions import embed_prompt


def take_diffusion_step(
    pipeline_bundle,
    latent,
    timestep,
    conditioning: dict,
    guidance_scale=7.5,
):
    # prepare the latent to match expected statistics
    latent_input = pipeline_bundle["scheduler"].scale_model_input(latent, timestep)

    # if cfg, we double the latent for single batch pass
    if guidance_scale > 1.0:
        model_input = torch.cat([latent_input] * 2)
    else:
        model_input = latent_input

    cond = conditioning.get("cond", None)
    uncond = conditioning.get("uncond", None)
    if isinstance(cond, str):
        cond = embed_prompt(pipeline_bundle, cond)
    if isinstance(uncond, str):
        uncond = embed_prompt(pipeline_bundle, uncond)

    # if cfg, concat the tensors for single batch pass
    if guidance_scale > 1.0:
        prompt_embeds = torch.cat([uncond, cond])
    else:
        prompt_embeds = cond

    model_input = model_input.to(dtype=pipeline_bundle["dtype"])
    prompt_embeds = prompt_embeds.to(dtype=pipeline_bundle["dtype"])

    # predict the noise
    model_output = pipeline_bundle["unet"](
        model_input, timestep, encoder_hidden_states=prompt_embeds
    ).sample

    # if cfg, split the batch and do cfg
    if guidance_scale > 1.0:
        noise_pred_uncond, noise_pred_text = model_output.chunk(2)

        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
    else:
        noise_pred = model_output

    return noise_pred
