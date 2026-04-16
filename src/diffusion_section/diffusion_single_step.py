import torch


def take_diffusion_step(
    model,
    latent,
    timestep,
    scheduler,
    conditioning: dict,
    guidance_scale=7.5,
):
    """
    Performs a single step of the diffusion process.

    Args:
        model: The U-Net or Transformer model.
        latent: The current latent tensor (x_t).
        timestep: The current timestep in the schedule.
        scheduler: The noise scheduler (e.g., DDIM, DPM, Euler).
        conditioning: dict containing 'cond' and 'uncond' pass tensors
        guidance_scale: The strength of Classifier-Free Guidance.
    Returns:
        The updated latent for the next step (x_t-1).
    """

    # prepare the latent to match expected statistics
    latent_input = scheduler.scale_model_input(latent, timestep)

    # if cfg, we double the latent for single batch pass
    if guidance_scale > 1.0:
        model_input = torch.cat([latent_input] * 2)
    else:
        model_input = latent_input

    cond = conditioning.get("cond", None)
    uncond = conditioning.get("nocond", None)

    # if cfg, concat the tensors for single batch pass
    if guidance_scale > 1.0:
        prompt_embeds = torch.cat([uncond, cond])
    else:
        prompt_embeds = cond

    # predict the noise
    model_output = model(
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

    # de-noise the latent with predicted noise
    step_output = scheduler.step(noise_pred, timestep, latent)

    return step_output.prev_sample
