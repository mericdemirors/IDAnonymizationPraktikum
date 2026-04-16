import torch
import cv2
import numpy as np


def retrieve_images_with_text(dataset_dict: dict, text_prompt: str, n: int):
    """Returns a list of read images"""
    image_paths = dataset_dict.get(text_prompt, [])[:n]
    images = [cv2.imread(p) for p in image_paths]
    return [img for img in images if img is not None]


def generate_image_with_text(
    model,
    scheduler,
    vae,
    text_encoder,
    tokenizer,
    text_prompt,
    num_steps=50,
    guidance_scale=7.5,
):
    """
    Standard implementation of a text-to-image diffusion loop.
    """
    device = next(model.parameters()).device

    # 1. Encode the text prompt
    text_inputs = tokenizer(
        text_prompt,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).to(device)
    prompt_embeds = text_encoder(text_inputs.input_ids)[0]

    # 2. Encode unconditional empty string for CFG
    uncond_inputs = tokenizer(
        "",
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).to(device)
    negative_prompt_embeds = text_encoder(uncond_inputs.input_ids)[0]

    # Concatenate for batch processing (uncond first, then cond)
    context = torch.cat([negative_prompt_embeds, prompt_embeds])

    # Initialize latents
    latents = torch.randn((1, 4, 64, 64), device=device)

    # Set timesteps
    scheduler.set_timesteps(num_steps)
    latents = latents * scheduler.init_noise_sigma

    # Denoising Loop
    for t in scheduler.timesteps:
        # Expand latents for Classifier-Free Guidance
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # Predict noise residual
        with torch.no_grad():
            noise_pred = model(
                latent_model_input, t, encoder_hidden_states=context
            ).sample

        # Perform Guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        # Compute previous noisy sample (x_t -> x_t-1)
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    # 6. Decode Latents
    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        image = vae.decode(latents).sample

    # Post-process to [0, 1] range
    image = (image / 2 + 0.5).clamp(0, 1)
    return image


def generate_images_with_text(pipeline_bundle, text_prompt, n=1, **kwargs):
    """
    Generates n images. pipeline_bundle contains (model, scheduler, vae, etc.)
    Returns a list of BGR numpy images.
    """
    generated_images = []
    for _ in range(n):
        # Using your existing logic but wrapped
        image_tensor = generate_image_with_text(
            **pipeline_bundle, text_prompt=text_prompt, **kwargs
        )

        # Convert Tensor [1, 3, H, W] -> BGR Numpy [H, W, 3]
        img_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        generated_images.append(img_bgr)

    return generated_images
