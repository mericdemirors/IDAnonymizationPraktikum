import torch
from tqdm import tqdm
import cv2
import numpy as np
from src.helper_functions import embed_prompt


def retrieve_images_with_text(dataset_dict: dict, text_prompt: str, n: int):
    """Returns a list of read images"""
    image_paths = dataset_dict.get(text_prompt, [])[:n]
    images = [cv2.imread(p) for p in image_paths]
    return [img for img in images if img is not None]


def generate_single_image_with_text(
    pipeline_bundle, text_prompt, num_inference_steps=150, guidance_scale=7.5
):
    """
    Standard implementation of a text-to-image diffusion loop.
    """
    # 1. Encode the text prompt
    prompt_embeds = embed_prompt(pipeline_bundle, text_prompt)

    # 2. Encode unconditional empty string for CFG
    negative_prompt_embeds = embed_prompt(pipeline_bundle, "")

    # Concatenate for batch processing (uncond first, then cond)
    context = torch.cat([negative_prompt_embeds, prompt_embeds])

    # Initialize latents
    latents = torch.randn((1, 4, 64, 64), device=pipeline_bundle["device"])

    # Set timesteps
    pipeline_bundle["scheduler"].set_timesteps(num_inference_steps)
    latents = latents * pipeline_bundle["scheduler"].init_noise_sigma

    latents = latents.to(dtype=pipeline_bundle["dtype"])
    context = context.to(dtype=pipeline_bundle["dtype"])

    # Denoising Loop
    for t in pipeline_bundle["scheduler"].timesteps:
        # Expand latents for Classifier-Free Guidance
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = pipeline_bundle["scheduler"].scale_model_input(
            latent_model_input, t
        )

        # Predict noise residual
        noise_pred = pipeline_bundle["unet"](
            latent_model_input, t, encoder_hidden_states=context
        ).sample

        # Perform Guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        # Compute previous noisy sample (x_t -> x_t-1)
        latents = pipeline_bundle["scheduler"].step(noise_pred, t, latents).prev_sample

    # 6. Decode Latents
    latents = 1 / 0.18215 * latents
    image = pipeline_bundle["vae"].decode(latents).sample

    # Post-process to [0, 1] range
    image = (image / 2 + 0.5).clamp(0, 1)
    return image


def generate_images_with_text(
    pipeline_bundle, text_prompt, num_inference_steps, guidance_scale, n
):
    generated_images = []
    for _ in tqdm(range(n)):
        # Using your existing logic but wrapped
        image_tensor = generate_single_image_with_text(
            pipeline_bundle=pipeline_bundle,
            text_prompt=text_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

        # Convert Tensor [1, 3, H, W] -> BGR Numpy [H, W, 3]
        img_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        generated_images.append(img_bgr)

    return generated_images
