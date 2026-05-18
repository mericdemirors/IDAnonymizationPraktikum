import torch
from PIL import Image
from torchvision import transforms
from diffusers import StableDiffusionPipeline, DDIMScheduler


def prepare_pipe_and_scheduler(model_version, device, num_inference_steps=150):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_version, torch_dtype=torch.float32
    ).to(device)

    scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler = scheduler
    scheduler.set_timesteps(num_inference_steps)

    return pipe, scheduler


def read_and_prepare_image(image_path, device):
    image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    return image_tensor


def embed_image_latent(pipeline_bundle, image_tensor):
    image_tensor = image_tensor.to(dtype=pipeline_bundle["dtype"])

    latents = pipeline_bundle["vae"].encode(image_tensor).latent_dist.sample()
    latents = latents * pipeline_bundle["vae"].config.scaling_factor
    latents = latents.to(dtype=pipeline_bundle["dtype"])
    return latents


def inversion_step(pipeline_bundle, latents, t, num_inference_steps, i, noise_pred):
    alpha_t = pipeline_bundle["scheduler"].alphas_cumprod[t]
    alpha_prev = (
        pipeline_bundle["scheduler"].alphas_cumprod[
            pipeline_bundle["scheduler"].timesteps[-i - 2]
        ]
        if i < num_inference_steps - 1
        else pipeline_bundle["scheduler"].alphas_cumprod[-1]
    )

    latents = (latents - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
    latents = alpha_prev.sqrt() * latents + (1 - alpha_prev).sqrt() * noise_pred
    latents = latents.to(dtype=pipeline_bundle["dtype"])
    return latents
