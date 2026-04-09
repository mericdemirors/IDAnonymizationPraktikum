import torch
from PIL import Image
from torchvision import transforms
from diffusers import StableDiffusionPipeline, DDIMScheduler


def prepare_pipe_and_scheduler(model_version, device, num_inference_steps):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_version, torch_dtype=torch.float16 if device == "cuda" else torch.float32
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


def embed_image_latent(pipe, image_tensor):
    with torch.no_grad():
        latents = pipe.vae.encode(image_tensor).latent_dist.sample()
        latents = latents * pipe.vae.config.scaling_factor
    return latents


def embed_prompt(pipe, prompt, device):
    text_input = pipe.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    )
    text_embeddings = pipe.text_encoder(text_input.input_ids.to(device))[0]
    return text_embeddings


def inversion_step(latents, scheduler, t, num_inference_steps, i, device, noise_pred):
    alpha_t = scheduler.alphas_cumprod[t]
    alpha_prev = (
        scheduler.alphas_cumprod[scheduler.timesteps[-i - 2]]
        if i < num_inference_steps - 1
        else torch.tensor(1.0, device=device)
    )

    latents = (latents - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
    latents = alpha_prev.sqrt() * latents + (1 - alpha_prev).sqrt() * noise_pred
    return latents
