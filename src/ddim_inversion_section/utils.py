import torch
from PIL import Image
from torchvision import transforms
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL


def prepare_pipe(model_version, vae_model_version, device):
    if vae_model_version:
        vae = AutoencoderKL.from_pretrained(
            vae_model_version, torch_dtype=torch.float32
        )
        pipe = StableDiffusionPipeline.from_pretrained(
            model_version, vae=vae, torch_dtype=torch.float32
        )
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_version, torch_dtype=torch.float32
        )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = None
    pipe.feature_extractor = None
    return pipe.to(device)


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


def embed_image_latent(pipeline_bundle_or_ipadapter, image_tensor):
    if isinstance(pipeline_bundle_or_ipadapter, dict):
        vae = pipeline_bundle_or_ipadapter["vae"]
        device = pipeline_bundle_or_ipadapter["device"]
        dtype = pipeline_bundle_or_ipadapter["dtype"]
    else:
        vae = pipeline_bundle_or_ipadapter.pipe.vae
        device = pipeline_bundle_or_ipadapter.device
        dtype = image_tensor.dtype

    image_tensor = image_tensor.to(device=device, dtype=dtype)
    latents = vae.encode(image_tensor).latent_dist.sample()
    latents = latents * vae.config.scaling_factor
    latents = latents.to(device=device, dtype=dtype)

    return latents


def inversion_step(scheduler, latents, t, num_inference_steps, i, noise_pred):
    alpha_t = scheduler.alphas_cumprod[t]
    alpha_prev = (
        scheduler.alphas_cumprod[scheduler.timesteps[-i - 2]]
        if i < num_inference_steps - 1
        else scheduler.alphas_cumprod[-1]
    )

    latents = (latents - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
    latents = alpha_prev.sqrt() * latents + (1 - alpha_prev).sqrt() * noise_pred
    return latents
