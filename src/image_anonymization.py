import torch
from insightface.app import FaceAnalysis

from .caption_retrieval import get_image_captions
from .caption_retrieval_section.utils import load_clip_model
from .ddim_inversion import ddim_invert_image
from .ddim_inversion_section.utils import prepare_pipe_and_scheduler
from .diffusion_generation import parallel_generate, single_generate
from .id_generation import get_id_from_textual_data
from .helper_functions import verbose


def aggregate_by_mean(embeddings):
    mean_emb = torch.mean(embeddings, dim=0)
    return mean_emb / mean_emb.norm()


def get_positive_and_negative_captions(captions_probs_attributelist):
    negative_caption = captions_probs_attributelist[0][0]
    positive_caption = captions_probs_attributelist[1][0]
    return negative_caption, positive_caption


def run_image_swap_pipeline(
    image_path,
    config_path,
    clip_version,
    pipe_version,
    device,
    face_app_name,
    face_app_det_size,
    face_app_ctx_id,
    n_coeff_init,
    p_coeff_init,
    n_coeff_update_fn,
    p_coeff_update_fn,
    image_save_path,
    num_inference_steps,
    guidance_scale,
    inversion_num_inference_steps,
    inversion_guidance_scale,
):
    # GET MODELS  GET MODELS # GET MODELS # GET MODELS # GET MODELS #
    verbose("get models...")
    clip_model, processor = load_clip_model(model_version=clip_version, device=device)
    verbose("got clip model")
    pipe, scheduler = prepare_pipe_and_scheduler(
        model_version=pipe_version,
        device=device,
        num_inference_steps=num_inference_steps,
    )
    pipeline_bundle = {
        "unet": pipe.unet,
        "scheduler": scheduler,
        "vae": pipe.vae,
        "text_encoder": pipe.text_encoder,
        "tokenizer": pipe.tokenizer,
        "device": device,
        "dtype": pipe.unet.dtype,
    }
    verbose("got pipeline")
    # face_app = FaceAnalysis(name=face_app_name, providers=["CUDAExecutionProvider"])
    # face_app.prepare(
    #     ctx_id=face_app_ctx_id, det_size=(face_app_det_size, face_app_det_size)
    # )
    # verbose("got faceapp")
    # GET MODELS  GET MODELS # GET MODELS # GET MODELS # GET MODELS #

    # PREPARE CONDITIONS # PREPARE CONDITIONS # PREPARE CONDITIONS #
    verbose(f"get captions for {image_path}...")
    captions_probs_attributelist = get_image_captions(
        image_path=image_path, config_path=config_path, model=clip_model, device=device
    )

    negative_caption, positive_caption = get_positive_and_negative_captions(
        captions_probs_attributelist=captions_probs_attributelist
    )
    verbose("got captions...")
    verbose(f"C1 (Negative Caption): {negative_caption}")
    verbose(f"C2 (Positive Caption): {positive_caption}")
    # PREPARE CONDITIONS # PREPARE CONDITIONS # PREPARE CONDITIONS #

    # PREPARE GENERATION # PREPARE GENERATION # PREPARE GENERATION #
    verbose("conditional invertion...")
    ddim_inverted_latent = ddim_invert_image(
        image_path=image_path,
        pipeline_bundle=pipeline_bundle,
        prompt=negative_caption,
        num_inference_steps=inversion_num_inference_steps,
        guidance_scale=inversion_guidance_scale,
    )
    verbose("conditional inverted")
    verbose(f"ddim_inverted_latent.shape {ddim_inverted_latent.shape}")
    # verbose("get ID for positive cond...")
    # identity_embedding = get_id_from_textual_data(
    #     method="generation",
    #     number_of_ids=5,
    #     text_prompt=positive_caption,
    #     id_aggregation_func=aggregate_by_mean,
    #     face_app=face_app,
    #     pipeline_bundle=pipeline_bundle,
    #     # dataset_dict={"a photo of a person": ["path/to/image1.jpg", "path/to/image2.jpg"],"a professional portrait": ["path/to/image3.jpg"],}
    #     num_inference_steps=num_inference_steps,
    #     guidance_scale=guidance_scale,
    # )
    # verbose("got ID for positive cond")
    # verbose(f"identity_embedding.shape {identity_embedding.shape}")
    # PREPARE GENERATION # PREPARE GENERATION # PREPARE GENERATION #

    # GENERATION # GENERATION # GENERATION # GENERATION # GENERATION #
    verbose(
        "generating new image... (replace the 'positive_caption' with 'identity_embedding')"
    )
    # latent = parallel_generate(
    #     pipeline_bundle=pipeline_bundle,
    #     latent=ddim_inverted_latent.clone(),
    #     negative_condition=negative_caption,
    #     positive_condition=positive_caption,
    #     n_coeff=n_coeff_init,
    #     p_coeff=p_coeff_init,
    #     n_coeff_update_fn=n_coeff_update_fn,
    #     p_coeff_update_fn=p_coeff_update_fn,
    #     num_inference_steps=num_inference_steps,
    #     guidance_scale=guidance_scale,
    # )
    latent = single_generate(
        pipeline_bundle=pipeline_bundle,
        latent=ddim_inverted_latent.clone(),
        positive_condition=positive_caption,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )
    verbose("got new image latents")
    verbose(f"latent.shape {latent.shape}")
    decoded_latents = (
        pipeline_bundle["vae"].decode((latent / 0.18215).to(pipe.dtype)).sample
    )
    image = pipe.image_processor.postprocess(decoded_latents, output_type="pil")
    final_image = image[0]
    final_image.save(image_save_path)
    verbose(f"new image saved to {image_save_path}")
    # GENERATION # GENERATION # GENERATION # GENERATION # GENERATION #


if __name__ == "__main__":
    GPU_ID = 0
    with torch.no_grad():
        final_image = run_image_swap_pipeline(
            image_path=f"/igd/a1/home/demiroer/IDAnonymizationPraktikum/datasets/toy_dataset/3/2.png",
            config_path="src/caption_retrieval_section/clip_configs/race_age_gender_config/race_age_gender_config.py",
            clip_version="ViT-L/14",
            pipe_version="runwayml/stable-diffusion-v1-5",
            device=f"cuda:{GPU_ID}",
            face_app_name="buffalo_l",
            face_app_det_size=640,
            face_app_ctx_id=GPU_ID,
            n_coeff_init=1,
            p_coeff_init=0,
            n_coeff_update_fn=lambda c, t: c,
            p_coeff_update_fn=lambda c, t: c,
            image_save_path=f"stable_diffusion_output.png",
            num_inference_steps=150,
            guidance_scale=7.5,
            inversion_num_inference_steps=50,
            inversion_guidance_scale=1.25,
        )
