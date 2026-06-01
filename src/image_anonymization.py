import os
import pickle
import torch
from insightface.app import FaceAnalysis

from .caption_retrieval import get_image_captions_and_pos_neg_caption_retrieval_fn
from .caption_retrieval_section.utils import load_clip_model
from .ddim_inversion import ddim_invert_image
from .ddim_inversion_section.utils import prepare_pipe
from .diffusion_generation import (
    parallel_text_generate,
    single_text_generate,
    single_id_generate,
    parallel_id_generate,
)
from .id_generation import get_id_from_textual_data
from .helper_functions import verbose, decode_latent_to_image_and_save
from .ip_adapter_section.prepare_ip_adapter import prepare_ip_adapter


def aggregate_by_mean(embeddings):
    mean_emb = torch.mean(embeddings, dim=0)
    return mean_emb / mean_emb.norm()


CLIP_MODEL = None
PIPELINE_BUNDLE = None
FACE_APP = None
IP_ADAPTER_MODEL = None


def run_image_swap_pipeline(
    image_path,
    config_path,
    clip_version,
    pipe_version,
    vae_version,
    device,
    face_app_name,
    face_app_det_size,
    face_app_ctx_id,
    ip_adapter_ckpt,
    ip_adapter_width,
    ip_adapter_height,
    new_id_retrieval_method,
    dataset_dict,
    n_coeff_init,
    p_coeff_init,
    n_coeff_update_fn,
    p_coeff_update_fn,
    number_of_ids,
    image_save_path,
    inversion_num_inference_steps,
    inversion_guidance_scale,
    text_generation_num_inference_steps,
    text_generation_guidance_scale,
    ip_adapter_num_inference_steps,
    ip_adapter_guidance_scale,
    generation_method,
    clip_model=None,
    pipeline_bundle=None,
    face_app=None,
    ip_adapter_model=None,
    captions_probs_attributelist=None,
    get_positive_and_negative_captions_fn=None,
):
    global CLIP_MODEL, PIPELINE_BUNDLE, FACE_APP, IP_ADAPTER_MODEL

    clip_model = CLIP_MODEL
    pipeline_bundle = PIPELINE_BUNDLE
    face_app = FACE_APP
    ip_adapter_model = IP_ADAPTER_MODEL

    # GET MODELS  GET MODELS # GET MODELS # GET MODELS # GET MODELS #
    if clip_model is None:
        clip_model, processor = load_clip_model(
            model_version=clip_version, device=device
        )
        CLIP_MODEL = clip_model
    if pipeline_bundle is None:
        pipe = prepare_pipe(
            model_version=pipe_version, vae_model_version=vae_version, device=device
        )
        pipeline_bundle = {
            "pipe": pipe,
            "unet": pipe.unet,
            "scheduler": pipe.scheduler,
            "vae": pipe.vae,
            "image_processor": pipe.image_processor,
            "text_encoder": pipe.text_encoder,
            "tokenizer": pipe.tokenizer,
            "device": device,
            "dtype": pipe.unet.dtype,
        }
        PIPELINE_BUNDLE = pipeline_bundle
    if face_app is None and "id" in generation_method:
        face_app = FaceAnalysis(name=face_app_name, providers=["CUDAExecutionProvider"])
        face_app.prepare(
            ctx_id=face_app_ctx_id, det_size=(face_app_det_size, face_app_det_size)
        )
        FACE_APP = face_app
    if ip_adapter_model is None and "id" in generation_method:
        # we have to create a new pipeline becuase ip adapter modifies the one we pass
        # we want to have the pipeline_bundle untouched because it's used everywhere
        pipe_for_ipadapter = prepare_pipe(
            model_version=pipe_version, vae_model_version=vae_version, device=device
        )
        ip_adapter_model = prepare_ip_adapter(
            pipe=pipe_for_ipadapter,
            ip_ckpt=ip_adapter_ckpt,
            device=pipeline_bundle["device"],
        )
        IP_ADAPTER_MODEL = ip_adapter_model
    # GET MODELS  GET MODELS # GET MODELS # GET MODELS # GET MODELS #
    #
    #
    #
    # PREPARE CONDITIONS # PREPARE CONDITIONS # PREPARE CONDITIONS #
    if (
        captions_probs_attributelist is None
        or get_positive_and_negative_captions_fn is None
    ):
        captions_probs_attributelist, get_positive_and_negative_captions_fn = (
            get_image_captions_and_pos_neg_caption_retrieval_fn(
                image_path=image_path,
                config_path=config_path,
                model=clip_model,
                device=device,
            )
        )
    negative_caption, positive_caption = get_positive_and_negative_captions_fn(
        captions_probs_attributelist=captions_probs_attributelist
    )
    verbose(f"C1 (Negative Caption): {negative_caption}")
    verbose(f"C2 (Positive Caption): {positive_caption}")
    # PREPARE CONDITIONS # PREPARE CONDITIONS # PREPARE CONDITIONS #
    #
    #
    #
    # PREPARE DDIM LATENT # PREPARE DDIM LATENT # PREPARE DDIM LATENT #
    ddim_inverted_latent = ddim_invert_image(
        generation_method=generation_method,
        image_path=image_path,
        pipeline_bundle=pipeline_bundle,
        ip_adapter_model=ip_adapter_model,
        prompt=negative_caption,
        num_inference_steps=inversion_num_inference_steps,
        guidance_scale=inversion_guidance_scale,
    )
    # PREPARE DDIM LATENT # PREPARE DDIM LATENT # PREPARE DDIM LATENT #
    #
    #
    #
    # PREPARE ID CONDITION # PREPARE ID CONDITION # PREPARE ID CONDITION #
    if "id" in generation_method:
        identity_embedding = get_id_from_textual_data(
            method=new_id_retrieval_method,
            number_of_ids=number_of_ids,
            text_prompt=positive_caption,
            id_aggregation_func=aggregate_by_mean,
            face_app=face_app,
            pipeline_bundle=pipeline_bundle,
            dataset_dict=dataset_dict,
        )
        if identity_embedding.sum() == 0:
            print("ABORT")
            return
    # PREPARE ID CONDITION # PREPARE ID CONDITION # PREPARE ID CONDITION #
    #
    #
    #
    # GENERATION # GENERATION # GENERATION # GENERATION # GENERATION #
    match generation_method:
        case "single_text":
            latent = single_text_generate(
                pipeline_bundle=pipeline_bundle,
                latent=ddim_inverted_latent.clone(),
                positive_condition=positive_caption,
                num_inference_steps=text_generation_num_inference_steps,
                guidance_scale=text_generation_guidance_scale,
            )
            decode_latent_to_image_and_save(latent, pipeline_bundle, image_save_path)
        case "parallel_text":
            latent = parallel_text_generate(
                pipeline_bundle=pipeline_bundle,
                latent=ddim_inverted_latent.clone(),
                negative_condition=negative_caption,
                positive_condition=positive_caption,
                n_coeff=n_coeff_init,
                p_coeff=p_coeff_init,
                n_coeff_update_fn=n_coeff_update_fn,
                p_coeff_update_fn=p_coeff_update_fn,
                num_inference_steps=text_generation_num_inference_steps,
                guidance_scale=text_generation_guidance_scale,
            )
            decode_latent_to_image_and_save(latent, pipeline_bundle, image_save_path)
        case "single_id":
            image_save_paths = (
                [image_save_path]
                if isinstance(image_save_path, str)
                else image_save_path
            )
            images = single_id_generate(
                ip_adapter_model=ip_adapter_model,
                latent=ddim_inverted_latent.clone(),
                positive_text_condition=[positive_caption],
                positive_id_condition=identity_embedding,
                width=ip_adapter_width,
                height=ip_adapter_height,
                guidance_scale=ip_adapter_guidance_scale,
                num_inference_steps=ip_adapter_num_inference_steps,
            )

            for img, image_save_path in zip(images, image_save_paths):
                img.save(image_save_path)
        case "parallel_id":
            parallel_id_generate()
        case _:
            raise Exception("Wrong generation method")
    # GENERATION # GENERATION # GENERATION # GENERATION # GENERATION #


# okay~ish resulting parameters:
"""
single_text parameters:

    pipe_version="runwayml/stable-diffusion-v1-5",
    vae_version=None,
    inversion_num_inference_steps=100,
    inversion_guidance_scale=1.25,
    text_generation_num_inference_steps=150,
    text_generation_guidance_scale=5.5,

    OR

    pipe_version="SG161222/Realistic_Vision_V4.0_noVAE",
    vae_version="stabilityai/sd-vae-ft-mse",
    inversion_num_inference_steps=50,
    inversion_guidance_scale=1.25,
    text_generation_num_inference_steps=100,
    text_generation_guidance_scale=3,

    

parallel_text_parameters:

    pipe_version="runwayml/stable-diffusion-v1-5",
    vae_version=None,
    inversion_num_inference_steps=100,
    inversion_guidance_scale=1.25,
    text_generation_num_inference_steps=150,
    text_generation_guidance_scale=5.5,

    OR

    pipe_version="SG161222/Realistic_Vision_V4.0_noVAE",
    vae_version="stabilityai/sd-vae-ft-mse",
    inversion_num_inference_steps=50, or 100
    inversion_guidance_scale=1.25,
    text_generation_num_inference_steps=100, or 50
    text_generation_guidance_scale=3, or 5.5



single_id_parameters:

    pipe_version="runwayml/stable-diffusion-v1-5",
    vae_version=None,
    DOESN'T WORK

    OR

    STILL DOESN'T WORK, BUT AT LEAST GIVES OKAY RESULTS
    pipe_version="SG161222/Realistic_Vision_V4.0_noVAE",
    vae_version="stabilityai/sd-vae-ft-mse",
    inversion_num_inference_steps=50
    inversion_guidance_scale=1.25,
    ip_adapter_num_inference_steps=50,
    ip_adapter_guidance_scale=5.5,

"""


if __name__ == "__main__":
    with open(
        "/igd/a1/home/demiroer/IDAnonymizationPraktikum/datasets/caption_to_images.pkl",
        "rb",
    ) as f:
        dataset_dict = pickle.load(f)

    GPU_ID = 0
    with torch.no_grad():
        image_save_path = f"tries/stable_diffusion_output.png"
        run_image_swap_pipeline(
            image_path=f"/igd/a1/home/demiroer/IDAnonymizationPraktikum/datasets/toy_dataset/3/2.png",
            config_path="src/caption_retrieval_section/clip_configs/race_age_gender_config/race_age_gender_config.py",
            clip_version="ViT-L/14",
            pipe_version="SG161222/Realistic_Vision_V4.0_noVAE",
            vae_version="stabilityai/sd-vae-ft-mse",
            device=f"cuda:{GPU_ID}",
            face_app_name="buffalo_l",
            face_app_det_size=512,
            face_app_ctx_id=GPU_ID,
            ip_adapter_ckpt="/igd/a1/home/demiroer/IDAnonymizationPraktikum/src/ip_adapter_section/ip-adapter-faceid_sd15.bin",
            ip_adapter_width=512,
            ip_adapter_height=512,
            # how to retrieve a new ID
            new_id_retrieval_method="RAG",
            dataset_dict="/igd/a1/home/demiroer/IDAnonymizationPraktikum/datasets/caption_to_images.pkl",
            # parallel generation coefficients
            n_coeff_init=0,
            p_coeff_init=1,
            n_coeff_update_fn=lambda c, t: c,
            p_coeff_update_fn=lambda c, t: c,
            # number of IDs to aggregate for ID conditioning
            number_of_ids=3,
            # where to save image
            image_save_path=image_save_path,
            # how to ddim inverse the image
            inversion_num_inference_steps=100,
            inversion_guidance_scale=1.25,
            # how to generate new image
            text_generation_num_inference_steps=150,
            text_generation_guidance_scale=5.5,
            ip_adapter_num_inference_steps=50,
            ip_adapter_guidance_scale=5.5,
            # what generation type to apply
            generation_method="single_id",
        )
