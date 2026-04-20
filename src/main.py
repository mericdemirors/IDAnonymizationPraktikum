import torch
from caption_retrieval import get_image_captions
from ddim_inversion import ddim_invert_image
from diffusion_generation import parallel_generate
from id_generation import get_id_from_textual_data


def run_image_swap_pipeline(
    image_path,
    config_path,
    model_version,
    device,
    face_app,
    pipeline_bundle,
    scheduler,
    n_coeff_init=1.0,
    p_coeff_init=1.0,
    n_coeff_update_fn=lambda c, t: c,
    p_coeff_update_fn=lambda c, t: c,
):
    """
    1. DDIM Inverts image
    2. Gets ranked captions
    3. Extracts ID from the 2nd best caption
    4. Generates new image using inverted latents + Parallel Diffusion
    """

    print("get captions...")
    caption_data = get_image_captions(image_path, config_path, model_version, device)
    if len(caption_data) < 2:
        raise ValueError("Could not find at least two fitting captions.")

    best_caption = caption_data[0][0]
    second_best_caption = caption_data[1][0]

    print(f"C1 (Negative): {best_caption}")
    print(f"C2 (Positive Source): {second_best_caption}")

    print("conditional invertion...")
    inverted_latent = ddim_invert_image(
        image_path=image_path,
        device=device,
        model_version=model_version,
        prompt=best_caption,
    )

    print("get ID for positive cond...")
    identity_embedding = get_id_from_textual_data(
        method="generation",
        number_of_ids=1,
        text_data=second_best_caption,
        id_aggregation_func=lambda x: torch.mean(x, dim=0),
        face_app=face_app,
        pipeline_bundle=pipeline_bundle,
    )

    print("generate new image...")
    final_latent = parallel_generate(
        negative_model=pipeline_bundle["negative_model"],
        positive_model=pipeline_bundle["positive_model"],
        latent=inverted_latent.clone(),
        scheduler=scheduler,
        negative_condition=best_caption,
        positive_condition=identity_embedding,
        n_coeff=n_coeff_init,
        p_coeff=p_coeff_init,
        n_coeff_update_fn=n_coeff_update_fn,
        p_coeff_update_fn=p_coeff_update_fn,
    )

    # Decode the final latent back to an image
    # final_image = decode_latent(latent, pipeline_bundle['vae'])
    return final_latent
