def verbose(text, **kwargs):
    print(f"\033[32m{text}\033[0m", **kwargs)


def embed_prompt(
    pipeline_bundle_or_ipadapter, prompt, do_classifier_free_guidance=False
):
    if isinstance(pipeline_bundle_or_ipadapter, dict):
        pipe = pipeline_bundle_or_ipadapter["pipe"]
        device = pipeline_bundle_or_ipadapter["device"]
    else:
        pipe = pipeline_bundle_or_ipadapter.pipe
        device = pipeline_bundle_or_ipadapter.device

    prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_classifier_free_guidance,
        negative_prompt=None,
    )[0]

    return prompt_embeds


def decode_latent_to_image_and_save(latent, pipeline_bundle, image_save_path):
    decoded_latent = (
        pipeline_bundle["vae"]
        .decode(
            (latent / pipeline_bundle["vae"].config.scaling_factor).to(
                pipeline_bundle["dtype"]
            )
        )
        .sample
    )
    image = pipeline_bundle["image_processor"].postprocess(
        decoded_latent, output_type="pil"
    )
    final_image = image[0]
    final_image.save(image_save_path)
