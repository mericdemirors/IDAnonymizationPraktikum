def verbose(text, **kwargs):
    print(f"\033[32m{text}\033[0m", **kwargs)


def embed_prompt(pipeline_bundle, prompt):
    text_input = pipeline_bundle["tokenizer"](
        [prompt],
        padding="max_length",
        max_length=pipeline_bundle["tokenizer"].model_max_length,
        return_tensors="pt",
    )
    text_embeddings = pipeline_bundle["text_encoder"](
        text_input.input_ids.to(pipeline_bundle["device"])
    )[0]

    text_embeddings = text_embeddings.to(dtype=pipeline_bundle["dtype"])

    return text_embeddings


def decode_latent_to_image_and_save(latent, pipeline_bundle, image_save_path):
    decoded_latent = (
        pipeline_bundle["vae"]
        .decode((latent / 0.18215).to(pipeline_bundle["dtype"]))
        .sample
    )
    image = pipeline_bundle["image_processor"].postprocess(
        decoded_latent, output_type="pil"
    )
    final_image = image[0]
    final_image.save(image_save_path)
