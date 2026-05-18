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


def verbose(text):
    print(f"\033[32m{text}\033[0m")
