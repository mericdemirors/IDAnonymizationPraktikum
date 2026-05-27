from .id_generation_section.extract_id import *
from .id_generation_section.retrieve_images import *


def get_id_from_textual_data(
    method,
    number_of_ids,
    text_prompt,
    id_aggregation_func,
    face_app,
    pipeline_bundle,
    dataset_dict=None,
    num_inference_steps=150,
    guidance_scale=7.5,
):
    """
    Main entry point to get a single ID embedding from text.
    """
    # get images to extract ID from
    if method.lower() == "rag":
        if dataset_dict is None:
            raise ValueError("dataset_dict is required for RAG method")
        images = retrieve_images_with_text(
            dataset_dict=dataset_dict, text_prompt=text_prompt, n=number_of_ids
        )
    elif method.lower() == "generation":
        if pipeline_bundle is None:
            raise ValueError("pipeline_bundle is required for Generation method")
        images = generate_images_with_text(
            pipeline_bundle=pipeline_bundle,
            text_prompt=text_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            n=number_of_ids,
        )
    else:
        raise ValueError("Method must be 'RAG' or 'Generation'")

    # extract ID
    id_embeddings = get_ids_from_images(face_app=face_app, images=images)

    print(
        f"got {len(images)} images using {method}, extracted {len(id_embeddings)} ID embeddings (RIGHT NOW FACES ARE NOT BEING ALIGNED BEFORE ID EMBED EXTRACT)"
    )

    # if no ID found in the images
    if id_embeddings.shape[0] == 0:
        return torch.zeros(512, dtype=torch.float32).to(pipeline_bundle["device"])

    # else aggregate the IDs into a single ID
    final_id = id_aggregation_func(id_embeddings)

    return final_id.to(pipeline_bundle["device"])
