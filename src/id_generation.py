from id_generation_section.extract_id import *
from id_generation_section.retrieve_images import *


def get_id_from_textual_data(
    method,
    number_of_ids,
    text_data,
    id_aggregation_func,
    face_app,
    dataset_dict=None,
    pipeline_bundle=None,
):
    """
    Main entry point to get a single ID embedding from text.
    """
    # 1. Image Acquisition
    if method.lower() == "rag":
        if dataset_dict is None:
            raise ValueError("dataset_dict is required for RAG method")
        images = retrieve_images_with_text(dataset_dict, text_data, number_of_ids)
    elif method.lower() == "generation":
        if pipeline_bundle is None:
            raise ValueError("pipeline_bundle is required for Generation method")
        images = generate_images_with_text(pipeline_bundle, text_data, n=number_of_ids)
    else:
        raise ValueError("Method must be 'RAG' or 'Generation'")

    if not images:
        print(f"Warning: No images found or generated for {text_data}")
        return None

    # 2. ID Extraction (Batch processing)
    id_embeddings = get_ids_from_images(face_app, images)

    # 3. Aggregation
    if id_embeddings.shape[0] == 0:
        return None

    # Apply the aggregation (e.g., torch.mean(id_embeddings, dim=0))
    final_id = id_aggregation_func(id_embeddings)

    return final_id


# import torch

# # 1. Setup FaceAnalysis once (Global or Class level)
# from insightface.app import FaceAnalysis
# face_app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])
# face_app.prepare(ctx_id=0, det_size=(640, 640))

# # 2. Prepare the Diffusion Bundle
# # Assuming you have already initialized your model, scheduler, etc.
# pipeline_bundle = {
#     "model": model,
#     "scheduler": scheduler,
#     "vae": vae,
#     "text_encoder": text_encoder,
#     "tokenizer": tokenizer,
# }

# # 3. Define your dataset (for RAG)
# dataset_dict = {
#     "a photo of a person": ["path/to/image1.jpg", "path/to/image2.jpg"],
#     "a professional portrait": ["path/to/image3.jpg"]
# }

# # 4. Define Aggregation Logic
# # Simple mean of embeddings is standard for creating a "Master ID"
# def aggregate_by_mean(embeddings):
#     # embeddings shape: (N, 512)
#     mean_emb = torch.mean(embeddings, dim=0)
#     # Re-normalize to keep it on the hypersphere
#     return mean_emb / mean_emb.norm()

# # --- ACTUAL CALLS ---

# # Example A: Using RAG
# rag_id = get_id_from_textual_data(
#     method='RAG',
#     number_of_ids=2,
#     text_data="a photo of a person",
#     id_aggregation_func=aggregate_by_mean,
#     face_app=face_app,
#     dataset_dict=dataset_dict
# )

# # Example B: Using Generation
# gen_id = get_id_from_textual_data(
#     method='Generation',
#     number_of_ids=3,
#     text_data="a portrait of a man with glasses",
#     id_aggregation_func=aggregate_by_mean,
#     face_app=face_app,
#     pipeline_bundle=pipeline_bundle
# )

# if gen_id is not None:
#     print(f"Generated ID Shape: {gen_id.shape}")
