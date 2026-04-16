import torch


def get_ids_from_images(face_app, images):
    """
    Extracts embeddings from a list of BGR images.
    Returns a stacked tensor of shape (N, 512).
    """
    embeddings = []
    for img in images:
        faces = face_app.get(img)
        if faces:
            embeddings.append(torch.from_numpy(faces[0].normed_embedding))

    return torch.stack(embeddings) if embeddings else torch.empty(0)
