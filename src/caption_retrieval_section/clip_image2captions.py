import torch
from caption_retrieval_section.clip_configs.clip_prepare_config import (
    generate_captions_and_embeddings_from_config,
)
from CLIPImageReaderDataset import CLIPImageReaderDataset


# function to get attribute probs of a single image
def get_single_image_probabilities(image_path, config_path, model, device):
    # get the config defined captions
    all_captions, idx_to_caption, caption_to_emb, cfg = (
        generate_captions_and_embeddings_from_config(config_path, model, device)
    )

    # create the matrix that holds caption embeddings
    all_text_embeds = torch.stack(
        [torch.from_numpy(caption_to_emb[cap]) for cap in all_captions]
    )
    all_text_embeds = all_text_embeds.to(device).float()
    all_text_embeds /= all_text_embeds.norm(dim=-1, keepdim=True)

    # we initialize the dataset so we can use it to read individual images
    reader = CLIPImageReaderDataset()
    img = reader.read_single_img(image_path)

    # add batch dimension and move to device
    img = img.unsqueeze(0).to(device)

    # run clip to get probabilities
    with torch.no_grad():
        image_embeds = model.encode_image(img).float()
        image_embeds /= image_embeds.norm(dim=-1, keepdim=True)

        probs = (100.0 * image_embeds @ all_text_embeds.T).softmax(dim=-1)

        # Move to cpu and convert to list
        probabilities = probs.squeeze(0).cpu().tolist()

    # map caption str to it's match probability
    caption_to_prob = {
        all_captions[i]: probabilities[i] for i in range(len(all_captions))
    }

    return caption_to_prob, cfg
