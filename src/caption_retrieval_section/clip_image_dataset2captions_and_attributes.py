import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from clip_prepare_config import generate_captions_and_embeddings_from_config
from CLIPImageReaderDataset import CLIPImageReaderDataset


def get_image_dataset_captions_and_attributes(config_path, dataset_path, model, device):
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

    # 4. Set up the dataset
    dataset = CLIPImageReaderDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # '<id>/<sample>' -> (caption_text, dict_of_attributes)
    image_to_caption_and_attributes = {}

    for id_indices, img_file_names, imgs in tqdm(dataloader, leave=False):
        imgs = imgs.to(device)

        with torch.no_grad():
            # Get the image embedding
            image_embeds = model.encode_image(imgs).float()
            image_embeds /= image_embeds.norm(dim=-1, keepdim=True)

            probs = (100.0 * image_embeds @ all_text_embeds.T).softmax(dim=-1)

            # Get the index of most probable match
            best_match_indices = torch.argmax(probs, dim=1).cpu().tolist()

        # Iterate through batch results (handling case where batch_size > 1)
        for i, match_idx in enumerate(best_match_indices):
            caption_text = idx_to_caption[match_idx]

            # Get the dict of attributes for that caption
            attributes_dict = cfg.extract_attributes(caption_text)

            # Build the key using the current batch element
            # Assuming id_indices and img_file_names are lists from the dataloader
            curr_id = id_indices[i]
            curr_name = img_file_names[i]

            key = f"{curr_id}/{curr_name}"
            image_to_caption_and_attributes[key] = (caption_text, attributes_dict)

    return image_to_caption_and_attributes


if __name__ == "__main__":
    # 1. Get Arguments
    parser = argparse.ArgumentParser(
        description="Process dataset with CLIP and dynamic config."
    )
    parser.add_argument(
        "--config_path", type=str, required=True, help="Path to CLIP attributes config"
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to the dataset folder"
    )
    parser.add_argument(
        "--model_version", type=str, default="ViT-L/14", help="CLIP model version"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="GPU ID (int) or 'cpu'"
    )
    args = parser.parse_args()

    get_image_dataset_captions_and_attributes(
        args.dataset_path, args.config_path, args.model_version, args.device
    )
