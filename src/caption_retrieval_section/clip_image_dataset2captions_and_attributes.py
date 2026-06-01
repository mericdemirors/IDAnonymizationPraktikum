import os
import torch
import argparse
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader
from .clip_configs.clip_prepare_config import (
    generate_captions_and_embeddings_from_config,
)
from .CLIPImageReaderDataset import CLIPImageReaderDataset
from .utils import load_clip_model


# function to get image-caption matchings to prepare the RAG dataset
def get_image_dataset_captions_and_attributes(config_path, dataset_path, model, device):
    # get the config defined captions
    all_captions, idx_to_caption, caption_to_emb, cfg = (
        generate_captions_and_embeddings_from_config(
            config_file_path=config_path, model=model, device=device
        )
    )

    # create the matrix that holds caption embeddings
    all_text_embeds = torch.stack(
        [torch.from_numpy(caption_to_emb[cap]) for cap in all_captions]
    )
    all_text_embeds = all_text_embeds.to(device).float()
    all_text_embeds /= all_text_embeds.norm(dim=-1, keepdim=True)

    # setup the dataset
    dataset = CLIPImageReaderDataset(dataset_folder=dataset_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # '<id>/<sample>' -> (caption_text, dict_of_attributes)
    image_to_caption_and_attributes = {}

    for id_indices, img_file_names, imgs in tqdm(dataloader, leave=False):
        imgs = imgs.to(device)

        # run clip to get probabilities
        with torch.no_grad():
            image_embeds = model.encode_image(imgs).float()
            image_embeds /= image_embeds.norm(dim=-1, keepdim=True)
            probs = (100.0 * image_embeds @ all_text_embeds.T).softmax(dim=-1)
            best_match_indices = torch.argmax(probs, dim=1).cpu().tolist()

        # map images to their caption probability, caption str, and list of attributes
        for i, match_idx in enumerate(best_match_indices):
            caption_prob = probs[i, match_idx].item()
            caption_text = idx_to_caption[match_idx]
            attributes_dict = cfg.extract_attributes(caption=caption_text)

            # Build the key using the current batch element
            # Assuming id_indices and img_file_names are lists from the dataloader
            curr_id = id_indices[i]
            curr_name = img_file_names[i]
            key = os.path.join(f"{curr_id}", f"{curr_name}")
            image_to_caption_and_attributes[key] = (
                caption_prob,
                caption_text,
                attributes_dict,
            )

    return image_to_caption_and_attributes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process dataset and save Top-10 mapping."
    )
    parser.add_argument(
        "--config_path", type=str, required=True, help="Path to CLIP attributes config"
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to the dataset folder"
    )
    parser.add_argument("--output_pickle", type=str, default="caption_to_images.pkl")
    parser.add_argument(
        "--model_version", type=str, default="ViT-L/14", help="CLIP model version"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="GPU ID (int) or 'cpu'"
    )
    args = parser.parse_args()
    device = f"cuda:{args.device}" if args.device.isdigit() else "cpu"
    parent_dir = os.path.dirname(os.path.abspath(args.dataset_path))

    # get image to caption mapping
    model, processor = load_clip_model(model_version=args.model_version, device=device)
    image_to_caption_and_attributes = get_image_dataset_captions_and_attributes(
        config_path=args.config_path,
        dataset_path=args.dataset_path,
        model=model,
        device=device,
    )

    # create caption to image mapping
    caption_to_all_matches = {}

    for img_path, (prob, caption, _) in image_to_caption_and_attributes.items():
        if caption not in caption_to_all_matches:
            caption_to_all_matches[caption] = []
        caption_to_all_matches[caption].append(
            (os.path.join(parent_dir, img_path), prob)
        )

    # get top 10 matches
    caption_to_top10_match_and_probs_dict = {}
    for caption, matches in caption_to_all_matches.items():
        sorted_matches = sorted(matches, key=lambda x: x[1], reverse=True)
        caption_to_top10_match_and_probs_dict[caption] = sorted_matches[:10]

    # save
    with open(os.path.join(parent_dir, args.output_pickle), "wb") as f:
        pickle.dump(caption_to_top10_match_and_probs_dict, f)
