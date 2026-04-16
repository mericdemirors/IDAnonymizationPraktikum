import os
import pickle
import torch
import clip
import importlib.util
from tqdm import tqdm


def generate_captions_and_embeddings_from_config(config_file_path: str, model, device):
    # get the given config
    spec = importlib.util.spec_from_file_location("dynamic_config", config_file_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)

    # setup paths relative to the config file
    config_folder = os.path.dirname(os.path.abspath(config_file_path))
    idx_to_caption_path = os.path.join(config_folder, "idx_to_caption.pkl")
    caption_to_emb_path = os.path.join(config_folder, "caption_to_emb.pkl")

    # create all possible captions
    all_captions = cfg.create_all_possible_captions()

    # read and return if pickles already exists
    if os.path.exists(idx_to_caption_path) and os.path.exists(caption_to_emb_path):
        with open(idx_to_caption_path, "rb") as f:
            idx_to_caption = pickle.load(f)
        with open(caption_to_emb_path, "rb") as f:
            caption_to_emb = pickle.load(f)
        return all_captions, idx_to_caption, caption_to_emb, cfg

    # create mapping from integer index to caption str
    idx_to_caption = {i: cap for i, cap in enumerate(all_captions)}

    # create mapping from caption str to torch tensor
    caption_to_emb = {}
    with torch.no_grad():
        for cap in tqdm(all_captions, leave=False):
            text_tokens = clip.tokenize([cap]).to(device)
            embedding = torch.squeeze(model.encode_text(text_tokens).cpu()).numpy()
            caption_to_emb[cap] = embedding

    # save for future access
    with open(idx_to_caption_path, "wb") as f:
        pickle.dump(idx_to_caption, f)
    with open(caption_to_emb_path, "wb") as f:
        pickle.dump(caption_to_emb, f)

    return all_captions, idx_to_caption, caption_to_emb, cfg
