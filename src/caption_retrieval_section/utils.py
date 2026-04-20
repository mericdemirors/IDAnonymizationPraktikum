import torch
import clip


def load_clip_model(model_version, device):
    model, preprocess = clip.load(model_version, device=device)
    model.eval()
    return model, preprocess
