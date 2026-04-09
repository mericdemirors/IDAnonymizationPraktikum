import torch
import clip


def load_clip_model(model_version, device):
    if device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{device}")

    model, preprocess = clip.load(model_version, device=device)
    model.eval()
    return model
