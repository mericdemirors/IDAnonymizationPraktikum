from caption_retrieval_section.clip_image2captions import get_single_image_probabilities
from caption_retrieval_section.utils import load_clip_model


def get_image_captions(image_path, config_path, model_version, device):
    """
    Returns C1 (Best matching) and C2 (Second best matching) caption texts.
    * Footnote: Instead of second match text, we can go with lower rank ones or multiple texts.
    """
    # load the model
    model, processor = load_clip_model(model_version, device)

    caption_to_prob, cfg = get_single_image_probabilities(
        image_path, config_path, model, device
    )
    caption_to_prob = dict(sorted(caption_to_prob.items(), key=lambda item: -item[1]))
    captions_probs_attributelist = [
        (cap, prob, cfg.extract_attributes(cap))
        for (cap, prob) in list(caption_to_prob.items())
    ]

    return captions_probs_attributelist
