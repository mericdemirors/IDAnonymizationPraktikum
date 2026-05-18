from .caption_retrieval_section.clip_image2captions import (
    get_single_image_probabilities,
)


def get_image_captions(image_path, config_path, model, device):
    """
    Returns C1 (Best matching) and C2 (Second best matching) caption texts.
    * Footnote: Instead of second match text, we can go with lower rank ones or multiple texts.
    """
    caption_to_prob, cfg = get_single_image_probabilities(
        image_path=image_path, config_path=config_path, model=model, device=device
    )
    caption_to_prob = dict(sorted(caption_to_prob.items(), key=lambda item: -item[1]))
    captions_probs_attributelist = [
        (cap, prob, cfg.extract_attributes(caption=cap))
        for (cap, prob) in list(caption_to_prob.items())
    ]

    return captions_probs_attributelist
