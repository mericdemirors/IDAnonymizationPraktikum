from .caption_retrieval_section.clip_image2captions import (
    get_single_image_probabilities,
)


def get_image_captions_and_pos_neg_caption_retrieval_fn(
    image_path, config_path, model, device
):
    caption_to_prob, cfg = get_single_image_probabilities(
        image_path=image_path, config_path=config_path, model=model, device=device
    )
    caption_to_prob = dict(sorted(caption_to_prob.items(), key=lambda item: -item[1]))
    captions_probs_attributelist = [
        (cap, prob, cfg.extract_attributes(caption=cap))
        for (cap, prob) in list(caption_to_prob.items())
    ]

    return captions_probs_attributelist, cfg.get_positive_and_negative_captions
