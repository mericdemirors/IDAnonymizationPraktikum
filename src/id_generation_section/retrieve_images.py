import torch
from tqdm import tqdm
import cv2
import numpy as np


def retrieve_images_with_text(dataset_dict: dict, text_prompt: str, n: int):
    """Returns a list of read images"""
    image_paths_and_probs = dataset_dict.get(text_prompt, [])[:n]
    images = [cv2.imread(path) for (path, prob) in image_paths_and_probs]
    return [img for img in images if img is not None]


def generate_images_with_text(pipeline_bundle, text_prompt, n):
    generated_images = []
    for _ in tqdm(range(n)):
        result = pipeline_bundle["pipe"](
            prompt=text_prompt,
            negative_prompt="",
            num_inference_steps=150,
            guidance_scale=7.25,
            output_type="pil",
        )

        # Extract the image
        pil_img = result.images[0]

        # Convert PIL (RGB) to OpenCV (BGR) for your saving configuration
        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        generated_images.append(img_bgr)

    return generated_images
