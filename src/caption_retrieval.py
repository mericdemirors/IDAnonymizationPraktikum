# caption_retrieval.py
import torch
# Placeholder for CLIP model imports
# from transformers import CLIPProcessor, CLIPModel

def load_clip_model():
    # Placeholder: Load CLIP model and processor
    # return model, processor
    pass

def get_caption_probabilities(image, list_of_captions, clip_model, clip_processor):
    """
    Passes the image and captions through CLIP to get match probabilities.
    """
    # Placeholder: Preprocess image and text captions
    # inputs = clip_processor(text=list_of_captions, images=image, return_tensors="pt", padding=True)
    # outputs = clip_model(**inputs)
    # logits_per_image = outputs.logits_per_image
    # probs = logits_per_image.softmax(dim=1)
    
    # Mocking sorted probabilities for the pipeline
    sorted_indices = [0, 1, 2] # Assuming sorted by highest probability
    return sorted_indices

def extract_conditions(image, list_of_captions):
    """
    Returns C1 (Best matching) and C2 (Second best matching) caption texts.
    * Footnote: Instead of second match text, we can go with lower rank ones or multiple texts.
    """
    model, processor = load_clip_model()
    sorted_indices = get_caption_probabilities(image, list_of_captions, model, processor)
    
    # Extract C1 and C2 based on highest probabilities
    c1_text = list_of_captions[sorted_indices[0]] # Best matching caption text
    c2_text = list_of_captions[sorted_indices[1]] # Second best matching caption text
    
    return c1_text, c2_text
