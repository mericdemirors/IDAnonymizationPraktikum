import re
import itertools

attributes_and_values_dict = {
    "race": ["Black", "White", "Asian", "Indian"],
    "age": ["18", "25", "35", "45", "55", "65"],
    "gender": ["Male", "Female"],
}

empty_caption = "A portrait photo of a {age} year old {race} {gender}."


def fill_attributes(selected_values_dict: dict):
    """Fills the template using the dictionary keys"""
    return empty_caption.format(**selected_values_dict)


def extract_attributes(caption: str):
    """Regex pattern to capture the values based on the known template structure"""
    pattern = (
        r"A portrait photo of a (?P<age>\d+) year old (?P<race>\w+) (?P<gender>\w+)."
    )
    match = re.search(pattern, caption)
    return match.groupdict() if match else {}


def create_all_possible_captions():
    """Generates all combinations of captions based on the dict and template."""
    keys = attributes_and_values_dict.keys()
    values = attributes_and_values_dict.values()
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return sorted([fill_attributes(selected_values_dict=c) for c in combinations])


def get_positive_and_negative_captions(captions_probs_attributelist):
    negative_caption = captions_probs_attributelist[0][0]
    negative_race = captions_probs_attributelist[0][2]["race"]
    for i in range(1, len(captions_probs_attributelist)):
        if negative_race != captions_probs_attributelist[i][2]["race"]:
            positive_caption = captions_probs_attributelist[i][0]
            break
    return negative_caption, positive_caption
