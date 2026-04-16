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
    return sorted([fill_attributes(c) for c in combinations])
