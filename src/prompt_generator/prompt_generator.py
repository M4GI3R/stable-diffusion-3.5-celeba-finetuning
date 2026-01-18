"""
CelebA Prompt Generator v9
Based on Dataset_Analysis.md Unified Bucket Structure
"""
import random

# Attribute to natural language mapping
ATTR_TO_TEXT = {
    # Facial Hair
    "5_o_Clock_Shadow": "a five o'clock shadow",
    "Goatee": "a goatee",
    "Mustache": "a mustache",
    "Sideburns": "sideburns",
    # Makeup & Style
    "Wearing_Lipstick": "wearing lipstick",
    "Heavy_Makeup": "heavy makeup",
    "Wearing_Earrings": "wearing earrings",
    "Wearing_Necklace": "wearing a necklace",
    # Hair Structure
    "Straight_Hair": "straight hair",
    "Wavy_Hair": "wavy hair",
    "Receding_Hairline": "a receding hairline",
    "Bangs": "bangs",
    "Bald": "a bald head",
    # Hair Color
    "Black_Hair": "black hair",
    "Brown_Hair": "brown hair",
    "Blond_Hair": "blond hair",
    "Gray_Hair": "gray hair",
    # Face Shape
    "High_Cheekbones": "high cheekbones",
    "Oval_Face": "an oval face",
    "Double_Chin": "a double chin",
    "Chubby": "chubby cheeks",
    # Eyebrows
    "Bushy_Eyebrows": "bushy eyebrows",
    "Arched_Eyebrows": "arched eyebrows",
    # Nose
    "Big_Nose": "a prominent nose",
    "Pointy_Nose": "a pointy nose",
    # Other
    "Young": "a youthful appearance",
    "Attractive": "an attractive look",
    "Smiling": "a smile",
    "Mouth_Slightly_Open": "slightly parted lips",
    "Bags_Under_Eyes": "bags under the eyes",
    "Big_Lips": "full lips",
    "Narrow_Eyes": "narrow eyes",
    "Rosy_Cheeks": "rosy cheeks",
    "Pale_Skin": "pale skin",
    "Eyeglasses": "wearing eyeglasses",
    "Wearing_Hat": "wearing a hat",
    "Wearing_Necktie": "wearing a necktie",
}

# Unified bucket definitions with weights (from Dataset_Analysis.md)
# Un-normalized weights are used here; logic normalizes them on the fly.
BUCKETS = {
    "male": {
        "facial_hair": {"prob": 15.9, "attrs": {"5_o_Clock_Shadow": 40.9, "Goatee": 23.2, "Sideburns": 20.8, "Mustache": 15.1}},
        "hair_structure": {"prob": 18.7, "attrs": {"Straight_Hair": 38.8, "Wavy_Hair": 21.7, "Receding_Hairline": 18.3, "Bangs": 12.6, "Bald": 8.5}},
        "hair_color": {"prob": 21.1, "attrs": {"Black_Hair": 53.8, "Brown_Hair": 27.4, "Gray_Hair": 15.3, "Blond_Hair": 3.5}},
        "face_shape": {"prob": 19.7, "attrs": {"High_Cheekbones": 41.2, "Oval_Face": 29.6, "Chubby": 16.1, "Double_Chin": 13.1}},
        "eyebrows": {"prob": 11.8, "attrs": {"Bushy_Eyebrows": 81.9, "Arched_Eyebrows": 18.1}},
        "nose": {"prob": 22.6, "attrs": {"Big_Nose": 72.1, "Pointy_Nose": 27.9}},
        "other": {"prob": 11.9, "attrs": {"Young": 24.5, "Mouth_Slightly_Open": 16.4, "Smiling": 15.7, "Bags_Under_Eyes": 13.7, "Attractive": 11.3, "Wearing_Necktie": 6.8, "Big_Lips": 6.1, "Eyeglasses": 4.8, "Narrow_Eyes": 4.4, "Wearing_Hat": 3.1, "Pale_Skin": 1.0, "Rosy_Cheeks": 0.1}}
    },
    "female": {
        "makeup_style": {"prob": 22.9, "attrs": {"Wearing_Lipstick": 40.7, "Heavy_Makeup": 33.8, "Wearing_Earrings": 15.7, "Wearing_Necklace": 9.8}},
        "hair_structure": {"prob": 17.6, "attrs": {"Wavy_Hair": 50.5, "Bangs": 22.7, "Straight_Hair": 20.8, "Receding_Hairline": 5.9}},
        "hair_color": {"prob": 17.6, "attrs": {"Brown_Hair": 35.5, "Blond_Hair": 34.4, "Black_Hair": 28.7, "Gray_Hair": 1.4}},
        "face_shape": {"prob": 18.9, "attrs": {"High_Cheekbones": 61.2, "Oval_Face": 36.5, "Chubby": 1.3, "Double_Chin": 1.0}},
        "eyebrows": {"prob": 12.8, "attrs": {"Arched_Eyebrows": 85.7, "Bushy_Eyebrows": 14.3}},
        "nose": {"prob": 12.1, "attrs": {"Pointy_Nose": 77.9, "Big_Nose": 22.1}},
        "other": {"prob": 16.5, "attrs": {"Young": 25.3, "Attractive": 20.3, "Smiling": 15.6, "Mouth_Slightly_Open": 15.1, "Big_Lips": 8.8, "Rosy_Cheeks": 3.3, "Narrow_Eyes": 3.1, "Bags_Under_Eyes": 3.0, "Pale_Skin": 1.6, "Wearing_Hat": 0.7, "Eyeglasses": 0.6}}
    }
}


def weighted_choice(weights_dict):
    """Sample one key from a dict of {key: weight}."""
    items = list(weights_dict.keys())
    weights = list(weights_dict.values())
    total = sum(weights)
    normalized = [w / total for w in weights]
    return random.choices(items, weights=normalized, k=1)[0]


def generate_prompt(gender=None):
    """Generate a random prompt based on Dataset_Analysis.md probabilities enforcing bucket mutual exclusion."""
    if gender is None:
        gender = random.choice(["male", "female"])
    
    gender_word = "man" if gender == "male" else "woman"
    
    # Track available buckets
    available_buckets = list(BUCKETS[gender].keys())
    
    selected_buckets = []
    selected_attrs = []
    
    # Pick 3 attributes from 3 distinct buckets
    for _ in range(3):
        # Normalize current available bucket weights
        current_weights = [BUCKETS[gender][b]["prob"] for b in available_buckets]
        total = sum(current_weights)
        norm_weights = [w / total for w in current_weights]
        
        # Select bucket
        chosen_bucket = random.choices(available_buckets, weights=norm_weights, k=1)[0]
        selected_buckets.append(chosen_bucket)

        # Get attributes for this bucket
        attrs_dict = BUCKETS[gender][chosen_bucket]["attrs"]

        # If we selected 'other', filter out attributes we have already chosen
        # to prevent duplicates (e.g. "Young" and "Young")
        if chosen_bucket == "other":
            attrs_dict = {k: v for k, v in attrs_dict.items() if k not in selected_attrs}

        # Select attribute
        chosen_attr = weighted_choice(attrs_dict)
        selected_attrs.append(chosen_attr)

        # Remove bucket so it can't be picked again,
        # UNLESS it is 'other', which allows multiple selections.
        if chosen_bucket != "other":
            available_buckets.remove(chosen_bucket)
        
    # Convert to text
    text_attrs = [ATTR_TO_TEXT.get(attr, attr.lower().replace("_", " ")) for attr in selected_attrs]
    
    prompt = f"A realistic portrait of a {gender_word}, with {text_attrs[0]}, {text_attrs[1]}, and {text_attrs[2]}."
    
    return {
        "prompt": prompt,
        "gender": gender,
        "selected_buckets": selected_buckets,
        "attributes": selected_attrs
    }


def get_all_buckets(gender):
    """Get all buckets and their attributes for the dashboard."""
    result = {}
    for cat_name, cat_info in BUCKETS[gender].items():
        result[cat_name] = list(cat_info["attrs"].keys())
    return result


if __name__ == "__main__":
    # Test generation
    for _ in range(5):
        result = generate_prompt()
        print(result["prompt"])
        print(f"  Buckets: {result['selected_buckets']}")
        print(f"  Attrs: {result['attributes']}\n")
