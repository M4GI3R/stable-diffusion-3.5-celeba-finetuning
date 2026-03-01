import csv
import random
import os
import collections

# Configuration
ANNOTATION_FILE = r'q:\_0_Projects\00_Fraunhofer\finetune_SD_3.5_on_CelebA\res\img_celeba_hq\CelebAMask-HQ\CelebAMask-HQ-attribute-anno.txt'
OUTPUT_DIR = r'q:\_0_Projects\00_Fraunhofer\finetune_SD_3.5_on_CelebA\src\prepare_celebA_hq'
PROMPTS_FILE = os.path.join(OUTPUT_DIR, 'prompts.csv')
STATS_FILE = os.path.join(OUTPUT_DIR, 'prompts_stats.csv')

# Attribute Buckets Mapping
BUCKETS = {
    "Facial_Hair": ["5_o_Clock_Shadow", "Goatee", "Sideburns", "Mustache"],
    "Makeup_Style": ["Wearing_Lipstick", "Heavy_Makeup", "Wearing_Earrings", "Wearing_Necklace"],
    "Hair_Structure": ["Straight_Hair", "Wavy_Hair", "Receding_Hairline", "Bangs", "Bald"],
    "Hair_Color": ["Black_Hair", "Brown_Hair", "Gray_Hair", "Blond_Hair"],
    "Face_Shape": ["High_Cheekbones", "Oval_Face", "Chubby", "Double_Chin"],
    "Eyebrows": ["Bushy_Eyebrows", "Arched_Eyebrows"],
    "Nose": ["Big_Nose", "Pointy_Nose"],
    "Other": ["Young", "Smiling", "Mouth_Slightly_Open", "Attractive", "Bags_Under_Eyes", "Big_Lips", "Narrow_Eyes", "Rosy_Cheeks", "Pale_Skin", "Eyeglasses", "Wearing_Hat", "Wearing_Necktie"]
}

# Bucket Rules
EXCLUSIVE_BUCKETS = {"Hair_Structure", "Hair_Color", "Eyebrows", "Nose"}
GENDER_EXCLUSIVE_BUCKETS = {
    "Male": ["Makeup_Style"], 
    "Female": ["Facial_Hair"] 
}

# Natural Language Mapping
# Categories: ADJ, WEARING, FEATURE
# 'target' used for accumulation (e.g. "hair"). 'order' used for sorting adjectives (lower = first).
ATTR_MAP = {
    "5_o_Clock_Shadow": {"type": "FEATURE", "phrase": "with 5 o'clock shadow"},
    "Arched_Eyebrows": {"type": "FEATURE", "phrase": "with arched eyebrows"},
    "Attractive": {"type": "ADJ", "phrase": "attractive"},
    "Bags_Under_Eyes": {"type": "FEATURE", "phrase": "with bags under eyes"},
    "Bald": {"type": "ADJ", "phrase": "bald"},
    "Bangs": {"type": "FEATURE", "phrase": "with bangs"},
    "Big_Lips": {"type": "FEATURE", "phrase": "with big lips"},
    "Big_Nose": {"type": "FEATURE", "phrase": "with a big nose"},
    "Black_Hair": {"type": "FEATURE", "phrase": "black", "target": "hair", "order": 2},
    "Blond_Hair": {"type": "FEATURE", "phrase": "blond", "target": "hair", "order": 2},
    "Blurry": {"type": "ADJ", "phrase": "blurry"}, 
    "Brown_Hair": {"type": "FEATURE", "phrase": "brown", "target": "hair", "order": 2},
    "Bushy_Eyebrows": {"type": "FEATURE", "phrase": "with bushy eyebrows"},
    "Chubby": {"type": "ADJ", "phrase": "chubby"},
    "Double_Chin": {"type": "FEATURE", "phrase": "with a double chin"},
    "Eyeglasses": {"type": "WEARING", "phrase": "wearing eyeglasses"},
    "Goatee": {"type": "FEATURE", "phrase": "with a goatee"},
    "Gray_Hair": {"type": "FEATURE", "phrase": "gray", "target": "hair", "order": 2},
    "Heavy_Makeup": {"type": "WEARING", "phrase": "wearing heavy makeup"},
    "High_Cheekbones": {"type": "FEATURE", "phrase": "with high cheekbones"},
    "Male": {"type": "GENDER", "phrase": "man"}, 
    "Mouth_Slightly_Open": {"type": "FEATURE", "phrase": "with mouth slightly open"},
    "Mustache": {"type": "FEATURE", "phrase": "with a mustache"},
    "Narrow_Eyes": {"type": "FEATURE", "phrase": "with narrow eyes"},
    "No_Beard": {"type": "FEATURE", "phrase": "with no beard"},
    "Oval_Face": {"type": "FEATURE", "phrase": "with an oval face"},
    "Pale_Skin": {"type": "FEATURE", "phrase": "with pale skin"},
    "Pointy_Nose": {"type": "FEATURE", "phrase": "with a pointy nose"},
    "Receding_Hairline": {"type": "FEATURE", "phrase": "with a receding hairline"},
    "Rosy_Cheeks": {"type": "FEATURE", "phrase": "with rosy cheeks"},
    "Sideburns": {"type": "FEATURE", "phrase": "with sideburns"},
    "Smiling": {"type": "ADJ", "phrase": "smiling"}, 
    "Straight_Hair": {"type": "FEATURE", "phrase": "straight", "target": "hair", "order": 1},
    "Wavy_Hair": {"type": "FEATURE", "phrase": "wavy", "target": "hair", "order": 1},
    "Wearing_Earrings": {"type": "WEARING", "phrase": "wearing earrings"},
    "Wearing_Hat": {"type": "WEARING", "phrase": "wearing a hat"},
    "Wearing_Lipstick": {"type": "WEARING", "phrase": "wearing lipstick"},
    "Wearing_Necklace": {"type": "WEARING", "phrase": "wearing a necklace"},
    "Wearing_Necktie": {"type": "WEARING", "phrase": "wearing a necktie"},
    "Young": {"type": "ADJ", "phrase": "young"}
}

def parse_row(line):
    parts = line.strip().split()
    filename = parts[0]
    values = [int(v) for v in parts[1:]]
    return filename, values

def generate_prompts():
    print(f"Reading annotations from: {ANNOTATION_FILE}")
    with open(ANNOTATION_FILE, 'r') as f:
        lines = f.readlines()
    
    header_line = lines[1]
    attributes = header_line.strip().split()
    attr_to_idx = {name: i for i, name in enumerate(attributes)}
    
    prompts_data = []
    
    # Stats counters
    stats_buckets = collections.Counter()
    stats_attributes = collections.Counter()
    
    for line in lines[2:]:
        filename, values = parse_row(line)
        
        # Gender
        male_val = values[attr_to_idx["Male"]]
        gender = "man" if male_val == 1 else "woman"
        gender_key = "Male" if male_val == 1 else "Female"
        
        # Identify True attributes
        true_attrs = set()
        for i, val in enumerate(values):
            if val == 1:
                true_attrs.add(attributes[i])
        
        # Valid buckets
        valid_buckets = {}
        for bucket_name, bucket_attrs in BUCKETS.items():
            if gender_key in GENDER_EXCLUSIVE_BUCKETS and bucket_name in GENDER_EXCLUSIVE_BUCKETS[gender_key]:
                continue
                
            true_in_bucket = [attr for attr in bucket_attrs if attr in true_attrs]
            if not true_in_bucket:
                continue
                
            if bucket_name in EXCLUSIVE_BUCKETS:
                if len(true_in_bucket) > 1: continue 
                valid_buckets[bucket_name] = true_in_bucket
            else:
                valid_buckets[bucket_name] = true_in_bucket

        # Select 3 attributes
        selected_raw_attrs = []
        available_bucket_names = list(valid_buckets.keys())
        
        for _ in range(3):
            if not available_bucket_names:
                break
                
            chosen_bucket = random.choice(available_bucket_names)
            candidates = valid_buckets[chosen_bucket]
            # Special Logic: Facial_Hair -> Select ALL valid candidates
            if chosen_bucket == "Facial_Hair":
                # Add all facial hair traits
                for cand in candidates:
                    if cand not in selected_raw_attrs:
                        selected_raw_attrs.append(cand)
                        stats_attributes[cand] += 1
                stats_buckets[chosen_bucket] += 1
                available_bucket_names = [b for b in available_bucket_names if b != chosen_bucket]
                continue
            
            # Standard Logic: Pick 1
            valid_candidates = [c for c in candidates if c not in selected_raw_attrs]
            
            if not valid_candidates:
                available_bucket_names = [b for b in available_bucket_names if b != chosen_bucket]
                continue
                
            chosen_attr = random.choice(valid_candidates)
            selected_raw_attrs.append(chosen_attr)
            
            # Stats tracking
            stats_buckets[chosen_bucket] += 1
            stats_attributes[chosen_attr] += 1
            
            if chosen_bucket != "Other":
                available_bucket_names = [b for b in available_bucket_names if b != chosen_bucket]

        # Construct Natural Language Prompt
        adj_list = []
        wear_list = []
        feat_list = []
        
        # Accumulation Dict: target -> list of (order, phrase) tuples
        accumulated_features = collections.defaultdict(list)
        
        for attr in selected_raw_attrs:
            mapping = ATTR_MAP.get(attr)
            if not mapping:
                # Fallback
                feat_list.append(f"with {attr.replace('_', ' ').lower()}")
                continue
                
            if mapping["type"] == "ADJ":
                adj_list.append(mapping["phrase"])
            elif mapping["type"] == "WEARING":
                wear_list.append(mapping["phrase"])
            elif mapping["type"] == "FEATURE":
                if "target" in mapping:
                    accumulated_features[mapping["target"]].append((mapping.get("order", 99), mapping["phrase"]))
                else:
                    feat_list.append(mapping["phrase"])
        
        # Process Accumulated Features (e.g. Hair)
        for target, phrase_tuples in accumulated_features.items():
            # Sort by order
            phrase_tuples.sort(key=lambda x: x[0])
            phrases = [p[1] for p in phrase_tuples]
            
            # Join phrases: "wavy, blond"
            combined_desc = ", ".join(phrases)
            feat_list.append(f"with {combined_desc} {target}")
        
        # Build Sentence
        # Format: "A photo-realistic portrait of a [ADJ] {gender} [WEARING] [FEATURE]."
        
        # 1. Adjectives + Gender
        # e.g. "young, attractive woman"
        subject_phrase = gender
        if adj_list:
            adjs_str = ", ".join(adj_list)
            subject_phrase = f"{adjs_str} {gender}"
            
        # Article "a" or "an"
        # Simple heuristic for vowel start (young -> a, attractive -> an)
        first_word = subject_phrase.split()[0].lower()
        article = "an" if first_word[0] in "aeiou" else "a"
        
        base_prompt = f"A photo-realistic portrait of {article} {subject_phrase}"
        
        # 2. Wearing
        # e.g. "wearing lipstick and wearing a hat" -> "wearing lipstick and a hat" (simplified: just list them)
        # To make it fluid, we can chain them.
        modifiers = []
        modifiers.extend(wear_list)
        
        # 3. Features
        # "with big lips and with black hair" -> "with big lips and black hair"
        # We can clean up "with" repetition if we want, but for now let's just list them to ensure existence.
        # Better: keep 'with' for the first feature, join others? 
        # Simpler approach: Just comma separate all modifiers.
        
        # Refined Logic for Features:
        # If we have multiple "with ...", we can merge them: "with X, Y, and Z"
        # If we have multiple "wearing ...", merge: "wearing X, Y"
        

                
        # Assemble
        parts = [base_prompt]
        
        # Helper to join lists with commas and 'and'
        def join_natural(items):
            if not items: return ""
            if len(items) == 1: return items[0]
            if len(items) == 2: return f"{items[0]} and {items[1]}"
            return f"{', '.join(items[:-1])}, and {items[-1]}"

        if wear_list:
            # Clean "wearing" prefix for subsequent items if we want to merge?
            # Actually, "wearing lipstick, wearing a hat" is Repetitive.
            # "wearing lipstick and a hat".
            clean_wear = []
            for i, w in enumerate(wear_list):
                 if i > 0 and w.startswith("wearing "):
                     clean_wear.append(w.replace("wearing ", ""))
                 else:
                     clean_wear.append(w)
            
            # If the first item didn't start with wearing (unlikely given map), ensure it flows.
            # But the map has "wearing ..." for all WEARING types.
            
            final_wear_str = join_natural(clean_wear)
            parts.append(final_wear_str)

        if feat_list:
            # "with big nose, with black hair" -> "with big nose and black hair"
            clean_feat = []
            for i, f in enumerate(feat_list):
                if i > 0 and f.startswith("with "):
                    clean_feat.append(f.replace("with ", ""))
                else:
                    clean_feat.append(f)
            
            final_feat_str = join_natural(clean_feat)
            parts.append(final_feat_str)
        
        full_prompt = " ".join(parts) + "."
        
        # Clean up double spaces
        full_prompt = full_prompt.replace("  ", " ")
        
        prompts_data.append([filename, full_prompt])

    # Write CSV
    print(f"Writing {len(prompts_data)} prompts to {PROMPTS_FILE}")
    os.makedirs(os.path.dirname(PROMPTS_FILE), exist_ok=True)
    with open(PROMPTS_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "prompt"])
        writer.writerows(prompts_data)
        
    # Write Stats
    print(f"Writing stats to {STATS_FILE}")
    with open(STATS_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Type", "Name", "Count"])
        for bucket, count in stats_buckets.most_common():
            writer.writerow(["Bucket", bucket, count])
        for attr, count in stats_attributes.most_common():
            writer.writerow(["Attribute", attr, count])
            
    print("Done.")

if __name__ == "__main__":
    generate_prompts()
