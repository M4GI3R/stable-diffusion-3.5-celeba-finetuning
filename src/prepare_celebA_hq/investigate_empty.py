import csv
import collections

# Config
ANNOTATION_FILE = r'q:\_0_Projects\00_Fraunhofer\finetune_SD_3.5_on_CelebA\res\img_celeba_hq\CelebAMask-HQ\CelebAMask-HQ-attribute-anno.txt'
PROMPTS_FILE = r'q:\_0_Projects\00_Fraunhofer\finetune_SD_3.5_on_CelebA\src\prepare_celebA_hq\prompts.csv'

# Re-define logic to trace it
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

EXCLUSIVE = {"Facial_Hair", "Hair_Structure", "Hair_Color", "Eyebrows", "Nose"}
GENDER_EXCL = {
    "Male": ["Makeup_Style"], 
    "Female": ["Facial_Hair"] 
}

def analyze_empty():
    # 1. Identify Empty Prompts
    print("Reading prompts...")
    empty_files = []
    with open(PROMPTS_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = row['prompt'].strip()
            # Check if it ends with "portrait of a woman." or "portrait of aa man." or similar
            # Robust check: look for "with" or "wearing" or adjectives
            # Actually, our empty matches: "A photo-realistic portrait of a {gender}."
            if p.endswith(" woman.") or p.endswith(" man."):
                empty_files.append(row['filename'])
                
    print(f"Found {len(empty_files)} empty prompts.")
    
    if not empty_files:
        return

    # 2. Read Annotations for these files
    print("Reading annotations...")
    with open(ANNOTATION_FILE, 'r') as f:
        lines = f.readlines()
        
    header = lines[1].strip().split()
    attr_map = {k: v for v, k in enumerate(header)}
    
    # Store reasons
    reasons = collections.Counter()
    example_cases = []
    
    for line in lines[2:]:
        parts = line.strip().split()
        fname = parts[0]
        if fname not in empty_files:
            continue
            
        vals = [int(x) for x in parts[1:]]
        
        # Logic Trace
        is_male = vals[attr_map["Male"]] == 1
        gender_key = "Male" if is_male else "Female"
        
        true_attrs = [header[i] for i, v in enumerate(vals) if v == 1 and header[i] != "Male"]
        
        # Check why buckets failed
        bucket_status = {}
        valid_buckets_found = 0
        
        for b_name, b_attrs in BUCKETS.items():
            # 1. Gender Excl
            if gender_key in GENDER_EXCL and b_name in GENDER_EXCL[gender_key]:
                bucket_status[b_name] = "Gender Excluded"
                continue
                
            # 2. Empty?
            true_in_metrics = [a for a in b_attrs if a in true_attrs]
            if not true_in_metrics:
                bucket_status[b_name] = "No Attributes"
                continue
                
            # 3. Exclusive Violation
            if b_name in EXCLUSIVE and len(true_in_metrics) > 1:
                bucket_status[b_name] = f"Exclusive Violation ({len(true_in_metrics)} found: {true_in_metrics})"
                continue
                
            valid_buckets_found += 1
            bucket_status[b_name] = "Valid"
            
        # If valid_buckets_found > 0, then randomizer just happened to not pick them?
        # Wait, the code says "if not selected_attributes".
        # If there are valid buckets, the code *should* pick something unless the 'Other' logic drained it?
        
        # Let's see if there were ANY valid buckets
        if valid_buckets_found == 0:
            reasons["No Valid Buckets"] += 1
            if len(example_cases) < 10:
                example_cases.append({
                    "file": fname,
                    "gender": gender_key,
                    "true_attrs": true_attrs,
                    "status": bucket_status
                })
        else:
            reasons["Had Valid Buckets (Script Error?)"] += 1
            
    # Report
    print("\n--- Analysis Report ---")
    print(f"Total Empty: {len(empty_files)}")
    for r, c in reasons.items():
        print(f"{r}: {c}")
        
    print("\n--- Examples of 'No Valid Buckets' ---")
    for case in example_cases:
        print(f"\nFile: {case['file']} ({case['gender']})")
        print(f"True Attributes: {case['true_attrs']}")
        print("Bucket Analysis:")
        for b, s in case['status'].items():
            if s != "No Attributes":
                print(f"  - {b}: {s}")

if __name__ == "__main__":
    analyze_empty()
