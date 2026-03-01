import os
import collections

# Configuration
ANNOTATION_FILE = r'q:\_0_Projects\00_Fraunhofer\finetune_SD_3.5_on_CelebA\res\img_celeba_hq\CelebAMask-HQ\CelebAMask-HQ-attribute-anno.txt'
DASHBOARD_FILE = r'q:\_0_Projects\00_Fraunhofer\finetune_SD_3.5_on_CelebA\src\prompt_generator\Analysis_Dashboard.md'

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

def analyze():
    print(f"Reading annotations from: {ANNOTATION_FILE}")
    with open(ANNOTATION_FILE, 'r') as f:
        lines = f.readlines()
    
    total_images = int(lines[0].strip())
    attributes = lines[1].strip().split()
    attr_to_idx = {name: i for i, name in enumerate(attributes)}
    
    # Init counters
    # Global counts
    attr_counts = collections.Counter()
    # Gender counts
    male_counts = collections.Counter()
    female_counts = collections.Counter()
    total_male = 0
    total_female = 0
    
    for line in lines[2:]:
        parts = line.strip().split()
        values = [int(v) for v in parts[1:]]
        
        is_male = values[attr_to_idx["Male"]] == 1
        
        if is_male:
            total_male += 1
        else:
            total_female += 1
            
        for i, val in enumerate(values):
            if val == 1:
                attr_name = attributes[i]
                attr_counts[attr_name] += 1
                if is_male:
                    male_counts[attr_name] += 1
                else:
                    female_counts[attr_name] += 1
                    
    # Generate Dashboard
    print(f"Generating dashboard to {DASHBOARD_FILE}")
    
    with open(DASHBOARD_FILE, 'w', encoding='utf-8') as f:
        f.write("# CelebA-HQ Analysis Dashboard\n\n")
        f.write(f"**Total Images**: {total_images}\n")
        f.write(f"- **Male**: {total_male} ({total_male/total_images:.1%})\n")
        f.write(f"- **Female**: {total_female} ({total_female/total_images:.1%})\n\n")
        
        f.write("## Attribute Frequency (Top 20)\n")
        f.write("| Attribute | Count | Percentage |\n")
        f.write("| :--- | :--- | :--- |\n")
        for attr, count in attr_counts.most_common(20):
            f.write(f"| {attr} | {count} | {count/total_images:.1%} |\n")
        f.write("\n")
        
        f.write("## Bucket Distribution\n")
        for bucket, bucket_attrs in BUCKETS.items():
            f.write(f"### {bucket}\n")
            f.write("| Attribute | Total % | Male % | Female % |\n")
            f.write("| :--- | :--- | :--- | :--- |\n")
            
            # Sort by total frequency
            sorted_attrs = sorted(bucket_attrs, key=lambda x: attr_counts[x], reverse=True)
            
            for attr in sorted_attrs:
                total_p = attr_counts[attr] / total_images
                male_p = male_counts[attr] / total_male if total_male > 0 else 0
                female_p = female_counts[attr] / total_female if total_female > 0 else 0
                
                f.write(f"| {attr} | {total_p:.1%} | {male_p:.1%} | {female_p:.1%} |\n")
            f.write("\n")
            
    print("Done analysis.")

if __name__ == "__main__":
    analyze()
