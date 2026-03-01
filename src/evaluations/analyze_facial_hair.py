import collections

ANNOTATION_FILE = r'q:\_0_Projects\00_Fraunhofer\finetune_SD_3.5_on_CelebA\res\img_celeba_hq\CelebAMask-HQ\CelebAMask-HQ-attribute-anno.txt'
FACIAL_HAIR = ["5_o_Clock_Shadow", "Goatee", "Sideburns", "Mustache"]

def analyze():
    print(f"Reading {ANNOTATION_FILE}...")
    with open(ANNOTATION_FILE, 'r') as f:
        lines = f.readlines()
    
    header = lines[1].strip().split()
    attr_map = {name: i for i, name in enumerate(header)}
    
    counts = collections.Counter()
    
    for line in lines[2:]:
        parts = line.strip().split()
        vals = [int(v) for v in parts[1:]]
        
        active_hair = []
        for fh in FACIAL_HAIR:
            if vals[attr_map[fh]] == 1:
                active_hair.append(fh)
        
        count = len(active_hair)
        counts[count] += 1
        
        if count >= 3:
            print(f"File {parts[0]}: {active_hair}")

    print("\n--- Facial Hair Counts ---")
    for k in sorted(counts.keys()):
        print(f"{k} attributes: {counts[k]}")

if __name__ == "__main__":
    analyze()
