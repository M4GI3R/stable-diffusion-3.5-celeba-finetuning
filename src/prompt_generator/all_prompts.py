import csv
import os
import itertools
from collections import Counter

import matplotlib.pyplot as plt
from prompt_generator import BUCKETS, ATTR_TO_TEXT

# Paths
CSV_PATH = "../../res/list_attr_celeba.csv"
OUTPUT_CSV = "celeba_prompt_stats.csv"
OUTPUT_IMG = "prompt_matches_distribution.png"


def load_attribute_sets():
    """
    Loads CelebA CSV and converts columns into Sets of Image IDs.
    This allows for O(1) intersection speed when checking combinations.
    """
    print(f"Loading dataset from {CSV_PATH}...")

    attr_sets = {}
    all_attr_names = []

    with open(CSV_PATH, 'r') as f:
        reader = csv.reader(f)

        # Handle headers
        first_line = next(reader)
        if len(first_line) == 1 and first_line[0].isdigit():
            headers = next(reader)
        else:
            headers = first_line

        headers = [h.strip() for h in headers]
        if len(headers) == 1 and ' ' in headers[0]:
            headers = headers[0].split()

        attr_start_idx = 1
        all_attr_names = headers[attr_start_idx:]

        for name in all_attr_names:
            attr_sets[name] = set()

        blurry_idx = headers.index('Blurry') if 'Blurry' in headers else -1

        count = 0
        for line in reader:
            parts = line[0].split() if len(line) == 1 and ' ' in line[0] else line

            # Skip blurry
            if blurry_idx != -1 and int(parts[blurry_idx]) == 1:
                continue

            img_id = parts[0]

            # Map attributes (1 = True)
            for i, val in enumerate(parts[attr_start_idx:]):
                if int(val) == 1:
                    attr_name = all_attr_names[i]
                    attr_sets[attr_name].add(img_id)
            count += 1

    print(f"Processed {count:,} images.")
    return attr_sets, all_attr_names


def is_valid_combination(bucket_list):
    """
    Validates a combination of 3 buckets based on v9 rules:
    - Can only use a specific structural bucket ONCE (e.g. max 1 'hair_color').
    - Can use 'other' bucket multiple times.
    """
    counts = Counter(bucket_list)
    for bucket, count in counts.items():
        # Rule: Only 'other' is allowed to appear more than once
        if bucket != "other" and count > 1:
            return False
    return True


def generate_stats():
    attr_sets, all_attr_names = load_attribute_sets()
    results = []

    print("Calculating combinations (this may take a moment)...")

    for gender in ["male", "female"]:
        print(f"Processing {gender}...")

        # 1. Flatten all available attributes for this gender into a list: [(bucket, attr_name), ...]
        available_items = []
        bucket_data = BUCKETS[gender]

        for bucket_name, info in bucket_data.items():
            for attr_name in info["attrs"].keys():
                available_items.append((bucket_name, attr_name))

        # 2. Generate ALL possible combinations of 3 attributes
        # itertools.combinations picks unique items, so we won't get (Young, Young, ...)
        all_combos = list(itertools.combinations(available_items, 3))

        # 3. Filter valid combinations based on bucket rules
        valid_combos = []
        for combo in all_combos:
            # combo is tuple of ((b1, a1), (b2, a2), (b3, a3))
            buckets = [item[0] for item in combo]
            if is_valid_combination(buckets):
                valid_combos.append(combo)

        print(f"  > Found {len(valid_combos)} valid combinations for {gender}.")

        # 4. Calculate matches
        for combo in valid_combos:
            # Extract names
            attrs = [item[1] for item in combo]
            buckets = [item[0] for item in combo]

            a1, a2, a3 = attrs

            # Get Sets
            set_a1 = attr_sets.get(a1, set())
            set_a2 = attr_sets.get(a2, set())
            set_a3 = attr_sets.get(a3, set())

            # Intersection
            base_match = set_a1.intersection(set_a2).intersection(set_a3)

            # Gender Filter
            male_set = attr_sets.get("Male", set())
            if gender == "male":
                final_match = base_match.intersection(male_set)
            else:
                final_match = base_match.difference(male_set)

            results.append({
                "gender": gender,
                "attrs": attrs,
                "buckets": buckets,
                "count": len(final_match)
            })

    # Sort results
    results.sort(key=lambda x: x['count'])

    total_valid = len(results)
    print(f"Total valid combinations generated: {total_valid:,}")

    # === SAVE TO CSV ===
    print(f"Saving to {OUTPUT_CSV}...")
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Gender", "Count", "Attribute_1", "Attribute_2", "Attribute_3", "Buckets", "Prompt_Preview"])

        for r in results:
            text_attrs = [ATTR_TO_TEXT.get(a, a) for a in r['attrs']]
            prompt = f"A realistic portrait of a {r['gender']}, with {text_attrs[0]}, {text_attrs[1]}, and {text_attrs[2]}."

            writer.writerow([
                r['gender'],
                r['count'],
                r['attrs'][0],
                r['attrs'][1],
                r['attrs'][2],
                str(r['buckets']),
                prompt
            ])

    # === GENERATE PLOT ===
    print(f"Generating plot {OUTPUT_IMG}...")

    counts = [r['count'] for r in results]

    plt.figure(figsize=(12, 6))
    plt.plot(counts, color='#4CAF50', linewidth=1.5)
    plt.fill_between(range(len(counts)), counts, color='#E8F5E9')

    plt.title(
        f'Distribution of Matches per Prompt\nTotal Combinations: {total_valid:,}')
    plt.xlabel('Prompt Combination Index (Sorted Low to High)')
    plt.ylabel('Number of Matching Images')
    plt.grid(True, linestyle='--', alpha=0.5)

    zero_count = counts.count(0)
    plt.annotate(f'Prompts with 0 matches: {zero_count:,}',
                 xy=(0, 0), xytext=(len(counts) * 0.05, max(counts) * 0.8),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    plt.tight_layout()
    plt.savefig(OUTPUT_IMG)
    print("Done!")


if __name__ == "__main__":
    generate_stats()
