"""
CelebA Prompt Explorer Dashboard
Streamlit UI for building and exploring prompts

IMPORTANT NOTE (keep this in no matter what you change): 
    - Please replace `use_container_width` with `width`. 
    - `use_container_width` will be removed after 2025-12-31.
    - For `use_container_width=True`, use `width='stretch'`. For `use_container_width=False`, use `width='content'`.
"""
import streamlit as st
import random
import csv
import os
from PIL import Image

# Import generator
from prompt_generator import (
    generate_prompt, 
    ATTR_TO_TEXT, 
    get_all_buckets
)

st.set_page_config(page_title="CelebA Prompt Explorer", layout="wide")

# Paths
CSV_PATH = "res/list_attr_celeba.csv"
IMG_ALIGNED_PATH = "res/img_align_celeba_png"
IMG_FULL_PATH = "res/img_celeba"

@st.cache_data
def load_dataset():
    """Load and parse the CelebA attribute CSV."""
    rows = []
    attr_names = []
    
    with open(CSV_PATH, 'r') as f:
        reader = csv.reader(f)
        first_line = next(reader)
        if len(first_line) == 1 and first_line[0].isdigit():
            attr_names = next(reader)
        else:
            attr_names = first_line
        
        attr_names = [h.strip() for h in attr_names]
        if len(attr_names) == 1 and ' ' in attr_names[0]:
            attr_names = attr_names[0].split()
        
        id_col = attr_names[0]
        attr_names = attr_names[1:] if 'image' in id_col.lower() or len(attr_names) == 41 else attr_names
        
        blurry_idx = attr_names.index('Blurry') if 'Blurry' in attr_names else -1
        
        for line in reader:
            parts = line[0].split() if len(line) == 1 and ' ' in line[0] else line
            image_id = parts[0]
            attrs = [1 if int(val) == 1 else 0 for val in parts[1:]]
            
            if len(attrs) == len(attr_names):
                # Exclude blurry
                if blurry_idx >= 0 and attrs[blurry_idx] == 1:
                    continue
                rows.append({"image_id": image_id, "attrs": attrs})
    
    return attr_names, rows


def find_matching_images(attr_names, rows, selected_attrs, gender):
    """Find images matching the selected attributes."""
    gender_idx = attr_names.index("Male") if "Male" in attr_names else -1
    
    matching = []
    for row in rows:
        if gender_idx >= 0:
            is_male = row["attrs"][gender_idx] == 1
            if (gender == "male" and not is_male) or (gender == "female" and is_male):
                continue
        
        match = True
        for attr in selected_attrs:
            if attr in attr_names and row["attrs"][attr_names.index(attr)] != 1:
                match = False
                break
        
        if match:
            matching.append(row["image_id"])
    
    return matching


# === UI ===
st.title("🖼️ CelebA Prompt Explorer (v9)")

attr_names, rows = load_dataset()
st.caption(f"Dataset: {len(rows):,} images (Blurry excluded)")

generate_btn = st.button("🎲 Generate Random Prompt", type="primary")

st.divider()

# === MANUAL BUILDER (Unified Buckets) ===
st.subheader("Build Prompt")
st.caption("Auto-generation enforces mutual exclusion (one per bucket). Manual mode allows you to freely experiment.")

# 4 Columns for layout: Gender, Slot 1, Slot 2, Slot 3
col_g, col_1, col_2, col_3 = st.columns(4)

with col_g:
    gender = st.selectbox("Gender", ["male", "female"])

buckets = get_all_buckets(gender)
bucket_options = list(buckets.keys())

# Helper to create category/attribute dropdowns for a slot
def render_slot(col, slot_name, default_cat_idx):
    with col:
        st.markdown(f"**{slot_name}**")
        cat = st.selectbox(f"Category {slot_name}", bucket_options, index=default_cat_idx, label_visibility="collapsed")
        attr = st.selectbox(f"Attribute {slot_name}", buckets[cat], label_visibility="collapsed")
        return attr

attr_1 = render_slot(col_1, "Slot #1", 0)
attr_2 = render_slot(col_2, "Slot #2", 1)
attr_3 = render_slot(col_3, "Slot #3", 2)

selected_attrs = [attr_1, attr_2, attr_3]

# Build prompt text
gender_word = "man" if gender == "male" else "woman"
text_attrs = [ATTR_TO_TEXT.get(a, a.lower().replace("_", " ")) for a in selected_attrs]
prompt_text = f"A realistic portrait of a {gender_word}, with {text_attrs[0]}, {text_attrs[1]}, and {text_attrs[2]}."

# Handle random generation
if generate_btn:
    result = generate_prompt()
    prompt_text = result["prompt"]
    gender = result["gender"]
    selected_attrs = result["attributes"]
    st.session_state["last_random"] = result
    st.session_state.pop("sampled_images", None)

st.markdown(f"### 📝 Prompt")
st.code(prompt_text, language=None)

st.divider()

# === MATCHING IMAGES ===
st.subheader("Matching Images")

prompt_key = f"{gender}_{'-'.join(sorted(selected_attrs))}"
matching_images = find_matching_images(attr_names, rows, selected_attrs, gender)

ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 2])
with ctrl_col1:
    st.metric("Matching", f"{len(matching_images):,}")
with ctrl_col2:
    resample_btn = st.button("🔄 Resample Images")
with ctrl_col3:
    use_aligned = st.toggle("Use Aligned Faces", value=True)

img_base_path = IMG_ALIGNED_PATH if use_aligned else IMG_FULL_PATH

current_prompt_key = st.session_state.get("current_prompt_key", None)
sampled_images = st.session_state.get("sampled_images", None)

should_resample = resample_btn or current_prompt_key != prompt_key or sampled_images is None

if should_resample and len(matching_images) > 0:
    sample_size = min(20, len(matching_images))
    sampled_images = random.sample(matching_images, sample_size)
    st.session_state["sampled_images"] = sampled_images
    st.session_state["current_prompt_key"] = prompt_key

# Display grid
if sampled_images and len(sampled_images) > 0:
    cols = st.columns(5)
    for i, img_id in enumerate(sampled_images):
        img_path = os.path.join(img_base_path, img_id)
        if not os.path.exists(img_path):
            base, ext = os.path.splitext(img_id)
            for try_ext in [".png", ".jpg", ".jpeg"]:
                test_path = os.path.join(img_base_path, base + try_ext)
                if os.path.exists(test_path):
                    img_path = test_path
                    break
        
        with cols[i % 5]:
            if os.path.exists(img_path):
                img = Image.open(img_path)
                # Replaced use_container_width=True with width='stretch' per instruction
                st.image(img, caption=img_id, width='stretch')
            else:
                st.warning(f"Not found: {img_id}")
elif len(matching_images) == 0:
    st.info("No matching images found for this combination.")
    