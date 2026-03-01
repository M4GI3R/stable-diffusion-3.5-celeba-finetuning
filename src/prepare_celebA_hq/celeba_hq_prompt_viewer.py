import streamlit as st
import pandas as pd
import os
from PIL import Image

# Config
# Adjust paths for your environment if necessary
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

# Assuming standard structure relative to script location
IMG_DIR = os.path.join(PROJECT_ROOT, "res", "img_celeba_hq", "CelebAMask-HQ", "CelebA-HQ-img")
PROMPTS_FILE = os.path.join(BASE_DIR, "prompts.csv")

st.set_page_config(layout="wide", page_title="CelebA-HQ Prompt Viewer")

st.title("CelebA-HQ Prompt Viewer")
st.write("Displays random images and their generated prompts to verify quality.")

@st.cache_data
def load_data():
    if not os.path.exists(PROMPTS_FILE):
        return pd.DataFrame()
    return pd.read_csv(PROMPTS_FILE)

df = load_data()

if df.empty:
    st.error(f"Prompts file not found at: {PROMPTS_FILE}")
else:
    # State Management
    # 'current_indices': The list of 3 filename IDs (strings) currently being displayed.
    if 'current_indices' not in st.session_state:
        # Initialize with random
        rand_rows = df.sample(3)
        st.session_state['current_indices'] = [f.replace(".jpg", "") for f in rand_rows['filename'].tolist()]

    # Global Random Button
    if st.button("Randomize All 3 Images"):
        rand_rows = df.sample(3)
        st.session_state['current_indices'] = [f.replace(".jpg", "") for f in rand_rows['filename'].tolist()]
        st.rerun()

    # Display Columns
    cols = st.columns(3)
    
    for i in range(3):
        with cols[i]:
            # Input Control for this slot
            # Use a form or just columns to align button next to input? 
            # Columns inside column work well.
            c_input, c_btn = st.columns([0.7, 0.3])
            
            with c_input:
                # Value is bound to a throwaway key, we read it when button is pressed
                new_val = st.text_input(f"ID {i+1}", key=f"input_{i}", placeholder="e.g. 123", label_visibility="collapsed")
            
            with c_btn:
                if st.button("Go", key=f"btn_{i}"):
                    if new_val.strip():
                        st.session_state['current_indices'][i] = new_val.strip()
                        st.rerun()

            # Display Logic
            current_id = st.session_state['current_indices'][i]
            filename = f"{current_id}.jpg"
            img_path = os.path.join(IMG_DIR, filename)
            
            # Find prompt
            row = df[df['filename'] == filename]
            
            if not os.path.exists(img_path):
                st.warning(f"Image not found: {filename}")
            else:
                image = Image.open(img_path)
                st.image(image, use_container_width=True)
                st.markdown(f"**{filename}**")
                
                if not row.empty:
                    st.success(row.iloc[0]['prompt'])
                else:
                    st.warning("No prompt generated.")

    with st.expander("Show Full Data Sample"):
        st.dataframe(df.head(50))
