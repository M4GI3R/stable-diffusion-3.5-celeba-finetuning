import streamlit as st
import os
import time
import json
import glob
import pandas as pd
from PIL import Image
import sys
import subprocess

# IMPORTANT NOTE (keep this in no matter what you change): 
#     - Please replace `use_container_width` with `width`. 
#     - `use_container_width` will be removed after 2025-12-31.
#     - For `use_container_width=True`, use `width='stretch'`. For `use_container_width=False`, use `width='content'`.

# Configure page
st.set_page_config(page_title="SD 3.5 Comparison", layout="wide")

# Constants
OUTPUT_ROOT = "out/comparison"
LEGACY_DIR = os.path.join(OUTPUT_ROOT, "legacy")

# Ensure directories exist
os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(LEGACY_DIR, exist_ok=True)

# --- LAYOUT CONTAINERS ---
# We define these first so we can populate them in a specific order but run logic whenever
main_container = st.container()
st.divider()
history_container = st.container()
st.divider()
stats_container = st.container()

# --- INPUT SECTION (Inside Main) ---
with main_container:
    st.title("Stable Diffusion 3.5 Model Comparison")
    
    prompt = st.text_area("Prompt", "A realistic portrait of a young woman with black hair, narrow eyes, and wearing a necklace. Neutral background, natural lighting, close-up face.")

    generate_btn = st.button("Run Comparison", type="primary")

    # Model Configuration
    col1, col2, col3 = st.columns(3)
    models_config = []

    with col1:
        st.header("Large")
        st.caption("High quality, slow generation")
        with st.expander("Settings"):
            steps_large = st.slider("Steps", 10, 50, 28, key="steps_large", help="**Rule of Thumb**:\n- Large/Medium: 20-30 steps.")
            guidance_large = st.slider("Guidance", 1.0, 20.0, 7.0, key="guid_large", help="**Rule of Thumb**:\n- Medium: 5-7 (Balanced).")
        models_config.append({
            "name": "Large", "id": "stabilityai/stable-diffusion-3.5-large",
            "steps": steps_large, "guidance": guidance_large, "suffix": "large"
        })

    with col2:
        st.header("Turbo")
        st.caption("Fastest, 4-step generation")
        with st.expander("Settings"):
            steps_turbo = st.slider("Steps", 1, 10, 4, key="steps_turbo", help="**Rule of Thumb**:\n- Turbo: 4-8 steps only.")
            guidance_turbo = st.slider("Guidance", 1.0, 20.0, 2.0, key="guid_turbo", help="**Rule of Thumb**:\n- Turbo: 1.5-3.5 (Low).")
        models_config.append({
            "name": "Large Turbo", "id": "stabilityai/stable-diffusion-3.5-large-turbo",
            "steps": steps_turbo, "guidance": guidance_turbo, "suffix": "large_turbo"
        })

    with col3:
        st.header("Medium")
        st.caption("Balanced performance")
        with st.expander("Settings"):
            steps_medium = st.slider("Steps", 10, 50, 28, key="steps_medium", help="**Rule of Thumb**:\n- Large/Medium: 20-30 steps.")
            guidance_medium = st.slider("Guidance", 1.0, 20.0, 7.0, key="guid_medium", help="**Rule of Thumb**:\n- Medium: 5-7 (Balanced).")
        models_config.append({
            "name": "Medium", "id": "stabilityai/stable-diffusion-3.5-medium",
            "steps": steps_medium, "guidance": guidance_medium, "suffix": "medium"
        })

# --- HISTORY DISPLAY (Populate BEFORE generation loop) ---
# Find all run folders (exclude legacy)
run_dirs = [d for d in os.listdir(OUTPUT_ROOT) if os.path.isdir(os.path.join(OUTPUT_ROOT, d)) and d != "legacy"]
run_dirs.sort(reverse=True) # Newest first

all_stats = []

with history_container:
    st.subheader("History")
    
    # Show history items
    for run_id in run_dirs:
        full_path = os.path.join(OUTPUT_ROOT, run_id)
        
        # Container for this run
        with st.container():
            st.markdown(f"#### 🗓️ Run: {run_id}")
            
            # Try to find one metadata file to get the prompt
            prompt_display = "Unknown Prompt"
            files = glob.glob(os.path.join(full_path, "*.json"))
            if files:
                try:
                    with open(files[0], "r") as f:
                        prompt_display = json.load(f).get("prompt", prompt_display)
                except: pass
            
            st.caption(f"📝 {prompt_display}")
            
            hc1, hc2, hc3 = st.columns(3)
            cols = [hc1, hc2, hc3]
            
            suffixes = ["large", "large_turbo", "medium"]
            
            for i, suffix in enumerate(suffixes):
                img_path = os.path.join(full_path, f"{suffix}.png")
                # Fix: Check for .json now, not .png.json
                meta_path = os.path.join(full_path, f"{suffix}.json")
                
                with cols[i]:
                    if os.path.exists(img_path):
                        st.image(Image.open(img_path), use_container_width=True)
                        if os.path.exists(meta_path):
                            try:
                                with open(meta_path, "r") as f:
                                    s = json.load(f)
                                    s['run_id'] = run_id # IMPORTANT: Add run_id for pivoting
                                    all_stats.append(s) # Collect for global table
                                    
                                    sc1, sc2, sc3 = st.columns(3)
                                    sc1.metric("Time", f"{s['duration']:.1f}s")
                                    sc2.metric("Stp", s['steps'])
                                    sc3.metric("Gdn", s['guidance'])
                            except: pass
                    else:
                        st.warning("Missing")
            
            st.divider()

# --- GLOBAL PERFORMANCE STATS (Populate BEFORE generation loop) ---
with stats_container:
    if all_stats:
        st.subheader("Performance Analytics")
        
        df = pd.DataFrame(all_stats)
        if not df.empty and "run_id" in df.columns:
            # 1. Average Speed Metrics (Restored as requested)
            avg_speed = df.groupby("model")["duration"].mean().reset_index()
            
            ac1, ac2, ac3 = st.columns(3)
            # Safe metrics loop
            metrics_cols = [ac1, ac2, ac3]
            for idx, row in avg_speed.iterrows():
                 # Find correct column index based on model name simple hash or just cycle
                 with metrics_cols[idx % 3]:
                    st.metric(f"Avg Time: {row['model']}", f"{row['duration']:.2f}s")
            
            st.divider()

            # 2. Pivot Table (The "Table with run_id Large...")
            # Index=Run, Columns=Model, Values=Duration
            pivot_df = df.pivot_table(index="run_id", columns="model", values="duration", aggfunc="first")
            
            # Get Prompt (one per run)
            prompts = df.groupby("run_id")["prompt"].first()
            
            # Join
            final_df = pivot_df.join(prompts)
            
            # Sort by run_id (newest first)
            final_df = final_df.sort_index(ascending=False)
            
            # Display without extra header
            st.dataframe(final_df, use_container_width=True)
            
        else:
            st.write("No valid data found yet.")

# --- GENERATION LOGIC (Runs Last, updates Main) ---
if generate_btn:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(OUTPUT_ROOT, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    
    st.info(f"� Starting Run: {timestamp}. Check terminal for logs.")
    
    tasks = []
    # Prepare Tasks
    for config in models_config:
        filename = f"{config['suffix']}.png"
        filepath = os.path.join(run_dir, filename)
        tasks.append({
            "name": config['name'],
            "id": config['id'],
            "path": filepath,
            "steps": config['steps'],
            "guidance": config['guidance']
        })
    
    # Launch Subprocess
    args = [sys.executable, "src/comparison/run_batch.py", "--prompt", prompt, "--tasks", json.dumps(tasks)]
    
    try:
        process = subprocess.Popen(args, creationflags=0x00000010, close_fds=True)
        
        # Poll for results logic (Inline visualization for active run)
        active_placeholders = [col1.empty(), col2.empty(), col3.empty()]
        completed_paths = set()
        
        progress_bar = st.progress(0)
        
        while process.poll() is None:
            for i, task in enumerate(tasks):
                path = task['path']
                # UPDATED: Check for .json
                meta_path = os.path.splitext(path)[0] + ".json"
                if path not in completed_paths and os.path.exists(path) and os.path.exists(meta_path):
                    try:
                        img = Image.open(path)
                        img.load()
                        with open(meta_path, "r") as f: stats = json.load(f)
                        
                        with active_placeholders[i].container():
                            st.image(img, use_container_width=True)
                            c_a, c_b, c_c = st.columns(3)
                            c_a.metric("Time", f"{stats['duration']:.1f}s")
                            c_b.metric("Steps", stats['steps'])
                            c_c.metric("Gym", stats['guidance'])
                        completed_paths.add(path)
                        progress_bar.progress(len(completed_paths) / len(tasks))
                    except: pass
            time.sleep(1)
            
        progress_bar.empty()
        st.success("Generation Complete!")
        time.sleep(1) # Give a moment to see success
        st.rerun() # Rerun to load into history view
        
    except Exception as e:
        st.error(f"Failed to launch: {e}")

# (History and Stats blocks moved to top)

# (Stats handled above)
