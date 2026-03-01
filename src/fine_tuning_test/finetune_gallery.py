import streamlit as st
from pathlib import Path
from PIL import Image

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

BASE_DIR = Path(__file__).parent              # fine_tuning_test/
ROOT_DIR = BASE_DIR / "out"                   # fine_tuning_test/out/

st.set_page_config(page_title="SD3.5 LoRA Gallery", layout="wide")
st.title("SD 3.5 Fine-Tune Gallery")

if not ROOT_DIR.exists():
    st.warning("No runs found yet.")
    st.stop()

# --------------------------------------------------
# LOAD RUNS
# --------------------------------------------------

run_dirs = sorted(
    [d for d in ROOT_DIR.iterdir() if d.is_dir()],
    key=lambda x: x.name,
    reverse=True
)

if not run_dirs:
    st.warning("No runs available.")
    st.stop()

run_names = [d.name for d in run_dirs]

st.sidebar.header("Run Selection")
selected_runs = st.sidebar.multiselect(
    "Select runs to display",
    run_names,
    default=[run_names[0]]
)

if not selected_runs:
    st.stop()

# --------------------------------------------------
# GRID HELPER
# --------------------------------------------------

def display_grid(image_paths, columns=5):
    """
    Display images in a fixed-width grid.
    Wraps automatically if more than `columns`.
    """
    for row_start in range(0, len(image_paths), columns):
        cols = st.columns(columns)
        for i in range(columns):
            idx = row_start + i
            with cols[i]:
                if idx < len(image_paths):
                    st.image(
                        Image.open(image_paths[idx]),
                        use_container_width=True
                    )
                else:
                    st.empty()

# --------------------------------------------------
# DISPLAY SELECTED RUNS
# --------------------------------------------------

for run_name in selected_runs:

    run_path = ROOT_DIR / run_name
    st.subheader(f"Run: {run_name}")

    base_images = sorted((run_path / "base").glob("*.png"))
    lora_images = sorted((run_path / "lora").glob("*.png"))

    st.markdown("### Base")
    display_grid(base_images, columns=5)

    st.markdown("### LoRA")
    display_grid(lora_images, columns=5)

    st.divider()

# --------------------------------------------------
# HISTORY GRID
# --------------------------------------------------

st.subheader("History")

for run_path in run_dirs:

    base_images = sorted((run_path / "base").glob("*.png"))
    lora_images = sorted((run_path / "lora").glob("*.png"))

    st.markdown(f"#### {run_path.name}")

    st.markdown("Base")
    display_grid(base_images, columns=5)

    st.markdown("LoRA")
    display_grid(lora_images, columns=5)

    st.divider()