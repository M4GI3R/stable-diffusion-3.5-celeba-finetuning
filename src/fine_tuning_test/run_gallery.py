import subprocess
import sys
import os

def main():
    # Resolve project root
    current_dir = os.path.dirname(os.path.abspath(__file__))

    dashboard_path = os.path.join(
        current_dir,
        "finetune_gallery.py"
    )

    if not os.path.exists(dashboard_path):
        print(f"Error: {dashboard_path} not found.")
        sys.exit(1)

    print("Launching Streamlit gallery...")
    print(f"File: {dashboard_path}\n")

    # Use same Python interpreter
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", dashboard_path],
        check=True
    )

if __name__ == "__main__":
    main()