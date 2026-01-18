import argparse
import sys
import os
import time
import json
from sd35_runner import SDRunner
from constants import HF_TOKEN

def run_batch(prompt, tasks_json):
    """
    Runs generation for multiple models sequentially with per-task settings.
    tasks_json: JSON string list of dicts:
    [
      {
        "name": "Large",
        "id": "...", 
        "path": "...",
        "steps": 28,
        "guidance": 7.0
      },
      ...
    ]
    """
    print("\n" + "="*60)
    print("  STABLE DIFFUSION 3.5 COMPARISON - BATCH EXECUTION")
    print("="*60 + "\n")
    print(f"Prompt: {prompt}")
    print("-" * 60)

    try:
        # Initialize Runner
        runner = SDRunner(auth_token=HF_TOKEN)
        
        # Parse tasks
        tasks = json.loads(tasks_json)
        
        for i, task in enumerate(tasks):
            name = task['name']
            model_id = task['id']
            path = task['path']
            steps = task.get('steps', 28)
            guidance = task.get('guidance', 7.0)
            
            print(f"\n[{i+1}/{len(tasks)}] Generating with {name}...")
            print(f"Model: {model_id}")
            print(f"Settings: Steps={steps}, Guidance={guidance}")
            
            try:
                # Optimized for memory: The runner handles unloading/loading
                image, duration, _ = runner.generate(
                    prompt, 
                    model_id, 
                    steps=steps, 
                    guidance_scale=guidance,
                    output_path=path
                )
                print(f"✅ Success! Saved to: {path}")
                print(f"⏱️ Duration: {duration:.2f}s")
                
                # Write metadata sidecar within the same directory, same basename
                # e.g. image.png -> image.json
                meta_path = os.path.splitext(path)[0] + ".json"
                
                metadata = {
                    "model": name,
                     # We can also save the model_id if we want
                    "duration": duration,
                    "steps": steps,
                    "guidance": guidance,
                    "prompt": prompt
                }
                with open(meta_path, "w") as f:
                    json.dump(metadata, f, indent=2)

                
            except Exception as e:
                print(f"❌ Error generating {name}: {e}")
                import traceback
                traceback.print_exc()
                
            print("-" * 60)
            
        print("\n✨ All tasks completed.")
        
    except Exception as ie:
         print(f"CRITICAL ERROR IN BATCH RUNNER: {ie}")
         import traceback
         traceback.print_exc()

    # Keep terminal open
    print("\n" + "="*60)
    input("Press Enter to close this window...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--tasks", type=str, required=True, help="JSON string of tasks config")
    
    args = parser.parse_args()
    
    run_batch(
        args.prompt,
        args.tasks
    )
