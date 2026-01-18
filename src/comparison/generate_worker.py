import argparse
import sys
import os
import time
import torch
import gc

# --- CONFIGURATION FROM ENV ---
# Ideally getting this from the same constants or env
from dotenv import load_dotenv
load_dotenv()

# Setup Cache (Important for isolating the worker)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
cache_dir = os.path.join(project_root, ".cache")
os.environ["HF_HOME"] = cache_dir

from diffusers import StableDiffusion3Pipeline

def generate(prompt, model_id, steps, guidance_scale, output_path):
    print(f"Worker: Initializing for {model_id}...")
    print(f"Worker: Cache dir: {cache_dir}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    try:
        print("Worker: Loading pipeline...")
        # Suppress tokenizers warning
        import logging
        logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
        
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            dtype=dtype,
            token=os.getenv("HF_TOKEN")
        )
        
        # Memory optimization
        pipe.enable_model_cpu_offload() 
        # pipe.enable_xformers_memory_efficient_attention() # Optional optimization
        
        print("Worker: Generating image...")
        start_time = time.time()
        image = pipe(
            prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale
        ).images[0]
        end_time = time.time()
        
        print(f"Worker: Saving to {output_path}...")
        image.save(output_path)
        
        duration = end_time - start_time
        print(f"Worker: Done. Duration: {duration:.2f}s")
        
        # Explicit cleanup
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nCRITICAL WORKER ERROR: {e}", file=sys.stderr)
        print("\n" + "!"*50)
        input("Press Enter to close this window...")
        sys.exit(1)

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--prompt", type=str, required=True)
        parser.add_argument("--model_id", type=str, required=True)
        parser.add_argument("--steps", type=int, default=28)
        parser.add_argument("--guidance_scale", type=float, default=7.0)
        parser.add_argument("--output_path", type=str, required=True)
        
        args = parser.parse_args()
        
        print("\n" + "="*50)
        print(f"Worker started for model: {args.model_id}")
        print("="*50 + "\n")
        
        generate(
            args.prompt,
            args.model_id,
            args.steps,
            args.guidance_scale,
            args.output_path
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        input("\nScript crashed. Press Enter to close...")
        sys.exit(1)
