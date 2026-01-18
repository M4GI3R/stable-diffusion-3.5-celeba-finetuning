import os
import sys
from dotenv import load_dotenv
load_dotenv()

OFFLINE_MODE = os.getenv("OFFLINE", "False").lower() in ("true", "1", "yes")

# --- CONFIGURE CACHE PATH (MUST BE FIRST) ---
# Get the absolute path to the current file (src/comparison/sd35_runner.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate up two levels to get to the project root
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
cache_dir = os.path.join(project_root, ".cache")

# Set the environment variable BEFORE importing diffusers/huggingface
os.environ["HF_HOME"] = cache_dir

if OFFLINE_MODE:
    os.environ["HF_HUB_OFFLINE"] = "1"
    print(f"Hugging Face Cache: {cache_dir}")
    print("🌍 Mode: OFFLINE (Network disabled, using local cache only)")
else:
    # Ensure offline mode is NOT forced if we want to be online
    os.environ.pop("HF_HUB_OFFLINE", None)
    print(f"Hugging Face Cache: {cache_dir}")
    print("🌍 Mode: ONLINE (Will check Hugging Face for updates)")

import torch
import time
import gc
from diffusers import StableDiffusion3Pipeline

class SDRunner:
    def __init__(self, output_dir="out/comparison", auth_token=None):
        self.output_dir = output_dir
        self.auth_token = auth_token
        os.makedirs(self.output_dir, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.current_model_id = None
        self.pipeline = None

        if not OFFLINE_MODE and self.auth_token:
            try:
                from huggingface_hub import login
                login(token=self.auth_token)
                print("✅ Logged in to Hugging Face")
            except Exception as e:
                print(f"⚠️ Login failed: {e}")
                print("Continuing, but model downloads might fail if repositories are gated.")

    def load_model(self, model_id):
        if self.current_model_id == model_id:
            return

        # Unload previous model
        if self.pipeline:
            del self.pipeline
            gc.collect()
            torch.cuda.empty_cache()
        
        print(f"Loading model: {model_id}")
        try:
            t0 = time.time()
            
            # Suppress tokenizers warning
            import logging
            logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
            
            self.pipeline = StableDiffusion3Pipeline.from_pretrained(
                model_id, 
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                local_files_only = OFFLINE_MODE
            )
            t1 = time.time()
            print(f"Model loaded from disk in {t1-t0:.2f}s")
            
            if self.device == "cuda":
                # Memory optimization: Offload model components to CPU when not in use
                # This is crucial for running SD 3.5 Large on consumer GPUs
                self.pipeline.enable_model_cpu_offload()
                print("Enabled CPU Offload.")
            else:
                self.pipeline.to(self.device)
                
            self.current_model_id = model_id
        except OSError:
            if OFFLINE_MODE:
                print(f"\n❌ ERROR: Model '{model_id}' not found in cache.", file=sys.stderr)
                print(f"   Path checked: {cache_dir}", file=sys.stderr)
                print("   Set OFFLINE=False in .env to download it.\n", file=sys.stderr)
            raise
        except Exception as e:
            print(f"Error loading model {model_id}: {e}")
            raise e

    def generate(self, prompt, model_id, steps=28, guidance_scale=7.0, output_path=None):
        self.load_model(model_id)
        
        start_time = time.time()
        image = self.pipeline(
            prompt, 
            num_inference_steps=steps, 
            guidance_scale=guidance_scale
        ).images[0]
        end_time = time.time()
        
        generation_time = end_time - start_time
        
        # Save image
        if output_path:
            filepath = output_path
        else:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            model_name = model_id.split("/")[-1]
            filename = f"{timestamp}_{model_name}.png"
            filepath = os.path.join(self.output_dir, filename)
            
        image.save(filepath)
        
        return image, generation_time, filepath
