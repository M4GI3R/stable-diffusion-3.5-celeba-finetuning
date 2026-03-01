import os
import time
import torch
from diffusers import StableDiffusion3Pipeline
from pathlib import Path
from datetime import datetime

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

BASE_MODEL_ID = "stabilityai/stable-diffusion-3.5-large"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LORA_PATH = PROJECT_ROOT / "output" / "ultimate_consumer"

PROMPT = "A photo-realistic portrait of a man with 5 o'clock shadow, sideburns, a mustache, a big nose, and brown hair."

NUM_IMAGES = 2
STEPS = 28
GUIDANCE = 7.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Output directory
timestamp = time.strftime("%Y%m%d-%H%M%S")
BASE_DIR = Path(__file__).parent  # fine_tuning_test/
OUT_DIR = BASE_DIR / "out"

run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
run_dir = OUT_DIR / run_id

base_dir = run_dir / "base"
lora_dir = run_dir / "lora"

base_dir.mkdir(parents=True, exist_ok=True)
lora_dir.mkdir(parents=True, exist_ok=True)

print(f"Saving results to: {OUT_DIR}")
print("Loading pipeline...")

# --------------------------------------------------
# LOAD PIPELINE
# --------------------------------------------------

pipe = StableDiffusion3Pipeline.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=DTYPE
)

if DEVICE == "cuda":
    pipe.enable_model_cpu_offload()
else:
    pipe.to(DEVICE)

print("Pipeline loaded.\n")

# --------------------------------------------------
# GENERATE BASE IMAGES
# --------------------------------------------------

print("Generating BASE images...")

for i in range(NUM_IMAGES):
    seed = 42 + i
    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    start = time.time()
    image = pipe(
        PROMPT,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        generator=generator
    ).images[0]
    duration = time.time() - start

    filename = os.path.join(base_dir, f"base_{i+1}_seed_{seed}.png")
    image.save(base_dir / f"base_{i}_seed_{seed}.png")

    print(f"[BASE {i+1}] Saved ({duration:.2f}s)")

# --------------------------------------------------
# LOAD LORA
# --------------------------------------------------

print("\nLoading LoRA...")
pipe.load_lora_weights(
    str(LORA_PATH),
    weight_name="adapter_model.safetensors",
    local_files_only=True
)
print("LoRA loaded.\n")

# --------------------------------------------------
# GENERATE LORA IMAGES
# --------------------------------------------------

print("Generating LoRA images...")

for i in range(NUM_IMAGES):
    seed = 42 + i
    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    start = time.time()
    image = pipe(
        PROMPT,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        generator=generator
    ).images[0]
    duration = time.time() - start

    filename = os.path.join(lora_dir, f"lora_{i+1}_seed_{seed}.png")
    image.save(lora_dir / f"lora_{i}_seed_{seed}.png")

    print(f"[LORA {i+1}] Saved ({duration:.2f}s)")

pipe.unload_lora_weights()

print("\nDone.")