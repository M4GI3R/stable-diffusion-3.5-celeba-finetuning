"""
Preprocess CelebA-HQ images and prompts into precomputed latents and embeddings.

Encodes images → VAE latents and prompts → text embeddings using the full
StableDiffusion3Pipeline, then saves .pt files for lightweight training.

Usage:
    python preprocess.py --config configs/consumer.yaml
    python preprocess.py --config configs/consumer.yaml --overwrite
"""

import os
import argparse

import torch
import yaml
from tqdm import tqdm
from diffusers import StableDiffusion3Pipeline

# Resolve parent package for CelebADataset
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.fine_tuning.dataset import CelebADataset


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_torch_dtype(precision: str) -> torch.dtype:
    """Map precision string to torch dtype."""
    return {"fp16": torch.float16, "bf16": torch.bfloat16}[precision]


def main():
    # ── CLI ──────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Precompute latents & text embeddings for SD3.5 LoRA training.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--overwrite", action="store_true", default=None, help="Re-encode files that already exist")
    cli = parser.parse_args()

    cfg = load_config(cli.config)

    # CLI overrides take precedence
    overwrite = cli.overwrite if cli.overwrite is not None else cfg.get("overwrite_preprocess", False)

    # ── Reproducibility ─────────────────────────────────────────────────
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)

    # ── Device & Precision ──────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = get_torch_dtype(cfg["precision"])
    print(f"Device: {device} | Precision: {cfg['precision']} | Seed: {seed}")

    # ── Load Pipeline (encoding only) ───────────────────────────────────
    model_id = cfg["model_id"]
    print(f"Loading SD3 Pipeline: {model_id}")
    pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    # ── Dataset ─────────────────────────────────────────────────────────
    dataset = CelebADataset(cfg["csv_path"], cfg["img_dir"], size=cfg["resolution"])
    output_dir = cfg["processed_dir"]
    os.makedirs(output_dir, exist_ok=True)

    max_samples = cfg.get("max_samples")
    total = min(len(dataset), max_samples) if max_samples else len(dataset)
    print(f"Processing {total}/{len(dataset)} items → {output_dir}")

    # ── Encode ──────────────────────────────────────────────────────────
    with torch.no_grad():
        for idx in tqdm(range(total), desc="Encoding"):
            item = dataset[idx]
            filename = item["filename"]
            stem = os.path.splitext(filename)[0]
            save_path = os.path.join(output_dir, f"{stem}.pt")

            # Skip existing unless overwrite
            if os.path.exists(save_path) and not overwrite:
                continue

            pixel_values = item["pixel_values"].unsqueeze(0).to(device, dtype=dtype)
            prompt = item["prompt"]

            # A. Image → Latents
            latents = pipe.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * pipe.vae.config.scaling_factor

            # B. Prompt → Embeddings
            prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(
                prompt=prompt,
                prompt_2=prompt,
                prompt_3=prompt,
                device=device,
                do_classifier_free_guidance=False,
            )

            # C. Save
            torch.save(
                {
                    "latents": latents.cpu(),
                    "prompt_embeds": prompt_embeds.cpu(),
                    "pooled_prompt_embeds": pooled_prompt_embeds.cpu(),
                    "filename": filename,
                },
                save_path,
            )

    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
