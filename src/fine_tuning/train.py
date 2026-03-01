"""
Unified LoRA fine-tuning script for SD3.5 Large (Transformer only).

Loads precomputed latents + prompt embeddings and trains LoRA adapters via
flow matching. Supports both consumer (RTX 4090) and cluster (A100) hardware
through configuration — zero code branching.

Usage:
    accelerate launch --config_file configs/accelerate_consumer.yaml \\
        train.py --config configs/consumer.yaml

    accelerate launch --config_file configs/accelerate_cluster.yaml \\
        train.py --config configs/cluster.yaml
"""

import os
import argparse

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset, DataLoader
from diffusers import SD3Transformer2DModel
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════

class ProcessedDataset(Dataset):
    """Loads precomputed .pt files (latents + prompt embeddings)."""

    def __init__(self, data_dir: str, max_samples: int | None = None):
        self.files = sorted(
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".pt")
        )
        if not self.files:
            raise FileNotFoundError(f"No .pt files found in {data_dir}")
        if max_samples is not None:
            self.files = self.files[:max_samples]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        return torch.load(self.files[idx], weights_only=False)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_torch_dtype(precision: str) -> torch.dtype:
    """Map precision string to torch dtype."""
    return {"fp16": torch.float16, "bf16": torch.bfloat16}[precision]


def build_optimizer(params, cfg: dict) -> torch.optim.Optimizer:
    """Build optimizer from config. Tries 8-bit AdamW when requested."""
    lr = cfg["learning_rate"]
    opt_type = cfg.get("optimizer", "adamw")

    if opt_type == "adamw_8bit":
        try:
            import bitsandbytes as bnb
            print("Using 8-bit AdamW (bitsandbytes)")
            return bnb.optim.AdamW8bit(params, lr=lr)
        except ImportError:
            print("bitsandbytes not available — falling back to standard AdamW")

    print("Using standard AdamW")
    return torch.optim.AdamW(params, lr=lr)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    # ── CLI ──────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Train SD3.5 LoRA on precomputed latents.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    cli = parser.parse_args()

    cfg = load_config(cli.config)

    # ── Reproducibility ─────────────────────────────────────────────────
    seed = cfg.get("seed", 42)
    set_seed(seed)

    # ── Accelerator ─────────────────────────────────────────────────────
    weight_dtype = get_torch_dtype(cfg["precision"])

    report_to = cfg.get("report_to", "none")
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        mixed_precision=cfg["precision"],
        log_with=report_to if report_to != "none" else None,
    )

    if accelerator.is_main_process:
        os.makedirs(cfg["output_dir"], exist_ok=True)
        if report_to != "none":
            accelerator.init_trackers(
                cfg.get("tracker_project", "celeba_sd35_finetune"),
                config=cfg,
            )

    # ── Model (Transformer only) ────────────────────────────────────────
    print("Loading SD3 Transformer...")
    transformer = SD3Transformer2DModel.from_pretrained(
        cfg["model_id"],
        subfolder="transformer",
        torch_dtype=weight_dtype,
    )
    transformer.requires_grad_(False)

    # ── LoRA ────────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=cfg["rank"],
        lora_alpha=cfg["lora_alpha"],
        init_lora_weights="gaussian",
        target_modules=cfg["target_modules"],
    )
    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()
    transformer.enable_gradient_checkpointing()

    # ── Optimizer ───────────────────────────────────────────────────────
    optimizer = build_optimizer(transformer.parameters(), cfg)

    # ── Dataset ─────────────────────────────────────────────────────────
    dataset = ProcessedDataset(cfg["processed_dir"], max_samples=cfg.get("max_samples"))
    dataloader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
    )

    # ── Prepare ─────────────────────────────────────────────────────────
    transformer, optimizer, dataloader = accelerator.prepare(
        transformer, optimizer, dataloader
    )

    # ── Training Loop ───────────────────────────────────────────────────
    checkpoint_every = cfg.get("checkpoint_every", 500)
    global_step = 0

    for epoch in range(cfg["num_epochs"]):
        transformer.train()
        progress_bar = tqdm(
            total=len(dataloader),
            disable=not accelerator.is_local_main_process,
            desc=f"Epoch {epoch}",
        )

        for batch in dataloader:
            with accelerator.accumulate(transformer):
                # Unpack precomputed tensors
                latents = batch["latents"].squeeze(1).to(dtype=weight_dtype)
                prompt_embeds = batch["prompt_embeds"].squeeze(1).to(dtype=weight_dtype)
                pooled_prompt_embeds = batch["pooled_prompt_embeds"].squeeze(1).to(dtype=weight_dtype)

                # ── Flow Matching ────────────────────────────────────
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.rand((bsz,), device=latents.device)

                # Interpolate: z_t = (1-t)*x_0 + t*noise
                t = timesteps.view(-1, 1, 1, 1)
                z_t = (1 - t) * latents + t * noise

                # Target velocity: v = noise - x_0
                target = noise - latents

                # ── Forward ──────────────────────────────────────────
                model_pred = transformer(
                    hidden_states=z_t,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False,
                )[0]

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # ── Backward ─────────────────────────────────────────
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            # ── Logging & Checkpointing ──────────────────────────────
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % checkpoint_every == 0 and accelerator.is_main_process:
                    save_path = os.path.join(cfg["output_dir"], f"checkpoint-{global_step}")
                    accelerator.unwrap_model(transformer).save_pretrained(save_path)
                    print(f"Saved checkpoint to {save_path}")

            logs = {"loss": loss.detach().item()}
            if report_to != "none":
                accelerator.log(logs, step=global_step)
            progress_bar.set_postfix(**logs)

        progress_bar.close()

    # ── Save Final ──────────────────────────────────────────────────────
    if accelerator.is_main_process:
        print("Saving final LoRA weights...")
        accelerator.unwrap_model(transformer).save_pretrained(cfg["output_dir"])
        if report_to != "none":
            accelerator.end_training()

    print("Training complete.")


if __name__ == "__main__":
    main()
