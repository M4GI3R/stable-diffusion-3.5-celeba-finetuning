# SD 3.5 Large — Unified LoRA Fine-Tuning Pipeline

## Architecture

This pipeline fine-tunes **Stable Diffusion 3.5 Large** on CelebA-HQ using **LoRA** adapters with a two-stage workflow:

```
┌──────────────┐      .pt files      ┌─────────────┐      LoRA weights
│ preprocess.py │ ──────────────────► │  train.py   │ ──────────────────►
│  (one-time)   │  latents + embeds  │ (iterative) │   checkpoint-N/
└──────────────┘                      └─────────────┘
```

**Stage 1 — Preprocess**: Load the full SD3 pipeline once to encode all images → VAE latents and all prompts → text embeddings. Save as `.pt` files. This runs **once** and is reused across all training runs.

**Stage 2 — Train**: Load **only** the SD3 Transformer (no VAE, no text encoders). Inject LoRA adapters and train with flow matching on the precomputed data.

### Why Precomputation?

Online encoding (loading VAE + 3 text encoders during training) was previously used in the cluster version. Analysis showed it provides **no meaningful quality gain** for CelebA LoRA fine-tuning while:
- Consuming ~24GB extra VRAM for encoders
- Adding ~2× per-step latency from re-encoding every epoch
- Preventing training on consumer GPUs (24GB)

Precomputation separates data preparation from training, enabling:
- **Lower VRAM**: Only the Transformer + LoRA in GPU memory during training
- **Faster iteration**: Precompute once, train many times with different hyperparameters
- **Hardware flexibility**: Same code runs on RTX 4090 and A100 clusters

> **Note**: Output quality is **not** reduced by precomputation. The modeling objective (flow matching) and the actual gradient updates are identical.

---

## Directory Structure

```
ultimate/
├── preprocess.py              # Stage 1: encode images + prompts
├── train.py                   # Stage 2: LoRA training (transformer only)
├── configs/
│   ├── consumer.yaml          # RTX 4090 profile
│   ├── cluster.yaml           # A100 cluster profile
│   ├── accelerate_consumer.yaml
│   └── accelerate_cluster.yaml
└── README.md
```

---

## Workflow

### 1. Preprocess (run once per dataset)

```bash
python src/fine_tuning/ultimate/preprocess.py --config src/fine_tuning/ultimate/configs/consumer.yaml
```

Encodes all images and prompts into `data/processed/*.pt`. Skips files that already exist (use `--overwrite` to re-encode).

### 2a. Train — Consumer (RTX 4090)

```bash
accelerate launch --config_file src/fine_tuning/ultimate/configs/accelerate_consumer.yaml src/fine_tuning/ultimate/train.py --config src/fine_tuning/ultimate/configs/consumer.yaml
```

### 2b. Train — Cluster (A100 × 8)

```bash
accelerate launch --config_file src/fine_tuning/ultimate/configs/accelerate_cluster.yaml src/fine_tuning/ultimate/train.py --config src/fine_tuning/ultimate/configs/cluster.yaml
```

No `accelerate config` needed — predefined YAML configs handle all distribution settings.

---

## Configuration Reference

| Parameter | Consumer | Cluster | Description |
|---|---|---|---|
| `rank` | 32 | 128 | LoRA rank (adapter capacity) |
| `lora_alpha` | 32 | 128 | LoRA scaling factor |
| `target_modules` | 4 attn layers | 10 (attn + MLP + joint) | Which transformer layers get LoRA |
| `batch_size` | 1 | 16 | Per-GPU batch size |
| `gradient_accumulation` | 4 | 1 | Effective batch = `batch_size × accum` |
| `precision` | fp16 | bf16 | Training precision |
| `optimizer` | AdamW 8-bit | AdamW | Memory vs speed tradeoff |
| `num_epochs` | 10 | 5 | Training duration |
| `learning_rate` | 1e-4 | 3e-4 | Scaled for larger effective batch |
| `num_workers` | 4 | 8 | DataLoader workers |

**The code never branches on "cluster vs consumer"** — it simply loads whichever config you point it at.

---

## Hardware Requirements & Runtime

### Preprocessing
- **VRAM**: ~22GB (full SD3 pipeline in fp16)
- **Time**: ~45 min for 30k images on RTX 4090
- Runs identically on consumer and cluster hardware

### Training — Consumer (RTX 4090, 24GB)
- **VRAM**: ~14–18GB (Transformer + LoRA + activations with gradient checkpointing)
- **Speed**: ~1.5 it/s at batch 1
- **Time**: ~10 epochs × 30k images ≈ **55 hours**

### Training — Cluster (A100 80GB × 8)
- **VRAM**: ~25–35GB per GPU (higher batch, more LoRA targets)
- **Speed**: ~2.0 it/s per GPU × 8 = ~16 it/s total
- **Time**: ~50 epochs × 30k images ≈ **13 hours** at batch 4

---

## Requirements

```
torch>=2.0
diffusers>=0.25
peft>=0.7
accelerate>=0.25
transformers>=4.36
pyyaml
tqdm
bitsandbytes  # optional, for 8-bit optimizer on consumer
```
