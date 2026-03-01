# Fine-Tuning Strategy & Recommendations
**Dataset**: CelebA-HQ (30k images)
**Model**: Stable Diffusion 3.5 Large (SD3.5L)

## 1. Prompt Strategy Evaluation
### Current Approach: "3 Attributes + Gender"
**Verdict**: **Excellent starting point.**
*   **Why it works**: 
    *   **Focus**: Limiting to 3 attributes prevents "concept bleeding" where the model confuses which part of the prompt corresponds to which feature.
    *   **Combinatorics**: 30k images with randomized 3-attribute combinations covers a vast combinatorial space, ensuring the model learns features in isolation and in context.
    *   **Natural Language**: The shift to natural phrasing ("who is wearing...") helps SD3.5's T5 text encoder understand the *relationship* between the person and the attribute, rather than just a bag of tags.

### Considerations for Iteration
*   **Token Length**: SD3.5 supports up to 512 tokens (via T5). Your current prompts are short. This is good for *feature learning*, but for *aesthetic* fine-tuning, you might later want to enrich them with style descriptors if the generated images look too "plain" compared to HQ photos.
*   **"Empty" Prompts**: We have a few prompts like "A photo-realistic portrait of a man." (when no valid buckets are selected). This is actually **beneficial**. It acts as a form of regularization, teaching the model to generate a basic "man" without forcing specific features.

## 2. Fine-Tuning Recommendations

### 2.1. Training Parameters
*   **Resolution**: **1024x1024**. CelebA-HQ is high res. Do not downscale below 1024 if you can avoid it. SD3.5 is native at 1024.
*   **Batch Size**: As high as your VRAM allows (usually 1-4 for consumer cards, higher for A100s). Gradient accumulation can simulate larger batches.
*   **Learning Rate**: 
    *   Start **low** (e.g., `1e-5` or `5e-6`) for full fine-tuning.
    *   If using **LoRA** (recommended for efficiency), use standard LoRA rates (e.g., `1e-4` for UNet, `5e-5` for Text Encoder).

### 2.2. Caption Dropout
**CRITICAL**: Implement **Caption Dropout** (e.g., 10%).
*   **What**: For 10% of training steps, pass an empty string "" instead of the prompt.
*   **Why**: This teaches the model to generate high-quality faces *even without* specific prompts. It forces the model to internalize the "CelebA-HQ Style" (high quality, centered alignment) into its unconditional distribution.

### 2.3. Text Encoder Training
*   **SD3.5 has two/three text encoders** (CLIP G, CLIP L, T5).
*   **Recommendation**: 
    *   **Fine-tune T5?** Optional. It's huge. Often keeping it frozen and training the UNet (MMDiT) is sufficient and saves VRAM.
    *   **CLIP**: Usually kept frozen or trained with very low LR. Capturing "Red Lipstick" usually doesn't require retraining CLIP, just mapping the existing concept to the new visual data in the UNet.

### 2.4. Validation Strategy
*   **Fixed Seeds**: During training, generate images for a fixed set of prompts (e.g., one for each major attribute) every 500 steps.
*   **Overfitting Check**: Watch if the background starts becoming identical to training data or if artifacts appear.

## 3. Potential Pitfalls
*   **"Plasticy" Skin**: CelebA-HQ is heavily processed/cleaned. The model might learn to generate overly smooth, airbrushed skin.
    *   *Mitigation*: If this happens, mix in a few hundred images of "gritter" real faces with a regularization class, or simply accept it as the style of this dataset.
*   **Attribute Leakage**: If "Blond Hair" co-occurs 100% of the time with "Female" in the training set (it doesn't, but hypothetically), the model might lose the ability to generate "Blond Man".
    *   *Mitigation*: Your randomization strategy helps here, but ensure you test "counter-stereotypical" prompts (e.g., "Man with heavy makeup") during validation to ensure disentanglement.

## 4. Summary
You are in a **very strong position**. The dataset is clean, the prompts are natural but structured, and the distribution is balanced. 
*   **Next Step**: Start training.
*   **First Run**: Use LoRA (Rank 32 or 64). It's faster, safer, and easier to iterate than full fine-tuning.
