# Prompt Redundancy & Accumulation Analysis

## 1. Objective
Reduce redundancy in prompts by merging attributes that describe the same feature (e.g., "wavy hair" + "blond hair" -> "wavy, blond hair").

## 2. Redundancy Candidates

### 2.1. Hair (The Primary Candidate)
Images can have attributes from both `Hair_Color` and `Hair_Structure` buckets.
*   **Colors**: Black, Brown, Blond, Gray.
*   **Structures**: Straight, Wavy.
*   **Other Hair**: Receding_Hairline, Bangs, Bald.
    *   *Bald* is treated as an ADJ ("bald man"), so no conflict.
    *   *Bangs* is a noun ("with bangs"). "with bangs and blond hair" flows well. Merging "bangs, blond hair" is awkward.
    *   *Receding Hairline* is a noun phrase. "with a receding hairline and gray hair" flows well.

**Strategy**: Merge **Structure (Adjective) + Color (Adjective)**.
*   *Target*: `Hair`
*   *Pattern*: "with {structure}, {color} hair".

### 2.2. Face / Skin
*   `Pale_Skin` ("with pale skin")
*   `Rosy_Cheeks` ("with rosy cheeks")
*   `High_Cheekbones`
*   `Oval_Face`
*   `Double_Chin`
These are distinct features on the face.
*   "with pale skin and rosy cheeks" -> Good.
*   "with pale, rosy cheeks skin" -> Bad.
*   **Strategy**: Keep separate.

### 2.3. Eyes
*   `Narrow_Eyes`
*   `Bags_Under_Eyes`
*   `Eyeglasses` (Wearing)
*   **Strategy**: Keep separate. "with narrow eyes and bags under eyes".

### 2.4. Nose
*   `Big_Nose`
*   `Pointy_Nose`
*   **Constraint**: `Nose` is an **Exclusive Bucket**. An image has max 1.
*   **Result**: No merging needed.

### 2.5. Eyebrows
*   `Bushy`
*   `Arched`
*   **Constraint**: `Eyebrows` is an **Exclusive Bucket**.
*   **Result**: No merging needed.

## 3. Implementation Plan

### 3.1. New Attribute Mapping
Refactor `ATTR_MAP` to include a `target` field for hair attributes.

```python
"Black_Hair": {"type": "FEATURE", "phrase": "black", "target": "hair"}
"Wavy_Hair":  {"type": "FEATURE", "phrase": "wavy", "target": "hair"}
"Bangs":      {"type": "FEATURE", "phrase": "with bangs"} # No target
```

### 3.2. Accumulation Logic
During prompt construction:
1.  Collect all `FEATURE` items.
2.  Check for common `target`.
3.  If multiple have `target="hair"`, collect their phrases.
4.  Format: `f"with {', '.join(phrases)} hair"`.
5.  Treat this combined string as a single feature item in the final list.

### 3.3. Ordering
Adjectives should follow English order (Opinion-Size-Age-Shape-Color-Origin-Material-Purpose). 
*   Structure (Shape) comes before Color.
*   "Wavy (Shape) Blond (Color) Hair" -> Correct.
*   "Blond Wavy Hair" -> Less natural.
*   **fix**: Ensure sort order or specific checking.

## 4. Expected Improvements
*   Old: "... with wavy hair and blond hair."
*   New: "... with wavy, blond hair."

## 5. Implementation Results
The logic successfully merges attributes targeting `Hair`:
- **Example 1**: `"A photo-realistic portrait of a smiling woman with wavy, blond hair."` (Merged `Wavy_Hair` and `Blond_Hair`).
- **Example 2**: `"A photo-realistic portrait of a young woman with straight, brown hair."` (Merged `Straight_Hair` and `Brown_Hair`).
- **Ordering**: Adjectives are sorted (Structure first, Color second) via the `order` field in `ATTR_MAP`.
- **Single Trait**: `"with wavy hair"` remains unchanged if no color is selected.
