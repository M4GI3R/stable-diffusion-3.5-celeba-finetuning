# CelebA Prompt Generator - Unified Bucket Structure

## Dataset

|                              |            |
| ---------------------------------- | --------------- |
| **Total Images**                   | 202,599         |
| **Total Images (Blurry Excluded)** | 192,287         |
| **Male**                           | 79,601 (41.4%)  |
| **Female**                         | 112,686 (58.6%) |

---

# PROMT GENERATOR WORKFLOW

## Step 1: Choose Gender
**Probability**: 50% Male | 50% Female

## Step 2-4: Pick 3 Attributes

For each slot (Attribute #1, #2, #3):
1. Select a **bucket** using normalized bucket probabilities
2. Select an **attribute** from that bucket using normalized attribute weights
3. **Exclusion rule**: If an attribute from a bucket is already selected, other attributes from the SAME bucket cannot be selected

---

# BUCKET SELECTION PROBABILITY SUMMARY

**Male** (sum = 100%):

| Bucket | Probability |
| :--- | :--- |
| Nose | 22.6% |
| Hair Color | 21.1% |
| Face Shape | 19.7% |
| Hair Structure | 18.7% |
| Facial Hair | 15.9% |
| Other | 11.9% |
| Eyebrows | 11.8% |
| Makeup & Style | 0% |

**Female** (sum = 100%):

| Bucket | Probability |
| :--- | :--- |
| Makeup & Style | 22.9% |
| Face Shape | 18.9% |
| Hair Structure | 17.6% |
| Hair Color | 17.6% |
| Other | 16.5% |
| Eyebrows | 12.8% |
| Nose | 12.1% |
| Facial Hair | 0% |

---

# ALL BUCKETS

## 1. Facial Hair (Male Exclusive)

| Attribute | Male % | Female % |
| :--- |:-------|:---------|
| 5_o_Clock_Shadow | 40.9%  | —        |
| Goatee | 23.2%  | —        |
| Sideburns | 20.8%  | —        |
| Mustache | 15.1%  | —        |

---

## 2. Makeup & Style (Female Exclusive)

| Attribute | Male % | Female % |
| :--- |:-------|:---------|
| Wearing_Lipstick | —      | 40.7%    |
| Heavy_Makeup | —      | 33.8%    |
| Wearing_Earrings | —      | 15.7%    |
| Wearing_Necklace | —      | 9.8%     |

---

## 3. Hair Structure

| Attribute | Male % | Female % |
| :--- |:-------|:---------|
| Straight_Hair | 38.8%  | 20.8%    |
| Wavy_Hair | 21.7%  | 50.5%    |
| Receding_Hairline | 18.3%  | 5.9%     |
| Bangs | 12.6%  | 22.7%    |
| Bald | 8.5%   | —        |

---

## 4. Hair Color

| Attribute | Male % | Female % |
| :--- |:-------|:---------|
| Black_Hair | 53.8%  | 28.7%    |
| Brown_Hair | 27.4%  | 35.5%    |
| Gray_Hair | 15.3%  | 1.4%     |
| Blond_Hair | 3.5%   | 34.4%    |

---

## 5. Face Shape

| Attribute | Male % | Female % |
| :--- |:-------|:---------|
| High_Cheekbones | 41.2%  | 61.2%    |
| Oval_Face | 29.6%  | 36.5%    |
| Chubby | 16.1%  | 1.3%     |
| Double_Chin | 13.1%  | 1.0%     |

---

---

## 6. Eyebrows

| Attribute | Male % | Female % |
| :--- |:-------|:---------|
| Bushy_Eyebrows | 81.9%  | 14.3%    |
| Arched_Eyebrows | 18.1%  | 85.7%    |

---

## 7. Nose

| Attribute | Male % | Female % |
| :--- |:-------|:---------|
| Big_Nose | 72.1%  | 22.1%    |
| Pointy_Nose | 27.9%  | 77.9%    |

---

## 8. Other

| Attribute | Male % | Female % |
| :--- |:-------|:---------|
| Young | 24.5%  | 25.3%    |
| Smiling | 15.7%  | 15.6%    |
| Mouth_Slightly_Open | 16.4%  | 15.1%    |
| Attractive | 11.3%  | 20.3%    |
| Bags_Under_Eyes | 13.7%  | 3.0%     |
| Big_Lips | 6.1%   | 8.8%     |
| Narrow_Eyes | 4.4%   | 3.1%     |
| Rosy_Cheeks | 0.1%   | 3.3%     |
| Pale_Skin | 1.0%   | 1.6%     |
| Eyeglasses | 4.8%   | 0.6%     |
| Wearing_Hat | 3.1%   | 0.7%     |
| Wearing_Necktie | 6.8%   | —        |

Note: This is the only **non-exclusive** bucket, meaning in a prompt multiple attributes from this bucket could be sampled

---

# OUTPUT TEMPLATE

```
A realistic portrait of a {gender}, with {attr_1}, {attr_2}, and {attr_3}.
```

---

# DASHBOARD BEHAVIOR

## Manual Builder Mode
- User can freely select any attribute from any bucket for each slot
- No duplicate checking required (user responsibility)
- Allows experimentation with unusual combinations

## Auto-Generate Mode
- Randomly selects 3 attributes following bucket probabilities
- **Enforces exclusion**: Once an attribute is selected, that bucket is disabled for remaining slots
- Prevents illogical combinations (e.g., "black hair and blond hair")

---

# WHY BUCKETS?

Buckets enforce **mutual exclusion** for logically conflicting attributes:
- Hair Color: Only one can be true (black XOR blond XOR brown XOR gray)
- Facial Hair: Typically one style per face
- Nose: Can't be both big AND pointy
- Eyebrows: Arched vs Bushy are opposites
