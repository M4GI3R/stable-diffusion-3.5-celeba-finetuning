# Analysis of Empty Prompts & Facial Hair Extremes

## 1. Empty Prompts (Gender Only)
*   **Count**: 31 images (0.1%).
*   **Cause**: Exclusive bucket conflicts (e.g., Beard + Mustache) combined with no other valid attributes.
*   **Action**: Accepted as regularization.

## 2. Facial Hair Extremes
We analyzed `Facial_Hair` attributes: `5_o_Clock_Shadow`, `Goatee`, `Sideburns`, `Mustache`.

*   **Total Images with Facial Hair**: 6,511 (out of 30k).
*   **Distribution**:
    *   **1 attribute**: 3,642
    *   **2 attributes**: 1,557
    *   **3 attributes**: 1,058
    *   **4 attributes (All)**: 254

### Impact on Prompts
*   **Previous**: Only 1 attribute was selected (e.g., "with a mustache"), ignoring the goatee/sideburns.
*   **New Strategy**: If `Facial_Hair` is selected, **all** true attributes from that bucket are included.
*   **Example (4 attrs)**: "with 5 o'clock shadow, a goatee, sideburns, and a mustache."
*   **Benefit**: Higher prompt fidelity for complex facial hair styles.
