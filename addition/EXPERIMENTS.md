# Addition Project - Experiment Logs

This document records the detailed history, settings, and observations of experiments conducted within the Addition project.

## Experiment 1: Basic Addition (Single Digit)
*   **Goal:** Verify if nanoGPT can learn single-digit addition (0+0 to 9+9) via rote memorization.
*   **Dataset:** All combinations of 1-digit addition (100 samples total).
*   **Initial Settings:** `max_iters = 2000`, `lr_decay_iters = 2000`, `batch_size = 12`.
*   **Initial Observation (2000 iters):** 
    *   Train loss 0.0672, val loss 0.0648.
    *   **Success Cases:** `1+1=2`, `5+3=8`, `2+2=4` (100% correct).
    *   **Failure Case:** `3+5=` outputs varied (`7`, `6`, `8`) unlike other stable cases.
    *   *Hypothesis:* Under-fitting or unlucky initialization for specific samples.

### Follow-up (5000 iters)
*   **Settings:** Increased `max_iters` to 5000.
*   **Observations:**
    *   Train/Val loss stable around ~0.064.
    *   **Most cases**: Still generated correctly.
    *   **Persistent Errors**: `3+5=` -> consistently output `9`. `1+3=` -> mostly `6`.
*   **Analysis:** Classic "memorization failure" in small networks. The model settled into a local minimum where specific keys were mapped to wrong values, likely due to small batch size or capacity constraints.
*   **Final Evaluation (N=100):**
    *   Result: 99/100 Correct.
    *   Accuracy: **99.00%**

### Train/Loss Graph
<img width="494" height="343" alt="image" src="https://github.com/user-attachments/assets/8e068168-cd60-4f5f-9ca1-00fa14ef529c" />

### Val/Loss Graph
<img width="494" height="343" alt="image" src="https://github.com/user-attachments/assets/83f2bbcd-53bb-438b-a118-13ec384dcce1" />


---

## Experiment 2: Intermediate (Two-Digit Addition)
*   **Goal:** Test capacity and generalization on a larger dataset (00-99 addition).
*   **Dataset:** All combinations of 0-99 addition (10,000 samples). Format: `a+b=c\n`.
*   **Settings:** 5000 iters, `n_layer=4`, `n_embd=128`.
*   **Training Results:** `train loss ~1.01`, `val loss ~1.01`.
*   **Sampling Results:**
    *   **Accuracy:** Surprisingly high despite loss around 1.0. Correctly answered prompts like `31+13=44`.
    *   **Continuous Generation:** Not only answered the prompt but also generated subsequent valid equations (e.g., `91+20=111`, `67+7=74`).
*   **Analysis:**
    *   **Loss Interpretation:** A loss of ~1.0 suggests some confusion (perplexity ~2.7), but `argmax` sampling consistently picks the correct digits. The model has mastered the syntax (`+`, `=`) perfectly.
    *   **Generalization vs Memorization:** Since we used the whole dataset, this is technically memorization. However, the fact it generates *new* valid equations immediately after the answer suggests it has learned the *structure* of the data perfectly.
*   **Final Evaluation (N=100):**
    *   Result: 91/100 Correct.
    *   Accuracy: **91.00%**

### Out-of-Distribution Test (3-Digit Inputs)
*   **Prompt:** `100+1=`, `100+13=`, `101+1=`
*   **Result:**
    *   `100+1=`: Often outputs `101` (Correct!), but sometimes `117`.
    *   `100+13=`: Fails (Outputs `33`, `23`, `32`).
*   **Insight:** The model shows partial generalization capabilities (getting close to the answer or getting simple carry cases right) even though it was never trained on 3-digit inputs, but struggles with consistent arithmetic on unseen digits.

---

## Conclusions (Intermediate Stage)

1.  **Framework Validated:** The `Prompt -> Response` framework using nanoGPT works reliably for arithmetic tasks.
2.  **Model Capacity:** Small Transformers can memorise arithmetic tables (1-digit, 2-digit) effectively.
3.  **Loss vs Accuracy:** Low loss is desirable, but functional accuracy can be achieved even with moderate loss values in structured tasks if the "best guess" is consistently correct.
