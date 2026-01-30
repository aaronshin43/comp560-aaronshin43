# Addition Project - Experiment Logs

This document records the detailed history, settings, and observations of experiments conducted within the Addition project.

## Experiment 1: Basic Addition (Single Digit)
*   **Goal:** Verify if nanoGPT can learn single-digit addition (0+0 to 9+9) via rote memorization.
*   **Dataset:** All combinations of 1-digit addition (100 samples total).

### Phase 1: Initial Attempts (Splitting Issue)
*   **Context:** Initially used a 90/10 random split for train/val sets.
*   **Test 1 (Blue Line):** `iters=2000`. Val loss increased significantly (~3.0).
    *   *Analysis:* The model memorized the training set (low train loss) but failed completely on the validation set because the validation set contained specific arithmetic pairs (e.g., `3+5`) that were never seen during training. This led to high overfitting.
*   **Test 2 (Red Line):** `iters=5000`. Val loss increased even further (~3.5).
    *   *Analysis:* Extending training only worsened the overfitting. The model became extremely confident in its correct answers for the training set, but completely wrong on the unseen validation set pairs.

### Phase 2: Memorization Strategy (No Split)
*   **Context:** Changed `prepare.py` to use the **same full dataset** (100 samples) for both train and validation to force rote memorization of the entire addition table.
*   **Test 3 (Green Line):** `iters=2000`. Val loss dropped dramatically to ~0.06.
    *   *Analysis:* Removing the split solved the high loss issue. The model successfully memorized most pairs.
*   **Test 4 (Purple Line):** `iters=5000`. Val loss stable low (~0.06).
    *   *Analysis:* Longer training stabilized the loss, but specific errors (e.g., `3+5=9`) persisted due to local minima or insufficient batch size.

### Phase 3: Final Optimization
*   **Test 5 (Pink Line):** `iters=7000`, Increased `batch_size`.
    *   *Observation:* Val loss dropped even faster and lower.
    *   *Result:* **99% Accuracy**. The combination of sufficient iterations, larger batch size, and full dataset visibility allowed the model to perfectly memorize the arithmetic table, eliminating the persistent errors found in Test 4.

### Train/Loss Graph
<img width="494" height="343" alt="image" src="https://github.com/user-attachments/assets/8e068168-cd60-4f5f-9ca1-00fa14ef529c" />

### Val/Loss Graph
<img width="494" height="343" alt="image" src="https://github.com/user-attachments/assets/83f2bbcd-53bb-438b-a118-13ec384dcce1" />

### Final Evaluation (N=100)
*   Result: 99/100 Correct.
*   Accuracy: **99.00%**

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
