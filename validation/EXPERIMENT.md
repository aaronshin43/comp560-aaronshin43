# Experiment Report: Impact of Target Masking and Context Window on Autoregressive Generation

## 1. Experimental Setup

* **Model Configuration:** `n_layer=6, n_head=4, n_embd=128`
* **Dataset:** 10,000 2-digit arithmetic samples (9,000 Train / 1,000 Val)
* **Task:** Short Input -> Short Output (e.g., `1+1=2\n`)
* **Evaluation Metric (TF Exact Match):** A custom evaluation metric that feeds the entire sequence (input + separator + output + stop_token) through the model in a single forward pass (Teacher Forcing). It isolates the model's argmax predictions strictly after the separator token, requiring a 100% token-for-token match against the ground-truth output to register as correct.
* **Objective:** To analyze how target masking (`ignore_index = -1`) and context length (`block_size`) influence training efficiency and true autoregressive (AR) generation performance.

## 2. Experiment 1: The `block_size=16` Anomaly

The first experiment tested the masking logic within a highly constrained context window.

<img width="500" height="333" alt="Train / Val Loss — No Mask vs. Target Mask (block_size=16)" src="https://github.com/user-attachments/assets/20e508a9-6416-492b-9fca-2367b4ca1372" /> 

<img width="500" height="333" alt="TF Exact Match — No Mask vs. Target Mask (block_size=16)" src="https://github.com/user-attachments/assets/d5ed6761-9506-4941-a344-f06be5e34ff5" />

* **Observation:**

| Configuration | Train Loss | Val Loss | Train TF Match | Val TF Match | Train AR Test | Val AR Test |
|---|---|---|---|---|---|---|
| No Mask | 0.989 | 1.160 | 89.3% | 92.2% | 87.9% | 89.8% |
| Target Mask | 0.531 | 0.541 | 83.1% | 84.1% | 78.8% | 79.1% |


* **Analysis (Loss vs Accuracy):**
    * Applying the target mask drastically reduced the loss, yet the exact match accuracy dropped.
    * This paradox was caused by sequence truncation. With `block_size=16`, many blocks begin mid-equation. The masking logic misinterprets these fragmented sequences and incorrectly masks the *actual target tokens* with `-1`.
    * The model effectively suffered from data loss. The loss value appeared lower simply because the model was being evaluated on far fewer tokens per block, creating an illusion of improvement.
    * **Autoregressive (AR) Test Performance:** Contrary to the expectation that target masking should improve AR generation (as demonstrated in Experiment 2), the masked model scored *lower* on the AR test (78.8% / 79.1%) than the unmasked model (87.9% / 89.8%). This is a direct consequence of the same truncation bug: the model trained with a broken mask learned from corrupted supervision signals — actual answer tokens were silenced, leaving the model to optimize over fragmented, context-incomplete blocks. The result is a model that is demonstrably *less capable* of autoregressive generation, confirming that the root cause is faulty masking logic rather than any fundamental limitation of the masking approach itself.

## 3. Experiment 2: The `block_size=64` Resolution

To resolve the truncation edge cases, `block_size` was increased to 64, allowing the model to process multiple complete equations per forward pass.

<img width="500" height="333" alt="Train / Val Loss — No Mask vs. Target Mask (block_size=64)" src="https://github.com/user-attachments/assets/c9961cc1-491e-4a77-932e-4bfc0994ae03" />

<img width="500" height="333" alt="Autoregressive Accuracy — No Mask vs. Target Mask (block_size=64)" src="https://github.com/user-attachments/assets/548a3386-cce2-4da0-a264-98861a8fafbe" />

* **Observation:**

| Configuration | Train Loss | Val Loss | Train TF Match | Val TF Match | Train AR Test | Val AR Test |
|---|---|---|---|---|---|---|
| No Mask | 0.989 | 1.160 | 88.4% | 89.6% | 73.4% | 76.7% |
| Target Mask | 0.123 | 0.125 | 88.9% | 90.5% | 87.2% | 88.0% |


* **Analysis (The AR Collapse without Masking):**
    * With a larger block size, the model observes a continuous stream of multiple equations. Without masking, it expends parameter capacity attempting to predict the arbitrary numbers of the *next* equation in the sequence.
    * When tested in a AR environment, the unmasked model is destabilized by the absence of continuous streaming context, leading to poor generation accuracy.
    * Conversely, **Target Masking** forces the model to disregard input distributions entirely. It directs its capacity exclusively toward computing the correct output after the `=` token. This prevents overfitting to the concatenated training stream and yields highly stable, superior performance during AR generation.

## 4. The Loss vs. Exact Match Paradox

A consistent anomaly was observed across all test runs: **Train Loss was always lower (better) than Val Loss, yet Train Exact Match (TF Eval) was occasionally lower (worse) than Val Exact Match.** After verifying the evaluation logic, this was confirmed to be a mathematically sound phenomenon driven by the difference in metric mechanics and dataset scale:

* **Confidence (Continuous) vs. Perfection (Discrete):** Cross-Entropy Loss measures average predictive confidence, whereas Exact Match requires 100% sequence perfection. The model exhibits high overall confidence across the vast training data (yielding low Train Loss). However, making even a single-token mistake on an equation zeroes out the Exact Match score.
* **Scale and Edge Cases:** The Train set (9,000 samples) is nine times larger than the Val set (1,000 samples). Statistically, the massive Train set contains a significantly higher volume of extreme "edge cases" (e.g., highly complex multi-digit carryovers). Failing these complex cases drops the overall Train Exact Match percentage. Meanwhile, the smaller Val set randomly sampled a slightly more uniform distribution, allowing for a higher exact match rate despite lower average confidence (higher Val Loss).

## 5. Experiment 3: Verifying the Edge Case Hypothesis (8:2 Split)

To directly verify the edge case hypothesis proposed in Section 4, the same `block_size=64` setup from Experiment 2 was reused, with only the dataset split ratio changed from **9:1 to 8:2** (8,000 Train / 2,000 Val).

<img width="500" height="333" alt="TF Exact Match — No Mask vs. Target Mask (block_size=64, 8:2 Split)" src="https://github.com/user-attachments/assets/cce58bff-877e-4c22-860e-e9a04c755122" />

* **Observation:**

| Configuration | Train Loss | Val Loss | Train TF Match | Val TF Match | Train AR Test | Val AR Test |
|---|---|---|---|---|---|---|
| No Mask | 0.982 | 1.170 | 88.0% | 86.6% | 80.1% | 78.4% |
| Target Mask | 0.121 | 0.125 | 89.2% | 88.4% | 87.4% | 86.9% |

* **Analysis (Edge Case Distribution as the Confounding Variable):**
    * With the validation set now twice as large (2,000 samples), it naturally contains a proportionally greater number of edge cases (e.g., carry-heavy equations). As a result, both No Mask and Target Mask configurations showed Val TF Match falling **below** Train TF Match far more frequently during training — the inverse of what was observed in the 9:1 split.
    * Similarly, AR test results on the val set were no longer consistently higher than the train set; val accuracy was sometimes lower and sometimes higher, fluctuating around the train value without a systematic advantage.
    * This directly confirms that the original paradox (Val > Train on Exact Match) was not a structural property of the model or loss function, but a statistical artifact of the small 9:1 val set randomly sampling a more uniform, edge-case-light distribution. By equalizing the proportion of edge cases across both splits, the gap disappeared as expected.

## 6. Conclusion

Computing cross-entropy loss over an entire sequence forces the model to learn irrelevant input patterns and transitioning noise, which actively harms true autoregressive generation. **Target Masking** successfully restricts the optimization objective purely to output tokens. By eliminating input-prediction penalties, the model allocates its parameters entirely toward problem-solving (generalization), making it a critical requirement for instruction-following and short-response tasks.
