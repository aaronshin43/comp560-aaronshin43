# Experiment Report: Scratchpad as a Length Generalization Strategy

## 1. Experimental Setup

* **Model Configuration:** `n_layer=6, n_head=4, n_embd=128` (identical across all phases for fair comparison)
* **Training Data:** 1~2-digit addition, exhaustive pairs (10,100 samples: 100 1-digit + 10,000 2-digit)
* **Train / Val Split:** 90:10 shuffle split (~9,090 train / ~1,010 val)
* **Out-of-distribution(OOD) Test Data:** 3,000 random 3-digit samples (`100`–`999`) and 3,000 random 4-digit samples (`1000`–`9999`)
* **Training Setting:** `target_mask=True`, `enable_tf_eval=True`
* **Evaluation Metric (AR):** Autoregressive generation — model is prompted with `input=` only; the final contiguous digit sequence in the generated output is compared to the true answer.
* **Objective:** Show that a plain model that memorizes 1~2-digit arithmetic completely fails to generalize to longer inputs (Phase 1), and that training with a scratchpad-based carry algorithm enables the model to zero-shot generalize to unseen lengths (Phase 3).

---

## 2. Phase 1: Baseline — Plain Model

**Hypothesis:** A model trained on plain 1~2-digit addition will score >80% on in-distribution val, but collapse to ~0% on 3~4-digit OOD inputs. Without an explicit algorithm, the model memorizes input-output mappings and cannot generalize the carry operation to new digit lengths.

**Config:** `config/phase1_plain.py` — `block_size=64`, `max_iters=5000`, `batch_size=128`

**Plain format:** `{"input": "15+27", "output": "42"}`

<img width="500" height="333" alt="Train / Val Loss — Phase 1 Plain" src="https://github.com/user-attachments/assets/e23f6c53-0717-46e9-a26f-1e236913c663" />
<img width="500" height="333" alt="TF Exact Match — Phase 1 Plain" src="https://github.com/user-attachments/assets/e5267869-ab09-4654-9f83-5d2652850bd5" />

### Results

| Eval Target | Type | Accuracy | Correct / Total |
|---|---|---|---|
| Val (1~2-digit) | AR (in-dist) | 82.9% | 837 / 1,010 |
| Test (3-digit) | AR (OOD) | 0.6% | 17 / 3,000 |
| Test (4-digit) | AR (OOD) | 0.0% | 0 / 3,000 |

### Analysis

* The model achieves **82.9%** on in-distribution 1~2-digit val, confirming it has learned the basic task.
* On 3-digit OOD inputs, accuracy collapses to **0.6%** — almost entirely noise (17 lucky guesses out of 3,000). On 4-digit inputs, it scores **0.0%** with zero correct answers.
* This confirms the core hypothesis: a plain model learns input-output mappings by rote, not the underlying carry algorithm. Faced with an unfamiliar number of digits, it has no mechanism to extend its learned behavior.

---

## 3. Phase 2: Scratchpad Training — Teaching the Algorithm

**Hypothesis:** By training on a scratchpad format that externalizes the digit-by-digit carry computation, the model can learn the *algorithm* rather than the output distribution.

**Config:** `config/phase2_scratchpad.py` — `block_size=128`, `max_iters=10000`, `batch_size=128`

**Scratchpad format:** `{"input": "15+27", "output": "[5+7=12,C1][1+2+1=4,C0]42"}`

Each bracket `[dA+dB(+carry)=sum,Cnew_carry]` encodes one digit position (right-to-left). The structure is identical regardless of operand length.

<img width="500" height="333" alt="Train / Val Loss — Phase 2 Scratchpad" src="https://github.com/user-attachments/assets/92a58311-7b9c-4fab-8982-0196be758060" />

<img width="500" height="333" alt="TF Exact Match — Phase 2 Scratchpad" src="https://github.com/user-attachments/assets/bf7eaf51-b22f-42d3-8cdb-de633602f6f4" />

### Results

| Eval Target | Type | Accuracy | Correct / Total |
|---|---|---|---|
| Val (1~2-digit) | AR (in-dist) | 88.5% | 894 / 1,010 |

**Qualitative verification:** To confirm the model is genuinely generating scratchpad chains rather than outputting memorized strings, individual samples were prompted and the outputs inspected manually. The model correctly produces bracket-by-bracket carry chains before emitting the final answer.

<img width="430" height="90" alt="Sample output — Phase 2 Scratchpad generation" src="https://github.com/user-attachments/assets/55c25870-02f0-40af-a29e-28c9191e6690" />

### Analysis

* The model achieves **88.5%** on in-distribution 1~2-digit val, matching Phase 1 performance despite the much longer and more structured output format. This confirms the model has successfully internalized the bracket-based carry algorithm.
* Manual sampling confirms the model is **genuinely computing** the scratchpad chain — digit extraction, per-position addition, carry propagation, and final answer emission all appear in sequence — rather than retrieving memorized strings.
* The slight gap below 100% is expected: edge cases involving multiple cascading carries (e.g., `99+1`) and zero-padding edge cases account for most failures, consistent with the difficulty distribution observed in Phase 1.

---

## 4. Phase 3: Zero-Shot Length Generalization

**Hypothesis:** The Phase 2 scratchpad model, with no additional training, can generalize to 3\~4-digit inputs by applying the same bracket-based carry algorithm it learned on 1\~2-digit data.

**Checkpoint:** `out/phase2_scratchpad/ckpt.pt` — no additional training performed.

### Results

| Eval Target | Phase 1 (Plain) | Phase 3 (Scratchpad, no retraining) |
|---|---|---|
| Test (3-digit OOD) | 0.6% (17/3,000) | 0.0% (0/3,000) |
| Test (4-digit OOD) | 0.0% (0/3,000) | 0.0% (0/3,000) |

**Qualitative verification:** Individual 3\~4-digit samples were prompted and outputs inspected. The model consistently generates exactly **two brackets** regardless of input length — the same fixed depth seen in all 1\~2-digit training samples — before emitting an incorrect answer.

<img width="430" height="90" alt="Sample output — Phase 3 OOD generation showing fixed 2-bracket pattern" src="https://github.com/user-attachments/assets/fed767c7-f7ec-472b-92f6-3b67554f924b" />
<img width="430" height="90" alt="Sample output2 — Phase 3 OOD generation showing fixed 2-bracket pattern" src="https://github.com/user-attachments/assets/89640a45-d338-4b8d-b21f-16f4217a7b25" />

### Analysis

* Contrary to the original hypothesis, the scratchpad model scores **0.0%** on all OOD inputs — no improvement over Phase 1 on 3-digit, and no benefit on 4-digit either.
* Manual sampling reveals the failure mode: the model has memorized **bracket depth = 2** as a structural invariant. Every generated output contains exactly two brackets, regardless of how many digit positions the input requires. The carry algorithm itself appears correct within those two steps, but the model has no mechanism to decide when to stop generating brackets and when to continue.
* This distinguishes the failure from Phase 1 in an important way:

| | Phase 1 (Plain) | Phase 3 (Scratchpad) |
|---|---|---|
| Knows carry rule? | No | Yes |
| Knows when to stop? | N/A | **No — fixed at 2** |
| Failure mode | No algorithm | Algorithm depth not generalized |

* The root cause is a **length generalization failure** compounded by overfitting. During training, the scratchpad sequences for 1~2-digit inputs always contain at most 2 brackets (1-digit: 1 bracket, 2-digit with carry: 2 brackets + optional `[1]`). The model overfits to this depth distribution rather than learning that bracket count is a function of operand length. The divergence between train loss (0.097) and val loss (0.601) observed late in training is consistent with this: the model is increasingly fitting the fixed structural pattern of training samples rather than the generative rule.
* This is a known limitation of standard Transformer architectures with learned absolute positional embeddings: systematic length generalization beyond the training distribution requires either explicit length signals, relative position encodings, or curriculum exposure to variable-depth sequences.

---

## 5. Phase 4 Design: Curriculum Learning

**Hypothesis:** If the training set includes scratchpad examples across all target digit lengths (1~4 digits), the model will learn that bracket count is determined by operand length and successfully generalize the carry algorithm to any depth.

**Key change from Phase 2:** Add 3-digit and 4-digit scratchpad samples to the training mix.

### Proposed Dataset

| Split | Contents | Approx. Samples |
|---|---|---|
| Train | 1~2-digit scratchpad (exhaustive) | 10,100 |
| Train | 3-digit scratchpad (random sample) | 5,000 |
| Train | 4-digit scratchpad (random sample) | 5,000 |
| **Total** | | **~20,100** |

### Config Changes

* `dataset = 'scratchpad_1_4digit'` — new combined dataset
* `block_size = 128` — unchanged (already accommodates 4-digit)
* `max_iters = 10000` — unchanged
* OOD test: held-out 3~4-digit samples **not seen during training**

### Evaluation Plan

1. In-dist val (1~4-digit scratchpad) — expect ~90%+
2. OOD test (held-out 3-digit) — expect significant improvement over Phase 3
3. OOD test (held-out 4-digit) — expect significant improvement over Phase 3
4. Stretch: OOD test on **5-digit** — does curriculum to 4 digits enable 5-digit generalization?

---

## 6. Conclusion
