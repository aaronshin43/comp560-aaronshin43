# Addition Experiment Project

This project aims to train a nanoGPT-based model to perform addition tasks. The goal is to verify if a character-level transformer can learn arithmetic operations through simple prompt-response pairs.

## Goals

1.  **Framework Verification**: Establish a framework for "Short Input -> Short Response" experiments using nanoGPT efficiently.
2.  **Associative Learning**: Verify if the model can memorize and generalize 1-digit addition (0-9) via rote memorization.
3.  **Future Expansion**: (Planned) Expand to multi-digit addition or other arithmetic operations (subtraction, etc.).

## Experiments

### 1. Basic (Single Digit Addition)
*   **ID**: `basic`
*   **Description**: Learning addition of single-digit numbers (0-9).
*   **Data Format**: `a+b=c\n` (e.g., `2+3=5\n`)
*   **Config**: `config/basic.py` (Small model, overfitting encouraged)

### 2. Intermediate (Two-Digit Addition)
*   **ID**: `intermediate`
*   **Description**: Learning addition of two-digit numbers (00-99).
*   **Data Format**: `a+b=c\n` (e.g., `12+34=46\n`). Total 10,000 samples.
*   **Config**: `config/intermediate.py` (Larger model, `n_layer=4`, `n_embd=128`)

## Useful Commands

Run these commands from the `addition/` directory.

### 1. Data Generation
Generate the `train.bin` and `val.bin` files.
```bash
python data/basic/prepare.py
```

### 2. Training
Train the model using the `nanoGPT` training script.
Pre-requisite: `comp560-nanoGPT` must exist in the parent directory.
```bash
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python -u ../../comp560-nanoGPT/train.py config/basic.py
```

### 3. Sampling (Inference)
Test the trained model.
```bash
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python -u ../../comp560-nanoGPT/sample.py config/basic.py
```
*Note: To specify a prompt (e.g., start with "2+2="), edit the `config/basic.py` file directly:*
```python
# config/basic.py
start = "2+2="
num_samples = 3
max_new_tokens = 5
```

### 4. Evaluation (Accuracy Test)
Automatically generate test prompts and calculate accuracy.
```bash
# Evaluate Intermediate (default)
python evaluate.py

# Evaluate Basic
python evaluate.py out_dir=out/basic dataset=basic
```

## Experiment Log

Detailed logs and observations from my experiments are recorded in a separate file to keep this README concise.

**[View Experiment Logs](EXPERIMENTS.md)**

## How to Use This Framework for Your Own Experiments

Research goal: *Create a transformer model to produce short responses in response to short inputs.* 
This repository provides a proven framework to adapt `nanoGPT` (originally designed for continuous text generation) for discrete **Prompt-Response** tasks.

To create your own experiment (e.g., Reverse String, Chain of Thought, Logic Gates), follow these steps:

### 1. Data Preparation (`data/your_experiment/prepare.py`)
The key is to format your data as a continuous stream of discrete events using separators.
*   **Format:** `Promt` + `Separator` + `Response` + `EndToken`
*   **Example (Addition):** `2+2` (Prompt) + `=` (Separator) + `4` (Response) + `\n` (EndToken)
*   **Example (Reverse):** `abc` (Prompt) + `:` (Separator) + `cba` (Response) + `\n` (EndToken)

**Checklist for `prepare.py`:**
*   [ ] Generate pairs of inputs and outputs.
*   [ ] Concatenate them into a single long string.
*   [ ] Save `train.bin`, `val.bin`, and `meta.pkl`.
*   [ ] **Crucial:** Ensure your `EndToken` (e.g., `\n`) is consistent so the model learns when to stop.

### 2. Configuration (`config/experiment.py`)
Create a new config file. You can copy `config/basic.py` and adjust:
*   `dataset`: Point to your new data folder name.
*   `out_dir`: Directory to save checkpoints.
*   `max_iters`: Increase for harder tasks (e.g., 2000 -> 5000+).
*   `n_layer`, `n_head`, `n_embd`:
    *   *Simple tasks (1-digit)*: Small is fine (`n_layer=3`, `n_embd=64`).
    *   *Complex tasks (2-digit, logic)*: Increase capacity (`n_layer=4~6`, `n_embd=128~256`).

### 3. Execution
Use the standard commands with your new config.
*   **Train:** `NANOGPT_CONFIG=... python train.py config/your_experiment.py`
*   **Sample:** `NANOGPT_CONFIG=... python sample.py config/your_experiment.py start="YourPrompt="`

---
