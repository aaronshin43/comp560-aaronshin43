# comp560-aaronshin43

This repo contains Aaron Shin's experiments for the COMP560 research project.

## Installation

### 0. Virtual Environment (Recommended)
It is recommended to use a virtual environment to manage dependencies locally.
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate
```

### 1. Install Dependencies
```bash
pip install numpy transformers datasets tiktoken wandb tqdm
```

### 2. Install PyTorch

**For CUDA 12.6 (GPU Support):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

**For CPU Only or non-CUDA machines:**
```bash
pip install torch
```

## Final Report

[FINAL_REPORT.md](FINAL_REPORT.md) — *Target Masking in a Tiny Transformer: Higher Ceiling, Not Faster Convergence.* Single-experiment writeup focused on the masking-benchmark result (Exp 6), with mechanism analysis and a cross-task input-fraction scaling rule.

## Projects

### 1. [English-Morse Translation](morse-code/README.md)
Experiments in learning the mapping between English text and Morse code sequences using nanoGPT.

### 2. [Addition Experiments](addition/README.md)
A specific application focusing on teaching a transformer model to perform arithmetic addition through simple prompt-response pairs.

### 3. [Short Input/Output Framework](framework/README.md)
A standardized framework designed for experiments mapping short inputs to short outputs. It simplifies data preparation, training, and sampling for tasks like arithmetic, logic gates, and simple Q&A.

### 4. [Validation: Target Masking Study](validation/README.md)
Experiments validating the impact of target masking and context window size on autoregressive generation accuracy. Shows that restricting loss to output tokens only is critical for instruction-following tasks.

### 5. [Addition Scratchpad: Length Generalization](addition_scratchpad/EXPERIMENT.md)
Investigates whether scratchpad-based chain-of-thought reasoning enables a tiny Transformer to generalize addition to digit lengths unseen during training. Covers baseline failure, scratchpad training, zero-shot OOD testing, and curriculum learning.

### 6. [Target Masking Benchmark](masking_benchmark/EXPERIMENT.md)
Convergence-curve study comparing masked vs unmasked training on plain and scratchpad 2-digit addition. Finds that target masking does not accelerate convergence but raises the accuracy ceiling by +6.4pp on plain addition, and that the benefit scales with the input token fraction of each sequence. Subject of the [final report](FINAL_REPORT.md).

### 7. [Masking Study: Input Fraction Hypothesis](masking_study/EXPERIMENT.md)
Two-phase follow-up attempting to causally isolate input token fraction as the driver of the masking benefit. Phase 1 extends digit length; Phase 2 manipulates input repetition. Both phases are inconclusive — Phase 1 changes too many variables simultaneously, and Phase 2 hits a ceiling effect from a [10, 99] digit-range design flaw. Hypothesis remains open.

### 8. [Pre-LN vs Post-LN Reproduction](autoresearch-preln/PLAN.md) *(in progress)*
Small-scale reproduction of Xiong et al. (ICML 2020) using Karpathy's `autoresearch` framework. 2×2 factorial design over LayerNorm placement and warmup, evaluated by validation bits-per-byte under a fixed 5-minute training budget.