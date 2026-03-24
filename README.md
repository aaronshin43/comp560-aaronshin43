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