# comp560-aaronshin43

This repo contains Aaron Shin's experiments for the COMP560 research project.

## Installation

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
A standardized framework designed for experiments mapping short inputs to short outputs. It simplifies data preparation and training for tasks like arithmetic, logic gates, and simple Q&A.