# Engine — comp560-nanoGPT

This document covers the `comp560-nanoGPT` repo: architecture, training scripts, config system, and eval utilities.

Repo root: `d:\03_Coding\comp560-nanoGPT\`

---

## Architecture (`model.py`)

Standard decoder-only GPT (GPT-2 style):

- **Token + positional embeddings** (`wte`, `wpe`) — learned, absolute
- **N Transformer blocks** (`Block`): pre-norm (`LayerNorm`) → causal self-attention (`CausalSelfAttention`) → pre-norm → MLP (4× expansion, GELU)
- **Output head** (`lm_head`): weight-tied to `wte`
- **Loss**: cross-entropy with `ignore_index=-1` — this is what target masking exploits

Key config fields (`GPTConfig`):

| Field | Description |
|---|---|
| `block_size` | Context window (max tokens per forward pass) |
| `n_layer` | Number of Transformer blocks |
| `n_head` | Number of attention heads |
| `n_embd` | Embedding / hidden dimension (must be divisible by `n_head`) |
| `vocab_size` | Set automatically from `meta.pkl` at training time |
| `dropout` | Dropout rate (0.0 in all comp560 experiments) |
| `bias` | Whether to include bias in Linear/LayerNorm layers |

---

## Config System

Configs are plain Python files that set global variables. Two mechanisms apply overrides in order:

1. **Config file** — positional arg without `--`, e.g. `config/basic.py`. Executed via `exec()` into `globals()`.
2. **CLI overrides** — `--key=value` flags after the config file. Parsed by `comp560ext.configure()` (or `configurator.py` for the vanilla `train.py`).

`comp560ext.configure()` fixes a bug in the original `configurator.py`: it uses `split('=', 1)` (maxsplit=1) so values containing `=` (e.g. `--start="1+1="`) parse correctly.

The `NANOGPT_CONFIG` environment variable tells `train.py`/`sample.py` where to find `configurator.py` so experiments can live outside the nanoGPT repo:

```bash
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python ../../comp560-nanoGPT/train.py config/my_config.py
```

---

## Training Scripts

### `train.py` — vanilla training

Standard nanoGPT training loop. Key behavior:
- Reads `data/<dataset>/train.bin` and `val.bin` (uint16 token IDs, from `prepare.py`)
- Samples random windows of length `block_size` from the flat binary file
- Cosine LR schedule with linear warmup (`warmup_iters`) and decay to `min_lr` at `lr_decay_iters`
- Saves checkpoint to `<out_dir>/ckpt.pt` when val loss improves (or always if `always_save_checkpoint=True`)
- Supports DDP (`torchrun`) and `torch.compile`

### `train_benchmark.py` — extended trainer (used from Exp 4 onward)

Superset of `train.py` with three extra features (all disabled by default):

| Flag | Default | Description |
|---|---|---|
| `--target_mask=True` | False | Applies target masking to `y` batches — sets input-region tokens to `-1` so loss is computed only on output tokens |
| `--enable_tf_eval=True` | False | Runs Teacher Forcing exact-match eval every `eval_interval` iters |
| `--early_stop_on_perfect_tf=True` | False | Halts training when TF val accuracy first reaches 100% |
| `--benchmark_target` | `'val'` | Which JSONL split(s) to use for TF eval: `'train'`, `'val'`, or `'both'` |
| `--tf_eval_max_samples` | 0 (all) | Cap on samples used per TF eval pass |

**Requires** `data/<dataset>/train.jsonl` and/or `val.jsonl` for TF eval (produced by `validation/prepare.py` and `addition_scratchpad/prepare.py` — NOT by `framework/prepare.py`).

---

## Evaluation Scripts

### `eval_generation.py` — standalone AR evaluation

Loads a saved `ckpt.pt` and runs autoregressive generation eval. Usage (run from the experiment directory):

```bash
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python ../../comp560-nanoGPT/eval_generation.py config/my_config.py \
  --benchmark_target=both --eval_max_samples=1000
```

- Prompts the model with `input + separator` only
- Stops generation at `stop_token` or `max_new_tokens`
- Reports exact string match between generated output and ground-truth `output` field
- Requires `data/<dataset>/meta.pkl` and `<split>.jsonl`

### `eval_scratchpad.py` — scratchpad-aware AR evaluation

Used exclusively in Exp 5. Same mechanics as `eval_generation.py` but uses **answer extraction** instead of full exact-match:
- Extracts the last contiguous digit sequence from the generated text (after stopping at first `\n`)
- Compares only the final numeric answer, ignoring scratchpad brackets
- Handles OOD tokens gracefully (skips samples with characters not in the tokenizer vocab)

```bash
python ../../comp560-nanoGPT/eval_scratchpad.py config/phase1_plain.py \
  --eval_file=data/plain_3_4digit/3digit.jsonl --max_samples=1000
```

---

## comp560 Extension Module (`comp560/comp560ext.py`)

All custom utilities live here to minimize changes to the original nanoGPT files.

| Function | Purpose |
|---|---|
| `configure(globals_dict)` | Parses CLI args, applies config file + `--key=value` overrides |
| `apply_target_mask(y, sep_id, stop_id)` | Vectorized cumsum masking — sets input-region tokens to `-1` |
| `run_tf_eval(model, eval_data, encode, ...)` | Single-pass Teacher Forcing exact-match eval → `(accuracy_pct, exact_matches, total)` |
| `run_final_gen_eval(model, jsonl_path, encode, decode, ...)` | AR generation eval from JSONL file → `(accuracy_pct, exact_matches, total)` |
| `generate(model, idx, max_new_tokens, ..., stop_token)` | AR generation with early stop at `stop_token` |
| `load_eval_dataset(jsonl_path, max_samples)` | Loads `[{'input': str, 'output': str}, ...]` from JSONL |
| `setup_char_encode_decode(meta, meta_vocab_size)` | Builds `encode`/`decode` callables from `meta.pkl` |
| `resolve_mask_token_ids(meta, ..., separator_token, stop_token)` | Returns `(sep_id, stop_id)` for target masking |

### Target Masking — `apply_target_mask`

Replaces input-region token IDs in the label tensor `y` with `-1` (the `ignore_index` in `model.py`'s cross-entropy call), so the loss is only computed over the output region.

**Masking rule:** position `t` is in the "input phase" when `stops_seen_before_t >= seps_seen_before_t`. Implemented with vectorized `cumsum` — no Python loops.

**Known edge case:** a context window that begins mid-output (before the first stop token) will incorrectly mask those leading output tokens. This is the `block_size=16` bug documented in Exp 4.

---

## Data Format

All comp560 experiments use character-level tokenization. Data pipeline:

1. Generate JSONL: `{"input": "15+27", "output": "42"}` per line
2. Run `prepare.py` → produces:
   - `train.bin` / `val.bin` — flat uint16 arrays of char IDs
   - `meta.pkl` — `{'vocab_size': int, 'stoi': dict, 'itos': dict}`
   - `train.jsonl` / `val.jsonl` — JSONL splits (only in `validation/` and `addition_scratchpad/` versions)
3. Training reads `.bin` files; TF/AR eval reads `.jsonl` files

Each sample in the binary stream is formatted as:
```
{input}{separator}{output}{stop_token}
```
Default separator: `=`, default stop token: `\n`.

---

## Typical Hyperparameter Ranges (comp560 experiments)

| Parameter | Small (Exp 1–3) | Medium (Exp 4–5) |
|---|---|---|
| `n_layer` | 3 | 6 |
| `n_head` | 3–4 | 4 |
| `n_embd` | 120–128 | 128 |
| `block_size` | 64 | 64–128 |
| `batch_size` | 12–32 | 128 |
| `max_iters` | 2000–7000 | 5000–10000 |
| `learning_rate` | 1e-3 | 3e-4 |
| `warmup_iters` | 0–100 | 200–400 |
