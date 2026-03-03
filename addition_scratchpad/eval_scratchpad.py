"""
Scratchpad-aware autoregressive generation evaluator.

Unlike eval_generation.py (which requires full exact-match of the output field),
this script extracts the *final numeric answer* from generated text and compares
it to the true answer. Compatible with both plain and scratchpad output formats.

Answer extraction rule:
  - Take all generated text up to the first \\n
  - Find all contiguous digit sequences with re.findall(r'\\d+', text)
  - The LAST sequence is the answer (final sum, appears after all brackets)

This makes all three phases comparable on the same metric:
  Phase 1: plain model  -> generates "42"               -> extracts "42"
  Phase 2: scratchpad   -> generates "[...C0]42"        -> extracts "42"
  Phase 3: scratchpad model on OOD data                 -> extracts final number

Usage:
  # Phase 1 in-dist (1~2-digit val):
  python eval_scratchpad.py config/phase1_plain.py \\
      --eval_file data/plain_1_2digit/val.jsonl --max_samples 1000

  # Phase 1 OOD (3-digit, expected ~0%):
  python eval_scratchpad.py config/phase1_plain.py \\
      --eval_file data/plain_3_4digit/3digit.jsonl --max_samples 1000

  # Phase 2 in-dist (scratchpad val):
  python eval_scratchpad.py config/phase2_scratchpad.py \\
      --eval_file data/scratchpad_1_2digit/val.jsonl --max_samples 1000

  # Phase 3 OOD — Phase 2 checkpoint, no extra training (expected: big jump):
  python eval_scratchpad.py config/phase2_scratchpad.py \\
      --eval_file data/plain_3_4digit/3digit.jsonl --max_samples 1000

  python eval_scratchpad.py config/phase2_scratchpad.py \\
      --eval_file data/plain_3_4digit/4digit.jsonl --max_samples 1000
"""

import os
import re
import json
import pickle
from contextlib import nullcontext

import torch

from model import GPTConfig, GPT
from comp560 import comp560ext

# ─── Configuration defaults (all overridable via --key=value CLI args) ────────
out_dir         = 'out'               # directory containing ckpt.pt
dataset         = 'plain_1_2digit'    # used to locate data/<dataset>/meta.pkl
device          = 'cuda'
dtype           = 'bfloat16'
compile         = False

eval_file       = ''                  # REQUIRED: path to JSONL file to evaluate
max_samples     = 0                   # 0 = evaluate all samples

separator_token = '='
stop_token      = '\n'
max_new_tokens  = 80                  # upper bound on tokens generated per sample
temperature     = 1.0
top_k           = None
# ──────────────────────────────────────────────────────────────────────────────

config_keys = [k for k, v in globals().items()
               if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
comp560ext.configure(globals())

# ── Validate required argument ────────────────────────────────────────────────
if not eval_file:
    raise ValueError(
        "You must specify --eval_file=<path/to/your.jsonl>\n"
        "Examples:\n"
        "  --eval_file=data/plain_3_4digit/3digit.jsonl\n"
        "  --eval_file=data/scratchpad_1_2digit/val.jsonl"
    )

if not os.path.exists(eval_file):
    raise FileNotFoundError(f"eval_file not found: {eval_file}")

# ── Device setup ──────────────────────────────────────────────────────────────
if device == 'cuda' and not torch.cuda.is_available():
    print("CUDA not available, falling back to CPU.")
    device = 'cpu'

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = (nullcontext() if device_type == 'cpu'
       else torch.amp.autocast(device_type=device_type, dtype=ptdtype))

# ── Load checkpoint ───────────────────────────────────────────────────────────
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
print(f"Loading checkpoint: {ckpt_path}")
checkpoint = torch.load(ckpt_path, map_location=device)

model_args = checkpoint['model_args']
gptconf    = GPTConfig(**model_args)
model      = GPT(gptconf)

state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.eval()
model.to(device)

if compile:
    print("Compiling model...")
    model = torch.compile(model)

ckpt_iter = checkpoint.get('iter_num', '?')
print(f"Model loaded  (iter={ckpt_iter}, "
      f"n_layer={model_args['n_layer']}, n_head={model_args['n_head']}, "
      f"n_embd={model_args['n_embd']})")

# ── Load tokenizer ─────────────────────────────────────────────────────────────
data_dir  = os.path.join('data', dataset)
meta_path = os.path.join(data_dir, 'meta.pkl')

if not os.path.exists(meta_path):
    raise FileNotFoundError(
        f"meta.pkl not found at {meta_path}.\n"
        f"Make sure --dataset points to a prepared directory (run prepare.py first)."
    )

with open(meta_path, 'rb') as f:
    meta = pickle.load(f)

stoi   = meta['stoi']
itos   = meta['itos']
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
print(f"Tokenizer ready  (vocab_size={meta['vocab_size']})")

# ── Answer extraction ─────────────────────────────────────────────────────────
def extract_answer(text: str) -> str:
    """
    Extract the final numeric answer from plain or scratchpad text.

    Works for:
      Plain:      "42"                       -> "42"
      Scratchpad: "[5+7=12,C1][1+2+1=4,C0]42" -> "42"
      OOD gen:    "[3+6=9,C0][2+5=7,C0][1+4=5,C0]579" -> "579"

    Rule: stop at first newline, find all digit sequences, return the last one.
    """
    text = text.split('\n')[0]
    matches = re.findall(r'\d+', text)
    return matches[-1] if matches else ''

# ── Load eval data ─────────────────────────────────────────────────────────────
print(f"\nLoading eval file: {eval_file}")
samples = []
with open(eval_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        samples.append(obj)

_max = max_samples if max_samples > 0 else None
if _max:
    samples = samples[:_max]

print(f"Evaluating {len(samples)} samples...")
print()

# ── Encode stop token ─────────────────────────────────────────────────────────
stop_ids = [stoi[c] for c in stop_token if c in stoi]
stop_id  = stop_ids[0] if stop_ids else None

_block_size = model_args['block_size']
_gen_tokens = min(max_new_tokens, _block_size)

# ── Run evaluation ─────────────────────────────────────────────────────────────
exact_matches = 0
skipped       = 0
total         = len(samples)

for sample in samples:
    prompt_str  = sample['input'] + separator_token
    true_answer = extract_answer(sample['output'])

    # Skip samples with OOD characters (e.g. evaluating a Phase 2 model on plain OOD
    # data — the tokenizer was built from scratchpad chars, so plain chars are subset)
    try:
        prompt_ids = encode(prompt_str)
    except KeyError as e:
        skipped += 1
        total   -= 1
        continue

    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        with ctx:
            out_tensor = comp560ext.generate(
                model, prompt_tensor,
                max_new_tokens=_gen_tokens,
                temperature=temperature,
                top_k=top_k,
                stop_token=stop_id,
            )

    # Decode only the newly generated tokens (after the prompt)
    new_ids     = out_tensor[0, len(prompt_ids):].tolist()
    gen_text    = decode(new_ids)
    pred_answer = extract_answer(gen_text)

    if pred_answer == true_answer:
        exact_matches += 1

# ── Summary ────────────────────────────────────────────────────────────────────
accuracy = 100.0 * exact_matches / total if total > 0 else 0.0

print(f"Eval file   : {eval_file}")
print(f"Checkpoint  : {ckpt_path}  (iter={ckpt_iter})")
print(f"Samples     : {total}  (skipped {skipped} due to OOD tokens)")
print(f"Exact-match : {accuracy:.1f}%  ({exact_matches}/{total} correct)")
