"""
Evaluate the accuracy of the trained model on addition tasks.
Generates prompts (e.g., "12+34=") and checks if the model's output matches the correct answer.
"""
import os
import pickle
from contextlib import nullcontext
import torch
import sys
# Add nanoGPT directory to sys.path so we can import model.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../comp560-nanoGPT')))

from model import GPTConfig, GPT
import random

# -----------------------------------------------------------------------------
# Configuration
init_from = 'resume' # 'resume' or 'gpt2'
out_dir = 'out/intermediate' # directory containing checkpoint
dataset = 'intermediate'     # data directory (for meta.pkl)
num_samples = 100            # how many examples to test
max_new_tokens = 10          # max tokens to generate for the answer
temperature = 0.8            # 0.8 is good for some randomness, or use ~0 for rigid argmax
top_k = 200
seed = 1337
device = 'cuda' # or 'cpu'
# -----------------------------------------------------------------------------
# Allow config override from command line
# e.g. python evaluate.py out_dir=out/basic dataset=basic num_samples=50
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
import sys
for arg in sys.argv[1:]:
    if '=' not in arg: continue
    key, val = arg.split('=', 1)
    if key in config_keys:
        # try int/float/bool conversion
        try:
            if val in ['True', 'False']: val = (val == 'True')
            else: val = float(val) if '.' in val else int(val)
        except ValueError: pass # keep as string
        globals()[key] = val

# -----------------------------------------------------------------------------
# Setup Model
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device_type = 'cuda' if 'cuda' in device else 'cpu'

# Load checkpoint
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
print(f"Loading model from {ckpt_path}...")
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to(device)

# Load meta (encoder/decoder)
meta_path = os.path.join('data', dataset, 'meta.pkl')
if os.path.exists(meta_path):
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    print(f"Error: meta.pkl not found at {meta_path}")
    sys.exit(1)

# -----------------------------------------------------------------------------
# Evaluation Logic
print(f"\nStarting evaluation (N={num_samples})...")
correct_count = 0
total_count = 0

# We need to know what kind of problem (1-digit or 2-digit) we are testing.
# We'll infer it roughly or just generate random additions suitable for the dataset.
# Since 'basic' is 1-digit and 'intermediate' is 2-digit, we can check dataset name
# or just default to 2-digit (0-99 covers 0-9 too).

is_basic = 'basic' in dataset
range_max = 10 if is_basic else 100

for i in range(num_samples):
    # Generate a random test case
    a = random.randint(0, range_max - 1)
    b = random.randint(0, range_max - 1)
    target = a + b
    
    prompt = f"{a}+{b}="
    
    # Encode
    start_ids = encode(prompt)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    # Generate
    with torch.no_grad():
        # Using a slightly customized generate loop could be faster/cleaner to stop at \n
        # but for simplicity we use model.generate and then parse string
        y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k, stop_token=stoi['\n'])
        
    output_str = decode(y[0].tolist())
    
    # Parsing logic:
    # prompt is "12+34=", output_str is "12+34=46\n" (ideally)
    # We want to extract "46"
    
    # Remove the prompt part if it's there (model.generate output usually includes input x)
    # But wait, model.generate returns full sequence including x.
    generated_part = output_str[len(prompt):]
    
    # Cut off at the first newline or end of string
    answer_str = generated_part.split('\n')[0].strip()
    
    try:
        predicted_val = int(answer_str)
    except ValueError:
        predicted_val = -9999 # invalid format
        
    is_correct = (predicted_val == target)
    if is_correct:
        correct_count += 1
    
    total_count += 1
    
    # Print some examples
    if i < 10:
        mark = "OK" if is_correct else "FAIL"
        print(f"[{mark}] {prompt} -> Predicted: '{answer_str}' (Target: {target})")

accuracy = (correct_count / total_count) * 100
print("-" * 30)
print(f"Final Result: {correct_count}/{total_count} Correct")
print(f"Accuracy: {accuracy:.2f}%")
