# train a miniature character-level model
# for single digit addition
# This will be good for testing simple prompt-response behavior

# out_dir must match the directory where you want checkpoints saved
out_dir = 'out/your-experiment'
eval_interval = 50 
eval_iters = 20
log_interval = 1 

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

# Weights & Biases logging
wandb_log = False # set to True to enable logging to wandb
wandb_project = 'your-wandb-project'
wandb_run_name = 'your-wandb-run-name'

dataset = 'your-experiment'
gradient_accumulation_steps = 1
batch_size = 32
block_size = 64

# very very small GPT model
n_layer = 3
n_head = 3
n_embd = 120  # need n_embd % n_head == 0
dropout = 0.0

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 2000
lr_decay_iters = 2000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 0 # 100 # not super necessary potentially

device = 'cpu'  # run on cpu only
compile = False # do not torch compile the model

########################################################################
### sampling-specific params
# init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')

# The starting string for generation (prompt)
start = ""
num_samples = 3
max_new_tokens = 100
seed = 1337
stop_token = '\n' # default stop token

# Attempt to load stop_token from meta.pkl if available
import os
import pickle
meta_path = os.path.join('data', dataset, 'meta.pkl')
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    if 'eos' in meta:
        stop_token = meta['eos']
        print(f"Loaded stop_token '{stop_token}' from {meta_path}")