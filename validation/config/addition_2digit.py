# train a miniature character-level model
# for 2-digit addition (0+0 to 99+99, max sequence: "99+99=198\n" = 10 chars)
# Dataset: 10,000 samples total, ~9,000 train / ~1,000 val

# out_dir must match the directory where you want checkpoints saved
out_dir = 'out/addition_2digit/test'
eval_interval = 500
eval_iters = 100   # enough iters for a stable loss estimate over ~1000 val samples

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

# Weights & Biases logging
wandb_log = True # set to True to enable logging to wandb
wandb_project = 'accuracy_benchmark'
wandb_run_name = 'test'

dataset = 'addition_2digit'
gradient_accumulation_steps = 1
batch_size = 128
block_size = 64

# small GPT model - 2-digit addition requires carry logic, so slightly deeper than 1-digit
n_layer = 6
n_head = 4
n_embd = 128  # need n_embd % n_head == 0
dropout = 0.0  # deterministic task; dropout hurts more than it helps here

learning_rate = 3e-4  # slightly conservative for stable convergence
max_iters = 3000
lr_decay_iters = 3000  # make equal to max_iters usually
min_lr = 3e-5  # learning_rate / 10
beta2 = 0.99   # good for small token-per-iter counts

warmup_iters = 200

# device = 'cpu'  # run on cpu only
compile = False # do not torch compile the model

########################################################################
### sampling-specific params
# The starting string for generation (prompt)
start = '\n' # or "1+1" or etc. Can also specify a file, use as: "FILE:prompt.txt"

# The token ID or string that stops generation.
stop_token = "\n"

num_samples = 1
max_new_tokens = 5
seed = 1337