# train a miniature character-level model
# for TWO digit addition (0-99)

out_dir = 'out/intermediate'
eval_interval = 200 # check every 200 iters
eval_iters = 20
log_interval = 10

always_save_checkpoint = False

wandb_log = True
wandb_project = 'shina-addition-intermediate'
wandb_run_name = '2digit-addition'

dataset = 'intermediate'
gradient_accumulation_steps = 1
batch_size = 32
block_size = 128 # increased block size for longer sequences

# Model config - slightly larger than basic to handle more patterns
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0

# 10,000 samples * ~8 chars = ~80,000 tokens.
# batch 32 * block 128 = ~4096 tokens per step? No, dependent on block_size utilization.
# Let's say we want to see the whole dataset ~50 times.
learning_rate = 1e-3
max_iters = 5000 
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100 

# device = 'cpu' # or 'cuda' if available
compile = False 

# -----------------------------------------------------------------------------
# Sampling config (for easy testing via basic.py style edit)
start = "55+77="
num_samples = 10
max_new_tokens = 5
seed = 1337
