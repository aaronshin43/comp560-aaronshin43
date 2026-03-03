# Phase 1: Baseline - plain 1~2-digit addition
# Train: 1-digit + 2-digit plain pairs (~10,100 samples, ~9,090 train / ~1,010 val)
# Goal: prove the plain model fails catastrophically on 3~4-digit OOD inputs

out_dir = 'out/phase1_plain'
eval_interval = 500
eval_iters = 100
always_save_checkpoint = False

wandb_log = True
wandb_project = 'scratchpad_generalization'
wandb_run_name = 'phase1-plain-1_2digit'

dataset = 'plain_1_2digit'
gradient_accumulation_steps = 1
batch_size = 128
block_size = 64   # max plain sequence: "99+99=198\n" = 10 chars; 64 is generous

# Same architecture as validation/ experiments for fair comparison
n_layer = 6
n_head  = 4
n_embd  = 128     # n_embd % n_head == 0
dropout = 0.0

learning_rate  = 3e-4
max_iters      = 5000
lr_decay_iters = 5000
min_lr         = 3e-5
beta2          = 0.99
warmup_iters   = 200

compile = False

########################################################################
### sampling / generation params
start          = '\n'
stop_token     = '\n'
num_samples    = 1
max_new_tokens = 5   # plain answer max: "199\n" = 4 chars
seed           = 1337
