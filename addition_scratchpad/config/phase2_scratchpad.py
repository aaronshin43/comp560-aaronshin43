# Phase 2: Scratchpad - teach the carry algorithm on 1~2-digit problems
# Train: 1-digit + 2-digit scratchpad pairs (~10,100 samples, ~9,090 train / ~1,010 val)
# Goal: model learns bracket-based carry reasoning on short inputs
#
# Phase 3 reuses this checkpoint directly on 3~4-digit OOD data (no extra training).
#
# block_size=128: accommodates 4-digit scratchpad at inference (~70 chars).
# max_iters=10000: 2x Phase 1 to compensate for longer sequence length per sample.

out_dir = 'out/phase2_scratchpad'
eval_interval = 500
eval_iters = 100
always_save_checkpoint = False

wandb_log = True
wandb_project = 'scratchpad_generalization'
wandb_run_name = 'phase2-scratchpad-1_2digit'

dataset = 'scratchpad_1_2digit'
gradient_accumulation_steps = 1
batch_size = 128
block_size = 128  # CRITICAL: must fit 4-digit scratchpad (~70 chars) for Phase 3 eval

# Identical architecture to phase1_plain - differences come from data format only
n_layer = 6
n_head  = 4
n_embd  = 128     # n_embd % n_head == 0
dropout = 0.0

learning_rate  = 3e-4
max_iters      = 10000   # 2x phase1 - scratchpad sequences are 3-5x longer
lr_decay_iters = 10000
min_lr         = 3e-5
beta2          = 0.99
warmup_iters   = 400

compile = False

########################################################################
### sampling / generation params
start          = '\n'
stop_token     = '\n'
num_samples    = 1
max_new_tokens = 80   # 4-digit scratchpad output: ~70 chars
seed           = 1337
