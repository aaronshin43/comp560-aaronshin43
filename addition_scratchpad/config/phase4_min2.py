# Phase 4b: Minimum Data Ablation - 1~4-digit scratchpad (500 samples each for 3~4-digit)
# Train: 1~2-digit exhaustive + 3-digit (500) + 4-digit (500) = ~11,100 samples
# Goal: find minimum 3~4-digit exposure needed to learn that bracket count = operand length
#
# Run AFTER Phase 4a to compare full (5000+5000) vs minimal (500+500) curriculum.

out_dir = 'out/phase4_min2'
eval_interval = 500
eval_iters = 100
always_save_checkpoint = False

wandb_log = True
wandb_project = 'scratchpad_generalization'
wandb_run_name = 'phase4b-min-n500_2'

dataset = 'scratchpad_1_4digit_min2'
gradient_accumulation_steps = 1
batch_size = 128
block_size = 128

# Identical architecture to all previous phases
n_layer = 6
n_head  = 4
n_embd  = 128     # n_embd % n_head == 0
dropout = 0.0

learning_rate  = 3e-4
max_iters      = 10000
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
max_new_tokens = 100
seed           = 1337
