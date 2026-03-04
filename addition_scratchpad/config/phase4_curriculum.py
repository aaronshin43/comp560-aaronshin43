# Phase 4a: Curriculum Learning - 1~4-digit scratchpad
# Train: 1~2-digit exhaustive + 3-digit (5,000) + 4-digit (5,000) = ~20,100 samples
# Goal: model learns bracket count = f(operand length), enabling depth generalization
#
# Key change from Phase 2: adds 3~4-digit scratchpad examples to training mix.
# max_new_tokens=100: 4-digit scratchpad output is ~70 chars; headroom for 5-digit OOD.

out_dir = 'out/phase4_curriculum'
eval_interval = 500
eval_iters = 100
always_save_checkpoint = False

wandb_log = True
wandb_project = 'scratchpad_generalization'
wandb_run_name = 'phase4a-curriculum-1_4digit'

dataset = 'scratchpad_1_4digit'
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
max_new_tokens = 100   # 4-digit scratchpad ~70 chars; extra room for 5-digit stretch
seed           = 1337
