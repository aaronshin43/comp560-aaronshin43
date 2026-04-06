# Exp 7 - Masking Study, Phase 1
# 3-digit scratchpad addition (100+100 to 999+999)
# Format: "123+456=[3+6+0=9,C0][2+5+0=7,C0][1+4+0=5,C0]579\n"
#
# Shared base config for conditions I (no mask) and J (target mask).
# Override per-run with CLI flags, e.g.:
#   --target_mask=True --seed=1337 --out_dir=out/cond_J_s1
#   --wandb_run_name=cond_J_s1

out_dir = 'out/cond_I_s1'
eval_interval = 500
eval_iters = 100
always_save_checkpoint = True

# Weights & Biases logging
wandb_log = True
wandb_project = 'masking_study'
wandb_run_name = 'cond_I_s1'    # override per run: cond_I_s1, cond_I_s2, cond_J_s1, ...

dataset = 'scratchpad_3digit'
gradient_accumulation_steps = 1
batch_size = 128
block_size = 64                 # scratchpad 3-digit max sequence: ~49 chars

# Model - same as Exp 6
n_layer = 6
n_head = 4
n_embd = 128
dropout = 0.0

learning_rate = 3e-4
max_iters = 10000
lr_decay_iters = 10000
min_lr = 3e-5
beta2 = 0.99

warmup_iters = 200

compile = False

# comp560 benchmark settings
separator_token = '='
stop_token = '\n'
enable_tf_eval = False
target_mask = False             # override with --target_mask=True for condition J
max_new_tokens = 50             # scratchpad 3-digit max output: ~41 chars

seed = 1337                     # override with --seed=... for each replicate
