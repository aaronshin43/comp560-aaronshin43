# Exp 7 - Masking Study, Phase 1
# 4-digit plain addition (1000+1000 to 9999+9999)
# Format: "1234+5678=6912\n"
#
# Shared base config for conditions G (no mask) and H (target mask).
# Override per-run with CLI flags, e.g.:
#   --target_mask=True --seed=1337 --out_dir=out/cond_H_s1
#   --wandb_run_name=cond_H_s1

out_dir = 'out/cond_G_s1'
eval_interval = 500
eval_iters = 100
always_save_checkpoint = True

# Weights & Biases logging
wandb_log = True
wandb_project = 'masking_study'
wandb_run_name = 'cond_G_s1'    # override per run: cond_G_s1, cond_G_s2, cond_H_s1, ...

dataset = 'plain_4digit'
gradient_accumulation_steps = 1
batch_size = 128
block_size = 64                 # plain 4-digit max sequence: ~17 chars

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
target_mask = False             # override with --target_mask=True for condition H
max_new_tokens = 10             # plain 4-digit max output: 5 digits + \n

seed = 1337                     # override with --seed=... for each replicate
