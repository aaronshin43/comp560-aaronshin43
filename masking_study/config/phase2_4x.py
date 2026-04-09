# Exp 7 - Masking Study, Phase 2
# 2-digit scratchpad, input repeated 4x (input fraction ~48%)
# Format: "12+34=12+34=12+34=12+34=[2+4=6,C0][1+3=4,C0]46\n"
#
# Shared base config for conditions S (no mask) and T (target mask).
# Override per-run with CLI flags, e.g.:
#   --target_mask=True --seed=1337 --out_dir=out/cond_T_s1
#   --wandb_run_name=cond_T_s1

out_dir = 'out/cond_S_s1'
eval_interval = 500
eval_iters = 100
always_save_checkpoint = True

# Weights & Biases logging
wandb_log = True
wandb_project = 'masking_study'
wandb_run_name = 'cond_S_s1'    # override per run: cond_S_s1, cond_S_s2, cond_T_s1, ...

dataset = 'phase2_4x'
gradient_accumulation_steps = 1
batch_size = 128
block_size = 64                 # 4x max sequence: ~55 chars

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
target_mask = False             # override with --target_mask=True for condition T
max_new_tokens = 35             # scratchpad 2-digit max output: ~30 chars

seed = 1337                     # override with --seed=... for each replicate
