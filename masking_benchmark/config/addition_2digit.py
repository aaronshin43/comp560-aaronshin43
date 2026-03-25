# Exp 6 - Target Masking Benchmark
# 2-digit addition (0+0 to 99+99)
# Format: "12+34=46\n"
#
# Shared base config for conditions A (no mask) and B (target mask).
# Override per-run with CLI flags, e.g.:
#   --target_mask=True --seed=1337 --out_dir=out/cond_B_s1
#   --wandb_run_name=cond_B_s1

out_dir = 'out/cond_A_s1'
eval_interval = 500
eval_iters = 100
always_save_checkpoint = True   # required: snapshots at every eval_interval for convergence curves

# Weights & Biases logging
wandb_log = True
wandb_project = 'masking_benchmark'
wandb_run_name = 'cond_A_s1'    # override per run: cond_A_s1, cond_A_s2, cond_B_s1, ...

dataset = 'addition_2digit'
gradient_accumulation_steps = 1
batch_size = 128
block_size = 64

# Model - same as Exp 4
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
enable_tf_eval = False      # disabled; accuracy collected post-hoc via eval_generation.py
target_mask = False         # override with --target_mask=True for condition B

seed = 1337                 # override with --seed=... for each replicate
