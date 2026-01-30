# train a miniature character-level model
# consisting of fixed strings consisting of a few characters in alphabetical order.
# This will be good for testing completions of different lengths.

out_dir = 'out/basic'
eval_interval = 50 # keep frequent because we'll overfit
eval_iters = 20
log_interval = 1 

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True # override via command line if you like
wandb_project = 'aaronshin-firstexperiment'
wandb_run_name = 'eng-to-morse'

dataset = 'basic'
gradient_accumulation_steps = 1
batch_size = 12
block_size = 64

# very very small GPT model
n_layer = 3
n_head = 3
n_embd = 120  # need n_embd % n_head == 0
dropout = 0.0

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 2000
lr_decay_iters = 2000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 0 # 100 # not super necessary potentially

# device = 'cpu'  # run on cpu only
compile = False # do not torch compile the model

########################################################################
### sampling-specific params
# init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
# start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
# num_samples = 10 # number of samples to draw
# max_new_tokens = 500 # number of tokens generated in each sample
# temperature = 1e-5 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
# top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
# seed = 1337