"""
Prepare the 2-digit addition dataset.
Examples: "12+34=46", "99+01=100", etc.
"""
import os
import pickle
import random
import numpy as np

# 1. Generate Data
# Range: 00-99 + 00-99
# We use zero-padding to keep length consistent (optional, but helps simple models)
# Format: "xx+yy=zzz\n" or "xx+yy=zz\n"
# Let's stick to standard integer format: "12+5=17"
# But considering tokenization, consistency helps. Let's do pure random sampling first.

data = []
# Total combinations: 100 * 100 = 10,000 samples.
# This is small enough to use ALL of them.
for a in range(100):
    for b in range(100):
        c = a + b
        s = f"{a}+{b}={c}\n"
        data.append(s)

# Shuffle
random.seed(42)
random.shuffle(data)

# Join
train_data = "".join(data)
# Use the same data for val to test memorization/capacity first
val_data = train_data 

print(f"Total samples: {len(data)}")
print("First 20 lines of data:")
lines = train_data.split('\n')
for i in range(min(20, len(lines))):
    print(lines[i])

print(f"length of dataset in characters: {len(train_data):,}")

# 2. Vocab
chars = sorted(list(set(train_data)))
vocab_size = len(chars)
print(f"Unique characters: {''.join(chars)}")
print(f"Vocab size: {vocab_size}")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

# 3. Save
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

output_dir = os.path.dirname(__file__)
train_ids.tofile(os.path.join(output_dir, 'train.bin'))
val_ids.tofile(os.path.join(output_dir, 'val.bin'))

meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(output_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)
