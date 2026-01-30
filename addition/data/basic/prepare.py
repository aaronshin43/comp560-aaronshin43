import os
import pickle
import random
import numpy as np

# 1. Generate Data
# We want to teach the model single digit addition: "a+b=c"
# Range: 0-9
data = []
for a in range(10):
    for b in range(10):
        c = a + b
        # Format: "a+b=c\n"
        # \n acts as the 'end of sequence' token
        s = f"{a}+{b}={c}\n"
        data.append(s)

# Shuffle the data
random.seed(42)
random.shuffle(data)

# Join into a single string for character-level processing
train_data = "".join(data)
# For this basic experiment (rote memorization), we use the SAME data for train and val
# to ensure the model sees every possible combination (0+0 to 9+9).
# We want to check if it can memorize the addition table perfectly.
val_data = train_data

print("First 20 lines of data:")
lines = train_data.split('\n')
for i in range(min(20, len(lines))):
    print(lines[i])

print(f"length of dataset in characters: {len(train_data):,}")

# 2. Build Vocabulary
chars = sorted(list(set(train_data)))
vocab_size = len(chars)
print(f"Unique characters: {''.join(chars)}")
print(f"Vocab size: {vocab_size}")

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

print(f"Train size: {len(train_data)}")
print(f"Val size: {len(val_data)}")

train_ids = encode(train_data)
val_ids = encode(val_data)

# 4. Save to binary
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

# Ensure directory exists
output_dir = os.path.dirname(__file__)
train_ids.tofile(os.path.join(output_dir, 'train.bin'))
val_ids.tofile(os.path.join(output_dir, 'val.bin'))

# 5. Save Meta
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(output_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)