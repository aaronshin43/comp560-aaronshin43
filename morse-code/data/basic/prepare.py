"""
Based closely on Karpathy Shakespeare character-level prepare.py.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import random
import numpy as np


building_blocks = [
    'HELP: .... . .-.. .--.',
    'SOS: ... --- ...',
    'AI: .- ..',
    'CODE: -.-. --- -.. .',
    'CAT: -.-. .- -',
    'DOG: -.. --- --.',
    'HELLO: .... . .-.. .-.. ---',
    'WORLD: .-- --- .-. .-.. -..',
    'LOVE: .-.. --- ...- .',
    'SKY: ... -.- -.--',
    'STAR: ... - .- .-.',
    'SUN: ... ..- -.',
    'SEA: ... . .-',
    'HI: .... ..',
    'BYE: -... -.-- .',
]

target_length = 1_000_000  # about 1MB will be plenty of training data

# construct a string consisting of repeated copies of the building_blocks in random order, 
# such that it has length at least target_length
total_length = 0
blocks = []
while total_length < target_length:
    new_block = random.choice(building_blocks)
    blocks.append(new_block)
    total_length += len(new_block)
data = '\n'.join(blocks)

print("First 20 lines of data:")
lines = data.split('\n')
for i in range(20):
    print(lines[i])

print(f"length of dataset in characters: {len(data):,}")

# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print(f"all the unique characters: |{'|'.join(map(repr, chars))}|")
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    # encoder: take a string, output a list of integers
    return [stoi[c] for c in s]


def decode(l):
    # decoder: take a list of integers, output a string
    return ''.join([itos[i] for i in l])


# jmac: For this experimental example, training and validation data are the same.
n = len(data)
train_data = data
val_data = data

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)
