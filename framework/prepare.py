"""
Generic Data Preparation Script for NanoGPT Experiments.

This script reads a JSONL dataset, tokenizes it at the character level,
and saves the training and validation data in binary format (.bin).
It also generates a meta.pkl file containing the vocabulary.

Protocol:
- Input file must be JSONL with 'input' and 'output' fields.
- Formats each sample as: {input}{sep}{output}{stop_token}
"""

import os
import json
import pickle
import numpy as np
import argparse
import random

def main():
    parser = argparse.ArgumentParser(description="Prepare data for NanoGPT.")
    parser.add_argument('--file', type=str, required=True, help='Path to input JSONL file')
    parser.add_argument('--out_dir', type=str, help='Output directory for bin files (default: same as input file directory)')
    parser.add_argument('--sep', type=str, default='=', help='Separator between input and output')
    parser.add_argument('--stop_token', type=str, default='\n', help='Token indicating end of a sample (EOS)')
    parser.add_argument('--test_size', type=float, default=0.1, help='Fraction of data to use for validation (0.0 for no split/memorization)')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle data before splitting')

    args = parser.parse_args()

    # Determine output directory if not provided
    if args.out_dir is None:
        args.out_dir = os.path.dirname(args.file)
        if args.out_dir == '':
            args.out_dir = '.'
    
    # 1. Read Input Data
    print(f"Reading data from {args.file}...")
    dataset = []
    with open(args.file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                obj = json.loads(line)
                if 'input' not in obj or 'output' not in obj:
                    print(f"Skipping invalid line (missing keys): {line.strip()}")
                    continue
                dataset.append(obj)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON: {line.strip()}")
    
    print(f"Loaded {len(dataset)} samples.")

    # 2. Format Data
    # Construct the full string: input + sep + output + stop_token
    samples_str = []
    for sample in dataset:
        samples_str.append(f"{sample['input']}{args.sep}{sample['output']}{args.stop_token}")
    
    raw_data = "".join(samples_str)
    
    print(f"Total characters in dataset: {len(raw_data)}")

    # 3. Build Vocabulary (Character-level)
    chars = sorted(list(set(raw_data)))
    vocab_size = len(chars)
    print(f"Unique characters: {vocab_size}")
    print(f"Vocab: {''.join(chars)}")

    # Create mappings
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }

    # Shuffle Data
    if args.shuffle:
        random.shuffle(samples_str)

    # 4. Split Train/Val
    if args.test_size > 0:
        num_val = int(len(samples_str) * args.test_size)
        if num_val == 0 and len(samples_str) > 1:
             num_val = 1
             print(f"Warning: Dataset is very small. Forcing 1 validation sample.")
        
        # Slicing logic
        if num_val == 0: # Case where len=1 and test_size small
            train_samples = samples_str
            val_samples = [] # Should ideally error out or just be 0
        else:
            train_samples = samples_str[:-num_val]
            val_samples = samples_str[-num_val:]
            
        print(f"Split: {len(train_samples)} training samples, {len(val_samples)} validation samples.")
    else:
        # If test_size is 0, we use the FULL dataset for both train and val (Memorization task)
        train_samples = samples_str
        val_samples = samples_str
        print(f"Split: Using full dataset ({len(train_samples)} samples) for both Train and Val (Memorization).")

    train_data = "".join(train_samples)
    val_data = "".join(val_samples)

    # Encode finalized strings
    train_ids = [stoi[c] for c in train_data]
    val_ids = [stoi[c] for c in val_data]

    print(f"Train tokens: {len(train_ids)}")
    print(f"Val tokens: {len(val_ids)}")

    # 6. Save Artifacts
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # Save meta for tokenizer
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi
    }
    meta_path = os.path.join(args.out_dir, 'meta.pkl')
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)

    # Save bins
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    
    train_bin_path = os.path.join(args.out_dir, 'train.bin')
    val_bin_path = os.path.join(args.out_dir, 'val.bin')
    
    train_ids.tofile(train_bin_path)
    val_ids.tofile(val_bin_path)

if __name__ == '__main__':
    main()
