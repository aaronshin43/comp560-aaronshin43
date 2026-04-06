"""
Generic Data Preparation Script for NanoGPT Experiments.

This script reads a JSONL dataset, tokenizes it at the character level,
and saves the training and validation data in binary format (.bin).
It also generates a meta.pkl file containing the vocabulary.

Protocol:
- Input file must be JSONL with 'input' and 'output' fields.
- Formats each sample as: {input}{sep}{output}{stop_token}

Usage (from masking_benchmark/):
    python prepare.py --file data/addition_2digit/addition_2digit.jsonl \
        --out_dir data/addition_2digit --shuffle
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
            if not line.strip():
                continue
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
    samples_str = [f"{s['input']}{args.sep}{s['output']}{args.stop_token}" for s in dataset]
    raw_data = "".join(samples_str)
    print(f"Total characters in dataset: {len(raw_data)}")

    # 3. Build Vocabulary (Character-level)
    chars = sorted(list(set(raw_data)))
    vocab_size = len(chars)
    print(f"Unique characters: {vocab_size}")
    print(f"Vocab: {''.join(chars)}")

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    # 4. Shuffle Data
    if args.shuffle:
        combined = list(zip(dataset, samples_str))
        random.shuffle(combined)
        dataset, samples_str = zip(*combined)
        dataset = list(dataset)
        samples_str = list(samples_str)

    # 5. Split Train/Val
    if args.test_size > 0:
        num_val = int(len(samples_str) * args.test_size)
        if num_val == 0 and len(samples_str) > 1:
            num_val = 1
            print("Warning: Dataset is very small. Forcing 1 validation sample.")

        if num_val == 0:
            train_samples, val_samples = samples_str, []
            train_dataset, val_dataset = dataset, []
        else:
            train_samples = samples_str[:-num_val]
            val_samples   = samples_str[-num_val:]
            train_dataset = dataset[:-num_val]
            val_dataset   = dataset[-num_val:]

        print(f"Split: {len(train_samples)} training samples, {len(val_samples)} validation samples.")
    else:
        train_samples = val_samples = samples_str
        train_dataset = val_dataset = dataset
        print(f"Split: Using full dataset ({len(train_samples)} samples) for both Train and Val (Memorization).")

    train_data = "".join(train_samples)
    val_data   = "".join(val_samples)

    train_ids = [stoi[c] for c in train_data]
    val_ids   = [stoi[c] for c in val_data]

    print(f"Train tokens: {len(train_ids)}")
    print(f"Val tokens:   {len(val_ids)}")

    # 6. Save Artifacts
    os.makedirs(args.out_dir, exist_ok=True)

    meta = {'vocab_size': vocab_size, 'itos': itos, 'stoi': stoi}
    meta_path = os.path.join(args.out_dir, 'meta.pkl')
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)
    print(f"Saved meta.pkl to {meta_path}")

    np.array(train_ids, dtype=np.uint16).tofile(os.path.join(args.out_dir, 'train.bin'))
    np.array(val_ids,   dtype=np.uint16).tofile(os.path.join(args.out_dir, 'val.bin'))
    print(f"Saved train.bin and val.bin to {args.out_dir}")

    # Save raw JSONL splits for evaluation
    with open(os.path.join(args.out_dir, 'train.jsonl'), 'w', encoding='utf-8') as f:
        for item in train_dataset:
            f.write(json.dumps(item) + '\n')

    with open(os.path.join(args.out_dir, 'val.jsonl'), 'w', encoding='utf-8') as f:
        for item in val_dataset:
            f.write(json.dumps(item) + '\n')

    print(f"Saved train.jsonl and val.jsonl to {args.out_dir}")


if __name__ == '__main__':
    main()
