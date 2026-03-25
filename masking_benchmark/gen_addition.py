"""
Generate the 2-digit addition dataset in JSONL format.

Produces an exhaustive set of all 10,000 combinations (0+0 to 99+99)
and writes them to data/addition_2digit/addition_2digit.jsonl.

Run from masking_benchmark/:
    python gen_addition.py
"""

import json
import os


def generate_addition_dataset(filename, num_digits=2):
    """
    Generate all (a, b) pairs for num_digits-digit addition and save as JSONL.
    For num_digits=2: 100 * 100 = 10,000 samples.
    """
    limit = 10 ** num_digits
    data = [{"input": f"{a}+{b}", "output": str(a + b)}
            for a in range(limit)
            for b in range(limit)]

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')

    print(f"Saved {len(data)} samples to {filename}")


if __name__ == '__main__':
    generate_addition_dataset('data/addition_2digit/addition_2digit.jsonl', num_digits=2)
