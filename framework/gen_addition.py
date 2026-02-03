"""
Generate addition dataset in JSONL format.
"""
import json
import random
import os

def generate_addition_dataset(filename, num_digits=2, num_samples=None):
    """
    Generates addition problems (a + b = c).
    
    Args:
        filename: Output JSONL filepath.
        num_digits: Maximimum number of digits for operands (e.g. 2 means 0-99).
        num_samples: Number of samples to generate. If None, generates ALL possible combinations.
    """
    
    # Range limit: 2 digits -> 10^2 = 100 (0-99)
    limit = 10 ** num_digits
    
    data = []
    
    if num_samples is None:
        # Exhaustive generation
        print(f"Generating exhaustive dataset for {num_digits}-digit addition...")
        for a in range(limit):
            for b in range(limit):
                data.append({"input": f"{a}+{b}", "output": str(a+b)})
    else:
        # Random sampling
        print(f"Generating {num_samples} random samples for {num_digits}-digit addition...")
        for _ in range(num_samples):
            a = random.randint(0, limit - 1)
            b = random.randint(0, limit - 1)
            data.append({"input": f"{a}+{b}", "output": str(a+b)})

    # Save to JSONL
    with open(filename, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")
            
    print(f"Saved {len(data)} samples to {filename}")

if __name__ == "__main__":
    # 1. 1-digit Addition (exhaustive) -> 100 samples
    os.makedirs('data/addition_1digit', exist_ok=True)
    generate_addition_dataset('data/addition_1digit/addition_1digit.jsonl', num_digits=1, num_samples=None)
    
    # 2. 2-digit Addition (exhaustive) -> 10000 samples
    os.makedirs('data/addition_2digit', exist_ok=True)
    generate_addition_dataset('data/addition_2digit/addition_2digit.jsonl', num_digits=2, num_samples=None)

    # 3. 3-digit Random Addition (10000 random samples)
    os.makedirs('data/addition_3digit', exist_ok=True)
    generate_addition_dataset('data/addition_3digit/addition_3digit.jsonl', num_digits=3, num_samples=10000)
