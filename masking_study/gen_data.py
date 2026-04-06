"""
Generate Phase 1 datasets for Exp 7 (Masking Study).

Produces 4 dataset variants, each with 10,000 randomly sampled addition pairs:
  - plain_3digit:       100-999  + 100-999,  format: ABC+DEF=GHIJ
  - plain_4digit:       1000-9999 + 1000-9999, format: ABCD+EFGH=IJKLM
  - scratchpad_3digit:  100-999  + 100-999,  scratchpad bracket format
  - scratchpad_4digit:  1000-9999 + 1000-9999, scratchpad bracket format

Each JSONL line: {"input": "<a>+<b>", "output": "<result>"}
90/10 train/val split applied by prepare.py.

Run from masking_study/:
    python gen_data.py
"""

import json
import os
import random
import sys

# ---------------------------------------------------------------------------
# Scratchpad builder (adapted from masking_benchmark/gen_scratchpad.py)
# ---------------------------------------------------------------------------

def build_scratchpad(a: int, b: int) -> str:
    """
    Build the full scratchpad output string for a + b.

    Examples:
        build_scratchpad(12, 34)  -> "[2+4=6,C0][1+3=4,C0]46"
        build_scratchpad(58, 47)  -> "[8+7=15,C1][5+4+1=10,C1][1]105"
        build_scratchpad(100, 200) -> "[0+0=0,C0][0+0=0,C0][1+2=3,C0]300"
    """
    a_str = str(a)
    b_str = str(b)
    max_len = max(len(a_str), len(b_str), 1)

    a_digits = [int(c) for c in reversed(a_str.zfill(max_len))]
    b_digits = [int(c) for c in reversed(b_str.zfill(max_len))]

    brackets = []
    carry = 0

    for i in range(max_len):
        da, db = a_digits[i], b_digits[i]
        s = da + db + carry
        new_carry = s // 10
        if carry > 0:
            brackets.append(f"[{da}+{db}+{carry}={s},C{new_carry}]")
        else:
            brackets.append(f"[{da}+{db}={s},C{new_carry}]")
        carry = new_carry

    if carry > 0:
        brackets.append(f"[{carry}]")

    return "".join(brackets) + str(a + b)


# ---------------------------------------------------------------------------
# Dataset generators
# ---------------------------------------------------------------------------

def generate_plain(lo: int, hi: int, n: int, seed: int = 42) -> list:
    """
    Sample n pairs from [lo, hi] x [lo, hi] without replacement (if feasible).
    Returns list of {"input": "a+b", "output": "sum"} dicts.
    """
    rng = random.Random(seed)
    total_pairs = (hi - lo + 1) ** 2

    if total_pairs <= n * 10:
        # Small enough: enumerate all pairs and sample without replacement
        pairs = [(a, b) for a in range(lo, hi + 1) for b in range(lo, hi + 1)]
        chosen = rng.sample(pairs, min(n, len(pairs)))
    else:
        # Large space: sample with replacement (collision probability negligible)
        seen = set()
        chosen = []
        attempts = 0
        max_attempts = n * 3
        while len(chosen) < n and attempts < max_attempts:
            a = rng.randint(lo, hi)
            b = rng.randint(lo, hi)
            key = (a, b)
            if key not in seen:
                seen.add(key)
                chosen.append(key)
            attempts += 1
        # Fill remaining with replacement if needed
        while len(chosen) < n:
            a = rng.randint(lo, hi)
            b = rng.randint(lo, hi)
            chosen.append((a, b))

    return [{"input": f"{a}+{b}", "output": str(a + b)} for a, b in chosen]


def generate_scratchpad(lo: int, hi: int, n: int, seed: int = 42) -> list:
    """
    Sample n pairs from [lo, hi] x [lo, hi] and build scratchpad sequences.
    Returns list of {"input": "a+b", "output": "<scratchpad>"} dicts.
    """
    rng = random.Random(seed)
    total_pairs = (hi - lo + 1) ** 2

    if total_pairs <= n * 10:
        pairs = [(a, b) for a in range(lo, hi + 1) for b in range(lo, hi + 1)]
        chosen = rng.sample(pairs, min(n, len(pairs)))
    else:
        seen = set()
        chosen = []
        attempts = 0
        max_attempts = n * 3
        while len(chosen) < n and attempts < max_attempts:
            a = rng.randint(lo, hi)
            b = rng.randint(lo, hi)
            key = (a, b)
            if key not in seen:
                seen.add(key)
                chosen.append(key)
            attempts += 1
        while len(chosen) < n:
            a = rng.randint(lo, hi)
            b = rng.randint(lo, hi)
            chosen.append((a, b))

    return [{"input": f"{a}+{b}", "output": build_scratchpad(a, b)} for a, b in chosen]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

VARIANTS = [
    {
        "name": "plain_3digit",
        "lo": 100,
        "hi": 999,
        "n": 10_000,
        "generator": "plain",
    },
    {
        "name": "plain_4digit",
        "lo": 1000,
        "hi": 9999,
        "n": 10_000,
        "generator": "plain",
    },
    {
        "name": "scratchpad_3digit",
        "lo": 100,
        "hi": 999,
        "n": 10_000,
        "generator": "scratchpad",
    },
    {
        "name": "scratchpad_4digit",
        "lo": 1000,
        "hi": 9999,
        "n": 10_000,
        "generator": "scratchpad",
    },
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    for v in VARIANTS:
        name = v["name"]
        lo, hi, n = v["lo"], v["hi"], v["n"]
        out_dir = os.path.join(BASE_DIR, "data", name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{name}.jsonl")

        print(f"Generating {name}: {n} samples from [{lo}, {hi}] x [{lo}, {hi}]...")

        if v["generator"] == "plain":
            data = generate_plain(lo, hi, n)
        else:
            data = generate_scratchpad(lo, hi, n)

        with open(out_path, "w", encoding="utf-8") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")

        print(f"  Saved {len(data)} samples -> {out_path}")

    print("\nAll variants generated.")


if __name__ == "__main__":
    main()
