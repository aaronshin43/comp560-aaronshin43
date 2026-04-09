"""
Generate datasets for Exp 7 (Masking Study).

Phase 1 -- 4 variants, each with 30,000 randomly sampled addition pairs:
  - plain_3digit:       100-999  + 100-999,  format: ABC+DEF=GHIJ
  - plain_4digit:       1000-9999 + 1000-9999, format: ABCD+EFGH=IJKLM
  - scratchpad_3digit:  100-999  + 100-999,  scratchpad bracket format
  - scratchpad_4digit:  1000-9999 + 1000-9999, scratchpad bracket format

Phase 2 -- 5 variants, each with 10,000 samples using 2-digit addition (10-99)
with the input equation repeated N times before the scratchpad output:
  - phase2_1x:  input repeated 1x  (e.g. "12+34")
  - phase2_2x:  input repeated 2x  (e.g. "12+34=12+34")
  - phase2_3x:  input repeated 3x
  - phase2_4x:  input repeated 4x
  - phase2_5x:  input repeated 5x

Each JSONL line: {"input": "<repeated_eq>", "output": "<scratchpad>"}
90/10 train/val split applied by prepare.py.

Run from masking_study/:
    python gen_data.py           # Phase 1 only
    python gen_data.py --phase2  # Phase 2 only
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


def generate_scratchpad_repeated(n: int, multiplier: int, seed: int = 42) -> list:
    """
    Sample n pairs from [10, 99] x [10, 99] and build Phase 2 sequences where
    the base equation 'a+b' is repeated `multiplier` times using '=' as separator.

    The input field contains the repeated equation:
      multiplier=1: "12+34"
      multiplier=2: "12+34=12+34"
      multiplier=3: "12+34=12+34=12+34"
      ...

    The output field is the scratchpad string WITHOUT a leading '=' (that is
    added by the training/eval pipeline when it joins input + sep + output).

    Returns list of {"input": "<repeated_eq>", "output": "<scratchpad>"} dicts.
    """
    lo, hi = 10, 99
    rng = random.Random(seed)
    total_pairs = (hi - lo + 1) ** 2  # 8100 pairs

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

    records = []
    for a, b in chosen:
        base_eq = f"{a}+{b}"
        repeated_input = "=".join([base_eq] * multiplier)
        output = build_scratchpad(a, b)
        records.append({"input": repeated_input, "output": output})
    return records


# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------

PHASE1_VARIANTS = [
    {
        "name": "plain_3digit",
        "lo": 100,
        "hi": 999,
        "n": 30_000,
        "generator": "plain",
    },
    {
        "name": "plain_4digit",
        "lo": 1000,
        "hi": 9999,
        "n": 30_000,
        "generator": "plain",
    },
    {
        "name": "scratchpad_3digit",
        "lo": 100,
        "hi": 999,
        "n": 30_000,
        "generator": "scratchpad",
    },
    {
        "name": "scratchpad_4digit",
        "lo": 1000,
        "hi": 9999,
        "n": 30_000,
        "generator": "scratchpad",
    },
]

PHASE2_VARIANTS = [
    {"name": "phase2_1x", "multiplier": 1, "n": 10_000},
    {"name": "phase2_2x", "multiplier": 2, "n": 10_000},
    {"name": "phase2_3x", "multiplier": 3, "n": 10_000},
    {"name": "phase2_4x", "multiplier": 4, "n": 10_000},
    {"name": "phase2_5x", "multiplier": 5, "n": 10_000},
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def run_phase1():
    for v in PHASE1_VARIANTS:
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

    print("\nPhase 1 variants generated.")


def run_phase2():
    for v in PHASE2_VARIANTS:
        name = v["name"]
        multiplier = v["multiplier"]
        n = v["n"]
        out_dir = os.path.join(BASE_DIR, "data", name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{name}.jsonl")

        print(f"Generating {name}: {n} samples, multiplier={multiplier}, range=[10, 99]...")

        data = generate_scratchpad_repeated(n, multiplier)

        with open(out_path, "w", encoding="utf-8") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")

        # Compute and report max sequence length:
        # full sequence = input + "=" + output + "\n"
        max_seq_len = max(len(d["input"]) + 1 + len(d["output"]) + 1 for d in data)
        print(f"  Saved {len(data)} samples -> {out_path}")
        print(f"  Max sequence length (input=sep=output=newline): {max_seq_len}")

    print("\nPhase 2 variants generated.")


def main():
    phase2_flag = "--phase2" in sys.argv

    if phase2_flag:
        run_phase2()
    else:
        run_phase1()


if __name__ == "__main__":
    main()
