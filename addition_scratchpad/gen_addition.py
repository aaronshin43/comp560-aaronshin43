"""
Generate addition datasets for scratchpad length generalization experiments.

Outputs:
  data/plain_1_2digit/plain_1_2digit.jsonl       - Phase 1 & 2 training (plain)
  data/scratchpad_1_2digit/scratchpad_1_2digit.jsonl - Phase 2 training (scratchpad)
  data/plain_3_4digit/3digit.jsonl               - Phase 1 & 3 OOD test (3-digit)
  data/plain_3_4digit/4digit.jsonl               - Phase 1 & 3 OOD test (4-digit)

JSONL formats:
  Plain:      {"input": "15+27", "output": "42"}
  Scratchpad: {"input": "15+27", "output": "[5+7=12,C1][1+2+1=4,C0]42"}

Scratchpad bracket structure: [dA+dB(+carry)=sum,Cnew_carry]
  - Digits processed right-to-left (least significant first)
  - carry > 0 adds "+carry" inside bracket
  - Final overflow carry appended as [1]
"""

import json
import random
import os


def build_scratchpad(a: int, b: int) -> str:
    """
    Build the full scratchpad output string for a + b.

    Examples:
        build_scratchpad(15, 27) -> "[5+7=12,C1][1+2+1=4,C0]42"
        build_scratchpad(99, 99) -> "[9+9=18,C1][9+9+1=19,C1][1]198"
        build_scratchpad(3, 4)   -> "[3+4=7,C0]7"
        build_scratchpad(0, 0)   -> "[0+0=0,C0]0"
    """
    result = a + b

    # Determine max digit length (pad shorter number with leading zeros)
    a_str = str(a)
    b_str = str(b)
    max_len = max(len(a_str), len(b_str), 1)

    # Pad both to equal length
    a_padded = a_str.zfill(max_len)
    b_padded = b_str.zfill(max_len)

    # Process right-to-left (least significant digit first)
    a_digits = [int(c) for c in reversed(a_padded)]
    b_digits = [int(c) for c in reversed(b_padded)]

    brackets = []
    carry = 0

    for i in range(max_len):
        da = a_digits[i]
        db = b_digits[i]
        s = da + db + carry
        new_carry = s // 10

        if carry > 0:
            brackets.append(f"[{da}+{db}+{carry}={s},C{new_carry}]")
        else:
            brackets.append(f"[{da}+{db}={s},C{new_carry}]")

        carry = new_carry

    # Remaining carry becomes its own bracket
    if carry > 0:
        brackets.append(f"[{carry}]")

    return "".join(brackets) + str(result)


def generate_plain(out_dir: str, max_digits: int = 2) -> None:
    """
    Generate exhaustive plain addition dataset covering 1-digit AND max_digits-digit operands.

    1-digit:      a, b in [0..9]   -> 100 samples
    max_digits-digit: a, b in [0..99] -> 10,000 samples (for max_digits=2)
    Total: 10,100 samples (includes overlap at 0..9)
    """
    os.makedirs(out_dir, exist_ok=True)
    data = []

    # 1-digit pairs (always included for pattern diversity)
    if max_digits >= 2:
        print("Generating 1-digit plain pairs (0..9 x 0..9)...")
        for a in range(10):
            for b in range(10):
                data.append({"input": f"{a}+{b}", "output": str(a + b)})

    # max_digits-digit exhaustive pairs
    limit = 10 ** max_digits
    print(f"Generating {max_digits}-digit plain pairs (0..{limit - 1} x 0..{limit - 1})...")
    for a in range(limit):
        for b in range(limit):
            data.append({"input": f"{a}+{b}", "output": str(a + b)})

    fname = os.path.join(out_dir, f"plain_1_{max_digits}digit.jsonl")
    with open(fname, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")

    print(f"Saved {len(data)} plain samples to {fname}")


def generate_scratchpad(out_dir: str, max_digits: int = 2) -> None:
    """
    Generate exhaustive scratchpad addition dataset covering 1-digit AND max_digits-digit operands.

    Same sample pairs as generate_plain, but output contains the full bracket chain.
    Total: 10,100 samples (for max_digits=2).
    """
    os.makedirs(out_dir, exist_ok=True)
    data = []

    # 1-digit pairs
    if max_digits >= 2:
        print("Generating 1-digit scratchpad pairs (0..9 x 0..9)...")
        for a in range(10):
            for b in range(10):
                data.append({
                    "input":  f"{a}+{b}",
                    "output": build_scratchpad(a, b),
                })

    # max_digits-digit exhaustive pairs
    limit = 10 ** max_digits
    print(f"Generating {max_digits}-digit scratchpad pairs (0..{limit - 1} x 0..{limit - 1})...")
    for a in range(limit):
        for b in range(limit):
            data.append({
                "input":  f"{a}+{b}",
                "output": build_scratchpad(a, b),
            })

    fname = os.path.join(out_dir, f"scratchpad_1_{max_digits}digit.jsonl")
    with open(fname, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")

    print(f"Saved {len(data)} scratchpad samples to {fname}")


def generate_ood_plain(out_dir: str, digit_count: int, num_samples: int = 3000) -> None:
    """
    Generate random plain OOD (out-of-distribution) addition test samples.

    Args:
        out_dir:      Output directory.
        digit_count:  Exact digit count for both operands.
                      digit_count=3 -> a, b in [100..999]
                      digit_count=4 -> a, b in [1000..9999]
        num_samples:  Number of random samples to generate.
    """
    os.makedirs(out_dir, exist_ok=True)

    lo = 10 ** (digit_count - 1)
    hi = 10 ** digit_count - 1

    print(f"Generating {num_samples} random {digit_count}-digit OOD plain samples ({lo}..{hi})...")

    data = []
    for _ in range(num_samples):
        a = random.randint(lo, hi)
        b = random.randint(lo, hi)
        data.append({"input": f"{a}+{b}", "output": str(a + b)})

    fname = os.path.join(out_dir, f"{digit_count}digit.jsonl")
    with open(fname, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")

    print(f"Saved {len(data)} OOD samples to {fname}")


if __name__ == "__main__":
    # ── Phase 1 & 2 training data ─────────────────────────────────────────────
    generate_plain('data/plain_1_2digit')
    generate_scratchpad('data/scratchpad_1_2digit')

    # ── Phase 1 & 3 OOD test data ─────────────────────────────────────────────
    generate_ood_plain('data/plain_3_4digit', digit_count=3, num_samples=3000)
    generate_ood_plain('data/plain_3_4digit', digit_count=4, num_samples=3000)

    print("\nAll datasets generated.")
