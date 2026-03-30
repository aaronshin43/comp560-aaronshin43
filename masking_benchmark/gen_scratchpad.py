"""
Generate the 2-digit scratchpad addition dataset in JSONL format.

Produces an exhaustive set of all 10,000 combinations (0+0 to 99+99) with
bracket-based scratchpad outputs, and writes them to
data/scratchpad_1_2digit/scratchpad_1_2digit.jsonl.

Scratchpad format: {"input": "15+27", "output": "[5+7=12,C1][1+2+1=4,C0]42"}
Bracket structure: [dA+dB(+carry)=sum,Cnew_carry] processed right-to-left.

Run from masking_benchmark/:
    python gen_scratchpad.py
"""

import json
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


def generate_scratchpad_dataset(out_dir: str, num_digits: int = 2) -> None:
    """Generate exhaustive scratchpad dataset for num_digits-digit operands."""
    limit = 10 ** num_digits
    data = [{"input": f"{a}+{b}", "output": build_scratchpad(a, b)}
            for a in range(limit)
            for b in range(limit)]

    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"scratchpad_1_{num_digits}digit.jsonl")
    with open(fname, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')

    print(f"Saved {len(data)} samples to {fname}")


if __name__ == '__main__':
    generate_scratchpad_dataset('data/scratchpad_1_2digit', num_digits=2)
