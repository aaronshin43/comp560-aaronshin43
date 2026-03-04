# 0. Create data
```bash
python gen_addition.py
```

# Phase 1
```bash
# prepare data
python prepare.py --file data/plain_1_2digit/plain_1_2digit.jsonl --shuffle --test_size=0.1

# train
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python ../../comp560-nanoGPT/train_benchmark.py config/phase1_plain.py --target_mask=True --enable_tf_eval=True --benchmark_target=both

# evlauate
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python ../../comp560-nanoGPT/eval_scratchpad.py config/phase1_plain.py --eval_file=data/plain_1_2digit/val.jsonl

NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python ../../comp560-nanoGPT/eval_scratchpad.py config/phase1_plain.py --eval_file=data/plain_3_4digit/3digit.jsonl

NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python ../../comp560-nanoGPT/eval_scratchpad.py config/phase1_plain.py --eval_file=data/plain_3_4digit/4digit.jsonl

# sample
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python ../../comp560-nanoGPT/sample.py config/phase1_plain.py --start='12+45='
```

# Phase 2
```bash
# prepare data
python prepare.py --file data/scratchpad_1_2digit/scratchpad_1_2digit.jsonl --shuffle --test_size=0.1

# train
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python ../../comp560-nanoGPT/train_benchmark.py config/phase2_scratchpad.py --target_mask=True --enable_tf_eval=True --benchmark_target=both

# evaluate
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python ../../comp560-nanoGPT/eval_scratchpad.py config/phase2_scratchpad.py --eval_file=data/scratchpad_1_2digit/val.jsonl

# sample
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python ../../comp560-nanoGPT/sample.py config/phase2_scratchpad.py --start='12+45='
```

# Phase 3 (No training)
```bash
# evaluate
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python ../../comp560-nanoGPT/eval_scratchpad.py config/phase2_scratchpad.py --eval_file=data/plain_3_4digit/3digit.jsonl

NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python ../../comp560-nanoGPT/eval_scratchpad.py config/phase2_scratchpad.py --eval_file=data/plain_3_4digit/4digit.jsonl

# sample
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python ../../comp560-nanoGPT/sample.py config/phase2_scratchpad.py --start='123+456='
```

# Phase 4a
```bash
# prepare data
python prepare.py --file data/scratchpad_1_4digit/scratchpad_1_4digit.jsonl --shuffle --test_size=0.1

# train
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python ../../comp560-nanoGPT/train_benchmark.py config/phase4_curriculum.py --target_mask=True --enable_tf_eval=True --benchmark_target=both

# evlauate
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python ../../comp560-nanoGPT/eval_scratchpad.py config/phase4_curriculum.py --eval_file=data/scratchpad_1_4digit/val.jsonl

NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python ../../comp560-nanoGPT/eval_scratchpad.py config/phase4_curriculum.py --eval_file=data/plain_3_4digit/3digit.jsonl

NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python ../../comp560-nanoGPT/eval_scratchpad.py config/phase4_curriculum.py --eval_file=data/plain_3_4digit/4digit.jsonl

NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python ../../comp560-nanoGPT/eval_scratchpad.py config/phase4_curriculum.py --eval_file=data/plain_5digit/5digit.jsonl

# sample
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python ../../comp560-nanoGPT/sample.py config/phase4_curriculum.py --start='1234+5678='
```

# Phase 4b min
```bash
# prepare data
python prepare.py --file data/scratchpad_1_4digit_min/scratchpad_1_4digit_min.jsonl --shuffle --test_size=0.1

# train
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python ../../comp560-nanoGPT/train_benchmark.py config/phase4_min.py --target_mask=True --enable_tf_eval=True --benchmark_target=both

# evlauate
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python ../../comp560-nanoGPT/eval_scratchpad.py config/phase4_min.py --eval_file=data/scratchpad_1_4digit_min/val.jsonl

NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python ../../comp560-nanoGPT/eval_scratchpad.py config/phase4_min.py --eval_file=data/plain_3_4digit/3digit.jsonl

NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python ../../comp560-nanoGPT/eval_scratchpad.py config/phase4_min.py --eval_file=data/plain_3_4digit/4digit.jsonl
```

# Phase 4b mid
```bash
# prepare data
python prepare.py --file data/scratchpad_1_4digit_mid/scratchpad_1_4digit_mid.jsonl --shuffle --test_size=0.1

# train
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python ../../comp560-nanoGPT/train_benchmark.py config/phase4_mid.py --target_mask=True --enable_tf_eval=True --benchmark_target=both

# evlauate
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python ../../comp560-nanoGPT/eval_scratchpad.py config/phase4_mid.py --eval_file=data/scratchpad_1_4digit_mid/val.jsonl

NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python ../../comp560-nanoGPT/eval_scratchpad.py config/phase4_mid.py --eval_file=data/plain_3_4digit/3digit.jsonl

NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python ../../comp560-nanoGPT/eval_scratchpad.py config/phase4_mid.py --eval_file=data/plain_3_4digit/4digit.jsonl
```