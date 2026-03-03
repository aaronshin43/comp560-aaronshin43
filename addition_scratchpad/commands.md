# 0. Create data
```
python gen_addition.py
```

# Phase 1
```
python prepare.py --file data/plain_1_2digit/plain_1_2digit.jsonl --shuffle --test_size=0.1
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python ../../comp560-nanoGPT/train_benchmark.py config/phase1_plain.py --target_mask=True --enable_tf_eval=True --benchmark_target=both --tf_eval_max_samples=1000
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python ../../comp560-nanoGPT/eval_scratchpad.py config/phase1_plain.py --eval_file=data/plain_1_2digit/val.jsonl
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python ../../comp560-nanoGPT/eval_scratchpad.py config/phase1_plain.py --eval_file=data/plain_3_4digit/3digit.jsonl  # expect: ~0%
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python ../../comp560-nanoGPT/eval_scratchpad.py config/phase1_plain.py --eval_file=data/plain_3_4digit/4digit.jsonl  # expect: ~0%
```

# Phase 2
```
python prepare.py --file data/scratchpad_1_2digit/scratchpad_1_2digit.jsonl --shuffle --test_size=0.1
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python ../../comp560-nanoGPT/train_benchmark.py config/phase2_scratchpad.py --target_mask=True --enable_tf_eval=True --benchmark_target=both --tf_eval_max_samples=1000
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python ../../comp560-nanoGPT/eval_scratchpad.py config/phase2_scratchpad.py --eval_file=data/scratchpad_1_2digit/val.jsonl
```

# Phase 3 (No training)
```
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python ../../comp560-nanoGPT/eval_scratchpad.py config/phase2_scratchpad.py --eval_file=data/plain_3_4digit/3digit.jsonl
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python ../../comp560-nanoGPT/eval_scratchpad.py config/phase2_scratchpad.py --eval_file=data/plain_3_4digit/4digit.jsonl
```