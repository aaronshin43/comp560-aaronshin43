```bash
# Dataset prepare
python prepare.py --file data/addition_2digit/addition_2digit.jsonl --shuffle --test_size=0.2

# Teacher Forcing evaluation
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python ../../comp560-nanoGPT/train_benchmark.py config/addition_2digit.py --enable_tf_eval=True --benchmark_target="both" --tf_eval_max_samples=1000

NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python ../../comp560-nanoGPT/train_benchmark.py config/addition_2digit.py --enable_tf_eval=True --benchmark_target="both" --tf_eval_max_samples=1000 --target_mask=True

# Autoagressive evaluation
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python ../../comp560-nanoGPT/eval_gen.py config/addition_2digit.py --benchmark_target="both" --eval_max_samples=1000
```