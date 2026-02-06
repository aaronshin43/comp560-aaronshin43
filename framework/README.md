# Framework for NanoGPT Experiments

This framework standardizes the data preparation process for "Short Input -> Short Output" experiments (e.g., arithmetic, logic gates, simple Q&A).

## Data Protocol (Input Format)

To evaluate a new task, provide a valid **JSONL (JSON Lines)** file.
Each line must be a valid JSON object containing `"input"` and `"output"` keys.

**Example `dataset.jsonl`:**
```json
{"input": "12+34", "output": "46"}
{"input": "5+5", "output": "10"}
```

## Workflow

This section describes the standard workflow to run experiments.
> **Note:** Ensure you are in the `framework` directory for all commands.

### 1. Data Preparation
Convert your JSONL dataset into binary format using `prepare.py`.

```bash
# Uses default stop_token ("\n") - Recommended for most cases
python prepare.py --file=path/to/dataset.jsonl --out_dir=data/my_experiment

# Uses custom stop_token (e.g. ";") - Careful with shell escaping!
python prepare.py --file=path/to/dataset.jsonl --out_dir=data/my_experiment --stop_token=";"
```

**Arguments:**
- `--file`: Path to input JSONL file.
- `--out_dir`: Output directory for artifacts (`train.bin`, `meta.pkl`, etc).
- `--sep`: Separator between input and output (default: `"="`).
- `--stop_token`: Token to stop generation (default: `"\n"`). **Note:** If you use a custom token here, you must also update `stop_token` in your `config.py` later.
- `--test_size`: Validation split ratio (default 0.1). Set to 0.0 for rote memorization.

**Output Files:**
The script generates the following in the `out_dir`:
- `train.bin`: Training data (uint16).
- `val.bin`: Validation data (uint16).
- `meta.pkl`: Pickled dictionary containing `stoi` (string-to-int) and `itos` (int-to-string), `sep`, and `eos`.

### 2. Training
Run training using the standard nanoGPT script. Ensure `config.py` points to your `out_dir`.

```bash
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python ../../comp560-nanoGPT/train.py config/your_experiment.py
```

### 3. Sampling
Test the model with a prompt. You can override `start` and `stop_token` from the command line.

> **Important:** If you used a custom `--stop_token` during preparation (e.g., `";"`), you must provide it here as well (or update your config file).

```bash
# Basic usage (uses defaults from config.py)
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python ../../comp560-nanoGPT/sample.py config/your_experiment.py

# Override prompt and stop_token
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python ../../comp560-nanoGPT/sample.py config/your_experiment.py --start="5+5=" --stop_token=";"
```

## Quick Start (Example: 2-Digit Addition)

Follow these steps to run the included addition example. 

0.  **Enter Framework Directory:**
    ```bash
    cd framework
    ```

1.  **Generate Data:**
    ```bash
    python gen_addition.py
    ```

2.  **Prepare Data:**
    ```bash
    python prepare.py --file data/addition_2digit/addition_2digit.jsonl --out_dir data/addition_2digit
    ```

4.  **Train:**
    ```bash
    NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python ../../comp560-nanoGPT/train.py config/addition_2digit.py
    ```

5. **Sample:**
    ```bash
    NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python ../../comp560-nanoGPT/sample.py config/addition_2digit.py
    ```

## Todo
- [ ] If we want to let users sample continuous output without `stop_token`, we want to initialize `stop_token=None`. But if we do this, in `configurator.py` on line 42, the type check throws AssertionError when user overrides `stop_token` in command-line. The easiest approach is either having default `stop_token` value such as `\n` or forcing users only to override `stop_token` in their config file. 
- [ ] Research about how train/loss val/loss is evaluated (Check if inputs are considered as well)
- [ ] If input are considered, figure out way to only evaluate on outputs 
- [ ] Implement evaluation script 