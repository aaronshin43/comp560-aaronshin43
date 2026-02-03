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
# Uses default EOS ("\n") - Recommended for most cases
python prepare.py --file path/to/dataset.jsonl --out_dir data/my_experiment

# Uses custom EOS (e.g. ";") - Careful with shell escaping!
python prepare.py --file path/to/dataset.jsonl --out_dir data/my_experiment --eos ";"
```

**Arguments:**
- `--file`: Path to input JSONL file.
- `--out_dir`: Output directory for artifacts (`train.bin`, `meta.pkl`, etc).
- `--sep`: Separator between input and output (default: `"="`).
- `--eos`: End-of-sequence token (default: `"\n"`). **Note:** If you use a custom EOS here, you must also update `stop_token` in your `config.py` later.
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
Test the model with a prompt
> **Important:** If you used a custom `--eos` during preparation (e.g., `";"`), make sure to update `stop_token = ';'` in your `config/your_experiment.py` before sampling.

```bash
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python ../../comp560-nanoGPT/sample.py config/your_experiment.py --start="YourPrompt="
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

2.  **Prepare Data (Protocol Encoding):**
    ```bash
    python prepare.py --file data/addition_2digit/addition_2digit.jsonl --out_dir data/addition_2digit
    ``

4.  **Train:**
    ```bash
    NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python ../../comp560-nanoGPT/train.py config/addition_2digit.py
    ```

5. **Sample:**
    ```bash
    NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python ../../comp560-nanoGPT/sample.py config/addition_2digit.py
    ```

## Todo
- [ ] Implement evaluation script 