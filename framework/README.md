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

# Uses custom stop_token (e.g. ";") - Careful with shell escaping! (See Tip below in Sampling section)
python prepare.py --file=path/to/dataset.jsonl --out_dir=data/my_experiment --stop_token=";"
```

**Arguments:**
- `--file`: Path to input JSONL file.
- `--out_dir`: Output directory for artifacts (`train.bin`, `meta.pkl`, etc).
- `--sep`: Separator between input and output (default: `"="`).
- `--stop_token`: Token indicating end of a sample (default: `"\n"`). **Note:** This token is used to mark the end of each sample in the dataset.
- `--test_size`: Validation split ratio (default 0.1). Set to 0.0 for rote memorization.

**Output Files:**
The script generates the following in the `out_dir`:
- `train.bin`: Training data (uint16).
- `val.bin`: Validation data (uint16).
- `meta.pkl`: Pickled dictionary containing `stoi` (string-to-int) and `itos` (int-to-string).

### 2. Training
Run training using the standard nanoGPT script. Ensure `config.py` points to your `out_dir`.

```bash
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python ../../comp560-nanoGPT/train.py config/your_experiment.py
```

### 3. Sampling
Test the model with a prompt. You can override `start` and `stop_token` from the command line.

> **Important:** The default `stop_token` is empty. If you want the model to stop generating at a specific token (e.g., your custom EOS `;`), enable it by passing `--stop_token=";"`.
>
> **Note on `start`:** The default `start` prompt is `"\n"`. If you are using a custom `stop_token` to delimit samples, you typically want generation to start *after* that delimiter. So, you should also override `start` to match your delimiter (e.g., `--start=";"`) or provide a specific prompt (`--start="1+1"`).

```bash
# Basic usage (no early stopping, generates until max tokens)
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python ../../comp560-nanoGPT/sample.py config/your_experiment.py

# Enable early stopping with custom token.
# Also updating 'start' to match the custom delimiter ';'.
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python ../../comp560-nanoGPT/sample.py config/your_experiment.py --start=";" --stop_token=";"
```

> **Tip (Shell Escaping):** Passing special characters like newline (`\n`) via command line requires specific escaping depending on your shell.
> *   **Git Bash (Windows):** Use `--stop_token=$'\n'`
> *   **PowerShell:** Use `` --stop_token="`n" `` (backtick n)

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

3.  **Train:**
    ```bash
    NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python ../../comp560-nanoGPT/train.py config/addition_2digit.py
    ```

4. **Sample:**
    ```bash
    NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python ../../comp560-nanoGPT/sample.py config/addition_2digit.py
    ```

5. **Sample with Early Stopping:**
    ```bash
    NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python ../../comp560-nanoGPT/sample.py config/addition_2digit.py --start="12+34" --stop_token=$'\n'
    ```