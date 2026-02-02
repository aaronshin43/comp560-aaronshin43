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
python prepare.py --file path/to/dataset.jsonl --out_dir data/my_experiment --sep "=" --eos "\n"
```

**Arguments:**
- `--file`: Path to input JSONL file.
- `--out_dir`: Output directory for artifacts (`train.bin`, `meta.pkl`, etc).
- `--sep`: Separator between input and output.
- `--eos`: End-of-sequence token.
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

### 3. Sampling (Inference)
Test the model with a prompt.

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
    python examples/gen_addition.py
    ```

2.  **Prepare Data (Protocol Encoding):**
    ```bash
    python prepare.py --file examples/addition_intermediate.jsonl --out_dir data/addition_intermediate
    ```

3.  **Change Directory (Just for Quickstart):**
    ```bash
    cd examples
    ```

4.  **Configure:**
    Update `config.py`:
    - `dataset = 'addition_intermediate'`
    - `out_dir = 'out/addition_intermediate'`

5.  **Train:**
    ```bash
    NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python ../../comp560-nanoGPT/train.py config.py
    ```

6. **Sample:**
    ```bash
    NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python ../../comp560-nanoGPT/sample.py config/addition_intermediate.py --start="12+34="
    ```

## Todo
- [ ] Implement evaluation script 