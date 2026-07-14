# 2026 UR AI & Sustainability via System Prompt

## Setup Instructions

### Step 1: Install Dependencies
In the project root, create a Python virtual environment, activate it and install the required packages.
```bash
python -m venv venv
source ./venv/bin/activate
pip install -r ./requirements.txt
```

### Step 2: Ensure input data are present
The `data/` folder must be present. It contains the system prompts and user prompts required for the experiment. The script will fail if the input files are missing.

### Step 3: Adjust Experiment Parameters
In `main.py` you can adjust the following parameters:
|Parameter|Description|
|---|---|
|`EPOCH_COUNT_PER_MODEL_PROMPT_COMBINATION`|How many times shall the model repeat for the same combination of model, user prompt, and system prompt.|
|`MODEL_LIST`|A list of Ollama model slugs to use.|
|`USER_PROMPT_SRC_PATH`|Path to the user prompt source file.|
|`USER_PROMPT_COUNT`|How many user prompts to use. Must be smaller or equal to the total no. of lines in the source file.|
|`GENERAL_SYSTEM_PROMPT_SRC_PATH`|Path to the general case system prompt.|
|`SUSTAINABLE_SYSTEM_PROMPT_SRC_PATH`|Path to the sustainable case system prompt.|
|`IDLE_PERIOD_BETWEEN_RUNS_SECONDS`|How much time to idle between each model run.|
|`CODECARBON_LOG_LEVEL`|CodeCarbon log level. Default is `error`.|

### Step 4: Pull models from Ollama
Pull the desired models before running the experiment or Ollama will return a not found error:
```bash
ollama pull <MODEL_SLUG>
```

To delete an installed model, use:
```bash
ollama rm <MODEL_SLUG>
```

### Step 5: Execute Script and Begin Experiment
Execute the main program `main.py`:
```bash
python main.py
```
You may be prompted to enter your system's password, as CodeCarbon needs to access the hardware information of the system.

### Step 6: Review Experiment Results
The `output/` folder will be automatically created if not exist. Each experiment creates a new subfolder which contains all output responses and metrics.

The `_metrics.txt` file contains measurements of each run. Each line represents one epoch and data is ordered as follows:
- Epoch ID
- Output Token Length
- Energy Consumption
- Carbon Emission
- Latency

The Epoch ID is formatted as:
`M<Model_Index><G/S>__UP<User_Prompt_Index>__E<Epoch_Number>`.

For instance, the ID `M1G__UP001__E001` means:
- Model #1
- General case system prompt
- User prompt #001
- Epoch #001

Each output response is saved in as a separate file in the format `M1G__UP001__E001.txt`.

## Notes
User and system prompts are intentionally loaded from disk to memory just before each run. Outputs are written to disk immediately after each run and then discarded from memory. This prevents loading large chunks of data into memory during run tracking, which may affect model execution and consume extra hardware resources.

## TODO
- Organize user prompt dataset
- Implement preheat logic before experiment
- Implement output response comparison logic
- Implement LLM judge panel