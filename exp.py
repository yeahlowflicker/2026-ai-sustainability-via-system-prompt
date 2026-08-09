import os
import csv
import time
import json
from pathlib import Path
from ollama import chat, generate, ChatResponse
from codecarbon import EmissionsTracker

from helpers import read_string_from_file, write_to_file, preload_codecarbon
from warmup import warmup

MODEL_LIST = [ 'granite4.1:8b', 'gemma4:12b', 'qwen3.5:9b' ]

USER_PROMPT_DIR = './data/exp3'
USER_PROMPT_COUNT = 100

GENERAL_SYSTEM_PROMPT_SRC_PATH = './data/sys_prompt_general.txt'
SUSTAINABLE_SYSTEM_PROMPT_SRC_PATH = './data/sys_prompt_sus.txt'

HARDWARE_WARMUP_PERIOD_SECONDS = 300
IDLE_PERIOD_BETWEEN_RUNS_SECONDS = 15

OLLAMA_TOKEN_LIMIT = 16384
CODECARBON_LOG_LEVEL = 'error'


# Send a dummy request to Ollama so it loads up the required model
def ollama_preload_model(model_slug: str):
    generate(model=model_slug, prompt='Hi')


def ollama_send_request(
    model_slug:str,
    system_prompt:str,
    user_prompt:str
)->object:
    response: ChatResponse = chat(
        model=model_slug,
        messages=[
            { 'role': 'system', 'content': system_prompt, },
            { 'role': 'user', 'content': user_prompt },
        ],
        options={
            'f16_kv': True,  # Enable KV caching
            'num_predict': OLLAMA_TOKEN_LIMIT,  # Set token limit
        }
    )
    return response


# prompt_id should be in the format of `UP001`
def load_user_prompt(prompt_id:str):
    with open(f'{USER_PROMPT_DIR}/{prompt_id}.json', "r", encoding="utf-8") as f:
        data = json.load(f)
        return data['task_category'], data['instruction'], data['context']


def construct_ollama_input(instruction:str, context:any=None):
    # Append context only if it exists
    if context:
        return f'{instruction}\n<context>{context}</context>'
    else:
        return instruction


def construct_slm_response_dump(run_id: str, task_category:str, ollama_raw_response:object):
    data = {
        'run_id':           run_id,
        'task_category':    task_category,
        'done_reason':      ollama_raw_response.done_reason,
        'model_slug':       ollama_raw_response.model,
        'content':          ollama_raw_response.message.content,
        'thinking':         getattr(ollama_raw_response.message, "thinking", None),
    }
    return data


def construct_ollama_codecarbon_metrics(run_id: str, task_category:str, latency:int, ollama_raw_response:object, codecarbon_result: object):
    
    is_thinking = getattr(ollama_raw_response.message, "thinking", None) is not None

    metrics_data = {
        "run_id":           run_id,
        "task_category":    task_category,

        # Extract Ollama metrics
        # @ref: https://github.com/ollama/ollama-python/blob/25b93290d8cd07b0d00732641f812ee34fd4c989/ollama/_types.py#L230
        # @ref: https://github.com/ollama/ollama-python/blob/25b93290d8cd07b0d00732641f812ee34fd4c989/ollama/_types.py#L304
        "model_slug":                   ollama_raw_response.model,                         # Model used to generate response
        "is_done":                      ollama_raw_response.done,                          # True if response is complete, otherwise False. Useful for streaming to detect the final response.
        "done_reason":                  ollama_raw_response.done_reason,                   # Reason for completion. Only present when done is True.
        "total_duration_s":             ollama_raw_response.total_duration / 1e9,          # Total duration, converted from ns to seconds
        "load_duration_s":              ollama_raw_response.load_duration / 1e9,           # Load duration, converted from ns to seconds
        "input_token_count":            ollama_raw_response.prompt_eval_count,             # Number of tokens evaluated in the prompt
        "input_token_duration_s":       ollama_raw_response.prompt_eval_duration / 1e9,    # Duration of evaluating the prompt, converted from ns to seconds
        "output_token_count":           ollama_raw_response.eval_count,                    # Number of tokens evaluated in inference
        "output_token_duration_s":      ollama_raw_response.eval_duration / 1e9,           # Duration of evaluating inference, converted from ns to seconds

        "is_thinking":                  is_thinking,
        "output_char_count":            len(ollama_raw_response.message.content),
        "thinking_char_count":          len(ollama_raw_response.message.thinking) if is_thinking else 0,
        
        "latency_s":                    latency,    # Manually-calculated latency from start to finish

        # Extract CodeCarbon metrics
        # @ref: https://docs.codecarbon.io/latest/reference/output/#csv
        # @ref: https://github.com/mlco2/codecarbon/blob/master/codecarbon/output_methods/emissions_data.py#L7
        "cc_duration_s":                codecarbon_result.duration,                # Duration of the compute, in seconds
        "carbon_emissions_kg":          codecarbon_result.emissions,               # Emissions as CO₂-equivalents (CO₂eq), in kg
        "emissions_rate_kgps":          codecarbon_result.emissions_rate,          # Emissions divided per duration, in Kg/s
        "mean_cpu_power_w":             codecarbon_result.cpu_power,               # Mean CPU power (W)
        "mean_gpu_power_w":             codecarbon_result.gpu_power,               # Mean GPU power (W)
        "mean_ram_power_w":             codecarbon_result.ram_power,               # Mean RAM power (W)
        "cpu_energy_kwh":               codecarbon_result.cpu_energy,              # Energy used per CPU (kWh)
        "gpu_energy_kwh":               codecarbon_result.gpu_energy,              # Energy used per GPU (kWh)
        "ram_energy_kwh":               codecarbon_result.ram_energy,              # Energy used per RAM (kWh)
        "total_energy_consumed_kwh":    codecarbon_result.energy_consumed,         # Sum of cpu_energy, gpu_energy and ram_energy (kWh)
        "cpu_util_percent":             codecarbon_result.cpu_utilization_percent, # Average CPU utilization during tracking period (%)
        "gpu_util_percent":             codecarbon_result.gpu_utilization_percent, # Average GPU utilization during tracking period (%)
        "ram_util_percent":             codecarbon_result.ram_utilization_percent, # Average RAM utilization during tracking period (%)
        "avg_ram_used_gb":              codecarbon_result.ram_used_gb,             # Average RAM used during tracking period (GB)
    }

    return metrics_data


def experiment_entry():

    # Use UNIX timestamp as experiment ID
    experiment_id = int(time.time())
    
    # For progress tracking
    cumulative_run_counter = 0
    total_runs = len(MODEL_LIST)*USER_PROMPT_COUNT*2
    
    # Preload CodeCarbon once before the experiment, as the init process could take a long time.
    preload_codecarbon()

    # Pre-heat the hardware by running a performance-intensive task
    warmup(HARDWARE_WARMUP_PERIOD_SECONDS)


    # Iterate the models
    for model_index0, model_slug in enumerate(MODEL_LIST):

        # It can take a long time to load up a model on the first try
        # Here preloading is performed, followed by a cool down period
        ollama_preload_model(model_slug)
        time.sleep(IDLE_PERIOD_BETWEEN_RUNS_SECONDS)
        
        # Iterate the user prompts
        for prompt_index0 in range(USER_PROMPT_COUNT):

            prompt_id = f'UP{prompt_index0+1:03d}'   # e.g. UP001
            task_category, instruction, context = load_user_prompt(prompt_id)
            
            # Build the input message to be sent to Ollama
            ollama_input = construct_ollama_input(instruction, context)
            
            # Log the construct SLM input to file
            write_to_file(f'./outputs/{experiment_id}/slm_inputs/{prompt_id}.txt', ollama_input)

            # Alternate the general/sustainability conditions based on the prompt index
            condition_order = ['G','S'] if prompt_index0 % 2 == 0 else ['S','G']

            # Iterate the general and the sustainable conditions
            for condition in condition_order:

                # Generate run ID
                run_id = f'M{model_index0 + 1}{condition}__{prompt_id}'

                # Progress logging
                cumulative_run_counter += 1
                print(f'Running: {run_id} ({cumulative_run_counter} of {total_runs})...')

                # Load system prompt from file
                system_prompt = read_string_from_file(GENERAL_SYSTEM_PROMPT_SRC_PATH if condition == 'G' else SUSTAINABLE_SYSTEM_PROMPT_SRC_PATH)

                # Start CodeCarbon tracking
                cc_tracker = EmissionsTracker(experiment_id=run_id, output_dir=f'./outputs/{experiment_id}', log_level=CODECARBON_LOG_LEVEL)
                cc_tracker.start()

                # Log time of start
                t_start = time.time()
                
                # Send chat request to Ollama
                ollama_raw_response = ollama_send_request(model_slug, system_prompt, ollama_input)

                # Log time of completion and stop tracking
                t_end = time.time()
                cc_tracker.stop()
                codecarbon_result = cc_tracker.final_emissions_data
                
                # Calculate latency
                latency = t_end - t_start

                # Dump Ollama's raw response to file
                write_to_file(f'./outputs/{experiment_id}/ollama_dumps/{run_id}.ollama.txt', str(ollama_raw_response))

                # Write SLM response to file
                slm_response_dump = construct_slm_response_dump(run_id, task_category, ollama_raw_response)
                write_to_file(f'./outputs/{experiment_id}/slm_responses/{run_id}.resp.txt', json.dumps(slm_response_dump))

                # Process Ollama and CodeCarbon metrics
                metrics_data = construct_ollama_codecarbon_metrics(run_id, task_category, latency, ollama_raw_response, codecarbon_result)

                # Append metrics to CSV file
                metrics_csv_path = f'./outputs/{experiment_id}/exp_metrics.csv'
                file_exists = (os.path.exists(metrics_csv_path) and os.path.getsize(metrics_csv_path) > 0)
                with open(metrics_csv_path, mode="a", newline="", encoding="utf-8") as file:
                    writer = csv.DictWriter(file, fieldnames=metrics_data.keys())
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(metrics_data)

                # Wait for hardware cooldown
                time.sleep(IDLE_PERIOD_BETWEEN_RUNS_SECONDS)


if __name__ == '__main__':
    experiment_entry()
