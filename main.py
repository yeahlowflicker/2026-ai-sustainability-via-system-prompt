import time
import linecache
from pathlib import Path

from experiment import run_exp_epoch
from helpers import read_string_from_file, preload_codecarbon
from model import ollama_preload_model
from warmup import warmup


MODEL_LIST = [
    'granite4.1:8b',
    'gemma4:12b',
    'qwen3.5:9b',
]

USER_PROMPT_SRC_PATH = './data/user_prompt_ds.txt'
USER_PROMPT_COUNT = 100

EPOCH_COUNT_PER_MODEL_PROMPT_COMBINATION=1

HARDWARE_WARMUP_PERIOD_SECONDS = 300

GENERAL_SYSTEM_PROMPT_SRC_PATH = './data/sys_prompt_general.txt'
SUSTAINABLE_SYSTEM_PROMPT_SRC_PATH = './data/sys_prompt_sus.txt'

IDLE_PERIOD_BETWEEN_RUNS_SECONDS = 15

CODECARBON_LOG_LEVEL = 'error'


if __name__ == '__main__':
    experiment_id = int(time.time())
    run_counter = 0
    total_runs = len(MODEL_LIST)*USER_PROMPT_COUNT*EPOCH_COUNT_PER_MODEL_PROMPT_COMBINATION

    # The script reads user prompts from the src file line by line
    # This is used to control which line to read, and shall be incremented
    current_user_prompt_index = 1

    # Preload CodeCarbon once before the experiment, as the init process could take a long time.
    preload_codecarbon()

    # Pre-heat the hardware by running a performance-intensive task
    warmup(HARDWARE_WARMUP_PERIOD_SECONDS)

    # Iterate models (e.g. 3)
    for i, model_slug in enumerate(MODEL_LIST):
        
        model_index = i+1

        # Reset to 1 for each model
        current_user_prompt_index = 1

        # It can take a long time to load up a model on the first try
        # Here preloading is performed, followed by a cool down period
        ollama_preload_model(model_slug)
        time.sleep(IDLE_PERIOD_BETWEEN_RUNS_SECONDS)

        # Iterate user prompts (e.g. 50)
        for _ in range(USER_PROMPT_COUNT):

            # Load next user prompt from dataset
            user_prompt = linecache.getline(USER_PROMPT_SRC_PATH, current_user_prompt_index)
            if len(user_prompt) == 0:
                break

            # Iterate epochs (e.g. 30)
            for j, epoch in enumerate(range(EPOCH_COUNT_PER_MODEL_PROMPT_COMBINATION)):
                epoch_index = j+1

                # Just for logging progress
                run_counter += 1
                print(f'Run {run_counter} of {total_runs}')

                # General system prompt
                epoch_id = f'M{model_index}G__UP{current_user_prompt_index:03d}__E{epoch_index:03d}'
                print(f'Epoch: {epoch_id}')

                system_prompt = read_string_from_file(GENERAL_SYSTEM_PROMPT_SRC_PATH)
                run_exp_epoch(
                    experiment_id=experiment_id,
                    epoch_id=epoch_id,
                    model_slug=model_slug,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    cc_loglevel=CODECARBON_LOG_LEVEL,
                )


                # Wait for a period of time to cool down the hardware
                time.sleep(IDLE_PERIOD_BETWEEN_RUNS_SECONDS)


                # Sustainable system prompt
                epoch_id = f'M{model_index}S__UP{current_user_prompt_index:03d}__E{epoch_index:03d}'
                print(f'Epoch: {epoch_id}')

                system_prompt = read_string_from_file(SUSTAINABLE_SYSTEM_PROMPT_SRC_PATH)
                run_exp_epoch(
                    experiment_id=experiment_id,
                    epoch_id=epoch_id,
                    model_slug=model_slug,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    cc_loglevel=CODECARBON_LOG_LEVEL,
                )

                # Wait for a period of time to cool down the hardware
                time.sleep(IDLE_PERIOD_BETWEEN_RUNS_SECONDS)

            # Increment user prompt to read the next line from src file
            current_user_prompt_index += 1
