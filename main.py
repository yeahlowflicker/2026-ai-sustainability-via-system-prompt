from experiment import run_exp_epoch
import time
from helpers import read_string_from_file
import linecache
from pathlib import Path


EPOCH_COUNT_PER_MODEL_PROMPT_COMBINATION=30

MODEL_LIST = [
    'granite3.2:2b',
    # 'qwen2:7b',
    # 'llama3:8b',
]

USER_PROMPT_SRC_PATH = './data/user_prompt_ds.txt'
USER_PROMPT_COUNT = 3

GENERAL_SYSTEM_PROMPT_SRC_PATH = './data/sys_prompt_general.txt'
SUSTAINABLE_SYSTEM_PROMPT_SRC_PATH = './data/sys_prompt_sus.txt'

IDLE_PERIOD_BETWEEN_RUNS_SECONDS = 60

CODECARBON_LOG_LEVEL = 'error'


if __name__ == '__main__':
    experiment_id = int(time.time())

    # The script reads user prompts from the src file line by line
    # This is used to control which line to read, and shall be incremented
    current_user_prompt_index = 1

    for i, model_slug in enumerate(MODEL_LIST):
        
        model_index = i+1

        # Load next user prompt from dataset
        user_prompt = linecache.getline(USER_PROMPT_SRC_PATH, current_user_prompt_index)
        if len(user_prompt) == 0:
            break

        for j, epoch in enumerate(range(EPOCH_COUNT_PER_MODEL_PROMPT_COMBINATION)):
            epoch_index = j+1

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
            time.sleep(IDLE_PERIOD_BETWEEN_RUNS_SECONDS = 60)


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
            time.sleep(IDLE_PERIOD_BETWEEN_RUNS_SECONDS = 60)

        # Increment user prompt to read the next line from src file
        current_user_prompt_index += 1
