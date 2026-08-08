from dotenv import load_dotenv
from openrouter import OpenRouter
import os
import re
import time
from helpers import read_string_from_file, write_response_to_file

load_dotenv() # loads env variables from .env file
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

EXPERIMENT_ID = '1786114246'
SLM_MODEL_COUNT = 3

# Change this to skip a certain number of runs
# This is useful when the script fails due to random API errors
INITIAL_SKIP=0

# How long to wait between each LLM api call
# Useful for dealing with api rate-limiting
ITERATION_DELAY_SECONDS=0

# The LLM judge models to use for evaluation
# These should be model slugs available on OpenRouter
LLM_JUDGE_MODELS = [
    'openai/gpt-oss-120b',
    'meta-llama/llama-3.3-70b-instruct',
    'deepseek/deepseek-v4-pro',
]

# How many user prompts are there per category
ITEMS_PER_CATEGORY = 20

# Task categories in order
task_categories = [
    "closed_qa",
    "open_qa",
    "classification",
    "information_extraction",
    "summarization",
]

def openrouter_evaluate(
    llm_model_slug: str,
    run_id: str,
    task: str,
    llm_input: str,
):
    print(f'{llm_model_slug} - {run_id} ({task})')

    with OpenRouter(api_key=OPENROUTER_API_KEY) as client:
        response = client.chat.send(
            model=llm_model_slug,
            provider={"sort": {"by": "price"}},  # choose the cheapest provider
            messages=[
                {"role": "system", "content": llmjudge_instructions},
                {"role": "user", "content": llm_input}
            ],
        )

        llm_response = response.choices[0].message.content
        print(llm_response)
        print()

        with open(f'./analysis/{EXPERIMENT_ID}/_llm_outcomes.txt', "a") as f:
            f.write(f'{run_id},{llm_model_slug},{task},{llm_response}')
            f.write('\n')


if __name__ == '__main__':
    llmjudge_instructions = read_string_from_file('./data/llmjudge_instructions.txt')
    
    cumulative_evaluations = 0

    with open('./data/user_prompt_ds.txt', "r", encoding="utf-8") as f:
        original_user_prompts = [line.rstrip("\n") for line in f]

    for model_index in range(SLM_MODEL_COUNT):
        current_user_prompt_index = 0

        for user_prompt in original_user_prompts:
            current_user_prompt_index += 1

            for condition in ['G', 'S']:
                run_id = f'M{model_index+1}{condition}__UP{current_user_prompt_index:03d}'

                # Extract the task category
                task = task_categories[(current_user_prompt_index - 1) // ITEMS_PER_CATEGORY]
                
                # Extract the context if present
                match = re.search(r'<context>(.*?)</context>', user_prompt, flags=re.DOTALL)
                context = match.group(1) if match else None

                # Load the SLM response for the current user prompt
                slm_response_path = f'./outputs/{EXPERIMENT_ID}/M{model_index+1}G__UP{current_user_prompt_index:03d}__E001.txt'
                slm_response = read_string_from_file(slm_response_path)

                # Construct the LLM evaluation prompt
                llm_input = f'<task>{task}</task>\n<input>{user_prompt}</input>\n<body>{slm_response}</body>'

                write_response_to_file(f'./analysis/{EXPERIMENT_ID}/{run_id}.llm_eval.txt', llm_input)

                if cumulative_evaluations >= INITIAL_SKIP:
                    for llm in LLM_JUDGE_MODELS:
                        openrouter_evaluate(llm, run_id, task, llm_input)
                        time.sleep(ITERATION_DELAY_SECONDS)

                cumulative_evaluations += 1

