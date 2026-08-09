import os
import re
import csv
import time
import json
import asyncio
from pathlib import Path

from dotenv import load_dotenv
from openrouter import OpenRouter

from helpers import read_string_from_file, write_to_file

load_dotenv() # loads env variables from .env file
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

EXPERIMENT_ID = '1786304595'
SLM_MODEL_COUNT = 3

# The LLM judge models to use for evaluation
# These should be model slugs available on OpenRouter
LLM_JUDGE_MODELS = [
    'openai/gpt-oss-120b',
    'meta-llama/llama-3.3-70b-instruct',
    'deepseek/deepseek-v4-pro',
]

OUTPUT_CSV_PATH = f'./outputs/{EXPERIMENT_ID}/llm_outcomes.csv'

llm_system_prompt = read_string_from_file('./data/llmjudge_instructions.txt')

csv_lock = asyncio.Lock()

MAX_CONCURRENT_REQUESTS = 10
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

async def openrouter_evaluate(
    llm_model_slug: str,
    run_id: str,
    llm_input: str,
):
    async with semaphore:
        print(f'{llm_model_slug} - {run_id}')

        # Helper function for synchronous SDK call
        def fetch_openrouter():
            with OpenRouter(api_key=OPENROUTER_API_KEY) as client:
                return client.chat.send(
                    model=llm_model_slug,
                    provider={"sort": {"by": "price"}},
                    messages=[
                        {"role": "system", "content": llm_system_prompt},
                        {"role": "user", "content": llm_input}
                    ],
                )

        # Offload the blocking network request to a separate thread
        response = await asyncio.to_thread(fetch_openrouter)
        llm_response = response.choices[0].message.content

        data = {
            'target_run_id':    run_id,
            'llm_model':        llm_model_slug,
            'llm_outcome':      llm_response,
        }

        # Safely acquire the lock and append to CSV
        async with csv_lock:
            file_exists = (os.path.exists(OUTPUT_CSV_PATH) and os.path.getsize(OUTPUT_CSV_PATH) > 0)
            with open(OUTPUT_CSV_PATH, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=data.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(data)



async def entrypoint():
    tasks = []

    slm_response_files = sorted(Path(f'./outputs/{EXPERIMENT_ID}/slm_responses').glob("*.txt"))

    for file_path in slm_response_files:

        with open(file_path, "r", encoding="utf-8") as file:
            slm_response_json = json.load(file)

        run_id = slm_response_json['run_id']    # e.g. M1G__UP001
        prompt_id = run_id.split('__')[1]       # extract user prompt ID, e.g. UP001

        task_category = slm_response_json['task_category']
        slm_model_slug = slm_response_json['model_slug']
        slm_response_content = slm_response_json['content']

        slm_input = read_string_from_file(f'./outputs/{EXPERIMENT_ID}/slm_inputs/{prompt_id}.txt')

        # Construct the LLM evaluation prompt
        llm_input = f'<task>{task_category}</task>\n<input>{slm_input}</input>\n<body>{slm_response_content}</body>'

        write_to_file(f'./outputs/{EXPERIMENT_ID}/llm_judge_inputs/{run_id}.txt', llm_input)

        for llm_model_slug in LLM_JUDGE_MODELS:
            counter += 1
            tasks.append(openrouter_evaluate(llm_model_slug, run_id, llm_input))
    
    await asyncio.gather(*tasks)

if __name__ == '__main__':
    asyncio.run(entrypoint())
