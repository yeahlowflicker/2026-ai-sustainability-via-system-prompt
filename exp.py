import os
import csv
import time
import json
from typing import Any

from ollama import chat, generate, ChatResponse
from codecarbon import EmissionsTracker

from helpers import read_string_from_file, write_to_file, preload_codecarbon
from warmup import warmup


# ============================================================
# Experiment configuration
# ============================================================

MODEL_LIST = [
    'granite4.1:8b',
    'gemma4:12b',
    'qwen3.5:9b',
]

USER_PROMPT_DIR = './data/exp3'
USER_PROMPT_COUNT = 100

GENERAL_SYSTEM_PROMPT_SRC_PATH = './data/sys_prompt_general.txt'
SUSTAINABLE_SYSTEM_PROMPT_SRC_PATH = './data/sys_prompt_sus.txt'

HARDWARE_WARMUP_PERIOD_SECONDS = 300

# No additional inter-run cooldown.
# Pilot testing showed stable inference throughput without it.
IDLE_PERIOD_BETWEEN_RUNS_SECONDS = 0

# Same maximum generation budget for all models and both conditions.
OLLAMA_TOKEN_LIMIT = 16384

CODECARBON_LOG_LEVEL = 'error'


# ============================================================
# Ollama helpers
# ============================================================

# Send a dummy request to Ollama so it loads the required model.
def ollama_preload_model(model_slug: str):
    generate(
        model=model_slug,
        prompt='Hi',
    )


def ollama_send_request(
    model_slug: str,
    system_prompt: str,
    user_prompt: str,
) -> ChatResponse:
    response: ChatResponse = chat(
        model=model_slug,
        messages=[
            {
                'role': 'system',
                'content': system_prompt,
            },
            {
                'role': 'user',
                'content': user_prompt,
            },
        ],
        options={
            'f16_kv': True,
            'num_predict': OLLAMA_TOKEN_LIMIT,
        },
    )

    return response


# ============================================================
# Prompt helpers
# ============================================================

# prompt_id should be in the format UP001
def load_user_prompt(prompt_id: str):
    with open(
        f'{USER_PROMPT_DIR}/{prompt_id}.json',
        'r',
        encoding='utf-8',
    ) as f:
        data = json.load(f)

        return (
            data['task_category'],
            data['instruction'],
            data['context'],
        )


def construct_ollama_input(
    instruction: str,
    context: Any = None,
):
    if context:
        return f'{instruction}\n<context>{context}</context>'

    return instruction


# ============================================================
# Generation-status helpers
# ============================================================

def get_generation_status(
    ollama_raw_response: ChatResponse,
):
    """
    Classify the outcome of a completed Ollama generation.

    A model hitting the generation limit is a model-generation
    outcome, not an infrastructure failure, and is not
    automatically rerun.
    """

    content = ollama_raw_response.message.content or ''

    thinking = (
        getattr(
            ollama_raw_response.message,
            'thinking',
            None,
        )
        or ''
    )

    generation_limit_reached = (
        ollama_raw_response.done_reason == 'length'
    )

    final_content_empty = (
        len(content.strip()) == 0
    )

    # The model exhausted its generation budget without returning
    # any usable final response.
    generation_failure = (
        generation_limit_reached
        and final_content_empty
    )

    # Specific failure pattern observed in the pilot:
    # generation budget consumed in thinking/reasoning, with no
    # final answer produced.
    reasoning_limit_failure = (
        generation_failure
        and len(thinking.strip()) > 0
    )

    # The model reached the generation cap but did produce some
    # final-answer content. Keep and judge the returned response.
    truncated_response = (
        generation_limit_reached
        and not final_content_empty
    )

    return {
        'content': content,
        'thinking': thinking,
        'generation_limit_reached': generation_limit_reached,
        'final_content_empty': final_content_empty,
        'generation_failure': generation_failure,
        'reasoning_limit_failure': reasoning_limit_failure,
        'truncated_response': truncated_response,
    }


# ============================================================
# Response output
# ============================================================

def construct_slm_response_dump(
    run_id: str,
    task_category: str,
    ollama_raw_response: ChatResponse,
):
    status = get_generation_status(
        ollama_raw_response
    )

    data = {
        'run_id':
            run_id,

        'task_category':
            task_category,

        'model_slug':
            ollama_raw_response.model,

        'is_done':
            ollama_raw_response.done,

        'done_reason':
            ollama_raw_response.done_reason,

        'content':
            status['content'],

        'thinking':
            status['thinking'],

        'generation_limit_reached':
            status['generation_limit_reached'],

        'final_content_empty':
            status['final_content_empty'],

        'generation_failure':
            status['generation_failure'],

        'reasoning_limit_failure':
            status['reasoning_limit_failure'],

        'truncated_response':
            status['truncated_response'],
    }

    return data


# ============================================================
# Metrics
# ============================================================

def construct_ollama_codecarbon_metrics(
    run_id: str,
    task_category: str,
    latency: float,
    ollama_raw_response: ChatResponse,
    codecarbon_result: object,
):
    status = get_generation_status(
        ollama_raw_response
    )

    content = status['content']
    thinking = status['thinking']

    is_thinking = (
        len(thinking) > 0
    )

    metrics_data = {
        # --------------------------------------------------------
        # Experiment identifiers
        # --------------------------------------------------------

        'run_id':
            run_id,

        'task_category':
            task_category,

        # --------------------------------------------------------
        # Ollama metrics
        # --------------------------------------------------------

        # Exact model slug actually returned/used by Ollama.
        'model_slug':
            ollama_raw_response.model,

        'is_done':
            ollama_raw_response.done,

        'done_reason':
            ollama_raw_response.done_reason,

        # Ollama durations are reported in nanoseconds.
        'total_duration_s':
            ollama_raw_response.total_duration / 1e9,

        'load_duration_s':
            ollama_raw_response.load_duration / 1e9,

        # Prompt/input processing.
        'input_token_count':
            ollama_raw_response.prompt_eval_count,

        'input_token_duration_s':
            ollama_raw_response.prompt_eval_duration / 1e9,

        # H1:
        # Total generated tokens reported by Ollama eval_count.
        # For thinking-capable models, this can include reasoning/
        # thinking tokens as well as final-answer generation.
        'output_token_count':
            ollama_raw_response.eval_count,

        'output_token_duration_s':
            ollama_raw_response.eval_duration / 1e9,

        # --------------------------------------------------------
        # Thinking / response diagnostics
        # --------------------------------------------------------

        'is_thinking':
            is_thinking,

        'output_char_count':
            len(content),

        'thinking_char_count':
            len(thinking),

        # --------------------------------------------------------
        # Generation outcome flags
        # --------------------------------------------------------

        'generation_limit_reached':
            status['generation_limit_reached'],

        'final_content_empty':
            status['final_content_empty'],

        'generation_failure':
            status['generation_failure'],

        'reasoning_limit_failure':
            status['reasoning_limit_failure'],

        'truncated_response':
            status['truncated_response'],

        # --------------------------------------------------------
        # H4: end-to-end request latency
        # --------------------------------------------------------

        'latency_s':
            latency,

        # --------------------------------------------------------
        # CodeCarbon metrics
        # --------------------------------------------------------

        'cc_duration_s':
            codecarbon_result.duration,

        # CodeCarbon emissions are returned in kg CO2eq.
        # Convert to grams for H3 / preregistered units.
        'carbon_emissions_grams':
            codecarbon_result.emissions * 1000,

        # CodeCarbon emissions_rate is kg CO2eq / second.
        'emissions_rate_grams_per_s':
            codecarbon_result.emissions_rate * 1000,

        'mean_cpu_power_w':
            codecarbon_result.cpu_power,

        'mean_gpu_power_w':
            codecarbon_result.gpu_power,

        'mean_ram_power_w':
            codecarbon_result.ram_power,

        'cpu_energy_kwh':
            codecarbon_result.cpu_energy,

        'gpu_energy_kwh':
            codecarbon_result.gpu_energy,

        'ram_energy_kwh':
            codecarbon_result.ram_energy,

        # H2:
        # Total CPU + GPU + RAM energy consumption.
        'total_energy_consumed_kwh':
            codecarbon_result.energy_consumed,

        'cpu_util_percent':
            codecarbon_result.cpu_utilization_percent,

        'gpu_util_percent':
            codecarbon_result.gpu_utilization_percent,

        'ram_util_percent':
            codecarbon_result.ram_utilization_percent,

        'avg_ram_used_gb':
            codecarbon_result.ram_used_gb,
    }

    return metrics_data


# ============================================================
# CSV helpers
# ============================================================

def append_dict_to_csv(
    csv_path: str,
    data: dict,
):
    file_exists = (
        os.path.exists(csv_path)
        and os.path.getsize(csv_path) > 0
    )

    with open(
        csv_path,
        mode='a',
        newline='',
        encoding='utf-8',
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=data.keys(),
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(data)


def log_technical_failure(
    experiment_id: int,
    run_id: str,
    task_category: str,
    model_slug: str,
    condition: str,
    error: Exception,
    failure_stage: str,
    failed_attempt_latency_s: float | None,
    codecarbon_result: object | None,
):
    """
    Record infrastructure/request/measurement failures separately.

    Any CodeCarbon values stored here are diagnostics for the
    failed attempt only. They are NOT part of H1-H4 and are not
    written to exp_metrics.csv.
    """

    failure_data = {
        'run_id':
            run_id,

        'task_category':
            task_category,

        'model_slug':
            model_slug,

        'condition':
            condition,

        'failure_type':
            'technical_failure',

        'failure_stage':
            failure_stage,

        'exception_type':
            type(error).__name__,

        'exception_message':
            str(error),

        # Diagnostic values only.
        'failed_attempt_latency_s':
            failed_attempt_latency_s,

        'failed_attempt_cc_duration_s':
            (
                codecarbon_result.duration
                if codecarbon_result is not None
                else None
            ),

        'failed_attempt_energy_kwh':
            (
                codecarbon_result.energy_consumed
                if codecarbon_result is not None
                else None
            ),

        'failed_attempt_emissions_g':
            (
                codecarbon_result.emissions * 1000
                if codecarbon_result is not None
                else None
            ),

        'timestamp':
            time.time(),
    }

    failure_csv_path = (
        f'./outputs/{experiment_id}/technical_failures.csv'
    )

    append_dict_to_csv(
        failure_csv_path,
        failure_data,
    )


# ============================================================
# Main experiment
# ============================================================

def experiment_entry():

    # Use UNIX timestamp as experiment ID.
    experiment_id = int(time.time())
    print(f'Experiment ID: {experiment_id}')

    # Progress tracking.
    cumulative_run_counter = 0

    total_runs = (
        len(MODEL_LIST)
        * USER_PROMPT_COUNT
        * 2
    )

    # Preload CodeCarbon once before the experiment because
    # initialization itself may take a relatively long time.
    preload_codecarbon()

    # Pre-heat hardware before experimental measurements.
    warmup(
        HARDWARE_WARMUP_PERIOD_SECONDS
    )

    # --------------------------------------------------------
    # Iterate models
    # --------------------------------------------------------

    for model_index0, model_slug in enumerate(MODEL_LIST):

        # Preload model so first measured inference does not include
        # the full model-loading operation.
        ollama_preload_model(
            model_slug
        )

        time.sleep(
            IDLE_PERIOD_BETWEEN_RUNS_SECONDS
        )

        # ----------------------------------------------------
        # Iterate prompts
        # ----------------------------------------------------

        for prompt_index0 in range(USER_PROMPT_COUNT):

            prompt_id = (
                f'UP{prompt_index0 + 1:03d}'
            )

            (
                task_category,
                instruction,
                context,
            ) = load_user_prompt(
                prompt_id
            )

            # Build final user message.
            ollama_input = construct_ollama_input(
                instruction,
                context,
            )

            # Save exact SLM input.
            write_to_file(
                (
                    f'./outputs/{experiment_id}'
                    f'/slm_inputs/{prompt_id}.txt'
                ),
                ollama_input,
            )

            # Counterbalance condition order:
            #
            # odd-numbered prompts:
            #   General -> Sustainability
            #
            # even-numbered prompts:
            #   Sustainability -> General
            condition_order = (
                ['G', 'S']
                if prompt_index0 % 2 == 0
                else ['S', 'G']
            )

            # ------------------------------------------------
            # Iterate experimental conditions
            # ------------------------------------------------

            for condition in condition_order:

                run_id = (
                    f'M{model_index0 + 1}'
                    f'{condition}__{prompt_id}'
                )

                cumulative_run_counter += 1

                print(
                    f'Running: {run_id} '
                    f'({cumulative_run_counter} '
                    f'of {total_runs})...'
                )

                # Load appropriate system prompt.
                system_prompt = read_string_from_file(
                    GENERAL_SYSTEM_PROMPT_SRC_PATH
                    if condition == 'G'
                    else SUSTAINABLE_SYSTEM_PROMPT_SRC_PATH
                )

                # ------------------------------------------------
                # Start CodeCarbon and run inference
                # ------------------------------------------------

                cc_tracker = EmissionsTracker(
                    experiment_id=run_id,
                    output_dir=(
                        f'./outputs/{experiment_id}'
                    ),
                    log_level=CODECARBON_LOG_LEVEL,
                )

                ollama_raw_response = None
                codecarbon_result = None

                request_error = None
                tracking_error = None

                t_start = None
                t_end = None

                try:
                    cc_tracker.start()

                    # perf_counter is monotonic and appropriate
                    # for elapsed-time measurements.
                    t_start = time.perf_counter()

                    ollama_raw_response = ollama_send_request(
                        model_slug,
                        system_prompt,
                        ollama_input,
                    )

                    t_end = time.perf_counter()

                except Exception as error:
                    request_error = error

                    if t_start is not None:
                        t_end = time.perf_counter()

                finally:
                    try:
                        cc_tracker.stop()

                        codecarbon_result = (
                            cc_tracker.final_emissions_data
                        )

                    except Exception as error:
                        tracking_error = error

                        # final_emissions_data may still exist
                        # even when stop() raises.
                        codecarbon_result = getattr(
                            cc_tracker,
                            'final_emissions_data',
                            None,
                        )

                # ------------------------------------------------
                # Handle genuine technical/infrastructure failures
                # ------------------------------------------------

                technical_error = (
                    request_error
                    if request_error is not None
                    else tracking_error
                )

                if technical_error is not None:

                    failed_attempt_latency = (
                        t_end - t_start
                        if t_start is not None
                        and t_end is not None
                        else None
                    )

                    if request_error is not None:
                        failure_stage = 'ollama_request'
                    else:
                        failure_stage = 'codecarbon_stop'

                    log_technical_failure(
                        experiment_id=experiment_id,
                        run_id=run_id,
                        task_category=task_category,
                        model_slug=model_slug,
                        condition=condition,
                        error=technical_error,
                        failure_stage=failure_stage,
                        failed_attempt_latency_s=failed_attempt_latency,
                        codecarbon_result=codecarbon_result,
                    )

                    print(
                        f'Technical failure in {run_id} '
                        f'[{failure_stage}]: '
                        f'{technical_error}'
                    )

                    # Do not create a normal experimental row from
                    # an incomplete technical/measurement failure.
                    # The run is logged separately and can be rerun.
                    continue

                # CodeCarbon completed without raising an exception,
                # but a missing result is still a measurement failure.
                if codecarbon_result is None:

                    measurement_error = RuntimeError(
                        'CodeCarbon returned no final_emissions_data.'
                    )

                    failed_attempt_latency = (
                        t_end - t_start
                        if t_start is not None
                        and t_end is not None
                        else None
                    )

                    log_technical_failure(
                        experiment_id=experiment_id,
                        run_id=run_id,
                        task_category=task_category,
                        model_slug=model_slug,
                        condition=condition,
                        error=measurement_error,
                        failure_stage='codecarbon_result',
                        failed_attempt_latency_s=failed_attempt_latency,
                        codecarbon_result=None,
                    )

                    print(
                        f'Technical failure in {run_id} '
                        f'[codecarbon_result]: '
                        f'{measurement_error}'
                    )

                    continue

                # ------------------------------------------------
                # Successful measured Ollama request
                # ------------------------------------------------

                latency = (
                    t_end - t_start
                )

                # Save complete raw Ollama response.
                write_to_file(
                    (
                        f'./outputs/{experiment_id}'
                        f'/ollama_dumps/'
                        f'{run_id}.ollama.txt'
                    ),
                    str(ollama_raw_response),
                )

                # Save structured SLM response.
                slm_response_dump = (
                    construct_slm_response_dump(
                        run_id,
                        task_category,
                        ollama_raw_response,
                    )
                )

                write_to_file(
                    (
                        f'./outputs/{experiment_id}'
                        f'/slm_responses/'
                        f'{run_id}.resp.txt'
                    ),
                    json.dumps(
                        slm_response_dump,
                        ensure_ascii=False,
                        indent=2,
                    ),
                )

                # Combine Ollama + CodeCarbon metrics.
                metrics_data = (
                    construct_ollama_codecarbon_metrics(
                        run_id=run_id,
                        task_category=task_category,
                        latency=latency,
                        ollama_raw_response=ollama_raw_response,
                        codecarbon_result=codecarbon_result,
                    )
                )

                # Save experiment metrics.
                metrics_csv_path = (
                    f'./outputs/{experiment_id}'
                    f'/exp_metrics.csv'
                )

                append_dict_to_csv(
                    metrics_csv_path,
                    metrics_data,
                )

                # ------------------------------------------------
                # Log important generation outcomes
                # ------------------------------------------------

                if metrics_data['reasoning_limit_failure']:

                    print(
                        f'WARNING: {run_id} exhausted the '
                        f'generation budget during reasoning '
                        f'and produced no final response.'
                    )

                elif metrics_data['generation_failure']:

                    print(
                        f'WARNING: {run_id} reached the '
                        f'generation limit and produced no '
                        f'final response.'
                    )

                elif metrics_data['truncated_response']:

                    print(
                        f'WARNING: {run_id} reached the '
                        f'generation limit but returned '
                        f'partial final content.'
                    )

                # No additional cooldown in the final design.
                time.sleep(
                    IDLE_PERIOD_BETWEEN_RUNS_SECONDS
                )


if __name__ == '__main__':
    experiment_entry()
