import time
from codecarbon import EmissionsTracker
from model import ollama_send_request
from helpers import Metrics, append_metrics_to_file, write_response_to_file


def run_exp_epoch(experiment_id:int, epoch_id:str, model_slug:str, system_prompt:str, user_prompt:str):

    # Start CodeCarbon tracking
    cc_tracker = EmissionsTracker(log_level='error')
    cc_tracker.start()

    # Log time of start
    t_start = time.time()

    # Send inference request to Ollama
    response = ollama_send_request(
        model_slug=model_slug,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    # Log time of completion and stop tracking
    t_end = time.time()
    emissions = cc_tracker.stop()
    
    output_message = response.message.content

    metrics = Metrics(
        epoch_id=epoch_id,
        output_length=len(output_message),
        energy_consumed=cc_tracker.final_emissions_data.energy_consumed,
        carbon_emission=emissions,
        latency=t_end-t_start,
    )

    append_metrics_to_file(f'./outputs/{experiment_id}/_metrics.txt', metrics)
    write_response_to_file(f'./outputs/{experiment_id}/{epoch_id}.txt', output_message)
