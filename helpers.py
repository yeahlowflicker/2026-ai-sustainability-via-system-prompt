from pathlib import Path

class Metrics:
    epoch_id: str
    input_token_count: int
    input_eval_duration: int
    output_char_length: int
    output_token_count: int
    output_eval_duration: int
    energy_consumed: float
    carbon_emission: float
    latency: int

    def __init__(self, epoch_id:str, input_token_count:int, input_eval_duration:int, output_char_length:int, output_token_count:int, output_eval_duration:int, energy_consumed:float, carbon_emission:float, latency:int):
        self.epoch_id = epoch_id
        self.input_token_count = input_token_count
        self.input_eval_duration = input_eval_duration
        self.output_char_length = output_char_length
        self.output_token_count = output_token_count
        self.output_eval_duration = output_eval_duration
        self.energy_consumed = energy_consumed
        self.carbon_emission = carbon_emission
        self.latency = latency

    def to_single_line(self):
        return f'{self.epoch_id},{self.input_token_count},{self.input_eval_duration},{self.output_char_length},{self.output_token_count},{self.output_eval_duration},{self.energy_consumed},{self.carbon_emission},{self.latency}'

def append_metrics_to_file(filename:str, metrics:Metrics):
    output_file = Path(filename)
    output_file.parent.mkdir(exist_ok=True, parents=True)

    with open(output_file, "a") as f:
        f.write(metrics.to_single_line())
        f.write('\n')

def write_response_to_file(filename:str, content:str):
    output_file = Path(filename)
    output_file.parent.mkdir(exist_ok=True, parents=True)
    output_file.write_text(content)


def read_string_from_file(filename:str)->str:
    with open(filename, 'r') as file:
        content = file.read()
    return content


def preload_codecarbon():
    from codecarbon import EmissionsTracker
    cc_tracker = EmissionsTracker(log_level='info')
    cc_tracker.start()
    cc_tracker.stop()