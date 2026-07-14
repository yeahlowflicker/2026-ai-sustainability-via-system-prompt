from pathlib import Path

class Metrics:
    epoch_id: str
    output_length: int
    energy_consumed: float
    carbon_emission: float
    latency: int

    def __init__(self, epoch_id:str, output_length:int, energy_consumed:float, carbon_emission:float, latency:int):
        self.epoch_id = epoch_id
        self.output_length = output_length
        self.energy_consumed = energy_consumed
        self.carbon_emission = carbon_emission
        self.latency = latency

    def to_single_line(self):
        return f'{self.epoch_id},{self.output_length},{self.energy_consumed},{self.carbon_emission},{self.latency}'

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

# Fibonacci sequence function for warm-up before experiments
def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)