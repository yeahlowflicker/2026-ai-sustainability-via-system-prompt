from pathlib import Path

def read_string_from_file(filename:str)->str:
    with open(filename, 'r') as file:
        content = file.read()
    return content

def write_to_file(path:str, content:str):
    output_file = Path(path)
    output_file.parent.mkdir(exist_ok=True, parents=True)
    output_file.write_text(content)

def preload_codecarbon():
    from codecarbon import EmissionsTracker
    cc_tracker = EmissionsTracker(log_level='info')
    cc_tracker.start()
    cc_tracker.stop()