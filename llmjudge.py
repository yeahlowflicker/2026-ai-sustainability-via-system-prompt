from dotenv import load_dotenv
from openrouter import OpenRouter

import csv
import json
import os
import re
import time
from pathlib import Path


load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Set this to the experiment ID produced by exp.py.
EXPERIMENT_ID = "REPLACE_WITH_FINAL_EXPERIMENT_ID"

EXPECTED_PROMPT_COUNT = 100
EXPECTED_MODEL_COUNT = 3
EXPECTED_RUN_COUNT = EXPECTED_PROMPT_COUNT * EXPECTED_MODEL_COUNT * 2

ITERATION_DELAY_SECONDS = 0
MAX_API_RETRIES = 3
API_RETRY_DELAY_SECONDS = 5

LLM_JUDGE_MODELS = [
    "openai/gpt-oss-120b",
    "meta-llama/llama-3.3-70b-instruct",
    "deepseek/deepseek-v4-pro",
]

QA_TASKS = {"closed_qa", "open_qa"}
SCORED_TASKS = {"summarization", "information_extraction", "classification"}
ALL_TASKS = QA_TASKS | SCORED_TASKS

RUN_ID_RE = re.compile(
    r"^M(?P<model_number>\d+)(?P<condition>[GS])__(?P<prompt_id>UP\d{3})$"
)

OUTPUT_ROOT = Path("./outputs") / EXPERIMENT_ID
ANALYSIS_ROOT = Path("./analysis") / EXPERIMENT_ID

METRICS_PATH = OUTPUT_ROOT / "exp_metrics.csv"
SLM_INPUT_DIR = OUTPUT_ROOT / "slm_inputs"
SLM_RESPONSE_DIR = OUTPUT_ROOT / "slm_responses"

JUDGE_OUTCOMES_PATH = ANALYSIS_ROOT / "judge_outcomes.csv"
JUDGE_FAILURES_PATH = ANALYSIS_ROOT / "judge_failures.csv"
JUDGE_PROMPT_DIR = ANALYSIS_ROOT / "judge_prompts"
JUDGE_RAW_DIR = ANALYSIS_ROOT / "judge_raw"

JUDGE_INSTRUCTIONS_PATH = Path("./data/llmjudge_instructions.txt")


def normalize_task(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0

    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def load_experiment_runs() -> list[dict]:
    """
    Load completed measured runs from exp_metrics.csv.

    Technical failures are not written to exp_metrics.csv. Requiring all
    600 rows here ensures any technical failures have been rerun before
    final judging starts.
    """
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"Experiment metrics file not found: {METRICS_PATH}")

    with METRICS_PATH.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if len(rows) != EXPECTED_RUN_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_RUN_COUNT} completed experiment rows, found {len(rows)}. "
            "Resolve/rerun technical failures before final judging."
        )

    required = {"run_id", "task_category", "model_slug"}
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"exp_metrics.csv is missing columns: {sorted(missing)}")

    seen_run_ids = set()
    seen_cells = set()
    prompt_ids = set()
    model_numbers = set()
    conditions = set()

    for row in rows:
        run_id = row["run_id"].strip()
        task = normalize_task(row["task_category"])
        match = RUN_ID_RE.fullmatch(run_id)

        if match is None:
            raise ValueError(f"Invalid run_id in exp_metrics.csv: {run_id!r}")
        if task not in ALL_TASKS:
            raise ValueError(f"Unexpected task category for {run_id}: {task!r}")
        if run_id in seen_run_ids:
            raise ValueError(f"Duplicate run_id in exp_metrics.csv: {run_id}")

        prompt_id = match.group("prompt_id")
        model_number = int(match.group("model_number"))
        condition = match.group("condition")
        cell = (prompt_id, model_number, condition)

        if cell in seen_cells:
            raise ValueError(f"Duplicate prompt × model × condition cell: {cell}")

        seen_run_ids.add(run_id)
        seen_cells.add(cell)
        prompt_ids.add(prompt_id)
        model_numbers.add(model_number)
        conditions.add(condition)

        row["run_id"] = run_id
        row["task_category"] = task
        row["prompt_id"] = prompt_id

    if len(prompt_ids) != EXPECTED_PROMPT_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_PROMPT_COUNT} unique prompts, found {len(prompt_ids)}."
        )
    if len(model_numbers) != EXPECTED_MODEL_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_MODEL_COUNT} model indices, found {len(model_numbers)}."
        )
    if conditions != {"G", "S"}:
        raise ValueError(f"Expected conditions G and S, found {sorted(conditions)}.")

    return rows


def load_slm_response(run_id: str, expected_task: str) -> dict:
    """
    Load the structured response written by exp.py.

    Only final `content` is judged. The model's internal `thinking`
    is deliberately not sent to the judges.
    """
    path = SLM_RESPONSE_DIR / f"{run_id}.resp.txt"
    if not path.exists():
        raise FileNotFoundError(f"SLM response file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if data.get("run_id") != run_id:
        raise ValueError(
            f"Run-ID mismatch in {path}: {data.get('run_id')!r} != {run_id!r}"
        )

    response_task = normalize_task(data.get("task_category", ""))
    if response_task != expected_task:
        raise ValueError(
            f"Task mismatch for {run_id}: metrics={expected_task!r}, "
            f"response={response_task!r}"
        )

    data["content"] = "" if data.get("content") is None else str(data["content"])
    return data


def build_judge_system_prompt(base_instructions: str, task: str) -> str:
    """
    Preserve the existing judge instructions, while enforcing a
    machine-readable output and a predefined rule for empty responses.
    """
    if task in QA_TASKS:
        rule = (
            "For this QA evaluation, return exactly one token: TRUE or FALSE. "
            "Do not include explanation or punctuation. If the <body> is empty "
            "or contains no usable final answer, return FALSE."
        )
    else:
        rule = (
            "For this evaluation, return exactly one integer from 1 to 5. "
            "Do not include explanation or punctuation. If the <body> is empty "
            "or contains no usable final answer, return 1."
        )

    return base_instructions.rstrip() + "\n\n" + rule


def build_judge_input(task: str, slm_input: str, slm_content: str) -> str:
    return (
        f"<task>{task}</task>\n"
        f"<input>{slm_input}</input>\n"
        f"<body>{slm_content}</body>"
    )


def parse_judge_outcome(task: str, raw_response: object) -> str:
    """
    Normalize only unambiguous judge outputs.

    We do not guess from explanatory text because the final analysis expects
    clean TRUE/FALSE votes or clean 1-5 ratings.
    """
    if raw_response is None:
        raise ValueError("Judge returned no response content.")

    text = str(raw_response).strip()

    # Tolerate a single Markdown code block around the answer.
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()

    if task in QA_TASKS:
        normalized = text.upper()
        if normalized not in {"TRUE", "FALSE"}:
            raise ValueError(
                f"QA judge response must be exactly TRUE or FALSE; received {text!r}"
            )
        return normalized

    if task in SCORED_TASKS:
        if not re.fullmatch(r"[1-5]", text):
            raise ValueError(
                f"Scored-task judge response must be exactly 1-5; received {text!r}"
            )
        return text

    raise ValueError(f"Unexpected task category: {task!r}")


def request_judge(
    llm_model_slug: str,
    system_prompt: str,
    judge_input: str,
):
    """
    Retry API/transport exceptions only.

    Invalid-but-successfully-returned judge text is logged separately rather
    than silently interpreted or automatically replaced.
    """
    last_error = None

    for attempt in range(1, MAX_API_RETRIES + 1):
        try:
            with OpenRouter(api_key=OPENROUTER_API_KEY) as client:
                response = client.chat.send(
                    model=llm_model_slug,
                    provider={"sort": {"by": "price"}},
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": judge_input},
                    ],
                )

            return response.choices[0].message.content

        except Exception as error:
            last_error = error
            if attempt < MAX_API_RETRIES:
                time.sleep(API_RETRY_DELAY_SECONDS)

    raise last_error


def load_completed_judge_pairs() -> set[tuple[str, str]]:
    """
    Allow safe restart after an API failure.

    Existing valid (run_id, judge_model) outcomes are skipped so rerunning
    the script does not duplicate already-completed evaluations.
    """
    if not JUDGE_OUTCOMES_PATH.exists():
        return set()

    with JUDGE_OUTCOMES_PATH.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        return set()

    required = {"Run ID", "LLM Judge Model", "Task", "Outcome"}
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(
            f"Existing judge_outcomes.csv is missing columns: {sorted(missing)}"
        )

    pairs = set()

    for row in rows:
        run_id = row["Run ID"].strip()
        judge_model = row["LLM Judge Model"].strip()
        task = normalize_task(row["Task"])

        # Validate old rows before treating them as complete.
        parse_judge_outcome(task, row["Outcome"])

        pair = (run_id, judge_model)
        if pair in pairs:
            raise ValueError(f"Duplicate judge result already present for {pair}.")

        pairs.add(pair)

    return pairs


def log_judge_failure(
    run_id: str,
    judge_model: str,
    task: str,
    stage: str,
    error: Exception,
    raw_response: object | None = None,
) -> None:
    append_csv(
        JUDGE_FAILURES_PATH,
        {
            "Run ID": run_id,
            "LLM Judge Model": judge_model,
            "Task": task,
            "Failure Stage": stage,
            "Exception Type": type(error).__name__,
            "Exception Message": str(error),
            "Raw Response": "" if raw_response is None else str(raw_response),
            "Timestamp": time.time(),
        },
    )


def main() -> None:
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Add it to the environment or .env file."
        )

    if EXPERIMENT_ID == "REPLACE_WITH_FINAL_EXPERIMENT_ID":
        raise RuntimeError(
            "Set EXPERIMENT_ID at the top of llmjudge.py before running."
        )

    if not JUDGE_INSTRUCTIONS_PATH.exists():
        raise FileNotFoundError(
            f"LLM judge instruction file not found: {JUDGE_INSTRUCTIONS_PATH}"
        )

    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    JUDGE_PROMPT_DIR.mkdir(parents=True, exist_ok=True)
    JUDGE_RAW_DIR.mkdir(parents=True, exist_ok=True)

    base_instructions = JUDGE_INSTRUCTIONS_PATH.read_text(encoding="utf-8")
    runs = load_experiment_runs()
    completed_pairs = load_completed_judge_pairs()

    total_judgements = len(runs) * len(LLM_JUDGE_MODELS)
    print(f"Experiment runs: {len(runs)}")
    print(f"Required judge evaluations: {total_judgements}")
    print(f"Already completed: {len(completed_pairs)}")

    completed_this_session = 0
    failures_this_session = 0

    for run in runs:
        run_id = run["run_id"]
        task = run["task_category"]
        prompt_id = run["prompt_id"]

        input_path = SLM_INPUT_DIR / f"{prompt_id}.txt"
        if not input_path.exists():
            raise FileNotFoundError(f"SLM input file not found: {input_path}")

        slm_input = input_path.read_text(encoding="utf-8")
        response_data = load_slm_response(run_id, task)

        # Judge final answer only, never internal thinking.
        slm_content = response_data["content"]

        judge_input = build_judge_input(task, slm_input, slm_content)
        (JUDGE_PROMPT_DIR / f"{run_id}.judge_prompt.txt").write_text(
            judge_input,
            encoding="utf-8",
        )

        system_prompt = build_judge_system_prompt(base_instructions, task)

        for judge_model in LLM_JUDGE_MODELS:
            pair = (run_id, judge_model)

            if pair in completed_pairs:
                continue

            print(f"{judge_model} - {run_id} ({task})")
            raw_response = None

            try:
                raw_response = request_judge(
                    llm_model_slug=judge_model,
                    system_prompt=system_prompt,
                    judge_input=judge_input,
                )
            except Exception as error:
                failures_this_session += 1
                log_judge_failure(
                    run_id,
                    judge_model,
                    task,
                    "api_request",
                    error,
                )
                print(f"  API failure: {error}")
                continue

            # Preserve raw judge output for auditability.
            raw_dir = JUDGE_RAW_DIR / safe_filename(judge_model)
            raw_dir.mkdir(parents=True, exist_ok=True)
            (raw_dir / f"{run_id}.txt").write_text(
                str(raw_response),
                encoding="utf-8",
            )

            try:
                outcome = parse_judge_outcome(task, raw_response)
            except Exception as error:
                failures_this_session += 1
                log_judge_failure(
                    run_id,
                    judge_model,
                    task,
                    "outcome_validation",
                    error,
                    raw_response,
                )
                print(f"  Invalid judge output: {raw_response!r}")
                continue

            # This is the file consumed by final_sustainability_analysis.py.
            # Keep one row per judge. Majority vote and unrounded mean are
            # computed later in the analysis script.
            append_csv(
                JUDGE_OUTCOMES_PATH,
                {
                    "Run ID": run_id,
                    "LLM Judge Model": judge_model,
                    "Task": task,
                    "Outcome": outcome,
                },
            )

            completed_pairs.add(pair)
            completed_this_session += 1
            print(f"  Outcome: {outcome}")

            time.sleep(ITERATION_DELAY_SECONDS)

    final_completed = len(completed_pairs)

    print()
    print("LLM judging pass complete.")
    print(f"Completed judge evaluations: {final_completed} / {total_judgements}")
    print(f"New successful evaluations: {completed_this_session}")
    print(f"Failures this session: {failures_this_session}")
    print(f"Judge outcomes: {JUDGE_OUTCOMES_PATH}")

    if final_completed != total_judgements:
        print(
            "Some evaluations are still missing. Inspect judge_failures.csv and "
            "rerun this script; completed (run_id, judge_model) pairs are skipped."
        )
    else:
        print("All required judge evaluations are complete.")


if __name__ == "__main__":
    main()
