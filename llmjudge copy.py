from dotenv import load_dotenv
from openrouter import OpenRouter

import asyncio
import csv
import json
import os
import re
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Set this to the experiment ID created by exp.py.
EXPERIMENT_ID = "1786407428"

EXPECTED_RUN_COUNT = 600
MAX_CONCURRENT_REQUESTS = 5
MAX_ATTEMPTS_PER_JUDGEMENT = 3

LLM_JUDGE_MODELS = [
    # "openai/gpt-oss-120b",
    "meta-llama/llama-3.3-70b-instruct",
    # "deepseek/deepseek-v4-pro",
]

QA_TASKS = {"closed_qa", "open_qa"}
SCORED_TASKS = {"summarization", "information_extraction", "classification"}
ALL_TASKS = QA_TASKS | SCORED_TASKS

RUN_ID_RE = re.compile(r"^M\d+[GS]__(UP\d{3})$")

OUTPUT_ROOT = Path("./outputs") / EXPERIMENT_ID
ANALYSIS_ROOT = Path("./analysis") / EXPERIMENT_ID

METRICS_PATH = OUTPUT_ROOT / "exp_metrics.csv"
INPUT_DIR = OUTPUT_ROOT / "slm_inputs"
RESPONSE_DIR = OUTPUT_ROOT / "slm_responses"

JUDGE_INSTRUCTIONS_PATH = Path("./data/llmjudge_instructions.txt")
JUDGE_OUTCOMES_PATH = ANALYSIS_ROOT / "judge_outcomes.csv"
JUDGE_FAILURES_PATH = ANALYSIS_ROOT / "judge_failures.csv"


# ============================================================
# Helpers
# ============================================================

def normalize_task(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0

    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def parse_outcome(task: str, raw_response: object) -> str:
    """Accept only the exact output type required for the task."""
    if raw_response is None:
        raise ValueError("Judge returned None.")

    text = str(raw_response).strip()

    if task in QA_TASKS:
        value = text.upper()
        if value not in {"TRUE", "FALSE"}:
            raise ValueError(
                f"Expected TRUE/FALSE for {task}, received {text!r}."
            )
        return value

    if task in SCORED_TASKS:
        if not re.fullmatch(r"[1-5]", text):
            raise ValueError(
                f"Expected integer 1-5 for {task}, received {text!r}."
            )
        return text

    raise ValueError(f"Unexpected task category: {task!r}")


def load_completed_pairs() -> set[tuple[str, str]]:
    """Allow safe resume without duplicating completed judge evaluations."""
    if not JUDGE_OUTCOMES_PATH.exists():
        return set()

    with JUDGE_OUTCOMES_PATH.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    completed = set()

    for row in rows:
        task = normalize_task(row["Task"])
        parse_outcome(task, row["Outcome"])

        pair = (
            row["Run ID"].strip(),
            row["LLM Judge Model"].strip(),
        )

        if pair in completed:
            raise ValueError(f"Duplicate judge result already exists for {pair}.")

        completed.add(pair)

    return completed


def load_runs() -> list[dict]:
    """
    Use exp_metrics.csv as the source of truth for run_id and task_category.

    This avoids reconstructing task categories from prompt order.
    """
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"Missing experiment metrics: {METRICS_PATH}")

    with METRICS_PATH.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if len(rows) != EXPECTED_RUN_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_RUN_COUNT} completed experiment runs, "
            f"found {len(rows)}. Resolve technical failures before judging."
        )

    required = {"run_id", "task_category"}
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"exp_metrics.csv is missing columns: {sorted(missing)}")

    seen = set()
    runs = []

    for row in rows:
        run_id = row["run_id"].strip()
        match = RUN_ID_RE.fullmatch(run_id)

        if match is None:
            raise ValueError(f"Invalid run_id: {run_id!r}")

        if run_id in seen:
            raise ValueError(f"Duplicate run_id: {run_id}")

        task = normalize_task(row["task_category"])
        if task not in ALL_TASKS:
            raise ValueError(f"Unexpected task category for {run_id}: {task!r}")

        seen.add(run_id)

        runs.append({
            "run_id": run_id,
            "prompt_id": match.group(1),
            "task": task,
        })

    return runs


def load_exact_input(prompt_id: str) -> str:
    path = INPUT_DIR / f"{prompt_id}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing SLM input: {path}")
    return path.read_text(encoding="utf-8")


def load_final_response(run_id: str, expected_task: str) -> str:
    """
    Load the final response content written by exp.py.

    Internal model `thinking` is deliberately not sent to judges.
    """
    path = RESPONSE_DIR / f"{run_id}.resp.txt"

    if not path.exists():
        raise FileNotFoundError(f"Missing SLM response: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if data.get("run_id") != run_id:
        raise ValueError(f"Run-ID mismatch in {path}.")

    response_task = normalize_task(data.get("task_category", ""))
    if response_task != expected_task:
        raise ValueError(
            f"Task mismatch for {run_id}: "
            f"metrics={expected_task!r}, response={response_task!r}."
        )

    content = data.get("content")
    return "" if content is None else str(content)


# ============================================================
# OpenRouter judging
# ============================================================

async def judge_once(
    judge_model: str,
    system_prompt: str,
    judge_input: str,
    semaphore: asyncio.Semaphore,
) -> object:
    """
    OpenRouter's SDK call is synchronous, so run it in a worker thread.
    The semaphore limits concurrent requests.
    """

    def send_request():
        with OpenRouter(api_key=OPENROUTER_API_KEY) as client:
            response = client.chat.send(
                model=judge_model,
                provider={"sort": {"by": "price"}},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": judge_input},
                ],
            )
        return response.choices[0].message.content

    async with semaphore:
        return await asyncio.to_thread(send_request)


async def evaluate_pair(
    run_id: str,
    task: str,
    judge_model: str,
    judge_input: str,
    system_prompt: str,
    semaphore: asyncio.Semaphore,
    csv_lock: asyncio.Lock,
    completed_pairs: set[tuple[str, str]],
) -> None:
    """
    Retry only when the API fails or the returned output is malformed.

    We keep the first valid standardized judgement and never choose between
    multiple valid scores.
    """
    pair = (run_id, judge_model)

    if pair in completed_pairs:
        return

    last_error = None
    last_raw = None

    for attempt in range(1, MAX_ATTEMPTS_PER_JUDGEMENT + 1):
        try:
            last_raw = await judge_once(
                judge_model=judge_model,
                system_prompt=system_prompt,
                judge_input=judge_input,
                semaphore=semaphore,
            )

            outcome = parse_outcome(task, last_raw)

            async with csv_lock:
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

            print(f"{judge_model} - {run_id}: {outcome}")
            return

        except Exception as error:
            last_error = error
            print(
                f"{judge_model} - {run_id}: attempt {attempt} failed: {error}"
            )

    async with csv_lock:
        append_csv(
            JUDGE_FAILURES_PATH,
            {
                "Run ID": run_id,
                "LLM Judge Model": judge_model,
                "Task": task,
                "Exception Type": type(last_error).__name__,
                "Exception Message": str(last_error),
                "Last Raw Response": "" if last_raw is None else str(last_raw),
            },
        )


# ============================================================
# Main
# ============================================================

async def main() -> None:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    if EXPERIMENT_ID == "REPLACE_WITH_FINAL_EXPERIMENT_ID":
        raise RuntimeError(
            "Set EXPERIMENT_ID at the top of llmjudge.py before running."
        )

    if not JUDGE_INSTRUCTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Missing judge instructions: {JUDGE_INSTRUCTIONS_PATH}"
        )

    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)

    system_prompt = JUDGE_INSTRUCTIONS_PATH.read_text(encoding="utf-8")
    runs = load_runs()
    completed_pairs = load_completed_pairs()

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    csv_lock = asyncio.Lock()

    tasks = []

    for run in runs:
        run_id = run["run_id"]
        prompt_id = run["prompt_id"]
        task = run["task"]

        if run_id != 'M3G__UP030':
            continue

        slm_input = load_exact_input(prompt_id)
        slm_content = load_final_response(run_id, task)

        judge_input = (
            f"<task>{task}</task>\n"
            f"<input>{slm_input}</input>\n"
            f"<body>{slm_content}</body>"
        )

        for judge_model in LLM_JUDGE_MODELS:
            if (run_id, judge_model) not in completed_pairs:
                tasks.append(
                    evaluate_pair(
                        run_id=run_id,
                        task=task,
                        judge_model=judge_model,
                        judge_input=judge_input,
                        system_prompt=system_prompt,
                        semaphore=semaphore,
                        csv_lock=csv_lock,
                        completed_pairs=completed_pairs,
                    )
                )

    total_required = len(runs) * len(LLM_JUDGE_MODELS)

    print(f"Experiment runs: {len(runs)}")
    print(f"Required judge evaluations: {total_required}")
    print(f"Already completed: {len(completed_pairs)}")
    print(f"Pending: {len(tasks)}")

    await asyncio.gather(*tasks)

    print()
    print(
        f"Completed valid judge evaluations: "
        f"{len(completed_pairs)} / {total_required}"
    )
    print(f"Outcomes: {JUDGE_OUTCOMES_PATH}")

    if len(completed_pairs) != total_required:
        print(
            f"Some evaluations failed. See {JUDGE_FAILURES_PATH} "
            "and rerun this script; completed pairs will be skipped."
        )


if __name__ == "__main__":
    asyncio.run(main())
