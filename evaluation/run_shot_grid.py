#!/usr/bin/env python3
"""Run prompt_optimization_fail_instances.py across engines and shot settings per task."""

import argparse
import subprocess
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from clear_cache import clear_cache


CACHE_LOCK = threading.Lock()


DEFAULT_EVAL_ENGINES = [
    "gemini-2.5-flash",
    "gpt-5",
    "gpt-4o",
]

DEFAULT_SHOTS = [0, 3]

DEFAULT_TASKS = [
    "MMLU_machine_learning",
    "DateUnderstanding",
    "BBH_object_counting",
    "MultiArith",
    "BBH_penguins_in_a_table",
    "BBH_geometric_shapes",
    "BBH_navigate",
    "GSM8K_DSPy",
]


def run_single(task: str, evaluation_engine: str, shots: int, script_path: Path, logs_dir: Path, python_exec: str):
    log_dir = logs_dir / task
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{task}_{evaluation_engine}_shots{shots}.log"

    cmd = [
        python_exec,
        str(script_path),
        "--task",
        task,
        "--evaluation_engine",
        evaluation_engine,
        "--shots",
        str(shots),
    ]

    start = time.time()
    print(f"[START] task={task} eval={evaluation_engine} shots={shots}")

    with log_file.open("w", encoding="utf-8") as logf:
        logf.write("$ " + " ".join(cmd) + "\n\n")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            logf.write(line)
            print(f"[{task}] {line.rstrip()}" )
        returncode = proc.wait()

    elapsed = time.time() - start
    status = "OK" if returncode == 0 else f"RC={returncode}"
    print(f"[DONE] task={task} eval={evaluation_engine} shots={shots} {status} ({elapsed:.1f}s) -> {log_file}")

    with CACHE_LOCK:
        clear_cache(confirm=False)

    return {
        "task": task,
        "evaluation_engine": evaluation_engine,
        "shots": shots,
        "returncode": returncode,
        "seconds": elapsed,
        "log": str(log_file),
    }


def run_for_task(task: str, evaluation_engines, shots_list, script_path: Path, logs_dir: Path, python_exec: str):
    results = []
    for eval_engine in evaluation_engines:
        for shots in shots_list:
            result = run_single(task, eval_engine, shots, script_path, logs_dir, python_exec)
            results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description="Run fail script across engines/shots per task")
    parser.add_argument("--python", default="python3", help="Python executable to use")
    parser.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS, help="Tasks to evaluate")
    parser.add_argument("--evaluation_engines", nargs="*", default=DEFAULT_EVAL_ENGINES, help="Evaluation engines to sweep")
    parser.add_argument("--shots", nargs="*", type=int, default=DEFAULT_SHOTS, help="Shot counts to sweep")
    parser.add_argument("--logs_dir", default=str(Path("evaluation") / "shots_runs"), help="Directory to store run logs")
    args = parser.parse_args()

    script_path = Path(__file__).resolve().parent / "prompt_optimization_fail_instances.py"
    logs_dir = Path(args.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    results = []
    with ThreadPoolExecutor(max_workers=len(args.tasks)) as executor:
        futures = {
            executor.submit(
                run_for_task,
                task,
                args.evaluation_engines,
                args.shots,
                script_path,
                logs_dir,
                args.python,
            ): task
            for task in args.tasks
        }

        for future in as_completed(futures):
            task = futures[future]
            try:
                task_results = future.result()
                results.extend(task_results)
            except Exception as exc:
                print(f"[ERROR] task={task} raised an exception: {exc}")

    summary_path = logs_dir / f"summary_{int(time.time())}.json"
    import json

    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
