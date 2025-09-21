#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess


HERE = Path(__file__).resolve().parent


def build_cmd(job: dict, python_exec: str) -> list:
    script_kind = job.get("script", "fail").lower()
    if script_kind in ("fail", "fail_instances", "fi"):
        script_path = HERE / "prompt_optimization_fail_instances.py"
    elif script_kind in ("gated", "fail_gated"):
        script_path = HERE / "prompt_optimization_fail_instances_gated.py"
    elif script_kind in ("standard", "prompt_optimization", "po"):
        script_path = HERE / "prompt_optimization.py"
    else:
        raise ValueError(f"Unknown script kind: {script_kind}")

    args = [python_exec, str(script_path)]

    # Common required/typical args
    if job.get("task"):
        args += ["--task", str(job["task"])]
    if job.get("evaluation_engine"):
        args += ["--evaluation_engine", str(job["evaluation_engine"])]
    if job.get("test_engine"):
        args += ["--test_engine", str(job["test_engine"])]

    # Optional common knobs
    for k in ["batch_size", "max_epochs", "seed", "num_threads"]:
        if k in job and job[k] is not None:
            args += [f"--{k}", str(job[k])]

    # Script-specific knobs
    if script_kind in ("fail", "fail_instances", "fi", "gated", "fail_gated"):
        for k in [
            "preserve_sample_size",
            "lambda_gating",
            "n_hard_examples",
            # gated only (will be ignored by non-gated script if not present)
            "gate_sample_size",
            "accept_epsilon",
            "resplit_60",
            "train_size",
            "val_size",
            "test_size",
        ]:
            if k in job and job[k] is not None:
                v = job[k]
                if isinstance(v, bool):
                    if v:
                        args += [f"--{k}"]
                else:
                    args += [f"--{k}", str(v)]

    # Extra passthrough args
    extra_args = job.get("extra_args", [])
    if extra_args:
        if not isinstance(extra_args, list):
            raise ValueError("extra_args must be a list of strings")
        args += [str(x) for x in extra_args]

    return args


def _snapshot_artifacts(globs: list[str]) -> dict:
    import glob
    snap = {}
    for g in globs:
        for p in glob.glob(g):
            try:
                st = os.stat(p)
                snap[p] = (st.st_mtime_ns, st.st_size)
            except FileNotFoundError:
                continue
    return snap


def _diff_artifacts(before: dict, after: dict) -> list[str]:
    changed = []
    for p, meta in after.items():
        if p not in before or before[p] != meta:
            changed.append(p)
    return changed


def run_job(job: dict, python_exec: str, logs_dir: Path, artifacts_globs: list[str], artifacts_root: Path) -> dict:
    name = job.get("name")
    if not name:
        # generate a friendly name
        script = job.get("script", "fail")
        task = job.get("task", "TASK")
        test_engine = job.get("test_engine", "ENGINE")
        name = f"{script}_{task}_{test_engine}".replace("/", "_").replace(":", "_")

    cmd = build_cmd(job, python_exec)
    log_path = logs_dir / f"{name}.log"
    # Snapshot artifacts before run
    before = _snapshot_artifacts([str(p) for p in artifacts_globs])
    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        cwd=str(HERE.parent),  # run from repo root
        env=os.environ.copy(),
    )
    prefix = f"[{name}] "
    with open(log_path, "w", encoding="utf-8") as f:
        header = "$ " + " ".join(cmd) + "\n\n"
        f.write(header)
        print(prefix + header.strip())
        for raw_line in proc.stdout:
            line = raw_line.rstrip("\n")
            formatted = prefix + line + "\n"
            f.write(formatted)
            print(prefix + line)
    rc = proc.wait()
    dt = time.time() - t0
    # Snapshot and collect changed artifacts
    after = _snapshot_artifacts([str(p) for p in artifacts_globs])
    changed = _diff_artifacts(before, after)
    # Copy into per-run folder
    run_dir = artifacts_root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for src in changed:
        try:
            dst = run_dir / Path(src).name
            # If name collides across jobs, prefix with timestamp
            if dst.exists():
                dst = run_dir / f"{int(time.time())}_{Path(src).name}"
            from shutil import copy2
            copy2(src, dst)
            copied.append(str(dst))
        except Exception:
            continue
    # Also copy the log into the run directory for easier grouping
    try:
        from shutil import copy2
        copy2(log_path, run_dir / Path(log_path).name)
    except Exception:
        pass
    return {"name": name, "returncode": rc, "seconds": dt, "log": str(log_path), "artifacts": copied, "run_dir": str(run_dir)}


def main():
    ap = argparse.ArgumentParser(description="Multi-thread runner for prompt optimization experiments")
    # Default config path under evaluation/configs
    default_cfg = HERE / "configs" / "multi_run.json"
    ap.add_argument("--config", default=str(default_cfg), help="Path to JSON config containing a list of jobs")
    ap.add_argument("--max_workers", type=int, default=3, help="Max concurrent jobs")
    ap.add_argument("--python", default=sys.executable, help="Python executable to use")
    ap.add_argument("--logs_dir", default=str(HERE.parent / "figures" / "multirun_logs"), help="Directory to write per-run logs")
    ap.add_argument(
        "--artifacts_globs",
        default="figures/results*.json,figures/*results*.json",
        help="Comma-separated globs of result files to collect per run",
    )
    ap.add_argument(
        "--artifacts_dir",
        default=str(HERE.parent / "figures" / "multirun_artifacts"),
        help="Directory to store grouped artifacts for each run",
    )
    ap.add_argument("--dry_run", action="store_true", help="Print commands without executing")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    # If config does not exist, scaffold a sample
    if not cfg_path.exists():
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        sample_jobs = [
            {
                "name": "date_understanding_fail",
                "script": "fail",
                "task": "DateUnderstanding",
                "evaluation_engine": "gpt-4o",
                "test_engine": "gpt-3.5-turbo-0125",
                "batch_size": 8,
                "n_hard_examples": 3
            },
            {
                "name": "mmlu_ml_gated_resplit",
                "script": "gated",
                "task": "MMLU_machine_learning",
                "evaluation_engine": "gpt-4o",
                "test_engine": "gpt-3.5-turbo-0125",
                "batch_size": 4,
                "n_hard_examples": 3,
                "gate_sample_size": 100,
                "accept_epsilon": -0.06,
                "resplit_60": True,
                "train_size": 200
            },
            {
                "name": "bbh_counting_standard",
                "script": "standard",
                "task": "BBH_object_counting",
                "evaluation_engine": "gpt-4o",
                "test_engine": "gpt-3.5-turbo-0125",
                "batch_size": 8,
                "max_epochs": 1
            }
        ]
        # Write sample config
        cfg_path.write_text(json.dumps(sample_jobs, indent=2), encoding="utf-8")
        print(f"No config provided/found. Wrote sample config to {cfg_path}")

    jobs = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(jobs, list):
        raise ValueError("Config must be a JSON array of job objects")

    logs_dir = Path(args.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        for j in jobs:
            print(" ".join(build_cmd(j, args.python)))
        return

    artifacts_globs = [g.strip() for g in args.artifacts_globs.split(",") if g.strip()]
    artifacts_root = Path(args.artifacts_dir)
    artifacts_root.mkdir(parents=True, exist_ok=True)

    results = []
    with ThreadPoolExecutor(max_workers=min(args.max_workers, max(1, len(jobs)))) as ex:
        futs = {ex.submit(run_job, j, args.python, logs_dir, artifacts_globs, artifacts_root): j for j in jobs}
        for fut in as_completed(futs):
            res = fut.result()
            results.append(res)
            status = "OK" if res["returncode"] == 0 else f"RC={res['returncode']}"
            print(f"[{status}] {res['name']} ({res['seconds']:.1f}s) -> {res['run_dir']}")

    # Write a summary JSON next to logs
    summary_path = logs_dir / f"summary_{int(time.time())}.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
