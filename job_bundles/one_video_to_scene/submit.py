#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["pyyaml"]
# ///
"""Submit the one_video_to_scene job, optionally restarting from a later stage of a previous job.

Fresh run:
    python submit.py --video tabletop.mp4

Re-run from a later stage, reusing the earlier stages' outputs of a previous job:
    python submit.py --from-job job-0123... --start-at Register

When restarting, this script:
  1. Reads the previous job's parameters, and reuses them unless you override them with -p.
  2. Downloads the previous job's outputs for every stage before --start-at.
  3. Copies those stage directories into a new directory that the new job takes as its
     PriorOutputDir input.
  4. Submits a copy of the bundle with the earlier steps removed.

Requires the Deadline Cloud CLI (`deadline`) on PATH and PyYAML. With uv, `uv run submit.py ...`
installs PyYAML automatically.
"""

import argparse
import datetime
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

BUNDLE_DIR = Path(__file__).resolve().parent
STAGES = ["Analyze", "Reconstruct", "Register", "Assemble", "Render"]
# Parameters that describe where a particular job reads and writes, so they're never reused.
PER_JOB_PARAMETERS = {"OutputDir", "PriorOutputDir", "InputVideo"}


def stage_dir_name(stage: str) -> str:
    return stage.lower()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", help="Input video. Required unless restarting after the Analyze stage.")
    parser.add_argument(
        "--output-dir",
        help="Output directory for the new job. Defaults to output/<timestamp> next to this script.",
    )
    parser.add_argument("--from-job", help="Previous job ID whose earlier-stage outputs to reuse.")
    parser.add_argument(
        "--start-at",
        choices=STAGES,
        type=lambda s: s.capitalize(),
        help="First stage to run. Requires --from-job when not Analyze.",
    )
    parser.add_argument(
        "-p", "--parameter", action="append", default=[], metavar="NAME=VALUE", help="Override a job parameter."
    )
    parser.add_argument("--name", help="Job name.")
    parser.add_argument("--max-retries-per-task", type=int, help="Passed to deadline bundle submit.")
    parser.add_argument("--farm-id")
    parser.add_argument("--queue-id")
    parser.add_argument("--profile")
    parser.add_argument("--region")
    parser.add_argument("--dry-run", action="store_true", help="Prepare the bundle and print the command only.")
    args = parser.parse_args()

    if args.start_at and args.start_at != "Analyze" and not args.from_job:
        parser.error("--start-at after Analyze requires --from-job")
    if args.from_job and not args.start_at:
        parser.error("--from-job requires --start-at")
    if not args.from_job and not args.video:
        parser.error("--video is required for a fresh run")
    return args


def deadline_cli(args, *cli_args) -> str:
    cmd = ["deadline", *cli_args]
    for flag in ("farm_id", "queue_id", "profile", "region"):
        value = getattr(args, flag)
        if value:
            cmd += [f"--{flag.replace('_', '-')}", value]
    print("+", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, text=True)
    return result.stdout


def get_job_parameters(args, job_id: str) -> dict:
    """Return {name: value} for the parameters of a previous job."""
    job = yaml.safe_load(deadline_cli(args, "job", "get", "--job-id", job_id))
    params = {}
    for name, typed_value in (job.get("parameters") or {}).items():
        # Each value is a single-key dict like {"string": "..."} or {"path": "..."}.
        params[name] = str(next(iter(typed_value.values())))
    return params


def download_stage_outputs(args, job_id: str, job_output_dir: str, stage: str):
    """Download one stage's outputs from a previous job. They land at the job's original OutputDir path."""
    include = f"{job_output_dir.rstrip('/')}/{stage_dir_name(stage)}/"
    deadline_cli(
        args,
        "job",
        "download-output",
        "--job-id",
        job_id,
        "--include",
        include,
        "--match-paths-by",
        "job",
        "--conflict-resolution",
        "overwrite",
        "--yes",
        "--output",
        "json",
    )


def gather_prior_outputs(args, prior_params: dict, stages_needed: list, prior_dir: Path):
    """Collect the given stages' outputs from the previous job into prior_dir/<stage>."""
    old_output = Path(prior_params["OutputDir"])
    # If the previous job was itself a restart, the stages it skipped are in its PriorOutputDir.
    old_prior = Path(prior_params["PriorOutputDir"]) if prior_params.get("PriorOutputDir") else None

    prior_dir.mkdir(parents=True, exist_ok=True)
    for stage in stages_needed:
        name = stage_dir_name(stage)
        download_stage_outputs(args, args.from_job, str(old_output), stage)
        candidates = [old_output / name] + ([old_prior / name] if old_prior else [])
        source = next((c for c in candidates if c.is_dir() and any(c.iterdir())), None)
        if source is None:
            sys.exit(
                f"ERROR: Could not find outputs for stage {stage} of {args.from_job}. Looked in: "
                + ", ".join(str(c) for c in candidates)
            )
        print(f"Using {stage} outputs from {source}")
        shutil.copytree(source, prior_dir / name, dirs_exist_ok=True)


def write_pruned_bundle(skip: list, dest: Path):
    """Copy the bundle to dest, removing the skipped steps and any dependencies on them."""
    shutil.copytree(BUNDLE_DIR / "scripts", dest / "scripts")
    template = yaml.safe_load((BUNDLE_DIR / "template.yaml").read_text())
    template["steps"] = [s for s in template["steps"] if s["name"] not in skip]
    for step in template["steps"]:
        deps = [d for d in step.get("dependencies", []) if d["dependsOn"] not in skip]
        if deps:
            step["dependencies"] = deps
        else:
            step.pop("dependencies", None)
    (dest / "template.yaml").write_text(yaml.safe_dump(template, sort_keys=False, width=1000))


def main():
    args = parse_args()
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_dir or BUNDLE_DIR / "output" / timestamp).resolve()
    start_at = args.start_at or "Analyze"
    skip = STAGES[: STAGES.index(start_at)]

    params = {}
    if args.from_job:
        prior_params = get_job_parameters(args, args.from_job)
        params.update({k: v for k, v in prior_params.items() if k not in PER_JOB_PARAMETERS})
        if skip:
            prior_dir = output_dir.parent / f"{output_dir.name}_prior"
            gather_prior_outputs(args, prior_params, skip, prior_dir)
            params["PriorOutputDir"] = str(prior_dir)
        if not args.video and start_at == "Analyze":
            args.video = prior_params.get("InputVideo")

    if args.video:
        params["InputVideo"] = str(Path(args.video).resolve())
    elif "Analyze" not in skip:
        sys.exit("ERROR: The Analyze stage needs --video")
    params["OutputDir"] = str(output_dir)
    for override in args.parameter:
        name, sep, value = override.partition("=")
        if not sep:
            sys.exit(f"ERROR: Expected NAME=VALUE, got {override!r}")
        params[name] = value

    name = args.name or f"One Video to Scene: {Path(params.get('InputVideo') or 'restart').stem}"
    if args.from_job:
        name += f" (from {start_at} of {args.from_job[:12]})"

    with tempfile.TemporaryDirectory(prefix="one_video_to_scene_") as tmp:
        bundle = Path(tmp) / "bundle"
        write_pruned_bundle(skip, bundle)
        cli = ["bundle", "submit", str(bundle), "--name", name, "--yes"]
        if args.max_retries_per_task is not None:
            cli += ["--max-retries-per-task", str(args.max_retries_per_task)]
        for key, value in params.items():
            cli += ["-p", f"{key}={value}"]
        if args.dry_run:
            print("Steps:", [s for s in STAGES if s not in skip])
            print("deadline", " ".join(cli))
            return
        out = deadline_cli(args, *cli)
        print(out)
        match = re.search(r"job-[0-9a-f]{32}", out)
        if match:
            print(f"Job ID: {match.group(0)}")
            print(f"Outputs will download to: {output_dir}")
            print(
                f"Re-run a later stage with: python {os.path.basename(__file__)} "
                f"--from-job {match.group(0)} --start-at <Stage>"
            )


if __name__ == "__main__":
    main()
