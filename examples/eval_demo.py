"""Run a MIKASA-Robo-VLA benchmark evaluation.

Replace ``DummyChunkPolicy`` with your own policy adapter.  The benchmark
runner handles CSV split loading, canonical per-task seeds, action chunk
queues, ``success_once`` latching, JSON files, and split aggregation.

Results are saved to a timestamped subdirectory under ``--output-dir`` so
successive runs never overwrite each other.

Examples (run from the repository root)
----------------------------------------
Smoke-test one task, one episode::

    uv run python examples/eval_demo.py \
        --num-episodes 1 --sim-backend cpu \
        --output-dir eval_results/dummy

Full canonical Short-split run, 50 episodes per task::

    uv run python examples/eval_demo.py \
        --split short \
        --output-dir results/my_model

Specific tasks::

    uv run python examples/eval_demo.py \
        --task RememberColor3-VLA-v0 \
        --task ShellGameTouch-VLA-v0 \
        --output-dir results/my_model

All 90 benchmark tasks::

    uv run python examples/eval_demo.py \
        --split all \
        --output-dir results/my_model

With per-episode rollout videos::

    uv run python examples/eval_demo.py \
        --split short \
        --save-videos \
        --output-dir results/my_model
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import argparse
import json
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple

import torch

from mikasa_robo_suite.vla.benchmarking import (
    NUM_EPISODES_PER_TASK,
    START_SEED,
    BenchmarkConfig,
    JsonDict,
    RichBenchmarkUI,
    evaluate_benchmark,
    make_run_dir,
    select_benchmark_tasks,
)


class DummyChunkPolicy:
    """Return random action chunks in the canonical 7D EE-delta action space."""

    def __init__(self, chunk_size: int = 8, action_dim: int = 7):
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
        self.chunk_size = int(chunk_size)
        self.action_dim = int(action_dim)

    @torch.no_grad()
    def forward(self, obs: Mapping[str, object]) -> torch.Tensor:
        proprio = obs.get("proprio")
        device = proprio.device if torch.is_tensor(proprio) else torch.device("cpu")
        return torch.empty(
            (self.chunk_size, self.action_dim),
            device=device,
            dtype=torch.float32,
        ).uniform_(-1.0, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--split",
        choices=("short", "medium", "long", "all"),
        help=(
            "Evaluate every task in the given horizon split, "
            "or 'all' for all 90 benchmark tasks."
        ),
    )
    selection.add_argument(
        "--task",
        action="append",
        dest="tasks",
        metavar="ENV_ID",
        help="Evaluate one env ID. Repeat to build an arbitrary subset.",
    )

    parser.add_argument("--num-episodes", type=int, default=NUM_EPISODES_PER_TASK)
    parser.add_argument("--start-seed", type=int, default=START_SEED)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval_results") / "dummy",
        help="Base directory. Results go into a timestamped subdirectory.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        metavar="RUN_DIR",
        help=(
            "Resume an interrupted run. Pass the timestamped run directory "
            "(e.g. eval_results/dummy/short/2026-05-21_18-52-16). "
            "Tasks that already have a result JSON there are skipped. "
            "The split is inferred from existing results; override with --split."
        ),
    )
    parser.add_argument(
        "--save-videos",
        action="store_true",
        help=(
            "Save a rollout video for every episode of every task. "
            "Videos are written to <output-dir>/<timestamp>/videos/<env_id>/."
        ),
    )
    parser.add_argument(
        "--sim-backend",
        default="gpu",
        help="ManiSkill sim backend ('cpu' or 'gpu'). Default: gpu.",
    )
    parser.add_argument("--render-mode", default="all")
    return parser.parse_args()


def _split_label(tasks: Sequence) -> str:
    splits = {task.split for task in tasks}
    return next(iter(splits)) if len(splits) == 1 else "custom"


def _load_resume_state(resume_dir: Path) -> Tuple[List[JsonDict], Optional[str]]:
    """Load completed task results from *resume_dir* and infer the benchmark split.

    Returns ``(completed_results, inferred_split)``.  *inferred_split* is one of
    ``"short"``, ``"medium"``, ``"long"``, ``"all"``, or ``None`` if it cannot
    be determined (user must pass ``--split`` explicitly in that case).
    """
    if not resume_dir.is_dir():
        raise FileNotFoundError(f"--resume directory not found: {resume_dir}")

    completed: List[JsonDict] = []
    for p in sorted(resume_dir.glob("*.json")):
        if p.stem == "summary":
            continue
        with p.open(encoding="utf-8") as f:
            completed.append(json.load(f))

    # Infer split from completed results: if all tasks share one split → that
    # split; if multiple splits are present → "all" (full benchmark run).
    splits = {r.get("split", "").lower() for r in completed}
    splits.discard("")
    if not splits:
        inferred_split: Optional[str] = None
    elif len(splits) == 1:
        inferred_split = next(iter(splits))
    else:
        inferred_split = "all"

    return completed, inferred_split


def main() -> None:
    args = parse_args()

    completed_results: List[JsonDict] = []

    if args.resume is not None:
        # ------------------------------------------------------------------ #
        # Resume mode: load finished tasks, determine remaining work          #
        # ------------------------------------------------------------------ #
        completed_results, inferred_split = _load_resume_state(args.resume)
        completed_ids = {r["env_id"] for r in completed_results}

        # Task selection: explicit flag wins; fall back to inferred split.
        if args.split:
            all_tasks = select_benchmark_tasks(split=args.split)
        elif args.tasks:
            all_tasks = select_benchmark_tasks(env_ids=args.tasks)
        elif inferred_split:
            all_tasks = select_benchmark_tasks(split=inferred_split)
        else:
            raise SystemExit(
                "Cannot infer the task split from the resume directory. "
                "Pass --split or --task explicitly."
            )

        tasks = [t for t in all_tasks if t.env_id not in completed_ids]
        run_dir = args.resume  # write results back into the same directory
    else:
        # ------------------------------------------------------------------ #
        # Fresh run                                                            #
        # ------------------------------------------------------------------ #
        if args.split:
            tasks = select_benchmark_tasks(split=args.split)
        elif args.tasks:
            tasks = select_benchmark_tasks(env_ids=args.tasks)
        else:
            tasks = select_benchmark_tasks(env_ids=["RememberColor3-VLA-v0"])

        run_dir = make_run_dir(args.output_dir / _split_label(tasks))

    config = BenchmarkConfig(
        start_seed=args.start_seed,
        n_episodes=args.num_episodes,
        sim_backend=args.sim_backend,
        save_videos=args.save_videos,
    )
    policy = DummyChunkPolicy(chunk_size=args.chunk_size)

    with RichBenchmarkUI(tasks, config.n_episodes, initial_results=completed_results or None) as ui:
        _, summary = evaluate_benchmark(
            tasks,
            policy,
            config,
            output_dir=run_dir,
            model={"name": "dummy-random-chunk-policy", "config": {"chunk_size": policy.chunk_size}},
            task_start_callback=ui.on_task_start,
            episode_callback=ui.on_episode_done,
            task_done_callback=ui.on_task_done,
            initial_results=completed_results if completed_results else None,
        )

    from rich.console import Console
    from rich.table import Table

    console = Console()
    console.print()

    summary_table = Table(
        title="[bold]Benchmark Summary[/bold]",
        show_header=True,
        header_style="bold cyan",
        border_style="blue",
    )
    summary_table.add_column("Memory Type", style="white")
    summary_table.add_column("SR", justify="right")

    for memory_type, sr in summary["sr_per_memory_type"].items():
        color = "bright_green" if sr >= 0.7 else "yellow" if sr >= 0.4 else "red"
        summary_table.add_row(memory_type, f"[{color}]{sr:.2%}[/{color}]")

    summary_table.add_section()
    sr_split = summary["sr_split"]
    split_color = "bright_green" if sr_split >= 0.7 else "yellow" if sr_split >= 0.4 else "red"
    summary_table.add_row(
        "[bold]Overall SR[/bold]",
        f"[bold {split_color}]{sr_split:.2%}[/bold {split_color}]",
    )

    console.print(summary_table)
    console.print(f"[dim]JSON results → {run_dir}[/dim]")
    if args.save_videos:
        console.print(f"[dim]Videos      → {run_dir / 'videos'}[/dim]")


if __name__ == "__main__":
    main()
