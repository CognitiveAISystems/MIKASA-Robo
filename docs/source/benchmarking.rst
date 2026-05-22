Benchmarking
============

MIKASA-Robo-VLA contains 90 tasks whose episode horizons range from **25 to
2160** simulation steps.  Training a VLA on all 90 simultaneously is hard
because the batch mixes episodes with vastly different lengths; it also
makes fair comparisons between models difficult.  To solve this we group
tasks into three **horizon splits** and define a canonical per-split
protocol.

.. contents:: On this page
   :depth: 2
   :local:

.. _horizon-splits:

Horizon Splits
--------------

Split assignment is deterministic from ``Max Length`` in
``mikasa_robo_vla_envs.csv``:

.. code-block:: text

   Max Length ≤ 200            → Short
   201 ≤ Max Length ≤ 601      → Medium
   Max Length > 601            → Long

.. list-table::
   :header-rows: 1
   :widths: 14 12 24 50

   * - Split
     - Tasks
     - Horizon (steps)
     - What it tests
   * - **Short**
     - 38
     - 25 – 200
     - Rapid cue encoding and short-term recall.
   * - **Medium**
     - 30
     - 201 – 601
     - Sustained working memory over moderately long episodes.
   * - **Long**
     - 22
     - 602 – 2160
     - Extended memory, multi-phase reasoning, procedural recall.

Memory-type distribution per split:

.. code-block:: text

   Short  (38): Spatial 14 · Object 9 · Negative 9 · Temporal 3 · Tracking 2 · Prospective 1
   Medium (30): Object 9 · Capacity 6 · Temporal 4 · Sequential 3 · Procedural 3 · Tracking 2 · Prospective 2 · Checklist 1
   Long   (22): Capacity 6 · Temporal 5 · Checklist 3 · Procedural 3 · Sequential 3 · Prospective 2

The full per-task table is in
`mikasa_robo_vla_envs.csv <https://github.com/CognitiveAISystems/MIKASA-Robo/blob/vla/mikasa_robo_vla_envs.csv>`_.

.. _canonical-protocol:

Canonical Protocol
------------------

The canonical benchmarking procedure for any VLA model:

1. **Choose a split** — ``short``, ``medium``, or ``long``.
2. **Download the datasets** for every task in that split from Hugging Face
   (one repository = one dataset = one task; available in
   `LeRobotDataset v3 <https://huggingface.co/datasets/mikasa-robo/mikasa-robo-vla-lerobot>`_
   and `RLDS <https://huggingface.co/datasets/mikasa-robo/mikasa-robo-vla-rlds>`_
   formats).
3. **Train in multi-task mode** on the combined data for the split.
4. **Evaluate on every task in that split** — 50 episodes per task, seeds
   ``4242424242 … 4242424291``.
5. **Report three numbers**: per-split SR (main metric), full per-task SR
   table, and per-memory-type SR breakdown within the split.  Optionally
   include mean return per task as a debug number.

See :doc:`evaluation_protocol` for the exact seed convention, metric
definitions, wrapper stack, action chunking rules, JSON output schema, and
the reproducibility checklist.

.. _running-eval:

Running an Evaluation
---------------------

The reference CLI is ``examples/eval_demo.py``.  It runs a checkpoint-free
``DummyChunkPolicy`` so you can verify the pipeline end-to-end before
attaching your own model.  Every run writes results to a **timestamped
subdirectory** under ``--output-dir`` so successive runs never overwrite each
other.

Smoke-test (one task, one episode, CPU backend)::

   uv run python examples/eval_demo.py \
       --num-episodes 1 --sim-backend cpu \
       --output-dir eval_results/dummy

Canonical Short-split run::

   uv run python examples/eval_demo.py \
       --split short \
       --output-dir eval_results/dummy

All 90 tasks::

   uv run python examples/eval_demo.py \
       --split all \
       --output-dir eval_results/dummy

Arbitrary subset of tasks::

   uv run python examples/eval_demo.py \
       --task RememberColor3-VLA-v0 \
       --task ShellGameTouch-VLA-v0 \
       --task BatteriesCheckerEasy-VLA-v0 \
       --output-dir eval_results/dummy

With per-episode rollout videos::

   uv run python examples/eval_demo.py \
       --split short \
       --save-videos \
       --output-dir eval_results/dummy

.. list-table:: ``eval_demo.py`` CLI flags
   :header-rows: 1
   :widths: 32 14 54

   * - Flag
     - Default
     - Description
   * - ``--split short|medium|long|all``
     - —
     - Evaluate every task in the given horizon split.  ``all`` runs all 90
       canonical tasks.  Mutually exclusive with ``--task``.
   * - ``--task ENV_ID``
     - —
     - Evaluate one specific env ID.  Repeat for a custom multi-task subset.
       Accepts any registered Gymnasium env, including IDs not in the
       canonical CSV.  Mutually exclusive with ``--split``.
   * - ``--num-episodes N``
     - ``50``
     - Episodes per task.
   * - ``--start-seed N``
     - ``4242424242``
     - Seed for episode ``i`` is ``start_seed + i``.
   * - ``--chunk-size K``
     - ``8``
     - Action chunk size for the built-in dummy policy.
   * - ``--output-dir PATH``
     - ``eval_results/dummy``
     - Base directory.  A timestamped subdir is created automatically.
   * - ``--resume RUN_DIR``
     - —
     - Continue an interrupted run.  Pass the existing timestamped directory
       (e.g. ``eval_results/dummy/short/2026-05-21_18-52-16``).  Tasks that
       already have a result ``.json`` there are skipped; results are written
       back to the same directory.  The split is inferred automatically from
       the existing files; override with ``--split`` if needed.
   * - ``--save-videos``
     - off
     - Record a video for every episode of every task.
       Videos go to ``<output-dir>/<timestamp>/videos/<env_id>/``.
   * - ``--sim-backend cpu|gpu``
     - auto
     - Override the ManiSkill simulation backend.
   * - ``--render-mode``
     - ``all``
     - Passed directly to ``gym.make``.

.. _resume:

Resuming an Interrupted Run
----------------------------

A full-split evaluation can take hours.  If a run is interrupted (OOM,
preemption, Ctrl-C), pass ``--resume`` to continue from exactly where it
stopped:

.. code-block:: bash

   # The split is inferred automatically from the existing result files.
   uv run python examples/eval_demo.py \
       --resume eval_results/dummy/short/2026-05-21_18-52-16

   # Override or extend — e.g. resume files that mixed splits:
   uv run python examples/eval_demo.py \
       --resume eval_results/dummy/all/2026-05-21_18-52-16 \
       --split all

How it works:

1. Every ``*.json`` file in ``RUN_DIR`` (except ``summary.json``) is loaded
   as a completed task result.
2. The completed env IDs are subtracted from the full task list — only the
   remaining tasks are evaluated.
3. New result files are written into the same ``RUN_DIR``; ``summary.json``
   is updated to include both old and new tasks after every completed task.
4. The :ref:`rich terminal display <rich-ui>` pre-populates the results table
   with already-completed tasks so you get a complete view at a glance.

.. note::

   Use the same ``--start-seed`` and ``--num-episodes`` values you used for
   the original run, or the resumed results will not be comparable with the
   already-saved ones.

.. note::

   **Mixed-split resume directories.**  When the existing ``*.json`` files
   come from different splits (e.g. you originally ran ``--split all``),
   ``--resume`` reads the ``"split"`` field of each result and:

   - if every file reports the same split, that split is selected;
   - if multiple distinct splits are present, ``"all"`` is inferred and the
     remaining 90 tasks are scheduled.

   Override either case by passing ``--split`` (or ``--task ENV_ID …``)
   explicitly.

.. _rich-ui:

Rich Terminal Display
---------------------

When running ``eval_demo.py`` (or any script that uses :class:`RichBenchmarkUI`)
the terminal shows a live panel with three parts that update after every episode:

.. code-block:: text

   ╭─── MIKASA-Robo-VLA Benchmark  4/38 tasks ───────────────────────────────╮
   │ ⠹ Tasks  [4/38] ShellGamePick-VLA-v0  ━━━━━━━━━━╸━━  4/38  10%  0:01:…  │
   │ ⠹ Episodes  ShellGamePick-VLA-v0      ━━━━━━━━━━━━━ 50/50 100%          │
   │ ──────────────────────────────────────────────────────────────────────  │
   │ ┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━┳━━━━━━━┳━━━━┳━━━━━━━━┓  │
   │ ┃ Task                     ┃ Split ┃ Memory    ┃  Eps  ┃ SR ┃ Return ┃  │
   │ ┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━╇━━━━━━━╇━━━━╇━━━━━━━━┩  │
   │ │ RememberColor3-VLA-v0    │ Short │ Object    │   50  │ 82%│ 0.6341 │  │
   │ │ ShellGameTouch-VLA-v0    │ Short │ Spatial   │   50  │ 56%│ 0.4120 │  │
   │ │ ShellGamePick-VLA-v0 …   │ Short │ Spatial   │ 32/50 │ 44%│ 0.3012 │  │
   │ └──────────────────────────┴───────┴───────────┴───────┴────┴────────┘  │
   ╰─────────────────────────────────────────────────────────────────────────╯

- **Top bar** — overall task progress with elapsed time and ETA.
- **Middle bar** — per-episode progress for the task currently running;
  resets to zero when a new task begins.
- **Table** — one row per completed task (with final SR coloured green /
  yellow / red) plus one live row for the task currently in progress
  (shown with ``(running…)`` label and values updated every episode).

When ``--resume`` is used, already-completed tasks appear in the table
immediately from the first frame so you always see the full picture.

To use :class:`RichBenchmarkUI` from your own script see the
:ref:`Python API <rich-ui-api>` section below.

.. _output-layout:

Output Layout
-------------

Each run creates its own directory under ``<output-dir>/<split>/``:

.. code-block:: text

   eval_results/
     my_model/
       short/
         2026-05-21_15-30-00/          ← one directory per run
           summary.json                ← updated after each completed task
           RememberColor3-VLA-v0.json
           ShellGameTouch-VLA-v0.json
           ...
           videos/                     ← only when --save-videos
             RememberColor3-VLA-v0/
               0.mp4                   ← episode 0
               1.mp4                   ← episode 1
               ...

``summary.json`` is rewritten after every task completes, so you can
inspect it at any point during a long run.

.. _python-api:

Python API
----------

Import the benchmark module directly to integrate evaluation into your own
scripts.

Selecting tasks
~~~~~~~~~~~~~~~

.. code-block:: python

   from mikasa_robo_suite.vla.benchmarking import select_benchmark_tasks

   tasks = select_benchmark_tasks(split="short")   # 38 tasks
   tasks = select_benchmark_tasks(split="medium")  # 30 tasks
   tasks = select_benchmark_tasks(split="long")    # 22 tasks
   tasks = select_benchmark_tasks(split="all")     # all 90 tasks

   # Explicit list — any subset, order preserved
   tasks = select_benchmark_tasks(env_ids=[
       "RememberColor3-VLA-v0",
       "ShellGameTouch-VLA-v0",
   ])

   # Env IDs not in the canonical CSV are accepted (split="custom",
   # memory_type="Unknown") so any registered Gymnasium env can be evaluated.
   tasks = select_benchmark_tasks(env_ids=["MyCustomTask-VLA-v0"])

Each returned :class:`~mikasa_robo_suite.vla.benchmarking.BenchmarkTask` is a
small dataclass carrying everything ``evaluate_benchmark`` needs to construct
and run a task:

.. list-table:: ``BenchmarkTask`` fields
   :header-rows: 1
   :widths: 28 18 54

   * - Field
     - Type
     - Meaning
   * - ``env_id``
     - ``str``
     - Gymnasium env ID, e.g. ``"RememberColor3-VLA-v0"``.
   * - ``split``
     - ``str``
     - Horizon split label: ``"Short"`` / ``"Medium"`` / ``"Long"``,
       or ``"custom"`` for env IDs not in ``mikasa_robo_vla_envs.csv``.
   * - ``memory_type``
     - ``str``
     - Memory category from the CSV (``"Object"``, ``"Spatial"``, ``"Capacity"``,
       …), or ``"Unknown"`` for custom env IDs.
   * - ``max_episode_steps``
     - ``int``
     - Episode horizon (``Max Length`` column in the CSV).  The runner uses
       this to size the per-episode progress bar and bound the rollout loop.

Running a benchmark
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mikasa_robo_suite.vla.benchmarking import (
       BenchmarkConfig,
       evaluate_benchmark,
       make_run_dir,
       select_benchmark_tasks,
   )

   tasks = select_benchmark_tasks(split="short")
   config = BenchmarkConfig(
       n_episodes=50,
       save_videos=True,        # record per-episode rollout videos
       sim_backend="gpu",
   )
   policy = MyPolicy(chunk_size=8)

   run_dir = make_run_dir("eval_results/my_model/short")
   results, summary = evaluate_benchmark(
       tasks,
       policy,
       config,
       output_dir=run_dir,
       model={"name": "my-vla", "config": {"checkpoint": "path/to/ckpt"}},
       progress=print,
   )
   print(f"SR_split = {summary['sr_split']:.2%}")

``evaluate_benchmark`` writes ``<env_id>.json`` and updates ``summary.json``
after every task, so partial runs are always readable on disk.

``make_run_dir(base)`` creates ``base/YYYY-MM-DD_HH-MM-SS/`` and returns the
path.  Call it once at the start of a run to avoid overwriting previous
results.

Plugging in a real policy
~~~~~~~~~~~~~~~~~~~~~~~~~

The benchmark runner expects any object with a ``chunk_size`` attribute and a
``forward(obs)`` method:

.. code-block:: python

   class MyPolicy:
       chunk_size = 8  # actions returned per forward pass

       def forward(self, obs: dict) -> torch.Tensor:
           # obs["rgb"]     shape (1, 128, 128, 6), uint8
           # obs["proprio"] shape (1, 7),           float32
           # info["language_instruction"]  str
           #
           # Return shape: (chunk_size, 7), float32, values in [-1, 1]
           ...

For ``chunk_size=1`` this is identical to standard per-step inference: the
internal FIFO action queue is consumed every step, so ``policy.forward(obs)``
is called on every simulator step.

.. note::

   **``success_once`` is a latch, not a count.**  The runner OR-accumulates
   ``info["success"]`` across an episode

   .. code-block:: python

      success_once = success_once or bool(info["success"])

   so the recorded value flips from ``False`` to ``True`` the *first* time
   the task succeeds and stays ``True`` for the rest of the episode.  See
   :doc:`evaluation_protocol` for full metric definitions.

.. _rich-ui-api:

Rich terminal UI
~~~~~~~~~~~~~~~~

:class:`~mikasa_robo_suite.vla.benchmarking.RichBenchmarkUI` is a context
manager that renders a live panel (progress bars + results table) in the
terminal.  It exposes three callbacks that wire directly into
:func:`~mikasa_robo_suite.vla.benchmarking.evaluate_benchmark`.

.. code-block:: python

   from mikasa_robo_suite.vla.benchmarking import (
       BenchmarkConfig,
       RichBenchmarkUI,
       evaluate_benchmark,
       make_run_dir,
       select_benchmark_tasks,
   )

   tasks  = select_benchmark_tasks(split="short")
   config = BenchmarkConfig(n_episodes=50)
   policy = MyPolicy(chunk_size=8)
   run_dir = make_run_dir("eval_results/my_model/short")

   with RichBenchmarkUI(tasks, config.n_episodes) as ui:
       results, summary = evaluate_benchmark(
           tasks,
           policy,
           config,
           output_dir=run_dir,
           task_start_callback=ui.on_task_start,
           episode_callback=ui.on_episode_done,
           task_done_callback=ui.on_task_done,
       )

To pre-populate the table when resuming a partial run, pass
``initial_results`` — a list of already-completed task result dicts loaded
from disk:

.. code-block:: python

   import json
   from pathlib import Path

   resume_dir = Path("eval_results/my_model/short/2026-05-21_18-52-16")
   completed = [
       json.loads(p.read_text())
       for p in sorted(resume_dir.glob("*.json"))
       if p.stem != "summary"
   ]
   remaining = [t for t in tasks if t.env_id not in {r["env_id"] for r in completed}]

   with RichBenchmarkUI(remaining, config.n_episodes, initial_results=completed) as ui:
       results, summary = evaluate_benchmark(
           remaining,
           policy,
           config,
           output_dir=resume_dir,
           task_start_callback=ui.on_task_start,
           episode_callback=ui.on_episode_done,
           task_done_callback=ui.on_task_done,
           initial_results=completed,
       )

:class:`RichBenchmarkUI` requires the ``rich`` package (``pip install rich``).
It is listed as a dependency of ``mikasa-robo-suite`` so it is installed
automatically when you install the package.

.. _flexible-usage:

Flexible Usage
--------------

The three-split protocol is the canonical procedure, but the 90 individual
datasets let you compose any training mixture:

- **Single-task**: download one dataset, train, evaluate on that task alone.
  Useful for ablations or environment-specific analyses.
- **Cross-split**: train on Short, evaluate on Medium to test generalisation.
  Must be labelled explicitly; not comparable to canonical results.
- **Full-benchmark**: train on all 90 tasks at once.

In all cases use the same ``eval_demo.py`` runner and the same seeding
convention so that individual task results remain comparable to the canonical
baseline.
