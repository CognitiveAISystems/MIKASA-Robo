Evaluation Protocol
===================

This page is the canonical specification for evaluating a trained VLA model on
MIKASA-Robo-VLA.  For how to run evaluations in practice — CLI flags, output
layout, Python API — see :doc:`benchmarking`.

.. contents:: On this page
   :depth: 2
   :local:

Protocol at a Glance
--------------------

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Parameter
     - Canonical value
   * - Split
     - Chosen explicitly by the user (``short`` / ``medium`` / ``long``).
       No auto-detection from a checkpoint.
   * - Episodes per task
     - **50**
   * - Seeds
     - ``4242424242, 4242424243, …, 4242424291`` (same stream for every task)
   * - Parallel envs
     - ``num_envs=1`` (default); ``num_envs > 1`` allowed as opt-in speedup
   * - ``obs_mode``
     - ``"rgb"``
   * - ``control_mode``
     - ``"pd_ee_delta_pose"``
   * - ``reward_mode``
     - ``"normalized_dense"``
   * - Wrapper stack
     - ``apply_mikasa_vla_wrappers(env, include_overlays=False)``
   * - Main metric
     - ``success_once`` per episode
   * - Debug metric
     - ``mean_return`` per task (not aggregated to split level)

Observation and Action Format
------------------------------

After ``gym.make`` + ``apply_mikasa_vla_wrappers(env, include_overlays=False)``,
``obs`` always contains ``obs["rgb"]`` (``(B, 128, 128, 6)`` uint8 — two
cameras concatenated on the channel axis) and ``obs["proprio"]``
(``(B, 7)`` float32 — absolute EEF pose + gripper opening).  The
*Rotate\** family additionally exposes ``obs["task_cue"]`` (target angle
for the RL oracle); for all other tasks this key is dropped by the
canonical helper.  ``oracle_info`` is never exposed in the VLA-facing
observation.  The action is a ``(B, 7)`` ``float32`` tensor in the
normalised ``pd_ee_delta_pose`` action space; all seven components are
clamped to ``[-1, 1]``.

The canonical, field-by-field specification — value ranges, units, the
exact gripper conventions, and the difference between
``obs["proprio"][..., 6]`` and ``action[6]`` — lives in
:doc:`observation_space`.  All evaluation pipelines (online runs, dataset
collection, published datasets) use exactly the same layout, so the
:doc:`observation_space` reference is authoritative for every context.

.. important::

   A canonical leaderboard policy consumes only the camera images,
   ``obs["proprio"]``, and ``info["language_instruction"]``.  ``oracle_info``
   is privileged and stripped from the VLA-facing observation.  ``task_cue``
   exists solely as an RL-training signal for the *Rotate\** family — the
   same target angle is also embedded in the natural-language instruction,
   so VLAs should ignore the ``task_cue`` key entirely.  Reading either
   field disqualifies a run from the canonical leaderboard.

Action chunking
~~~~~~~~~~~~~~~

VLA policies typically predict a chunk of ``K`` actions per forward pass.
The benchmark runner consumes them with a ``collections.deque`` (FIFO):

1. At the start of every episode a **fresh, empty** ``deque`` is created.
2. On each simulator step, if the deque is empty, ``policy.forward(obs)``
   is called to produce ``(K, action_dim)`` actions which are pushed onto
   the deque with ``extend``.
3. One action is popped from the front with ``popleft`` and passed to
   ``env.step(action)``.

For ``K=1`` this is identical to standard per-step inference: the deque is
refilled on every step.  See ``evaluate_task`` in
``mikasa_robo_suite/vla/benchmarking.py`` for the exact implementation.

Metrics
-------

``success_once`` (main metric)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``success_once`` is a **boolean latch**, not a count.  At each simulator
step the runner performs::

   success_once = success_once or bool(info["success"])   # OR-latch

so the value flips from ``False`` to ``True`` the *first* time the task
succeeds and stays ``True`` for the remainder of the episode.  The
canonical per-episode metric is the final value of this latch::

   success_once = any(info["success"] is True over all simulator steps)

This unifies tasks that succeed mid-episode (e.g. *ShellGameTouch*) and
tasks that only emit success on the final step (e.g. *TimedTransfer*) under
a single binary outcome.  See ``mikasa_robo_suite/vla/benchmarking.py``
(``evaluate_task``) for the exact code path.

``return`` (debug only)
~~~~~~~~~~~~~~~~~~~~~~~~

::

   return = sum(reward)  over all steps of the episode

Use this to inspect reward shaping or training quality.  Do not aggregate it
to split level or use it as a comparison number.

Reported SR levels
~~~~~~~~~~~~~~~~~~

Three numbers are required per evaluation run:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Metric
     - Definition
   * - **Per-task SR**
     - ``SR_task = mean(success_once)`` over the 50 episodes.
   * - **Per-split SR** *(main)*
     - ``SR_split = mean(SR_task  for all tasks in the split)``.
   * - **Per-memory-type SR**
     - For each memory type present in the split:
       ``SR_mem = mean(SR_task  for tasks with that memory_type)``.

Seeding
-------

For **every** task independently, run 50 episodes with seeds::

   env.reset(seed=4242424242 + i)   for i in 0, 1, …, 49

The starting seed ``4242424242`` matches the one used by muVLA when it was
evaluated on MIKASA-Robo, so results are directly comparable to that baseline.

.. note::

   Seeds are the same across tasks — each task independently starts from
   ``4242424242``.  This gives full determinism, but it means the per-episode
   initial states are **correlated across tasks**.  Two tasks evaluated under
   this protocol are not statistically independent.  Account for this when
   computing confidence intervals or aggregate statistics.

``num_envs``
------------

The canonical evaluation uses ``num_envs=1``.  The episode-to-seed mapping is
then trivial: episode ``i`` is the state produced by
``env.reset(seed=4242424242 + i)``.

For wall-clock speedups (especially for the Long split) you may use
``num_envs > 1`` as an opt-in optimisation, but each parallel env must be
seeded so that the set of 50 ``(env_id, episode_idx)`` pairs is identical to
the ``num_envs=1`` run.  The reported SR must match.

Wrapper Stack
-------------

Always construct the env with::

   env = gym.make(
       env_id,
       num_envs=1,
       obs_mode="rgb",
       control_mode="pd_ee_delta_pose",
       reward_mode="normalized_dense",
       render_mode="all",
   )
   env = apply_mikasa_vla_wrappers(env, include_overlays=False)

``include_overlays=False`` strips debug/render overlays so they cannot affect
timing or bleed into recorded observations.  The benchmark runner
(``evaluate_benchmark``) does this automatically.

JSON Output Schema
------------------

Per-task file ``<ENV_ID>.json``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One file per evaluated env ID:

.. code-block:: json

   {
     "env_id": "RememberColor3-VLA-v0",
     "split": "Short",
     "memory_type": "Object",
     "start_seed": 4242424242,
     "n_episodes": 50,
     "successes": [true, false, true, "..."],
     "returns":   [12.4, 0.0, 15.7, "..."],
     "sr": 0.74,
     "mean_return": 9.8,
     "benchmark_commit": "<git sha>",
     "control_mode": "pd_ee_delta_pose",
     "obs_mode": "rgb",
     "wrapper_chain": "apply_mikasa_vla_wrappers(include_overlays=False)",
     "action_chunk_size": 8,
     "model": {"name": "my-vla", "config": {}},
     "episode_lengths": [30, 28, 30, "..."],
     "episode_seeds":   [4242424242, 4242424243, "..."]
   }

``successes`` and ``returns`` have length ``n_episodes``.  ``sr`` and
``mean_return`` are their means.  ``episode_lengths`` (the number of
simulator steps in each episode before termination) and ``episode_seeds``
(``start_seed + i`` for ``i`` in ``0…n_episodes-1``) are always written by
``evaluate_benchmark`` and may be safely consumed by downstream tools, but
they are not required for the canonical SR comparison and may be omitted
from manually constructed result files.

Summary file ``summary.json``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One file per run, updated after every task completes:

.. code-block:: json

   {
     "split": "Short",
     "sr_split": 0.61,
     "sr_per_memory_type": {
       "Negative": 0.42,
       "Object": 0.55,
       "Prospective": 0.30,
       "Spatial": 0.72,
       "Temporal": 0.50,
       "Tracking": 0.40
     },
     "tasks": ["RememberColor3-VLA-v0", "..."],
     "per_task_sr": {"RememberColor3-VLA-v0": 0.74, "...": 0.0},
     "per_task_mean_return": {"RememberColor3-VLA-v0": 9.8, "...": 0.0}
   }

``summary.json`` is rewritten on disk after every completed task, so a run
that is interrupted mid-way leaves a valid partial summary.

Reference Evaluation Script
----------------------------

``examples/eval_demo.py`` is a working reference built around
``mikasa_robo_suite.vla.benchmarking``.  It uses ``DummyChunkPolicy``
(random actions) so you can run it end-to-end before plugging in a real model:

.. literalinclude:: ../../examples/eval_demo.py
   :language: python
   :caption: examples/eval_demo.py

Run from the repository root::

   # One-task smoke-test, one episode, CPU backend
   uv run python examples/eval_demo.py \
       --num-episodes 1 --sim-backend cpu \
       --output-dir eval_results/dummy

   # Canonical Short-split run
   uv run python examples/eval_demo.py \
       --split short \
       --output-dir results/my_model

   # With per-episode rollout videos
   uv run python examples/eval_demo.py \
       --split short \
       --save-videos \
       --output-dir results/my_model

See :doc:`benchmarking` for the full CLI reference and Python API.

What to Report in Papers
------------------------

Reproducibility checklist:

- [ ] Benchmark git commit.
- [ ] Split evaluated (Short / Medium / Long, or explicit subset + label).
- [ ] Start seed and episodes per task (default: ``4242424242`` × 50).
- [ ] GPU model and count.
- [ ] Training dataset (Hugging Face repo).
- [ ] Training compute (GPU hours or gradient steps) and config (LoRA rank, etc.).
- [ ] Action chunk size ``K``.
- [ ] **Required**: per-split SR, full per-task SR table, per-memory-type SR.
- [ ] Mean return per task (optional debug metric).
- [ ] Whether ``oracle_info`` or ``task_cue`` were passed to the policy.
  Both are RL-side fields (``task_cue`` only exists for *Rotate\** and is
  already encoded in the language instruction); reading either disqualifies
  the run from the canonical leaderboard.

Cross-Split and Partial Evaluation
-----------------------------------

Evaluating on a different split than the training split, or on a custom subset,
is allowed but must be reported separately with an explicit label such as
*"trained on Short, evaluated on Medium"*.  Do not mix such results into the
same SR table as canonical runs.
