Core Concepts
=============

This page explains the key ideas behind MIKASA-Robo-VLA: what makes the tasks
*memory-intensive*, how episodes are structured, and how the benchmark relates
to the original RL release.

.. contents:: On this page
   :depth: 2
   :local:

What Is MIKASA-Robo-VLA?
------------------------

MIKASA-Robo-VLA is a benchmark suite of 90 tabletop manipulation tasks for
evaluating **Vision-Language-Action (VLA) models** under partial observability.
In every task, the agent must retain, update, or continuously track information
across time in order to act correctly. The required memory may involve the
colour of a previously observed object, the order of targets in a sequence, the
location of a hidden item after shuffling, the number of events accumulated so
far, or other temporally distributed cues that are no longer directly observable
at decision time.

The benchmark extends the original **MIKASA-Robo** RL release
(`ICLR 2026 <https://arxiv.org/abs/2502.10550>`_) in many ways:

- **Task count** grows from 32 to **90 tasks**, covering a wider range of
  memory types (10 vs 4), horizon lengths (25 – 2160 steps), and difficulty
  levels.
- **New memory types**: Temporal, Prospective, Tracking, Checklist, Negative,
  and Procedural — not present in the original RL release.
- **Language instructions**: every task ships with a natural-language
  ``LANGUAGE_INSTRUCTION``, enabling VLA models to condition on text.
- **Calibrated dense rewards**: all environments in ``mikasa_robo_suite/vla/``
  have carefully tuned dense and normalised-dense reward functions, making them
  suitable not only for offline VLA evaluation but also for online RL training
  and reward-signal research.
- **Published datasets**: 22,500 trajectories (>6 M timesteps) in RLDS and
  LeRobotDataset v3 formats on Hugging Face, ready for imitation learning
  without any further conversion.

.. important::

   The earlier RL benchmark is available from the
   `mikasa-robo-rl branch <https://github.com/CognitiveAISystems/MIKASA-Robo/tree/mikasa-robo-rl>`_.
   Its implementation lives under ``mikasa_robo_suite/rl/`` and is kept for
   backwards compatibility.  All active development and new environments target
   ``mikasa_robo_suite/vla/``.

Episode Structure
-----------------

Most tasks in the benchmark follow a **three-phase structure**:

.. code-block:: text

   ┌──────────────┬──────────────────────┬──────────────────┐
   │  Cue phase   │   Memory phase       │  Action phase    │
   │  (observe)   │   (retain / track)   │  (act on memory) │
   └──────────────┴──────────────────────┴──────────────────┘

**Cue phase**
   The environment presents the information the agent must remember — a
   coloured light, a sequence of objects, a count of blinks.  The agent
   typically executes a no-op action during this phase.

**Memory phase**
   The cue disappears or the scene changes.  The agent must *retain* the
   relevant information internally.  This is the phase that separates
   memory-capable agents from reactive ones.

**Action phase**
   The agent uses the memorised information to complete the task —
   touching the correct cup, pressing the right button, returning an object
   to its original position.

.. note::

   Not all tasks follow this exact three-phase layout.  Some tasks require
   *continuous* memory or tracking across the entire episode:

   - **Tracking tasks** (*ShellGameShuffle*): the agent must maintain an
     up-to-date estimate of a hidden object's location as objects are
     shuffled throughout the episode.
   - **Procedural tasks** (*TraceShape*, *TraceShapeSeq*): the agent must
     recall a shape or sequence and re-execute it step-by-step with fine
     motor control.
   - **Prospective tasks** (*GatherAndRecall*): the agent observes a future
     goal, completes an intermediate task, then returns to fulfil the
     original intention.
   - **Temporal tasks** (*BlinkCountButtonPress*): information accumulates
     incrementally over time — the agent must integrate observations
     across steps rather than memorise a single cue.

The length of each phase, and therefore the horizon split of the task, is set
by class-level constants in the environment implementation.

Memory Types
------------

Tasks are grouped by the type of memory they exercise.  The full per-task
breakdown is in ``mikasa_robo_vla_envs.csv``.

.. list-table::
   :header-rows: 1
   :widths: 18 12 70

   * - Memory Type
     - # Tasks
     - What the agent must remember
   * - **Object**
     - 18
     - Identity, colour, or shape of a specific object shown during the
       cue phase (e.g. *RememberColor*, *FindImposter*).
   * - **Spatial**
     - 14
     - Location of an object that is then hidden from view
       (e.g. *ShellGameTouch*, *ShellGamePush*).
   * - **Capacity**
     - 12
     - An unordered set of items or a count (e.g. *BunchOfColors*,
       *BatteriesChecker*).
   * - **Temporal**
     - 12
     - A signal that accumulates over time, such as a blink count or a
       timed cue (e.g. *BlinkCountButtonPress*, *TimedTransfer*).
   * - **Negative**
     - 9
     - What the agent must *not* do — identify the odd-one-out
       (e.g. *FindImposterColor*, *FindImposterShape*).
   * - **Sequential**
     - 6
     - An ordered sequence of targets (e.g. *SeqOfColors*,
       *ChainOfColors*).
   * - **Procedural**
     - 6
     - A motor procedure that must be recalled and re-executed
       (e.g. *TraceShape*, *TraceShapeSeq*).
   * - **Prospective**
     - 5
     - A future intention — the agent sees a goal early in the episode,
       completes an intermediate task, then returns to fulfil the goal
       (e.g. *GatherAndRecall*).
   * - **Tracking**
     - 4
     - Multiple objects that are continuously shuffled or moved while
       hidden; the agent must maintain a dynamic estimate of their
       positions throughout the episode (e.g. *ShellGameShuffle*).
   * - **Checklist**
     - 4
     - A set of conditions all of which must be satisfied in any order
       (e.g. *BatteriesCheckerHard*).

Horizon Splits
--------------

Training a VLA on all 90 tasks simultaneously is difficult because episode
lengths range from 25 to 2160 steps.  MIKASA-Robo-VLA therefore defines
three **horizon splits** for reproducible multi-task evaluation:

.. list-table::
   :header-rows: 1
   :widths: 14 14 22 50

   * - Split
     - Tasks
     - Horizon range
     - Typical memory demand
   * - **Short**
     - 38
     - 25 – 200 steps
     - Rapid cue encoding and short-term recall.
   * - **Medium**
     - 30
     - 201 – 601 steps
     - Sustained working memory over moderately long episodes.
   * - **Long**
     - 22
     - 602 – 2160 steps
     - Extended memory, multi-phase reasoning, procedural recall.

See :doc:`benchmarking` for the canonical evaluation protocol that uses these
splits.

Observation Modes and the task_cue / oracle_info Fields
--------------------------------------------------------

VLA environments expose two observation modes:

``obs_mode="state"``
   Privileged simulator state as a flat tensor.

   .. warning::

      This mode is **not** used for VLA training or benchmarking.  It is
      intended solely for PPO oracle training, reward calibration, and
      generating ground-truth labels.  Always use ``obs_mode="rgb"`` for
      VLA evaluation.

``obs_mode="rgb"``
   RGB images from the top-down and wrist-mounted cameras, plus 7D
   proprioception ``obs["proprio"]``.  This is the standard mode for VLA
   training and evaluation.  After
   :func:`~mikasa_robo_suite.vla.utils.apply_wrappers.apply_mikasa_vla_wrappers`
   the proprioception vector is the **absolute EEF pose plus gripper opening**::

      obs["proprio"] = [eef_x, eef_y, eef_z, eef_roll, eef_pitch, eef_yaw, gripper_opening]

   and the action accepted by ``env.step(action)`` is a **relative delta plus
   gripper position command** (all values in ``[-1, 1]``)::

      action = [delta_eef_x, delta_eef_y, delta_eef_z,
                delta_eef_roll, delta_eef_pitch, delta_eef_yaw,
                gripper_command]

   See :doc:`observation_space` for the complete field-by-field reference with
   units and ranges.

Both modes are extended by
:class:`~mikasa_robo_suite.vla.utils.wrappers.StateOnlyTensorToDictWrapper`
with two extra fields:

- **``task_cue``** — a small numerical tensor exposed *only* by the
  *Rotate\** family of environments.  A single RGB frame cannot convey the
  exact target rotation angle, so RL agents trained on those tasks need
  the angle (in degrees) as an explicit observation channel.  The cue is
  consumed by the PPO oracles during dataset collection; it is **not**
  required by VLA policies, because the same information is embedded
  inside every task's ``language_instruction`` (e.g. *"Rotate the peg by
  30 degrees clockwise"*).  For all non-Rotate tasks
  ``StateOnlyTensorToDictWrapper`` returns the sentinel value
  ``4242424242``, and the canonical
  :func:`~mikasa_robo_suite.vla.utils.apply_wrappers.apply_mikasa_vla_wrappers`
  helper drops the key from the VLA-facing observation entirely.

- **``oracle_info``** — additional privileged information for evaluation or
  debugging (e.g. ground-truth object position).  Always ``4242424242`` when
  not available, and stripped from the VLA-facing observation by the
  canonical wrapper.

.. tip::

   **task_cue vs language_instruction.**  ``language_instruction`` is a
   human-readable text string (e.g. *"Observe the cube's colour, wait, then
   touch the cube of the same colour"*) and is the **only** task-conditioning
   channel a VLA model needs.  ``task_cue`` is a numeric encoding of the
   same information that exists solely for RL baselines on *Rotate\** —
   *VLA pipelines should ignore it.*

   The canonical wrapped observation therefore contains only ``rgb`` and
   ``proprio`` for almost all tasks, plus ``task_cue`` for *Rotate\**.
   See :doc:`observation_space` for the field-by-field layout and
   :doc:`wrappers_cookbook` for how the chain composes.

Dense Reward Functions
----------------------

A key property of MIKASA-Robo-VLA is that all 90 environments include
**calibrated dense reward functions**.  Unlike sparse rewards that provide
a signal only on task completion, the dense rewards break down each task into
interpretable sub-goals (approach, grasp, reach, release, …) and assign
incremental reward at each sub-step.

This has two practical consequences:

1. **PPO oracle training** converges reliably even for tasks with long
   horizons, making it possible to generate high-quality demonstration
   datasets.
2. Researchers who want to train RL agents online — rather than fine-tuning a
   VLA offline — can do so with the same environments without any reward
   engineering.

Use ``reward_mode="normalized_dense"`` (values in [0, 1]) for RL training, or
``reward_mode="sparse"`` if you only care about task success during evaluation.
See :doc:`observation_space` for the full reward-mode reference.

Dataset Formats
---------------

Trajectories collected with the PPO oracle or motion-planning scripts are
stored in three formats (in order of the collection pipeline):

.. code-block:: text

   NPZ (raw episodes)
     └─► RLDS / TensorFlow Datasets (episodic, Open-X style)
           └─► LeRobotDataset v3 (Parquet + MP4, modern imitation learning)

NPZ is the internal collection format.  RLDS and LeRobotDataset v3 are the
formats published on Hugging Face and recommended for VLA training.
See :doc:`datasets` for download links and conversion commands.
