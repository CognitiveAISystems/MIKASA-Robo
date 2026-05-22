Quick Start
===========

The examples below assume the project is installed with ``uv sync --frozen``
from the repository root.

Minimal VLA Run
---------------

.. important::

  Every MIKASA-Robo-VLA environment must be wrapped with
  :func:`~mikasa_robo_suite.vla.utils.apply_wrappers.apply_mikasa_vla_wrappers`
  immediately after ``gym.make`` and before the first ``env.reset()``. **This wrapper
  is required for correct benchmark behavior:** it applies the task-specific VLA
  logic and makes the environment inputs and outputs match the format used in the
  released datasets. The helper selects the correct wrapper chain for any of the
  90 benchmark tasks, so it should always be used instead of manually composing
  wrappers.

.. code-block:: python

   import gymnasium as gym
   import torch

   import mikasa_robo_suite.vla.memory_envs  # registers all VLA env IDs
   from mikasa_robo_suite.vla.utils.apply_wrappers import apply_mikasa_vla_wrappers

   env = gym.make(
       "RememberColor3-VLA-v0",
       num_envs=1,
       obs_mode="rgb",
       control_mode="pd_ee_delta_pose",
       reward_mode="normalized_dense",
       render_mode="all",
       sim_backend="gpu",
   )
   env = apply_mikasa_vla_wrappers(env, include_overlays=False)

   obs, info = env.reset(seed=42)

   for _ in range(env.max_episode_steps):
       action = torch.as_tensor(env.action_space.sample(), device=env.unwrapped.device)
       obs, reward, terminated, truncated, info = env.step(action)
       if torch.as_tensor(terminated | truncated).any():
           break

   env.close()

``include_overlays=False`` omits debug text overlays from rendered frames (if render).

``include_overlays=True`` when generating human-watchable videos.

Listing available tasks
-----------------------

The canonical task list is in ``mikasa_robo_vla_envs.csv``:

.. code-block:: python

   import csv

   with open("mikasa_robo_vla_envs.csv", newline="", encoding="utf-8") as f:
       for row in csv.DictReader(f, delimiter=";"):
           print(row["Name"], row["Horizon Split"], row["Max Length"])

``gym.make`` also accepts any env ID registered in
``mikasa_robo_suite/vla/memory_envs/``, including development variants not
in the CSV.

What ``obs`` contains
---------------------

After ``apply_mikasa_vla_wrappers(env)``, ``obs`` is a Python dict with the
following keys.  ``B`` is ``num_envs``.

.. code-block:: python

   obs["rgb"]            # shape (B, 128, 128, 6), uint8
   obs["proprio"]        # shape (B, 7),           float32
   obs.get("task_cue")   # Rotate* only: target angle for the RL oracle; VLAs ignore

RGB images
~~~~~~~~~~

``obs["rgb"]`` concatenates two cameras along the channel axis::

   top_rgb   = obs["rgb"][..., :3]   # base_camera,  top-down view
   wrist_rgb = obs["rgb"][..., 3:6]  # hand_camera,  wrist-mounted view

Both cameras produce 128 × 128 images by default.

Proprioception
~~~~~~~~~~~~~~

``obs["proprio"]`` is the 7D absolute end-effector state.
It is **not** joint angles and it is **not** action deltas::

   obs["proprio"] = [
       eef_x,           # [0]  TCP position x, metres, not normalised
       eef_y,           # [1]  TCP position y, metres, not normalised
       eef_z,           # [2]  TCP position z, metres, not normalised
       eef_roll,        # [3]  TCP roll,  radians, [-pi,   pi]
       eef_pitch,       # [4]  TCP pitch, radians, [-pi/2, pi/2]
       eef_yaw,         # [5]  TCP yaw,   radians, [-pi,   pi]
       gripper_opening, # [6]  sum of two Panda finger qpos, metres, [0.0, 0.08]
   ]

``gripper_opening`` is a single scalar.  The Panda gripper has two symmetric
fingers; each finger's qpos range is ``[0.0, 0.04]`` m, so the sum is
``[0.0, 0.08]`` m.  A value of ``0.0`` means fully closed; ``0.08`` means
fully open.  The value is **not** normalised.

task_cue and language_instruction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``obs["task_cue"]`` exists **only for the Rotate\* tasks**, where the
target rotation angle (in degrees) cannot be inferred from a single RGB
frame.  It is provided so PPO RL oracles can be trained from images; the
canonical VLA helper drops the key for every other task family.

For VLA policies the cue is redundant: the same numeric value is already
embedded in ``info["language_instruction"]`` (e.g. *"Rotate the peg by
30 degrees clockwise"*).  VLAs should therefore consume only the language
instruction and ignore ``obs["task_cue"]`` even when it is present::

   # VLA-style: ignore task_cue, use language_instruction
   text = info["language_instruction"]

   # RL-style: read task_cue directly (Rotate* only)
   if "task_cue" in obs:
       target_angle = obs["task_cue"]

The canonical helper does not expose ``oracle_info`` to the VLA policy
observation. That privileged field is only available in lower-level or manual
debug wrapper stacks.

Language instruction
~~~~~~~~~~~~~~~~~~~~

``info["language_instruction"]`` is a Python ``str`` describing the task
goal, e.g. *"Observe the cube's colour, wait, then touch the matching cube"*.
It is returned by both ``env.reset()`` and ``env.step()``.

Action format
-------------

Pass a 7D ``float32`` tensor to ``env.step(action)``.
Single-env shape: ``(7,)``.  Batched shape: ``(B, 7)``.

All seven values are in the normalised range ``[-1, 1]``::

   action = [
       delta_eef_x,      # [0]  EEF translation delta, normalised; Panda maps to [-0.1, 0.1] m
       delta_eef_y,      # [1]
       delta_eef_z,      # [2]
       delta_eef_roll,   # [3]  EEF rotation delta, normalised; magnitude capped at 0.1 rad
       delta_eef_pitch,  # [4]
       delta_eef_yaw,    # [5]
       gripper_command,  # [6]  normalised position target: -1 → close, +1 → open
   ]

``action[0:3]`` are *relative* translation commands, not absolute positions.
``action[3:6]`` are *relative* rotation commands, not absolute orientations.
``action[6]`` (``gripper_command``) is a *position target* sent to the Panda
gripper joint controller — it is **not** a delta of ``obs["proprio"][..., 6]``
(``gripper_opening``).  Sending ``-1`` drives the gripper toward the closed
target; sending ``+1`` drives it toward the open target.

The dataset ``action`` field stores these same values; see :doc:`datasets`
for the full training signal reference.

gym.make parameters
-------------------

.. list-table::
   :header-rows: 1
   :widths: 28 20 52

   * - Parameter
     - Canonical value
     - Notes
   * - ``obs_mode``
     - ``"rgb"``
     - Use ``"state"`` only for PPO oracle training and reward debugging.
       Never use ``"state"`` for VLA training or benchmark evaluation.
   * - ``control_mode``
     - ``"pd_ee_delta_pose"``
     - Required for all VLA training, evaluation, and dataset collection.
       Using a different control mode changes action semantics and breaks
       dataset compatibility.
   * - ``reward_mode``
     - ``"normalized_dense"``
     - Values in ``[0, 1]``. Use for RL training. Use ``"sparse"`` (0 or 1)
       for success-rate evaluation. ``"dense"`` is unscaled and varies per task.
   * - ``num_envs``
     - ``1`` for eval, larger for collection
     - GPU-parallelised. All returned tensors have a leading batch dim ``B``.
   * - ``render_mode``
     - ``"all"`` for video; omit for headless
     - Required for ``RecordEpisode`` and ``env.render()``.
   * - ``sim_backend``
     - ``"gpu"``
     - Use ``"cpu"`` only when no GPU is available. CPU simulation is significantly slower and is not recommended for benchmark use, as the CPU and GPU physics backends are not perfectly identical. For reproducible evaluation and dataset-consistent behavior, the GPU backend should be used whenever possible.

Video Recording
---------------

Wrap with ``RecordEpisode`` *after* ``apply_mikasa_vla_wrappers``:

.. code-block:: python

   import gymnasium as gym
   import torch
   from mani_skill.utils.wrappers import RecordEpisode

   import mikasa_robo_suite.vla.memory_envs
   from mikasa_robo_suite.vla.utils.apply_wrappers import apply_mikasa_vla_wrappers

   env_name = "RememberColor3-VLA-v0"

   env = gym.make(
       env_name,
       num_envs=1,
       obs_mode="rgb",
       control_mode="pd_ee_delta_pose",
       reward_mode="normalized_dense",
       render_mode="all",
       sim_backend="gpu",
   )
   env = apply_mikasa_vla_wrappers(env, include_overlays=True)
   env = RecordEpisode(
       env,
       output_dir=f"./videos/{env_name}",
       save_trajectory=False,
       max_steps_per_video=env.max_episode_steps,
   )

   obs, info = env.reset(seed=42)
   for _ in range(env.max_episode_steps):
       action = torch.as_tensor(env.action_space.sample(), device=env.unwrapped.device)
       obs, reward, terminated, truncated, info = env.step(action)

   env.close()
   # video written to ./videos/RememberColor3-VLA-v0/0.mp4

``include_overlays=True`` renders task-state overlays (step counter, reward,
task-specific debug info) on top of the video frames.

Benchmark Demo GIF/MP4
-----------------------

.. code-block:: bash

   uv run python utils/prepare_benchmark_demo_videos.py \
     --tasks RememberColor3-VLA-v0 \
     --output-dir videos/benchmark_demos \
     --max-attempts-per-task 8 \
     --overwrite
