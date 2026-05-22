Batteries Checker Hard
======================

.. admonition:: Language Instruction

   Find all working batteries by inserting each one into the socket, observing the lamp result, returning it from the socket to its initial slot, and then pressing the button to confirm.

.. admonition:: Task Description
   :class: tip

   Hard sequential verification: test each battery, remember the result, return it to its slot, and submit the working set.

.. image:: ../_static/videos/batteries_checker_hard.gif
   :alt: Render preview for Batteries Checker Hard
   :width: 720px
   :align: center

Source
------

- Module: ``mikasa_robo_suite.vla.memory_envs.batteries_checker_hard_vla``
- Source file: ``mikasa_robo_suite/vla/memory_envs/batteries_checker_hard_vla.py``

Difficulty and Parameters
-------------------------

Hard variants add return-to-slot manipulation and longer horizons on top of working-set memory.

Variants
--------

.. list-table::
   :header-rows: 1
   :widths: 38 12 16

   * - Env ID
     - Horizon
     - Data Source
   * - ``BatteriesCheckerHard-12-VLA-v0``
     - 4320
     - -
   * - ``BatteriesCheckerHard-15-VLA-v0``
     - 4320
     - -
   * - ``BatteriesCheckerHard-3-VLA-v0``
     - 1080
     - MP
   * - ``BatteriesCheckerHard-6-VLA-v0``
     - 2160
     - MP
   * - ``BatteriesCheckerHard-9-VLA-v0``
     - 3240
     - -


Run Example
-----------

.. code-block:: python

   import gymnasium as gym
   import torch

   import mikasa_robo_suite.vla.memory_envs  # registers VLA env IDs
   from mikasa_robo_suite.vla.utils.apply_wrappers import apply_mikasa_vla_wrappers

   env = gym.make(
       "BatteriesCheckerHard-3-VLA-v0",
       num_envs=1,
       obs_mode="rgb",
       control_mode="pd_ee_delta_pose",
       render_mode="all",
   )
   env = apply_mikasa_vla_wrappers(env)
   obs, info = env.reset(seed=42)
   for _ in range(env.max_episode_steps):
       action = env.action_space.sample()
       if not torch.is_tensor(action):
           action = torch.as_tensor(action, device=env.unwrapped.device)
       obs, reward, terminated, truncated, info = env.step(action)
   env.close()

Dataset Collection
------------------

Motion-planning MIKASA-Robo-90 variants use planner plus replay collection:

.. code-block:: bash

   uv run python mikasa_robo_suite/vla/dataset_collectors/get_mikasa_robo_datasets_motion_planning.py \
     --env-id BatteriesCheckerHard-3-VLA-v0 \
     --path-to-save-data data_mikasa_robo \
     --num-train-data 250 \
     --max-attempts 5000 \
     --seed 0

Additional local variants are useful for debugging or ablations. Report them separately from MIKASA-Robo-90 results.

Render Videos
-------------

Generate a fresh MP4/GIF render with:

.. code-block:: bash

   uv run python utils/prepare_benchmark_demo_videos.py \
     --tasks BatteriesCheckerHard-3-VLA-v0 \
     --output-dir videos/benchmark_demos \
     --max-attempts-per-task 8 \
     --overwrite

Generated media stays under ``videos/`` and should be published deliberately rather than committed by default.
