Timed Transfer
==============

.. admonition:: Language Instructions

   - When the white lamp turns green, start counting steps from that exact moment. Move the blue cube from the green disc to the red disc exactly on step 300 of that count.
   - When the white lamp turns green, start counting steps from that exact moment. Move the blue cube from the green disc to the red disc exactly on step 100 of that count.
   - When the white lamp turns green, start counting steps from that exact moment. Move the blue cube from the green disc to the red disc exactly on step 1000 of that count.
   - When the white lamp turns green, start counting steps from that exact moment. Move the blue cube from the green disc to the red disc exactly on step 200 of that count.
   - When the white lamp turns green, start counting steps from that exact moment. Move the blue cube from the green disc to the red disc exactly on step 500 of that count.
   - When the white lamp turns green, start counting steps from that exact moment. Move the blue cube from the green disc to the red disc exactly on step 150 of that count.

.. admonition:: Task Description
   :class: tip

   Delayed timing: after a lamp cue, move the cube at a precise future timestep.

.. image:: ../_static/videos/timed_transfer.gif
   :alt: Render preview for Timed Transfer
   :width: 720px
   :align: center

Source
------

- Module: ``mikasa_robo_suite.vla.memory_envs.timed_transfer_vla``
- Source file: ``mikasa_robo_suite/vla/memory_envs/timed_transfer_vla.py``

Difficulty and Parameters
-------------------------

Easy, Medium, Hard, and Long variants increase target count and horizon.

Variants
--------

.. list-table::
   :header-rows: 1
   :widths: 38 12 16

   * - Env ID
     - Horizon
     - Data Source
   * - ``TimedTransferEasy-Long-VLA-v0``
     - 600
     - MP
   * - ``TimedTransferEasy-VLA-v0``
     - 200
     - MP
   * - ``TimedTransferHard-Long-VLA-v0``
     - 1200
     - MP
   * - ``TimedTransferHard-VLA-v0``
     - 300
     - MP
   * - ``TimedTransferMedium-Long-VLA-v0``
     - 900
     - MP
   * - ``TimedTransferMedium-VLA-v0``
     - 250
     - MP


Run Example
-----------

.. code-block:: python

   import gymnasium as gym
   import torch

   import mikasa_robo_suite.vla.memory_envs  # registers VLA env IDs
   from mikasa_robo_suite.vla.utils.apply_wrappers import apply_mikasa_vla_wrappers

   env = gym.make(
       "TimedTransferEasy-VLA-v0",
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
     --env-id TimedTransferEasy-VLA-v0 \
     --path-to-save-data data_mikasa_robo \
     --num-train-data 250 \
     --max-attempts 5000 \
     --seed 0

Render Videos
-------------

Generate a fresh MP4/GIF render with:

.. code-block:: bash

   uv run python utils/prepare_benchmark_demo_videos.py \
     --tasks TimedTransferEasy-VLA-v0 \
     --output-dir videos/benchmark_demos \
     --max-attempts-per-task 8 \
     --overwrite

Generated media stays under ``videos/`` and should be published deliberately rather than committed by default.
