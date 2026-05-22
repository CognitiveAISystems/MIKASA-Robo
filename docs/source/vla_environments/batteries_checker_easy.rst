Batteries Checker Easy
======================

.. admonition:: Language Instruction

   Find all working batteries by inserting each one into the socket, observing the lamp result, and then pressing the button to confirm.

.. admonition:: Task Description
   :class: tip

   Sequential verification: insert batteries into a socket, observe the lamp result, remember which batteries work, and press the confirmation button.

.. image:: ../_static/videos/batteries_checker_easy.gif
   :alt: Render preview for Batteries Checker Easy
   :width: 720px
   :align: center

Source
------

- Module: ``mikasa_robo_suite.vla.memory_envs.batteries_checker_easy_vla``
- Source file: ``mikasa_robo_suite/vla/memory_envs/batteries_checker_easy_vla.py``

Difficulty and Parameters
-------------------------

Battery count controls memory load and horizon. Easy variants do not require returning each tested battery to its original slot.

Variants
--------

.. list-table::
   :header-rows: 1
   :widths: 38 12 16

   * - Env ID
     - Horizon
     - Data Source
   * - ``BatteriesCheckerEasy-12-VLA-v0``
     - 2160
     - -
   * - ``BatteriesCheckerEasy-15-VLA-v0``
     - 2400
     - -
   * - ``BatteriesCheckerEasy-3-VLA-v0``
     - 540
     - MP
   * - ``BatteriesCheckerEasy-6-VLA-v0``
     - 1080
     - MP
   * - ``BatteriesCheckerEasy-9-VLA-v0``
     - 1620
     - -


Run Example
-----------

.. code-block:: python

   import gymnasium as gym
   import torch

   import mikasa_robo_suite.vla.memory_envs  # registers VLA env IDs
   from mikasa_robo_suite.vla.utils.apply_wrappers import apply_mikasa_vla_wrappers

   env = gym.make(
       "BatteriesCheckerEasy-3-VLA-v0",
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
     --env-id BatteriesCheckerEasy-3-VLA-v0 \
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
     --tasks BatteriesCheckerEasy-3-VLA-v0 \
     --output-dir videos/benchmark_demos \
     --max-attempts-per-task 8 \
     --overwrite

Generated media stays under ``videos/`` and should be published deliberately rather than committed by default.
