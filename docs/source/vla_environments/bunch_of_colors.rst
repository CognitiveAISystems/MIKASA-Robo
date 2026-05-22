Bunch Of Colors
===============

.. admonition:: Language Instruction

   Observe which colored cubes appear during the cue, wait, then touch all of them in any order and press the center button.

.. admonition:: Task Description
   :class: tip

   Set memory: observe a set of colored cubes, then touch all remembered colors in any order and press the center button.

.. image:: ../_static/videos/bunch_of_colors.gif
   :alt: Render preview for Bunch Of Colors
   :width: 720px
   :align: center

Source
------

- Module: ``mikasa_robo_suite.vla.memory_envs.bunch_of_colors_vla``
- Source file: ``mikasa_robo_suite/vla/memory_envs/bunch_of_colors_vla.py``

Difficulty and Parameters
-------------------------

Number of colors and Long variants increase capacity and retention time.

Variants
--------

.. list-table::
   :header-rows: 1
   :widths: 38 12 16

   * - Env ID
     - Horizon
     - Data Source
   * - ``BunchOfColors3-Long-VLA-v0``
     - 700
     - MP
   * - ``BunchOfColors3-VLA-v0``
     - 400
     - MP
   * - ``BunchOfColors5-Long-VLA-v0``
     - 700
     - MP
   * - ``BunchOfColors5-VLA-v0``
     - 400
     - MP
   * - ``BunchOfColors7-Long-VLA-v0``
     - 700
     - MP
   * - ``BunchOfColors7-VLA-v0``
     - 400
     - MP


Run Example
-----------

.. code-block:: python

   import gymnasium as gym
   import torch

   import mikasa_robo_suite.vla.memory_envs  # registers VLA env IDs
   from mikasa_robo_suite.vla.utils.apply_wrappers import apply_mikasa_vla_wrappers

   env = gym.make(
       "BunchOfColors3-VLA-v0",
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
     --env-id BunchOfColors3-VLA-v0 \
     --path-to-save-data data_mikasa_robo \
     --num-train-data 250 \
     --max-attempts 5000 \
     --seed 0

Render Videos
-------------

Generate a fresh MP4/GIF render with:

.. code-block:: bash

   uv run python utils/prepare_benchmark_demo_videos.py \
     --tasks BunchOfColors3-VLA-v0 \
     --output-dir videos/benchmark_demos \
     --max-attempts-per-task 8 \
     --overwrite

Generated media stays under ``videos/`` and should be published deliberately rather than committed by default.
