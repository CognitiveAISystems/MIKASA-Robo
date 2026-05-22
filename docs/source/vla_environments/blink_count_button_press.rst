Blink Count Button Press
========================

.. admonition:: Language Instruction

   Count how many times the blue lamp blinks, press the red button exactly that many times when the red lamp turns green, then press the black button to submit your answer.

.. admonition:: Task Description
   :class: tip

   Temporal counting: count blue-lamp blinks, wait for execution, press the red button exactly that many times, then submit.

.. image:: ../_static/videos/blink_count_button_press.gif
   :alt: Render preview for Blink Count Button Press
   :width: 720px
   :align: center

Source
------

- Module: ``mikasa_robo_suite.vla.memory_envs.blink_count_button_press_vla``
- Source file: ``mikasa_robo_suite/vla/memory_envs/blink_count_button_press_vla.py``

Difficulty and Parameters
-------------------------

Easy, Medium, Hard, and Long variants increase count range, delay, and total horizon.

Variants
--------

.. list-table::
   :header-rows: 1
   :widths: 38 12 16

   * - Env ID
     - Horizon
     - Data Source
   * - ``BlinkCountButtonPressEasy-Long-VLA-v0``
     - 1200
     - MP
   * - ``BlinkCountButtonPressEasy-VLA-v0``
     - 150
     - MP
   * - ``BlinkCountButtonPressHard-Long-VLA-v0``
     - 1200
     - MP
   * - ``BlinkCountButtonPressHard-VLA-v0``
     - 300
     - MP
   * - ``BlinkCountButtonPressMedium-Long-VLA-v0``
     - 1200
     - MP
   * - ``BlinkCountButtonPressMedium-VLA-v0``
     - 200
     - MP


Run Example
-----------

.. code-block:: python

   import gymnasium as gym
   import torch

   import mikasa_robo_suite.vla.memory_envs  # registers VLA env IDs
   from mikasa_robo_suite.vla.utils.apply_wrappers import apply_mikasa_vla_wrappers

   env = gym.make(
       "BlinkCountButtonPressEasy-VLA-v0",
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
     --env-id BlinkCountButtonPressEasy-VLA-v0 \
     --path-to-save-data data_mikasa_robo \
     --num-train-data 250 \
     --max-attempts 5000 \
     --seed 0

Render Videos
-------------

Generate a fresh MP4/GIF render with:

.. code-block:: bash

   uv run python utils/prepare_benchmark_demo_videos.py \
     --tasks BlinkCountButtonPressEasy-VLA-v0 \
     --output-dir videos/benchmark_demos \
     --max-attempts-per-task 8 \
     --overwrite

Generated media stays under ``videos/`` and should be published deliberately rather than committed by default.
