Intercept
=========

.. admonition:: Language Instruction

   Intercept the rolling ball by moving to its path and deflecting it toward the target.

.. admonition:: Task Description
   :class: tip

   Predictive interception: infer a rolling ball path and deflect it toward the target.

.. image:: ../_static/videos/intercept.gif
   :alt: Render preview for Intercept
   :width: 720px
   :align: center

Source
------

- Module: ``mikasa_robo_suite.vla.memory_envs.intercept_vla``
- Source file: ``mikasa_robo_suite/vla/memory_envs/intercept_vla.py``

Difficulty and Parameters
-------------------------

Slow, Medium, and Fast variants change timing precision.

Variants
--------

.. list-table::
   :header-rows: 1
   :widths: 38 12 16

   * - Env ID
     - Horizon
     - Data Source
   * - ``InterceptFast-VLA-v0``
     - 60
     - PPO
   * - ``InterceptMedium-VLA-v0``
     - 60
     - PPO
   * - ``InterceptSlow-VLA-v0``
     - 60
     - PPO


Run Example
-----------

.. code-block:: python

   import gymnasium as gym
   import torch

   import mikasa_robo_suite.vla.memory_envs  # registers VLA env IDs
   from mikasa_robo_suite.vla.utils.apply_wrappers import apply_mikasa_vla_wrappers

   env = gym.make(
       "InterceptSlow-VLA-v0",
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

PPO-sourced MIKASA-Robo-90 variants use oracle checkpoints:

.. code-block:: bash

   uv run python mikasa_robo_suite/vla/dataset_collectors/get_mikasa_robo_datasets.py \
     --env-id InterceptSlow-VLA-v0 \
     --path-to-save-data data_mikasa_robo \
     --ckpt-dir . \
     --num-train-data 250

Render Videos
-------------

Generate a fresh MP4/GIF render with:

.. code-block:: bash

   uv run python utils/prepare_benchmark_demo_videos.py \
     --tasks InterceptSlow-VLA-v0 \
     --output-dir videos/benchmark_demos \
     --max-attempts-per-task 8 \
     --overwrite

Generated media stays under ``videos/`` and should be published deliberately rather than committed by default.
