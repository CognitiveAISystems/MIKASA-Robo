Find Imposter Shape And Color
=============================

.. admonition:: Language Instruction

   Observe the objects shown, wait, then touch the object whose shape and color combination was not present before.

.. admonition:: Task Description
   :class: tip

   Out-of-set binding detection: remember shape-color combinations and touch the novel combination.

.. image:: ../_static/videos/find_imposter_shape_and_color.gif
   :alt: Render preview for Find Imposter Shape And Color
   :width: 720px
   :align: center

Source
------

- Module: ``mikasa_robo_suite.vla.memory_envs.find_imposter_shape_and_color_vla``
- Source file: ``mikasa_robo_suite/vla/memory_envs/find_imposter_shape_and_color_vla.py``

Difficulty and Parameters
-------------------------

Larger shape-color grids increase binding-memory and distractor difficulty.

Variants
--------

.. list-table::
   :header-rows: 1
   :widths: 38 12 16

   * - Env ID
     - Horizon
     - Data Source
   * - ``FindImposterShapeAndColor3x2-VLA-v0``
     - 25
     - PPO
   * - ``FindImposterShapeAndColor3x3-VLA-v0``
     - 25
     - PPO
   * - ``FindImposterShapeAndColor5x3-VLA-v0``
     - 40
     - PPO


Run Example
-----------

.. code-block:: python

   import gymnasium as gym
   import torch

   import mikasa_robo_suite.vla.memory_envs  # registers VLA env IDs
   from mikasa_robo_suite.vla.utils.apply_wrappers import apply_mikasa_vla_wrappers

   env = gym.make(
       "FindImposterShapeAndColor3x2-VLA-v0",
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
     --env-id FindImposterShapeAndColor3x2-VLA-v0 \
     --path-to-save-data data_mikasa_robo \
     --ckpt-dir . \
     --num-train-data 250

Render Videos
-------------

Generate a fresh MP4/GIF render with:

.. code-block:: bash

   uv run python utils/prepare_benchmark_demo_videos.py \
     --tasks FindImposterShapeAndColor3x2-VLA-v0 \
     --output-dir videos/benchmark_demos \
     --max-attempts-per-task 8 \
     --overwrite

Generated media stays under ``videos/`` and should be published deliberately rather than committed by default.
