Environments API
================

All VLA environments live under ``mikasa_robo_suite.vla.memory_envs`` and
are registered with Gymnasium via ``@register_env`` decorators.  Import the
package to make the IDs available:

.. code-block:: python

   import mikasa_robo_suite.vla.memory_envs
   import gymnasium as gym
   from mikasa_robo_suite.vla.utils.apply_wrappers import apply_mikasa_vla_wrappers

   env = gym.make(
       "RememberColor3-VLA-v0",
       num_envs=1,
       obs_mode="rgb",
       control_mode="pd_ee_delta_pose",
   )
   env = apply_mikasa_vla_wrappers(env)  # canonical per-task wrapper chain

Common Patterns
---------------

Every environment exposes two class-level attributes that VLA wrappers rely on:

.. code-block:: python

   class MyEnv(BaseEnv):
       LANGUAGE_INSTRUCTION: str = "..."  # natural-language task description
       # self.task_cue        — cue tensor (or None)
       # self.oracle_info   — privileged hint (or None)

``LANGUAGE_INSTRUCTION`` is stored in ``mikasa_robo_vla_envs.csv`` (column
``language_instruction``) for all 90 registered tasks.

.. note::

   The full task list and split labels are in the :doc:`../vla_environments/index`
   section.  The pages below are an API reference, not a task catalogue.

Shell Game Environments
-----------------------

.. automodule:: mikasa_robo_suite.vla.memory_envs.shell_game_touch_vla
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mikasa_robo_suite.vla.memory_envs.shell_game_push_vla
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mikasa_robo_suite.vla.memory_envs.shell_game_pick_vla
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mikasa_robo_suite.vla.memory_envs.shell_game_color_lamp_touch_vla
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mikasa_robo_suite.vla.memory_envs.shell_game_shuffle_touch_vla
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mikasa_robo_suite.vla.memory_envs.shell_game_shuffle_color_lamp_touch_vla
   :members:
   :undoc-members:
   :show-inheritance:

Remember Environments
---------------------

.. automodule:: mikasa_robo_suite.vla.memory_envs.remember_color_vla
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mikasa_robo_suite.vla.memory_envs.remember_shape_vla
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mikasa_robo_suite.vla.memory_envs.remember_shape_and_color_vla
   :members:
   :undoc-members:
   :show-inheritance:

Find-Imposter Environments
--------------------------

.. automodule:: mikasa_robo_suite.vla.memory_envs.find_imposter_color_vla
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mikasa_robo_suite.vla.memory_envs.find_imposter_shape_vla
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mikasa_robo_suite.vla.memory_envs.find_imposter_shape_and_color_vla
   :members:
   :undoc-members:
   :show-inheritance:

Colors Sequence Environments
-----------------------------

.. automodule:: mikasa_robo_suite.vla.memory_envs.seq_of_colors_vla
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mikasa_robo_suite.vla.memory_envs.chain_of_colors_vla
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mikasa_robo_suite.vla.memory_envs.bunch_of_colors_vla
   :members:
   :undoc-members:
   :show-inheritance:

Intercept Environments
----------------------

.. automodule:: mikasa_robo_suite.vla.memory_envs.intercept_vla
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mikasa_robo_suite.vla.memory_envs.intercept_grab_vla
   :members:
   :undoc-members:
   :show-inheritance:

Rotate Environments
-------------------

.. automodule:: mikasa_robo_suite.vla.memory_envs.rotate_lenient_vla
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mikasa_robo_suite.vla.memory_envs.rotate_strict_vla
   :members:
   :undoc-members:
   :show-inheritance:

Trace-Shape Environments
------------------------

.. automodule:: mikasa_robo_suite.vla.memory_envs.trace_shape_vla
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mikasa_robo_suite.vla.memory_envs.trace_shape_seq_vla
   :members:
   :undoc-members:
   :show-inheritance:

Batteries-Checker Environments
------------------------------

.. automodule:: mikasa_robo_suite.vla.memory_envs.batteries_checker_easy_vla
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mikasa_robo_suite.vla.memory_envs.batteries_checker_hard_vla
   :members:
   :undoc-members:
   :show-inheritance:

Other Environments
------------------

.. automodule:: mikasa_robo_suite.vla.memory_envs.take_it_back_vla
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mikasa_robo_suite.vla.memory_envs.blink_count_button_press_vla
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mikasa_robo_suite.vla.memory_envs.gather_and_recall_vla
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mikasa_robo_suite.vla.memory_envs.timed_transfer_vla
   :members:
   :undoc-members:
   :show-inheritance:
