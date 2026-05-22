Wrappers Cookbook
=================

MIKASA-Robo-VLA ships 19 Gymnasium wrappers in
``mikasa_robo_suite/vla/utils/wrappers.py`` plus a one-call helper
:func:`~mikasa_robo_suite.vla.utils.apply_wrappers.apply_mikasa_vla_wrappers`
that picks the correct per-task chain automatically. This page groups the
individual wrappers by purpose and shows when and how to compose them
manually if you ever need to.

For the full API reference, see :doc:`api/wrappers`.

.. contents:: On this page
   :depth: 2
   :local:

The Default: ``apply_mikasa_vla_wrappers``
-------------------------------------------

For **any of the 90 MIKASA-Robo-VLA tasks**, the canonical wrapper stack —
state→dict, curriculum noop (where needed), task-specific overlays,
RGB-flatten, and EEF proprioception — is applied by a single function:

.. code-block:: python

   import gymnasium as gym
   import mikasa_robo_suite.vla.memory_envs  # registers VLA env IDs
   from mikasa_robo_suite.vla.utils.apply_wrappers import apply_mikasa_vla_wrappers

   env = gym.make(
       "RememberColor3-VLA-v0",
       num_envs=1,
       obs_mode="rgb",
       control_mode="pd_ee_delta_pose",
       render_mode="all",
   )
   env = apply_mikasa_vla_wrappers(env)  # default for every task

The helper is the recommended entry point. It guarantees:

- the same core obs format (``obs["rgb"]`` and ``obs["proprio"]``) that
  the published datasets use, plus ``obs["task_cue"]`` *only for the
  Rotate\* family* (the angle cue used by RL oracles; VLA policies should
  ignore it because the same value is already in
  ``info["language_instruction"]``);
- the correct cue-phase noop wrapper for VLA tasks that need one
  (``CurriculumPhaseNoopActionWrapper`` in the canonical
  ``pd_ee_delta_pose`` action space);
- task-specific render overlays during ``env.render()`` for human-watchable
  videos.

For headless metric-only evaluations pass ``include_overlays=False`` —
only the four functional wrappers are kept, no text on rendered frames,
but observations and reward are byte-identical to the default::

   env = apply_mikasa_vla_wrappers(env, include_overlays=False)

See :func:`mikasa_robo_suite.vla.utils.apply_wrappers.apply_mikasa_vla_wrappers`
for the full per-env mapping.

Composition Order (manual)
--------------------------

You only need this section if you are intentionally composing wrappers by
hand — for example, to reproduce the dataset collector or to experiment
with a non-standard chain. Otherwise use ``apply_mikasa_vla_wrappers``.

Always apply wrappers in this order:

.. code-block:: text

   gym.make(...)
     └─ StateOnlyTensorToDictWrapper          ← must be first
         └─ CurriculumPhaseNoopActionWrapper   (cue-phase VLA tasks)
             └─ <task-specific info wrappers>  (overlays)
                 └─ <render / debug wrappers>  (overlays, dev only)
                     └─ FlattenRGBDObservationWrapper(rgb=True, joints=True)
                         └─ ConvertJointsToEEFXyzRpyGripperWrapper
                             └─ RecordEpisode  (outermost)

The ``env_info`` helper inside the PPO collector returns the same
task-specific overlay chain that ``apply_mikasa_vla_wrappers`` builds
internally, so you can inspect or replicate it manually:

.. code-block:: python

   # env_info lives in the PPO collector module; the dataset_collectors
   # package is included with mikasa-robo-suite and is importable at runtime.
   from mikasa_robo_suite.vla.dataset_collectors.get_mikasa_robo_datasets import env_info

   wrappers_list, episode_timeout = env_info("RememberColor9-VLA-v0")
   for wrapper_class, wrapper_kwargs in wrappers_list:
       env = wrapper_class(env, **wrapper_kwargs)

Core Wrappers
-------------

These two wrappers are part of the standard pipeline and are almost always
required.

StateOnlyTensorToDictWrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Converts the raw tensor observation into a dict and injects the
``task_cue`` (filled with sentinel ``4242424242`` for tasks that do not
expose a numeric cue) and ``oracle_info`` fields:

.. code-block:: python

   from mikasa_robo_suite.vla.utils.wrappers import StateOnlyTensorToDictWrapper

   env = gym.make("RememberColor3-VLA-v0", num_envs=1, obs_mode="rgb",
                  control_mode="pd_ee_delta_pose", render_mode="all")
   env = StateOnlyTensorToDictWrapper(env)
   obs, info = env.reset(seed=0)
   print(obs.keys())  # dict_keys(['state'/'sensor_data', 'task_cue', 'oracle_info'])

**When to use**: always — it is the first wrapper in every recommended
stack.  Downstream wrappers in ``apply_mikasa_vla_wrappers`` then drop
``oracle_info`` and drop ``task_cue`` for every task *except* the
*Rotate\** family (where PPO oracles need the target angle, and where VLA
policies should still ignore it because the angle is already in
``info["language_instruction"]``).

ConvertJointsToEEFXyzRpyGripperWrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Converts the flattened raw joint-state input ``obs["joints"]`` into the
public VLA proprioception key ``obs["proprio"]`` with the 7D
end-effector representation ``xyz(3) + rpy(3) + gripper(1)``.
The raw ``obs["joints"]`` key is removed from the output dict.

.. code-block:: python

   from mikasa_robo_suite.vla.utils.wrappers import (
       StateOnlyTensorToDictWrapper,
       ConvertJointsToEEFXyzRpyGripperWrapper,
   )

   env = StateOnlyTensorToDictWrapper(env)
   env = ConvertJointsToEEFXyzRpyGripperWrapper(env)

After this wrapper, ``obs["proprio"]`` is the canonical 7D proprioception
vector used by all VLA datasets and evaluation scripts.  See
:doc:`observation_space` for the field-by-field reference (units, ranges,
how ``gripper_opening`` differs from the ``gripper_command`` action).

**When to use**: for dataset collection and any downstream pipeline that
expects the 7D ``proprio`` vector (the format used in all published
MIKASA-Robo-VLA datasets).

Action-Shaping Wrappers
-----------------------

InitialZeroActionWrapper
~~~~~~~~~~~~~~~~~~~~~~~~~

Executes a fixed number of zero-action steps at the start of each
episode.  Useful for tasks that require the robot to settle before the
cue phase begins.

CurriculumPhaseNoopActionWrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Replaces the agent's action with a no-op during the cue (and optional
empty) phase, while the cue is being shown.  This keeps the robot still
so that the cue is fully visible, which mirrors the behaviour of the
PPO oracle.

**When to use**: include it when reproducing the train-data rollout setup for
any task whose PPO ``env_info`` stack contains it. The
:func:`~mikasa_robo_suite.vla.utils.apply_wrappers.apply_mikasa_vla_wrappers`
helper adds it for those PPO-collected cue-phase tasks. Motion-planning data is
exported after replay through a plain ``pd_ee_delta_pose`` rollout, so the VLA
helper does not add a curriculum action filter for MP-only tasks.

CurriculumPhaseNoopActionWrapperPdJointPos
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``pd_joint_pos``-aware subclass of ``CurriculumPhaseNoopActionWrapper``.
Plain zeros aren't a "stand still" command in ``pd_joint_pos`` (they
would drive the robot toward qpos = [0, ..., 0]), so during the cue
phase this wrapper substitutes the robot's current ``qpos`` plus a
normalized gripper command — i.e. *hold the current pose*.

**When to use**: in the ``pd_joint_pos`` motion-planning oracle scripts for
``BlinkCountButtonPress*``. That hold wrapper is upstream of replay; the
published VLA train-data rollout and the VLA helper both expose the canonical
unfiltered ``pd_ee_delta_pose`` validation path.

CameraShutdownWrapper
~~~~~~~~~~~~~~~~~~~~~

Disables the cameras during the memory phase of tasks where the cameras
are explicitly turned off as part of the task design (e.g. *BatteriesChecker*).
This ensures that the agent cannot cheat by looking at occluded objects.

Render / Debug Wrappers
-----------------------

These wrappers overlay task-specific information on top of the rendered
video frame.  They are intended for local debugging and video generation;
do **not** use them during benchmark evaluation.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Wrapper
     - What it overlays
   * - ``RenderStepInfoWrapper``
     - Current step count.
   * - ``RenderRewardInfoWrapper``
     - Per-step reward value.
   * - ``RenderPressProgressInfoWrapper``
     - Button press progress bar (*BlinkCountButtonPress* tasks).
   * - ``RenderWorkingBatteriesInfoWrapper``
     - Ground-truth working battery positions (*BatteriesChecker* tasks).
   * - ``ShellGameRenderCupInfoWrapper``
     - Which cup hides the ball (*ShellGame* tasks).
   * - ``RotateRenderAngleInfoWrapper``
     - Target and current rotation angle (*Rotate* tasks).
   * - ``RenderTraceShapeDebugWrapper``
     - Target path overlay (*TraceShape* tasks).
   * - ``RenderTimedTransferInfoWrapper``
     - Timer countdown (*TimedTransfer* tasks).
   * - ``DebugRewardWrapper``
     - Breakdown of reward sub-terms for reward engineering.

Task-Specific Info Wrappers
---------------------------

These wrappers inject task-specific fields into the ``info`` dict returned
by ``env.step()``.  They are used during collection and can be useful for
evaluation scripts that need access to ground-truth labels.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Wrapper
     - Added ``info`` fields
   * - ``RememberColorInfoWrapper``
     - Ground-truth target colour index.
   * - ``RememberShapeInfoWrapper``
     - Ground-truth target shape index.
   * - ``RememberShapeAndColorInfoWrapper``
     - Ground-truth target shape and colour pair.
   * - ``MemoryCapacityInfoWrapper``
     - All memorised items for capacity-memory tasks.

Minimal Stacks for Common Scenarios
------------------------------------

The recommended stacks below all wrap through
:func:`~mikasa_robo_suite.vla.utils.apply_wrappers.apply_mikasa_vla_wrappers`.
For a manual chain see :doc:`api/wrappers` and the per-env "Recommended
Wrappers" sections in :doc:`vla_environments/index`.

Dataset collection / VLA training data (any task):

.. code-block:: python

   from mikasa_robo_suite.vla.utils.apply_wrappers import apply_mikasa_vla_wrappers

   env = gym.make(env_id, num_envs=N, obs_mode="rgb",
                  control_mode="pd_ee_delta_pose", render_mode="all")
   env = apply_mikasa_vla_wrappers(env)

VLA evaluation (clean observations, no rendered overlays):

.. code-block:: python

   from mikasa_robo_suite.vla.utils.apply_wrappers import apply_mikasa_vla_wrappers

   env = gym.make(env_id, num_envs=N, obs_mode="rgb",
                  control_mode="pd_ee_delta_pose")
   env = apply_mikasa_vla_wrappers(env, include_overlays=False)

Local debugging with video (overlays kept, ``RecordEpisode`` outermost):

.. code-block:: python

   from mani_skill.utils.wrappers import RecordEpisode
   from mikasa_robo_suite.vla.utils.apply_wrappers import apply_mikasa_vla_wrappers

   env = gym.make(env_id, num_envs=N, obs_mode="rgb",
                  control_mode="pd_ee_delta_pose", render_mode="all")
   env = apply_mikasa_vla_wrappers(env)
   env = RecordEpisode(env, f"./videos/{env_id}", max_steps_per_video=max_steps)
