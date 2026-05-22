Observation and Action Space
============================

MIKASA-Robo-VLA exposes two observation modes that can be selected when
constructing any environment.  The same action and reward interface applies
regardless of which mode you choose.

Observation Modes
-----------------

``obs_mode="state"``
   Privileged simulator state — an efficient flat tensor that contains the
   full physical state of the scene.  Use it for PPO oracle training, fast
   debugging, and reward sanity checks.  In this mode the raw ManiSkill
   observation is a single ``state`` tensor and there are **no camera images**.

``obs_mode="rgb"``
   RGB image observations from two cameras (top-down and wrist-mounted), plus
   proprioception.  This is the standard mode for VLA training and evaluation.
   In this mode the raw ManiSkill observation contains ``sensor_data`` (per
   camera RGB) and ``agent`` / ``extra`` joint state — no flat ``state``
   tensor.

The two modes are mutually exclusive: pick one when calling ``gym.make``.

Raw vs Wrapped Observations
---------------------------

The shapes you actually feed to a VLA model depend on which wrappers are
applied.  The canonical chain (used by
:func:`~mikasa_robo_suite.vla.utils.apply_wrappers.apply_mikasa_vla_wrappers`
and by every published dataset collector) is:

.. code-block:: text

   gym.make(obs_mode="rgb", control_mode="pd_ee_delta_pose")
     └─ StateOnlyTensorToDictWrapper          # adds task_cue + oracle_info
         └─ <task-specific info / overlay wrappers>
             └─ FlattenRGBDObservationWrapper(rgb=True, joints=True)
                  # collapses sensor_data → obs["rgb"] (concat 2 cams)
                  # exposes obs["joints"] from agent state
                 └─ ConvertJointsToEEFXyzRpyGripperWrapper
                      # rewrites obs["joints"] → obs["proprio"]  (7D EEF)

After the canonical chain the observation is a flat Python dict.  These are
the keys VLA training and evaluation code should consume (``B = num_envs``):

.. list-table:: Canonical wrapped observation (``apply_mikasa_vla_wrappers``)
   :header-rows: 1
   :widths: 22 22 56

   * - Key
     - Shape / dtype
     - Description
   * - ``obs["rgb"]``
     - ``(B, 128, 128, 6)``  uint8
     - Top-down and wrist cameras concatenated along the channel axis.
       ``obs["rgb"][..., :3]`` is ``base_camera`` (top-down);
       ``obs["rgb"][..., 3:6]`` is ``hand_camera`` (wrist).
   * - ``obs["proprio"]``
     - ``(B, 7)``  float32
     - Absolute end-effector pose + gripper opening — see the next section.
   * - ``obs["task_cue"]``
     - ``(B, P)``  float32
     - **RL-only.**  Numeric target-angle cue exposed exclusively by the
       *Rotate\** family (PPO oracles need it because the angle is not
       inferable from a single RGB frame).  The canonical VLA helper
       *drops* the key for all other tasks; for *Rotate\** tasks VLA
       policies may ignore it because the same information is already in
       ``info["language_instruction"]``.

.. note::

   If you skip the canonical chain and use only
   :class:`~mikasa_robo_suite.vla.utils.wrappers.StateOnlyTensorToDictWrapper`,
   the raw ManiSkill keys ``sensor_data`` (per camera) and ``agent`` /
   ``extra`` are preserved instead of the flattened ``obs["rgb"]`` /
   ``obs["proprio"]``.  In that case the per-camera RGB is reachable at
   ``obs["sensor_data"]["base_camera"]["rgb"]`` and
   ``obs["sensor_data"]["hand_camera"]["rgb"]``, each ``(B, 128, 128, 3)``
   uint8.  All published datasets and the benchmark runner expect the
   *canonical* layout above — prefer it for any VLA pipeline.

The sentinel value ``4242424242`` is documented in its own section below.

Canonical VLA Proprioception and Action
----------------------------------------

The online wrapped environment and the published VLA datasets use the same
7D proprioception and action format. After
:func:`~mikasa_robo_suite.vla.utils.apply_wrappers.apply_mikasa_vla_wrappers`,
use ``obs["proprio"]`` as the VLA proprioception input. It is **not** a vector
of arm joint angles and it is **not** an action delta. For one environment the
semantic layout is::

   obs["proprio"] =
   [
       eef_x,
       eef_y,
       eef_z,
       eef_roll,
       eef_pitch,
       eef_yaw,
       gripper_opening,
   ]

The batched online tensor has shape ``(B, 7)`` and dtype ``float32``; the
dataset signal has shape ``[T, 7]`` and dtype ``float32``. The values are
constructed from the absolute Panda TCP pose and finger qpos values by
:class:`~mikasa_robo_suite.vla.utils.wrappers.ConvertJointsToEEFXyzRpyGripperWrapper`.

.. list-table:: ``obs["proprio"]`` fields
   :header-rows: 1
   :widths: 18 30 52

   * - Fields
     - Range / units
     - Meaning
   * - ``[0:3]``
     - Absolute xyz in metres; not normalized; no finite wrapper clamp.
     - Absolute end-effector TCP position in the ManiSkill scene frame.
   * - ``[3]`` and ``[5]``
     - Radians, ``[-pi, pi]`` from quaternion-to-Euler conversion.
     - Absolute TCP roll and yaw.
   * - ``[4]``
     - Radians, ``[-pi/2, pi/2]`` from quaternion-to-Euler conversion.
     - Absolute TCP pitch.
   * - ``[6]``
     - Metres, Panda physical range ``[0.0, 0.08]`` after summing the two
       finger qpos values; not normalized.
     - Current gripper opening. ``0`` is closed and about ``0.08`` is open.

The action sent to ``env.step(action)`` is also 7D, but it has different
semantics. For one environment it is::

   action =
   [
       delta_eef_x,
       delta_eef_y,
       delta_eef_z,
       delta_eef_roll,
       delta_eef_pitch,
       delta_eef_yaw,
       gripper_command,
   ]

Use dtype ``float32``. A single action has shape ``(7,)`` and a batched action
has shape ``(B, 7)``. The action values stored in VLA datasets are in this
normalized ``pd_ee_delta_pose`` environment action space.

.. list-table:: ``action`` fields for the Panda ``pd_ee_delta_pose`` controller
   :header-rows: 1
   :widths: 18 30 52

   * - Fields
     - User-facing range
     - Meaning
   * - ``[0:3]``
     - Each value in ``[-1, 1]``. Panda maps this normalized range to
       xyz deltas in ``[-0.1, 0.1]`` metres per control action.
     - Relative end-effector translation command, not an absolute xyz pose.
   * - ``[3:6]``
     - Each value is supplied in ``[-1, 1]``. The Panda EEF-pose controller
       clips the rotation vector norm and applies its ``0.1`` radian rotation
       limit per control action.
     - Relative end-effector orientation command, not an absolute rpy pose.
   * - ``[6]``
     - ``[-1, 1]``. Panda maps ``-1`` toward the closed gripper target and
       ``+1`` toward the open gripper target.
     - Normalized gripper **position command**. It is not
       ``delta_gripper_opening`` and is not the same quantity as
       ``obs["proprio"][..., 6]``.

The Panda gripper has two symmetric fingers.  Each finger's qpos range is
``[0.0, 0.04]`` m, so ``obs["proprio"][..., 6]`` (the sum of both) is in
``[0.0, 0.08]`` m.  The gripper controller target is sent per-finger in
``[-0.01, 0.04]`` m; the slightly negative lower bound helps apply closing
force against a grasped object.  This internal controller detail does not
affect the normalised action interface — ``action[6]`` is always in ``[-1, 1]``.

Dataset Signals
---------------

In published NPZ, RLDS, and LeRobot VLA datasets, each timestep stores the
same signal semantics:

.. list-table::
   :header-rows: 1
   :widths: 24 20 56

   * - Signal
     - Shape/dtype
     - Description
   * - ``rgb``
     - ``[T, 128, 128, 6]`` uint8
     - Top and wrist RGB images **pre-concatenated** on the channel axis
       (``[..., :3]`` = ``base_camera``, ``[..., 3:6]`` = ``hand_camera``);
       this matches the online wrapped ``obs["rgb"]`` byte-for-byte, so a
       model trained on the dataset can be evaluated without any reshape.
   * - ``proprio``
     - ``[T, 7]`` float32
     - Same absolute vector as online ``obs["proprio"]``.
   * - ``action``
     - ``[T, 7]`` float32
     - Same normalized ``pd_ee_delta_pose`` action vector accepted by
       ``env.step(action)``.
   * - ``language_instruction``
     - string
     - Natural-language task instruction.
   * - ``reward``
     - ``[T]`` float32
     - Per-step reward in the chosen reward mode.
   * - ``success``
     - ``[T]`` bool
     - Whether the episode success condition was met at this step.

Always pass ``control_mode="pd_ee_delta_pose"`` when constructing
environments for dataset collection or VLA evaluation.

Reward Modes
------------

Pass ``reward_mode`` to ``gym.make`` (or the PPO script) to select the
reward signal:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Mode
     - Description
   * - ``sparse``
     - Binary reward: 1.0 on task success, 0.0 otherwise.
       Standard for imitation learning; not used during PPO training.
   * - ``dense``
     - Shaped reward summing task-specific sub-goal bonuses.
       Values vary per environment.
   * - ``normalized_dense``
     - Dense reward normalised to ``[0, 1]`` per environment.
       Recommended for PPO training as it keeps learning rates
       comparable across environments.

Sentinel Values
---------------

``StateOnlyTensorToDictWrapper`` uses the sentinel value **4242424242** for
``task_cue`` and ``oracle_info`` when the underlying environment does not
expose them.  Later wrappers in the canonical chain then drop the keys
that are not meaningful for VLA policies:

- ``oracle_info`` is removed for every task — it is privileged and never
  visible to a VLA policy in canonical evaluation.
- ``task_cue`` is kept **only for the Rotate\* tasks** (the rotation
  angle in degrees), since PPO oracles trained from RGB cannot recover
  the target angle from images alone.  For all other tasks the key is
  dropped after the canonical chain.

Check the final wrapped observation by key presence — VLA pipelines that
want strictly *images + proprioception* can simply not read it::

   task_cue = obs.get("task_cue")   # None for non-Rotate tasks
   # VLA policies may ignore task_cue: the same information is already
   # encoded in info["language_instruction"].
