Wrappers API
============

For usage patterns and composition guidance, see the :doc:`../wrappers_cookbook`.
This page is a full API reference generated from the source docstrings.

The 19 wrappers are organised into four groups, plus a one-call helper
that wires them up for any of the 90 MIKASA-Robo-VLA tasks:

- **One-call helper** — picks the correct chain for every task.
- **Core** — always required: dict conversion and proprioception remapping.
- **Action-shaping** — control robot behaviour during specific episode phases.
- **Task-specific info** — inject ground-truth fields into the ``info`` dict.
- **Render / debug** — overlay diagnostic information on rendered frames.

One-Call Helper
---------------

For every of the 90 MIKASA-Robo-VLA env IDs, use
:func:`~mikasa_robo_suite.vla.utils.apply_wrappers.apply_mikasa_vla_wrappers`
immediately after ``gym.make``. It applies the canonical wrapper chain that
matches the dataset-collection pipeline. The individual wrapper classes below
remain available for users that need to compose chains by hand.

.. autofunction:: mikasa_robo_suite.vla.utils.apply_wrappers.apply_mikasa_vla_wrappers

.. autodata:: mikasa_robo_suite.vla.utils.apply_wrappers.MIKASA_VLA_ENV_IDS
   :annotation:

Core Wrappers
-------------

.. autoclass:: mikasa_robo_suite.vla.utils.wrappers.StateOnlyTensorToDictWrapper
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: mikasa_robo_suite.vla.utils.wrappers.ConvertJointsToEEFXyzRpyGripperWrapper
   :members:
   :undoc-members:
   :show-inheritance:

Action-Shaping Wrappers
-----------------------

.. autoclass:: mikasa_robo_suite.vla.utils.wrappers.InitialZeroActionWrapper
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: mikasa_robo_suite.vla.utils.wrappers.CurriculumPhaseNoopActionWrapper
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: mikasa_robo_suite.vla.utils.wrappers.CurriculumPhaseNoopActionWrapperPdJointPos
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: mikasa_robo_suite.vla.utils.wrappers.CameraShutdownWrapper
   :members:
   :undoc-members:
   :show-inheritance:

Task-Specific Info Wrappers
---------------------------

.. autoclass:: mikasa_robo_suite.vla.utils.wrappers.RememberColorInfoWrapper
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: mikasa_robo_suite.vla.utils.wrappers.RememberShapeInfoWrapper
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: mikasa_robo_suite.vla.utils.wrappers.RememberShapeAndColorInfoWrapper
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: mikasa_robo_suite.vla.utils.wrappers.MemoryCapacityInfoWrapper
   :members:
   :undoc-members:
   :show-inheritance:

Render / Debug Wrappers
-----------------------

.. autoclass:: mikasa_robo_suite.vla.utils.wrappers.RenderStepInfoWrapper
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: mikasa_robo_suite.vla.utils.wrappers.RenderRewardInfoWrapper
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: mikasa_robo_suite.vla.utils.wrappers.RenderPressProgressInfoWrapper
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: mikasa_robo_suite.vla.utils.wrappers.RenderWorkingBatteriesInfoWrapper
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: mikasa_robo_suite.vla.utils.wrappers.ShellGameRenderCupInfoWrapper
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: mikasa_robo_suite.vla.utils.wrappers.RotateRenderAngleInfoWrapper
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: mikasa_robo_suite.vla.utils.wrappers.RenderTraceShapeDebugWrapper
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: mikasa_robo_suite.vla.utils.wrappers.RenderTimedTransferInfoWrapper
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: mikasa_robo_suite.vla.utils.wrappers.DebugRewardWrapper
   :members:
   :undoc-members:
   :show-inheritance:
