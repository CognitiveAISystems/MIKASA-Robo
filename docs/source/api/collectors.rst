Dataset Collectors API
======================

MIKASA-Robo-VLA provides two data collection methods:

**PPO Oracle** (``get_mikasa_robo_datasets.py``)
   Uses a pre-trained PPO checkpoint to collect expert trajectories.
   Suitable for all tasks where an oracle checkpoint is available (see the
   ``Source`` column in the :doc:`../vla_environments/index` table).

**Motion Planning** (``get_mikasa_robo_datasets_motion_planning.py``)
   Uses a geometric planner instead of a learned policy.  Required for
   tasks that cannot be solved by the PPO oracle (e.g. *TraceShape*).

Collection commands and the full pipeline are documented in :doc:`../datasets`.
The ``README.md`` inside
``mikasa_robo_suite/vla/dataset_collectors/`` contains additional notes on
parallel collection, checkpointing, and resuming interrupted runs.

PPO Oracle Collector
--------------------

.. automodule:: mikasa_robo_suite.vla.dataset_collectors.get_mikasa_robo_datasets
   :members:
   :undoc-members:

Motion-Planning Collector
-------------------------

.. automodule:: mikasa_robo_suite.vla.dataset_collectors.get_mikasa_robo_datasets_motion_planning
   :members:
   :undoc-members:
