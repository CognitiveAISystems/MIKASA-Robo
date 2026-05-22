Installation
============

System Requirements
-------------------

- **OS**: Linux x86_64 (required by the locked ManiSkill / SAPIEN stack;
  see ``pyproject.toml``).
- **Python**: ``>=3.9, <3.12``.
- **CUDA driver**: 11.8 or 12.x (driver must match your PyTorch CUDA
  build; ``uv sync --frozen`` picks the correct PyTorch wheel automatically).
- **OpenGL / Vulkan**: SAPIEN uses the system GPU graphics stack for rendering;
  ensure your driver provides Vulkan loader libraries.
- **Disk**: ~5 GB for the environment and SAPIEN assets; additional space
  for datasets (see :doc:`datasets`).

Recommended Setup
-----------------

MIKASA-Robo-VLA is installed from the repository with `uv <https://docs.astral.sh/uv/getting-started/installation/>`_.
If you do not have ``uv`` yet, install it first:

.. code-block:: bash

   curl -LsSf https://astral.sh/uv/install.sh | sh

After that, install the benchmark from source:

.. code-block:: bash

   git clone git@github.com:CognitiveAISystems/MIKASA-Robo.git
   cd MIKASA-Robo
   git submodule update --init --recursive
   uv sync --frozen

Alternatively, install the released package directly:

.. code-block:: bash

   uv add "mikasa-robo-suite>0.0.5"

Verify Installation
-------------------

Run the snippet below to confirm the package imports cleanly and the
canonical VLA wrapper path can be created:

.. code-block:: bash

   uv run python -c "
   import gymnasium as gym
   import mikasa_robo_suite.vla.memory_envs
   from mikasa_robo_suite.vla.utils.apply_wrappers import apply_mikasa_vla_wrappers

   env = gym.make('RememberColor3-VLA-v0', num_envs=1, obs_mode='rgb',
                  control_mode='pd_ee_delta_pose', render_mode='all',
                  sim_backend='gpu')
   env = apply_mikasa_vla_wrappers(env, include_overlays=False)
   obs, info = env.reset(seed=0)
   print('OK')
   env.close()
   "

Expected output includes: ``OK``

Legacy RL Version
-----------------

The default repository is moving to the VLA benchmark.  The ``pip`` command
below installs only the original RL benchmark release from PyPI; it does **not**
include the VLA environments or this documentation.  Use the
`mikasa-robo-rl branch <https://github.com/CognitiveAISystems/MIKASA-Robo/tree/mikasa-robo-rl>`_
or:

.. code-block:: bash

   pip install mikasa-robo-suite==0.0.5

.. note::

   If you run into errors during setup, check the :doc:`faq` page for common
   installation problems (``uv`` environment conflicts, CUDA mismatches, missing
   submodules).
