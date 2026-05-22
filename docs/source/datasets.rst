Datasets
========

MIKASA-Robo-VLA trajectories are available on Hugging Face and can be collected
locally.  The benchmark ships **22,500 trajectories** (250 per task × 90 tasks)
and more than **6,000,000 timesteps** across three interoperable formats.

.. contents:: On this page
   :depth: 2
   :local:

Choosing a Format
-----------------

.. list-table::
   :header-rows: 1
   :widths: 18 14 14 54

   * - Format
     - Random access
     - Framework
     - Best for
   * - **NPZ** (raw episodes)
     - No (sequential)
     - NumPy / PyTorch
     - Inspecting raw collection output; custom preprocessing.
   * - **RLDS / TFDS**
     - Yes
     - TensorFlow, JAX, Open-X pipelines
     - Open X-Embodiment style VLA training; any TF Dataset pipeline.
   * - **LeRobotDataset v3**
     - Yes
     - PyTorch, HuggingFace
     - LeRobot-based imitation learning; modern VLA fine-tuning (OpenVLA-OFT, π0.5).

For most users, **LeRobotDataset v3** is the recommended starting point because
it stores images as MP4 streams (compact), proprioception and actions as Parquet
files (fast random access), and is compatible with the latest VLA training
frameworks.

Public Datasets on Hugging Face
--------------------------------

The full 90-task collection is hosted on Hugging Face.  Each task is a
separate dataset repository so you can download only the split you need
(see :doc:`benchmarking` for split definitions).

.. |hf| image:: _static/hf-logo.svg
   :alt: Hugging Face
   :height: 1.1em
   :class: hf-inline-icon

- |hf| `MIKASA-Robo-VLA RLDS on Hugging Face <https://huggingface.co/datasets/mikasa-robo/mikasa-robo-vla-rlds>`_
- |hf| `MIKASA-Robo-VLA LeRobotDataset v3 on Hugging Face <https://huggingface.co/datasets/mikasa-robo/mikasa-robo-vla-lerobot>`_

Download a single task with ``huggingface_hub``:

.. code-block:: python

   from huggingface_hub import snapshot_download

   snapshot_download(
       repo_id="mikasa-robo/mikasa-robo-vla-lerobot",
       repo_type="dataset",
       allow_patterns="RememberColor3-VLA-v0/**",
       local_dir="data_mikasa_robo/data_lerobot",
   )

Download an entire split by listing the task names from
``mikasa_robo_vla_envs.csv``.  The CSV is **semicolon-delimited** with the
following header:

.. code-block:: text

   id;Name;Max Length;Memory Type;Horizon Split;Data Source;language_instruction
   1;RememberColor3-VLA-v0;30;Object;Short;PPO;"Observe the cube's color ..."

Iterate over the rows you want:

.. code-block:: python

   import csv
   from huggingface_hub import snapshot_download

   with open("mikasa_robo_vla_envs.csv") as f:
       tasks = [r["Name"] for r in csv.DictReader(f, delimiter=";")
                if r["Horizon Split"] == "Short"]

   for task in tasks:
       snapshot_download(
           repo_id="mikasa-robo/mikasa-robo-vla-lerobot",
           repo_type="dataset",
           allow_patterns=f"{task}/**",
           local_dir="data_mikasa_robo/data_lerobot",
       )

Training Signal Format
----------------------

Every dataset row stores the same canonical signals used by the online
wrapped environment: ``rgb`` (``[T, 128, 128, 6]`` uint8 — two cameras
concatenated on the channel axis), ``proprio`` (``[T, 7]`` float32 —
absolute EEF pose + gripper opening), ``action`` (``[T, 7]`` float32 in
the normalised ``pd_ee_delta_pose`` range ``[-1, 1]``), ``reward``,
``success``, and the natural-language ``language_instruction``.

The field-by-field reference (units, ranges, the difference between
``proprio[..., 6]`` and ``action[6]``) is in :doc:`observation_space`.

Dataset Layout
--------------

NPZ Source Layout
~~~~~~~~~~~~~~~~~

.. code-block:: text

   data_mikasa_robo/
     data_npz/
       <task>/
         train_data_000000.npz
         train_data_000001.npz
         ...
         metadata.json

Each NPZ episode stores: RGB observations, 7D end-effector proprioception,
7D ``pd_ee_delta_pose`` actions, rewards, success/done flags, and the language
instruction.

RLDS Layout
~~~~~~~~~~~

.. code-block:: text

   data_mikasa_robo/
     data_rlds/
       <task>/
         1.0.0/

RLDS stores trajectories as TensorFlow Datasets episodes.

LeRobotDataset v3 Layout
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   data_mikasa_robo/
     data_lerobot/
       <task>/
         data/          ← Parquet shards (proprio, action, reward, success)
         meta/          ← info.json, stats.json, tasks.jsonl, episodes.jsonl
         videos/        ← MP4 streams (observation.image, observation.wrist_image)

Collecting Your Own Datasets
-----------------------------

Choose a collector based on the task's ``Data Source`` column in
``mikasa_robo_vla_envs.csv``:

- ``Data Source = PPO`` — use ``get_mikasa_robo_datasets.py``.  Requires a
  trained PPO oracle checkpoint (see ``baselines/ppo/``); recommended for
  tasks that PPO solves reliably (most Object / Spatial / Sequential tasks).
- ``Data Source = MP`` — use ``get_mikasa_robo_datasets_motion_planning.py``.
  Uses scripted motion planning instead of an RL oracle; recommended for
  Procedural and Prospective tasks where PPO does not converge
  (e.g. *TraceShape*, *GatherAndRecall*).

Both collectors emit NPZ episodes in the same layout, so the downstream
RLDS and LeRobotDataset v3 conversion steps are identical.

Collect NPZ
~~~~~~~~~~~

PPO-oracle task:

.. code-block:: bash

   uv run python mikasa_robo_suite/vla/dataset_collectors/get_mikasa_robo_datasets.py \
     --env-id RememberColor3-VLA-v0 \
     --path-to-save-data data_mikasa_robo \
     --ckpt-dir . \
     --num-train-data 250

Motion-planning task:

.. code-block:: bash

   uv run python mikasa_robo_suite/vla/dataset_collectors/get_mikasa_robo_datasets_motion_planning.py \
     --env-id TraceShapeHard-VLA-v0 \
     --path-to-save-data data_mikasa_robo \
     --num-train-data 250 \
     --max-attempts 5000 \
     --seed 0

For parallel collection across multiple GPUs, see the collector README at
``mikasa_robo_suite/vla/dataset_collectors/README.md``.

Data Export Pipeline
~~~~~~~~~~~~~~~~~~~~

After NPZ collection, export to the training formats:

.. code-block:: text

   NPZ source episodes → RLDS / TFDS episodic dataset → LeRobotDataset v3

Convert NPZ to RLDS
~~~~~~~~~~~~~~~~~~~

Single task:

.. code-block:: bash

   uv sync --project utils/convert_npz_to_rlds/rlds_dataset_builder
   uv run --project utils/convert_npz_to_rlds/rlds_dataset_builder \
     python utils/convert_npz_to_rlds/convert_npz_task_to_rlds.py \
     --task RememberColor3-VLA-v0 \
     --overwrite-dest

All available NPZ task folders:

.. code-block:: bash

   for task_dir in data_mikasa_robo/data_npz/*; do
     task="$(basename "$task_dir")"
     uv run --project utils/convert_npz_to_rlds/rlds_dataset_builder \
       python utils/convert_npz_to_rlds/convert_npz_task_to_rlds.py \
       --task "$task" \
       --overwrite-dest
   done

Convert RLDS to LeRobotDataset v3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Single task:

.. code-block:: bash

   uv sync --project utils/convert_rlds_to_lerobot
   uv run --project utils/convert_rlds_to_lerobot \
     python utils/convert_rlds_to_lerobot/convert_rlds_to_lerobot.py \
     --task RememberColor3-VLA-v0 \
     --overwrite-dest

All available RLDS task folders:

.. code-block:: bash

   uv run --project utils/convert_rlds_to_lerobot \
     python utils/convert_rlds_to_lerobot/convert_rlds_to_lerobot.py \
     --all \
     --overwrite-dest
