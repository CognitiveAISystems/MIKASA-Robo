<h1 align="center">MIKASA-Robo (VLA Update)</h1>

<h3 align="center">A memory-intensive tabletop manipulation benchmark, reworked for Vision-Language-Action training</h3>

<div align="center">
  <a href="https://arxiv.org/abs/2502.10550">
    <img src="https://img.shields.io/badge/arXiv-2502.10550-b31b1b.svg"/>
  </a>
  <a href="https://openreview.net/forum?id=9cLPurIZMj">
    <img src="https://img.shields.io/badge/OpenReview-9cLPurIZMj-8c1aff.svg"/>
  </a>
  <a href="https://sites.google.com/view/memorybenchrobots/">
    <img src="https://img.shields.io/badge/Website-Project_Page-blue.svg"/>
  </a>
  <a href="https://pypi.org/project/mikasa-robo-suite/">
    <img src="https://img.shields.io/pypi/v/mikasa-robo-suite.svg"/>
  </a>
  <a href="https://github.com/CognitiveAISystems/MIKASA-Robo">
    <img src="https://img.shields.io/badge/GitHub-MIKASA--Robo-green.svg"/>
  </a>
  <a href="https://docs.astral.sh/ruff/">
    <img src="https://img.shields.io/badge/linting-ruff-46aef7.svg"/>
  </a>
</div>



---

## 🚨 Important Update: **MIKASA-Robo-90** (VLA Edition)

> [!IMPORTANT]  
> The `main` branch is now fully focused on **VLA research** and represents **MIKASA-Robo-90**.

### ✨ What is new in `main`
- ✅ **90 total tasks** (new + upgraded existing tasks)
- ✅ More **complex** and **long-horizon** environments
- ✅ VLA-ready dataset formats: **RLDS** and **LeRobotDataset-v3**
- ✅ Dedicated **motion-planning** data collection scripts

---

## 🔎 Looking for the old MIKASA-Robo?

> [!NOTE]  
> If you need the original benchmark from the paper (**arXiv:2502.10550**), use the legacy RL branch:

- 🌿 Branch: [`mikasa-robo-rl`](https://github.com/CognitiveAISystems/MIKASA-Robo/tree/mikasa-robo-rl)
- 📦 Package version: `pip install mikasa-robo-suite==0.0.5`
- 📄 Paper: https://arxiv.org/abs/2502.10550




---
## Roadmap
1. 🚧 **Google Colab Notebook**  
   Publish an interactive Colab notebook with practical demos for running every environment.
2. 🚧 **Sphinx Documentation**  
   Expand and polish full Sphinx docs (setup, environments, datasets, and training workflows).
3. 🚧 **PyPI Release**  
   Prepare and ship an updated PyPI package release with the latest VLA improvements.
4. 🚧 **RLDS Dataset Release (90/90 Tasks)**  
   Publish RLDS datasets for the complete 90-task benchmark suite.
5. 🚧 **LeRobotDataset v3 Release (90/90 Tasks)**  
   Publish LeRobotDataset v3 exports for all 90 benchmark tasks.
6. 🚧 **LIBERO-Style VLA Evaluation Protocol**  
   Define and release a standardized protocol (fixed seeds, success metrics, and reporting template) for consistent VLA benchmarking.

---

## Table of Contents
- [Roadmap](#roadmap)
- [Repository Update](#repository-update)
- [What Changed for VLA](#what-changed-for-vla)
- [Benchmark at a Glance](#benchmark-at-a-glance)
- [Task Families](#task-families)
- [Full Task Catalog](#full-task-catalog)
- [Repository Layout](#repository-layout)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset Pipeline (NPZ -> RLDS -> LeRobot)](#dataset-pipeline-npz---rlds---lerobot)
  - [Dataset Availability (Coming Soon)](#dataset-availability-coming-soon)
  - [Data Layout on Disk](#data-layout-on-disk)
  - [What Each Format Contains](#what-each-format-contains)
  - [Detailed Guides (Collection + Conversion)](#detailed-guides-collection--conversion)
- [Reproducibility Notes](#reproducibility-notes)
- [Citation](#citation)

## Repository Update
This repository has been significantly updated with a **VLA-first focus**.

The codebase now separates legacy RL-oriented modules and VLA-oriented modules:
- `mikasa_robo_suite/rl/*` for legacy RL setup,
- `mikasa_robo_suite/vla/*` for VLA environments, collectors, wrappers, and motion planning.

## What Changed for VLA
The VLA update introduces changes intended to improve dataset quality and training signal:

1. Episode structures were revised to reduce redundant “idle end-phase” behavior.
2. No-op/curriculum wrappers are used to avoid premature actions before manipulation phase.
3. Memory horizons are randomized for better temporal robustness.
4. Object layouts were adjusted to reduce ambiguity across language-conditioned goals.
5. We added new task families that cover additional memory capabilities beyond the original setup (e.g., counting, long-sequence recall, timed execution, and multi-step reasoning).
6. We introduced long-horizon variants for legacy tasks, extending the benchmark with `-Long-` versions for delayed decision and sustained-memory evaluation.
7. Data collection is split by horizon: we generate most short-horizon trajectories with PPO oracles, while long-horizon trajectories are collected with motion planning.
8. Saved proprio is standardized to **7D**: `xyz(3) + rpy(3) + gripper(1)`.
9. Saved action format is standardized to **`pd_ee_delta_pose`** (7D).

## Benchmark at a Glance
The canonical task registry is [`mikasa_robo_vla_envs.csv`](mikasa_robo_vla_envs.csv).

Current benchmark snapshot:
- **90 configured VLA environments**
- **250 successful trajectories per task** (target), for a total of **22,500 trajectories**
- Approximately **5 million timesteps** across the full dataset
- Data collection split by source:
  - **34 tasks** collected with **PPO oracle rollouts** (`Data Source = PPO`)
  - **56 tasks** collected with **motion planning + replay** (`Data Source = MP`)
- Episode horizon range: **25 to 2160 steps**


## Task Families
Below is a cognitively oriented grouping (not just naming-based), derived from the VLA environment definitions and task prompts. Use `mikasa_robo_vla_envs.csv` for the full per-task list and exact instructions. Full tasks list can be found [below](#full-task-catalog)

| Family (High-level View) | Number of tasks | Episode Length Range (min-max) | Data Source | Core Memory / Reasoning Challenge |
|---|---:|---:|---|---|
| Hidden-Object Tracking Under Occlusion (`ShellGame*` variants) | 7 | 30-600 | PPO + MP | Track hidden object identity through occlusion and shuffle permutations |
| Predictive Interception and Capture (`Intercept*`, `InterceptGrab*`) | 6 | 60-60 | PPO | Infer motion from brief observations and time interception/capture actions |
| Spatial Reference Restoration (`Rotate*`, `TakeItBack`) | 5 | 60-90 | PPO | Preserve a remembered reference state and restore target geometry |
| Delayed Attribute Recall (`RememberColor/Shape/Shape+Color`, short + long) | 18 | 25-600 | PPO + MP | Retain color/shape bindings across delays and select the matching target |
| Out-of-Set Detection (`FindImposter*`) | 9 | 25-25 | PPO | Remember initial set membership and identify the novel candidate |
| Set and Sequence Capacity (`BunchOfColors`, `SeqOfColors`, `ChainOfColors`, short + long) | 18 | 400-1200 | MP | Maintain larger item sets and, when required, reproduce temporal order |
| Demonstration Path Imitation (`TraceShape*`, `TraceShapeSeq*`) | 6 | 250-1500 | MP | Encode observed trajectories and reproduce single or multi-part traces |
| Count-to-Action Execution (`BlinkCountButtonPress*`, short + long) | 6 | 150-1200 | MP | Count temporal events and execute the exact number of delayed actions |
| Delayed-Time Actuation (`TimedTransfer*`, short + long) | 6 | 200-1200 | MP | Trigger manipulation at a precise future timestep after a cue |
| Sequential Verification Memory (`BatteriesCheckerEasy/Hard`) | 4 | 540-2160 | MP | Track which batteries were already tested, avoid repeats, and complete search within a step budget |
| Concurrent Manipulation + Cue Recall (`GatherAndRecall*`) | 5 | 200-1000 | MP | Keep a latent cue in memory while solving a concurrent manipulation subtask |



## Repository Layout
```text
MIKASA-Robo/                        # repository root
├── mikasa_robo_suite/              # main MIKASA-Robo Python package
│   ├── rl/                         # legacy RL branch code
│   └── vla/                        # VLA-focused environments and tooling
│       ├── memory_envs/            # tasks definitions and registration
│       ├── dataset_collectors/     # NPZ data collection pipelines (PPO + MP)
│       └── utils/                  # shared VLA utilities and wrappers
│           └── motion_planning/    # motion-planning scripts per task family
│
├── data_mikasa_robo/               # local dataset storage root
│   ├── data_npz/                   # source per-episode NPZ trajectories
│   ├── data_rlds/                  # converted RLDS datasets
│   └── data_lerobot/               # converted LeRobot v3 datasets
│
├── utils/                          # repository-level helper scripts
│   ├── convert_npz_to_rlds/        # NPZ -> RLDS conversion project
│   └── convert_rlds_to_lerobot/    # RLDS -> LeRobot v3 conversion project
│
└── mikasa_robo_vla_envs.csv        # canonical manifest of VLA tasks and metadata
```

## Installation
### Recommended (uv)
```bash
# Install uv: https://docs.astral.sh/uv/getting-started/installation/
git clone git@github.com:CognitiveAISystems/MIKASA-Robo.git
cd MIKASA-Robo
uv sync --frozen
```

### Alternative
```bash
pip install -e .
# or
pip install mikasa-robo-suite (not supported for VLA update now)
```



## Quick Start (TODO: update after release to pypi and creation of the google colab demo notebook)
```python
import gymnasium as gym
import torch

import mikasa_robo_suite.vla.memory_envs  # registers VLA env IDs
from mikasa_robo_suite.vla.utils.wrappers import StateOnlyTensorToDictWrapper

env_id = "RememberColor3-VLA-v0"
env = gym.make(
    env_id,
    num_envs=4,
    obs_mode="rgb",
    control_mode="pd_ee_delta_pose",
    render_mode="all",
)
env = StateOnlyTensorToDictWrapper(env)

obs, _ = env.reset(seed=42)
for _ in range(25):
    action = torch.from_numpy(env.action_space.sample())
    obs, reward, terminated, truncated, info = env.step(action)

env.close()
```

## Dataset Pipeline (NPZ -> RLDS -> LeRobot)

This benchmark uses a three-stage dataset pipeline:
- **NPZ** as source episode dumps from successful rollouts (oracle PPO / motion planning),
- **RLDS** as a standardized sequence dataset format for training/evaluation pipelines, [RLDS](https://github.com/google-research/rlds), [openvlaoft](https://openvla-oft.github.io/) uses this format
- **LeRobot v3** as an ecosystem-friendly format for policy training and tooling. [lerobot dataset v3](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)

All three formats represent the same core signals (vision, proprio, actions, reward/success/done, language), but with different packaging conventions optimized for different workflows.

### Dataset Availability (Coming Soon)

Public dataset releases are being prepared.

- 🤗 **Hugging Face (NPZ source trajectories):** _coming soon_
- 🤗 **Hugging Face (RLDS exports):** _coming soon_
- 🤗 **Hugging Face (LeRobot v3 exports):** _coming soon_

> [!NOTE]
> Add your release links above once the datasets are published.

### Data Layout on Disk

Default local layout:

```text
data_mikasa_robo/
  data_npz/<task>/train_data_*.npz
  data_npz/<task>/metadata.json
  data_rlds/<task>/1.0.0/...
  data_lerobot/<task>/...
```

### What Each Format Contains

#### NPZ (source trajectories)
Per-episode files at `data_mikasa_robo/data_npz/<task>/train_data_*.npz`.

- `rgb`: `uint8`, shape `[T, 128, 128, 6]` (top + wrist RGB concatenated by channel)
- `proprio`: `float32`, shape `[T, 7]` (`xyz + rpy + gripper`)
- `action`: `float32`, shape `[T, 7]` (`pd_ee_delta_pose`)
- `reward`: `float32`, shape `[T]`
- `success`: `int32`, shape `[T]`
- `done`: `int32`, shape `[T]`
- `language_instruction`: scalar string
- `success_once`: scalar bool
- `episode_length`: scalar int32
- `episode_seed`: scalar int64

Each task folder also stores `metadata.json` with aggregate episode statistics.

#### RLDS (standardized sequential dataset)
Versioned task datasets at `data_mikasa_robo/data_rlds/<task>/1.0.0/`.

Key mapped fields include:
- `steps.observation.image` (top/base camera)
- `steps.observation.wrist_image`
- `steps.observation.proprio`
- `steps.action`
- `steps.reward`
- `steps.is_first`, `steps.is_last`, `steps.is_terminal`
- `steps.language_instruction`

#### LeRobot v3 (ecosystem format)
Converted datasets at `data_mikasa_robo/data_lerobot/<task>/...`, with LeRobot-compatible metadata and shard layout for downstream training tooling.

### Detailed Guides (Collection + Conversion)

To keep this main README lightweight, detailed operational instructions are maintained in dedicated docs:

- **NPZ collection (PPO + motion planning, resume, parallel launchers):**  
  [mikasa_robo_suite/vla/dataset_collectors/README.md](mikasa_robo_suite/vla/dataset_collectors/README.md)
- **NPZ -> RLDS conversion:**  
  [utils/convert_npz_to_rlds/README.md](utils/convert_npz_to_rlds/README.md)
- **RLDS -> LeRobot v3 conversion:**  
  [utils/convert_rlds_to_lerobot/README.md](utils/convert_rlds_to_lerobot/README.md)

## Reproducibility Notes
1. Use `uv sync --frozen` for lockfile-consistent environments.
2. Keep `control_mode=pd_ee_delta_pose` consistent between collection and replay.
3. Prefer manifest-driven runs from `mikasa_robo_vla_envs.csv` (`Data Source` column).
4. For long-horizon MP tasks, keep `--max-attempts` sufficiently high.
5. Validate each stage before proceeding:
   - NPZ exists and includes `metadata.json`,
   - RLDS has `dataset_info.json`, `features.json`, and tfrecords,
   - LeRobot output has `meta/info.json` and data/video shards.


## Full Task Catalog
Complete list of tasks from `mikasa_robo_vla_envs.csv` (in manifest order).

| # | Env ID | Horizon (max length) | Data Source | Prompt |
|---:|---|---:|---|---|
| 1 | `ShellGameTouch-VLA-v0` | 30 | PPO | Observe which cup hides the ball, wait, then touch that cup. |
| 2 | `ShellGamePush-VLA-v0` | 30 | PPO | Observe which cup hides the ball, wait, then push that cup forward. |
| 3 | `InterceptSlow-VLA-v0` | 60 | PPO | Intercept the rolling ball by moving to its path and deflecting it toward the target. |
| 4 | `InterceptMedium-VLA-v0` | 60 | PPO | Intercept the rolling ball by moving to its path and deflecting it toward the target. |
| 5 | `InterceptFast-VLA-v0` | 60 | PPO | Intercept the rolling ball by moving to its path and deflecting it toward the target. |
| 6 | `InterceptGrabSlow-VLA-v0` | 60 | PPO | Intercept the rolling ball and grasp it to stop it. |
| 7 | `InterceptGrabMedium-VLA-v0` | 60 | PPO | Intercept the rolling ball and grasp it to stop it. |
| 8 | `InterceptGrabFast-VLA-v0` | 60 | PPO | Intercept the rolling ball and grasp it to stop it. |
| 9 | `RotateLenientPos-VLA-v0` | 60 | PPO | Rotate the peg by {angle_deg} degrees to match the target angle. |
| 10 | `RotateLenientPosNeg-VLA-v0` | 60 | PPO | Rotate the peg by {angle_deg} degrees to match the target angle. |
| 11 | `RotateStrictPos-VLA-v0` | 90 | PPO | Rotate the peg by {angle_deg} degrees to match the target angle while keeping the center of the peg in place. |
| 12 | `RotateStrictPosNeg-VLA-v0` | 90 | PPO | Rotate the peg by {angle_deg} degrees to match the target angle while keeping the center of the peg in place. |
| 13 | `TakeItBack-VLA-v0` | 60 | PPO | Push the cube onto the red target, and when the target changes color, return the cube to its original position. |
| 14 | `RememberColor3-VLA-v0` | 25 | PPO | Observe the cube's color, wait, then touch the cube of the same color. |
| 15 | `RememberColor5-VLA-v0` | 25 | PPO | Observe the cube's color, wait, then touch the cube of the same color. |
| 16 | `RememberColor9-VLA-v0` | 25 | PPO | Observe the cube's color, wait, then touch the cube of the same color. |
| 17 | `RememberShape3-VLA-v0` | 25 | PPO | Observe the object's shape, wait, then touch the object of the same shape. |
| 18 | `RememberShape5-VLA-v0` | 25 | PPO | Observe the object's shape, wait, then touch the object of the same shape. |
| 19 | `RememberShape9-VLA-v0` | 25 | PPO | Observe the object's shape, wait, then touch the object of the same shape. |
| 20 | `RememberShapeAndColor3x2-VLA-v0` | 25 | PPO | Observe the object's shape and color, wait, then touch the object of the same shape and color. |
| 21 | `RememberShapeAndColor3x3-VLA-v0` | 25 | PPO | Observe the object's shape and color, wait, then touch the object of the same shape and color. |
| 22 | `RememberShapeAndColor5x3-VLA-v0` | 25 | PPO | Observe the object's shape and color, wait, then touch the object of the same shape and color. |
| 23 | `BunchOfColors3-VLA-v0` | 400 | MP | Observe which colored cubes appear during the cue, wait, then touch all of them in any order and press the center button. |
| 24 | `BunchOfColors5-VLA-v0` | 400 | MP | Observe which colored cubes appear during the cue, wait, then touch all of them in any order and press the center button. |
| 25 | `BunchOfColors7-VLA-v0` | 400 | MP | Observe which colored cubes appear during the cue, wait, then touch all of them in any order and press the center button. |
| 26 | `SeqOfColors3-VLA-v0` | 400 | MP | Observe which colored cubes appear during the cue, wait, then touch all of them in any order and press the center button. |
| 27 | `SeqOfColors5-VLA-v0` | 400 | MP | Observe which colored cubes appear during the cue, wait, then touch all of them in any order and press the center button. |
| 28 | `SeqOfColors7-VLA-v0` | 400 | MP | Observe which colored cubes appear during the cue, wait, then touch all of them in any order and press the center button. |
| 29 | `ChainOfColors3-VLA-v0` | 400 | MP | Observe which colored cubes appear during the cue, wait, then touch all of them in the same order as the cubes were shown and press the center button. |
| 30 | `ChainOfColors5-VLA-v0` | 400 | MP | Observe which colored cubes appear during the cue, wait, then touch all of them in the same order as the cubes were shown and press the center button. |
| 31 | `ChainOfColors7-VLA-v0` | 400 | MP | Observe which colored cubes appear during the cue, wait, then touch all of them in the same order as the cubes were shown and press the center button. |
| 32 | `ShellGameShuffleTouch-VLA-v0` | 60 | PPO | Observe which cup hides the ball, track the cups as they shuffle, then touch the correct cup. |
| 33 | `ShellGameShuffleColorLampTouch-VLA-v0` | 60 | PPO | Observe which color is under each cup, track the cups as they shuffle, then touch the cup matching the lamp color. |
| 34 | `ShellGameColorLampTouch-VLA-v0` | 30 | PPO | Observe which color is under each cup, then touch the cup matching the lamp color. |
| 35 | `FindImposterColor3-VLA-v0` | 25 | PPO | Observe the cubes shown, wait, then touch the cube whose color was not present before. |
| 36 | `FindImposterColor5-VLA-v0` | 25 | PPO | Observe the cubes shown, wait, then touch the cube whose color was not present before. |
| 37 | `FindImposterColor9-VLA-v0` | 25 | PPO | Observe the cubes shown, wait, then touch the cube whose color was not present before. |
| 38 | `FindImposterShape3-VLA-v0` | 25 | PPO | Observe the shapes shown, wait, then touch the object whose shape was not present before. |
| 39 | `FindImposterShape5-VLA-v0` | 25 | PPO | Observe the shapes shown, wait, then touch the object whose shape was not present before. |
| 40 | `FindImposterShape9-VLA-v0` | 25 | PPO | Observe the shapes shown, wait, then touch the object whose shape was not present before. |
| 41 | `FindImposterShapeAndColor3x2-VLA-v0` | 25 | PPO | Observe the objects shown, wait, then touch the object whose shape and color combination was not present before. |
| 42 | `FindImposterShapeAndColor3x3-VLA-v0` | 25 | PPO | Observe the objects shown, wait, then touch the object whose shape and color combination was not present before. |
| 43 | `FindImposterShapeAndColor5x3-VLA-v0` | 25 | PPO | Observe the objects shown, wait, then touch the object whose shape and color combination was not present before. |
| 44 | `BatteriesCheckerEasy-3-VLA-v0` | 540 | MP | Find all working batteries by inserting each one into the socket, observing the lamp result, and then pressing the button to confirm. |
| 45 | `BatteriesCheckerEasy-6-VLA-v0` | 1080 | MP | Find all working batteries by inserting each one into the socket, observing the lamp result, and then pressing the button to confirm. |
| 46 | `BatteriesCheckerHard-3-VLA-v0` | 1080 | MP | Find all working batteries by inserting each one into the socket, observing the lamp result, returning it from the socket to its initial slot, and then pressing the button to confirm. |
| 47 | `BatteriesCheckerHard-6-VLA-v0` | 2160 | MP | Find all working batteries by inserting each one into the socket, observing the lamp result, returning it from the socket to its initial slot, and then pressing the button to confirm. |
| 48 | `BlinkCountButtonPressEasy-VLA-v0` | 150 | MP | Count how many times the blue lamp blinks, press the red button exactly that many times when the red lamp turns green, then press the black button to submit your answer. |
| 49 | `BlinkCountButtonPressMedium-VLA-v0` | 200 | MP | Count how many times the blue lamp blinks, press the red button exactly that many times when the red lamp turns green, then press the black button to submit your answer. |
| 50 | `BlinkCountButtonPressHard-VLA-v0` | 300 | MP | Count how many times the blue lamp blinks, press the red button exactly that many times when the red lamp turns green, then press the black button to submit your answer. |
| 51 | `RememberColor3-Long-VLA-v0` | 600 | MP | Observe the cube's color, wait, then touch the cube of the same color. |
| 52 | `RememberColor5-Long-VLA-v0` | 600 | MP | Observe the cube's color, wait, then touch the cube of the same color. |
| 53 | `RememberColor9-Long-VLA-v0` | 600 | MP | Observe the cube's color, wait, then touch the cube of the same color. |
| 54 | `RememberShape3-Long-VLA-v0` | 600 | MP | Observe the object's shape, wait, then touch the object of the same shape. |
| 55 | `RememberShape5-Long-VLA-v0` | 600 | MP | Observe the object's shape, wait, then touch the object of the same shape. |
| 56 | `RememberShape9-Long-VLA-v0` | 600 | MP | Observe the object's shape, wait, then touch the object of the same shape. |
| 57 | `RememberShapeAndColor3x2-Long-VLA-v0` | 600 | MP | Observe the object's shape and color, wait, then touch the object of the same shape and color. |
| 58 | `RememberShapeAndColor3x3-Long-VLA-v0` | 600 | MP | Observe the object's shape and color, wait, then touch the object of the same shape and color. |
| 59 | `RememberShapeAndColor5x3-Long-VLA-v0` | 600 | MP | Observe the object's shape and color, wait, then touch the object of the same shape and color. |
| 60 | `BunchOfColors3-Long-VLA-v0` | 700 | MP | Observe which colored cubes appear during the cue, wait, then touch all of them in any order and press the center button. |
| 61 | `BunchOfColors5-Long-VLA-v0` | 700 | MP | Observe which colored cubes appear during the cue, wait, then touch all of them in any order and press the center button. |
| 62 | `BunchOfColors7-Long-VLA-v0` | 700 | MP | Observe which colored cubes appear during the cue, wait, then touch all of them in any order and press the center button. |
| 63 | `SeqOfColors3-Long-VLA-v0` | 800 | MP | Observe which colored cubes appear during the cue, wait, then touch all of them in any order and press the center button. |
| 64 | `SeqOfColors5-Long-VLA-v0` | 1000 | MP | Observe which colored cubes appear during the cue, wait, then touch all of them in any order and press the center button. |
| 65 | `SeqOfColors7-Long-VLA-v0` | 1200 | MP | Observe which colored cubes appear during the cue, wait, then touch all of them in any order and press the center button. |
| 66 | `ChainOfColors3-Long-VLA-v0` | 800 | MP | Observe which colored cubes appear during the cue, wait, then touch all of them in the same order as the cubes were shown and press the center button. |
| 67 | `ChainOfColors5-Long-VLA-v0` | 1000 | MP | Observe which colored cubes appear during the cue, wait, then touch all of them in the same order as the cubes were shown and press the center button. |
| 68 | `ChainOfColors7-Long-VLA-v0` | 1200 | MP | Observe which colored cubes appear during the cue, wait, then touch all of them in the same order as the cubes were shown and press the center button. |
| 69 | `ShellGameShuffleTouch-Long-VLA-v0` | 600 | MP | Observe which cup hides the ball, track the cups as they shuffle, then touch the correct cup. |
| 70 | `ShellGameShuffleColorLampTouch-Long-VLA-v0` | 600 | MP | Observe which color is under each cup, track the cups as they shuffle, then touch the cup matching the lamp color. |
| 71 | `BlinkCountButtonPressEasy-Long-VLA-v0` | 1200 | MP | Count how many times the blue lamp blinks, press the red button exactly that many times when the red lamp turns green, then press the black button to submit your answer. |
| 72 | `BlinkCountButtonPressMedium-Long-VLA-v0` | 1200 | MP | Count how many times the blue lamp blinks, press the red button exactly that many times when the red lamp turns green, then press the black button to submit your answer. |
| 73 | `BlinkCountButtonPressHard-Long-VLA-v0` | 1200 | MP | Count how many times the blue lamp blinks, press the red button exactly that many times when the red lamp turns green, then press the black button to submit your answer. |
| 74 | `TraceShapeEasy-VLA-v0` | 250 | MP | Watch the red cube trace a shape on the table. When the lamp turns green, pick up the green cube and trace exactly the same shape. |
| 75 | `TraceShapeMedium-VLA-v0` | 300 | MP | Watch the red cube trace a shape on the table. When the lamp turns green, pick up the green cube and trace exactly the same shape. |
| 76 | `TraceShapeHard-VLA-v0` | 350 | MP | Watch the red cube trace a shape on the table. When the lamp turns green, pick up the green cube and trace exactly the same shape. |
| 77 | `TraceShapeSeqEasy-VLA-v0` | 1500 | MP | Watch the red cube trace a sequence of shapes. When the lamp turns green, pick up the green cube and trace the same sequence in order. After finishing all shapes, press the button to submit your answer. |
| 78 | `TraceShapeSeqMedium-VLA-v0` | 1500 | MP | Watch the red cube trace a sequence of shapes. When the lamp turns green, pick up the green cube and trace the same sequence in order. After finishing all shapes, press the button to submit your answer. |
| 79 | `TraceShapeSeqHard-VLA-v0` | 1500 | MP | Watch the red cube trace a sequence of shapes. When the lamp turns green, pick up the green cube and trace the same sequence in order. After finishing all shapes, press the button to submit your answer. |
| 80 | `TimedTransferEasy-VLA-v0` | 200 | MP | When the white lamp turns green, start counting steps from that exact moment. Move the blue cube from the green disc to the red disc exactly on step 100 of that count. |
| 81 | `TimedTransferMedium-VLA-v0` | 250 | MP | When the white lamp turns green, start counting steps from that exact moment. Move the blue cube from the green disc to the red disc exactly on step 150 of that count. |
| 82 | `TimedTransferHard-VLA-v0` | 300 | MP | When the white lamp turns green, start counting steps from that exact moment. Move the blue cube from the green disc to the red disc exactly on step 200 of that count. |
| 83 | `TimedTransferEasy-Long-VLA-v0` | 600 | MP | When the white lamp turns green, start counting steps from that exact moment. Move the blue cube from the green disc to the red disc exactly on step 300 of that count. |
| 84 | `TimedTransferMedium-Long-VLA-v0` | 900 | MP | When the white lamp turns green, start counting steps from that exact moment. Move the blue cube from the green disc to the red disc exactly on step 500 of that count. |
| 85 | `TimedTransferHard-Long-VLA-v0` | 1200 | MP | When the white lamp turns green, start counting steps from that exact moment. Move the blue cube from the green disc to the red disc exactly on step 1000 of that count. |
| 86 | `GatherAndRecall1-VLA-v0` | 200 | MP | Move all cubes onto the disc. A lamp will briefly flash while you work. After all cubes are placed, press the button matching the flash color. |
| 87 | `GatherAndRecall3-VLA-v0` | 400 | MP | Move all cubes onto the disc. A lamp will briefly flash while you work. After all cubes are placed, press the button matching the flash color. |
| 88 | `GatherAndRecall5-VLA-v0` | 600 | MP | Move all cubes onto the disc. A lamp will briefly flash while you work. After all cubes are placed, press the button matching the flash color. |
| 89 | `GatherAndRecall7-VLA-v0` | 800 | MP | Move all cubes onto the disc. A lamp will briefly flash while you work. After all cubes are placed, press the button matching the flash color. |
| 90 | `GatherAndRecall9-VLA-v0` | 1000 | MP | Move all cubes onto the disc. A lamp will briefly flash while you work. After all cubes are placed, press the button matching the flash color. |


## Citation
If you use MIKASA-Robo in research, please cite:

```bibtex
@inproceedings{
    cherepanov2026mikasarobo,
    title={Memory, Benchmark \& Robots: A Benchmark for Solving Complex Tasks with Reinforcement Learning},
    author={Egor Cherepanov and Nikita Kachaev and Alexey Kovalev and Aleksandr Panov},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=9cLPurIZMj}
}
```
