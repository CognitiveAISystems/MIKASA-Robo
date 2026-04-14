<h1 align="center">MIKASA-Robo (VLA Update)</h1>

<h3 align="center">A memory-centric tabletop manipulation benchmark, reworked for Vision-Language-Action training</h3>

<div align="center">
  <a href="https://arxiv.org/abs/2502.10550">
    <img src="https://img.shields.io/badge/arXiv-2502.10550-b31b1b.svg"/>
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
</div>

---
## Важное обновление!
Теперь этот репозиторий в `main` переквалифицирован под работу с VLA моделями и называется MIKASA-Robo-90: мы добавилии больше задач и обновили старые (теперь их 90), больше long-horizon tasks, VLA-совместимые datasets formats (RLDS, LerobotDataset-v3) и motion-planning скрипты. Если вам нужна предыдущая версия MIKASA-Robo, она сохранена в 


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
  - [Step 1: Collect NPZ Trajectories](#step-1-collect-npz-trajectories)
  - [Step 2: Convert NPZ to RLDS](#step-2-convert-npz-to-rlds)
  - [Step 3: Convert RLDS to LeRobot v3](#step-3-convert-rlds-to-lerobot-v3)
- [Data Schema](#data-schema)
- [Reproducibility Notes](#reproducibility-notes)
- [Citation](#citation)

## Repository Update
This repository has been significantly updated with a **VLA-first focus**.

The original MIKASA-Robo benchmark (memory-intensive tabletop tasks) is now extended into a reproducible pipeline for:
- environment benchmarking,
- large-scale VLA trajectory collection,
- conversion to RLDS,
- downstream export to LeRobot v3.

The codebase now separates legacy RL-oriented modules and VLA-oriented modules:
- `mikasa_robo_suite/rl/*` for legacy RL setup,
- `mikasa_robo_suite/vla/*` for VLA environments, collectors, wrappers, and motion planning.

## What Changed for VLA
The VLA update introduces changes intended to improve dataset quality and training signal:

1. Episode structures were revised to reduce redundant “idle end-phase” behavior.
2. No-op/curriculum wrappers are used to avoid premature actions before manipulation phase.
3. Memory horizons are randomized for better temporal robustness.
4. Object layouts were adjusted to reduce ambiguity across language-conditioned goals.
5. Oracle behavior was tuned for smoother, teleoperation-like trajectories with mild stochasticity.
6. Saved proprio is standardized to **7D**: `xyz(3) + rpy(3) + gripper(1)`.
7. Saved action format is standardized to **`pd_ee_delta_pose`** (7D).
8. Collection tooling supports recovery/resume workflows for interrupted long runs.

## Benchmark at a Glance
Source of truth for configured VLA tasks: `mikasa_robo_vla_envs.csv`.

Current manifest summary:
- `90` configured VLA env IDs.
- `64` short-horizon tasks and `26` long-horizon tasks (`-Long-`).
- Data collection source split:
  - `34` tasks via PPO-oracle rollout (`Data Source = PPO`),
  - `56` tasks via motion-planning + replay (`Data Source = MP`).
- Episode horizon range in manifest: from `25` to `2160` steps.

## Task Families
Below is a compact family-level overview. Use `mikasa_robo_vla_envs.csv` for the full per-task list and prompts.

| Family | # Env IDs | Short / Long | Data Source | Memory Skill |
|---|---:|---:|---|---|
| ShellGame Core | 2 | 2 / 0 | PPO | Object tracking under occlusion |
| Intercept + InterceptGrab | 6 | 6 / 0 | PPO | Spatial prediction and timing |
| Rotate (Lenient + Strict) | 4 | 4 / 0 | PPO | Pose memory and controlled rotation |
| TakeItBack | 1 | 1 / 0 | PPO | Return-to-origin memory |
| Remember (Color / Shape / Shape+Color) | 18 | 9 / 9 | PPO + MP | Delayed recall |
| FindImposter (Color / Shape / Shape+Color) | 9 | 9 / 0 | PPO | Negation-style memory |
| Memory Capacity (Bunch / Seq / Chain) | 18 | 9 / 9 | MP | Set and sequence memory |
| ShellGame Lamps + Shuffle variants | 5 | 3 / 2 | PPO + MP | Tracking + conditional target selection |
| Batteries Checker (Easy + Hard) | 4 | 4 / 0 | MP | Long-horizon hypothesis testing |
| Blink Count + Button Press | 6 | 3 / 3 | MP | Counting + delayed execution |
| TraceShape + TraceShapeSeq | 6 | 6 / 0 | MP | Demonstration recall and reproduction |
| TimedTransfer | 6 | 3 / 3 | MP | Delayed timing precision |
| GatherAndRecall | 5 | 5 / 0 | MP | Dual-task memory under manipulation |

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

## Repository Layout
```text
MIKASA-Robo/
├── mikasa_robo_suite/
│   ├── rl/
│   └── vla/
│       ├── memory_envs/
│       ├── dataset_collectors/
│       └── utils/
│           └── motion_planning/
├── data_mikasa_robo/
│   ├── data_npz/
│   ├── data_rlds/
│   └── data_lerobot/
├── utils/
│   ├── run_parallel_npz_collection.sh
│   ├── resume_interrupted_mp_collection.sh
│   ├── convert_npz_to_rlds/
│   └── convert_rlds_to_lerobot/
└── mikasa_robo_vla_envs.csv
```

## Installation
### Recommended (uv)
```bash
git clone git@github.com:CognitiveAISystems/MIKASA-Robo.git
cd MIKASA-Robo
uv sync --frozen
```

### Alternative
```bash
pip install -e .
# or
pip install mikasa-robo-suite
```

Python compatibility from `pyproject.toml`: `>=3.9,<3.12`.

## Quick Start
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

### Step 1: Collect NPZ Trajectories

#### A) Single-task collection (PPO-oracle tasks)
Use when `Data Source = PPO` in `mikasa_robo_vla_envs.csv`.

```bash
uv run python mikasa_robo_suite/vla/dataset_collectors/get_mikasa_robo_datasets.py \
  --env-id RememberColor3-VLA-v0 \
  --path-to-save-data data_mikasa_robo \
  --ckpt-dir . \
  --num-train-data 250
```

This script expects oracle checkpoints under `oracle_checkpoints/**/final_success_ckpt.pt`.

#### B) Single-task collection (Motion-planning tasks)
Use when `Data Source = MP`.

```bash
uv run python mikasa_robo_suite/vla/dataset_collectors/get_mikasa_robo_datasets_motion_planning.py \
  --env-id TraceShapeHard-VLA-v0 \
  --path-to-save-data data_mikasa_robo \
  --num-train-data 250 \
  --max-attempts 5000 \
  --seed 0
```

Collector behavior:
- planner generates raw trajectory,
- ManiSkill replay converts to `pd_ee_delta_pose`,
- successful replay rollouts are saved as per-episode `.npz`.

#### C) Parallel mixed PPO+MP collection
The helper launcher reads an env list file with format:
`<env_id> <max_length> <enabled(TRUE/FALSE)> <method(PPO/MP)>`

Generate it directly from the manifest:
```bash
python - <<'PY'
import csv

with open("mikasa_robo_vla_envs.csv", newline="") as f_in, open("envs.txt", "w", encoding="utf-8") as f_out:
    reader = csv.DictReader(f_in)
    for row in reader:
        f_out.write(
            f"{row['name']}\t{row['max length']}\t{row['Configured']}\t{row['Data Source']}\n"
        )
print("Wrote envs.txt")
PY
```

Example run:
```bash
GPU_LIST=0,1,2 JOBS_PER_GPU=2 NUM_TRAIN_DATA=250 MAX_ATTEMPTS_MP=5000 \
bash utils/run_parallel_npz_collection.sh envs.txt
```

#### D) Resume interrupted MP jobs
```bash
bash utils/resume_interrupted_mp_collection.sh
```
(Adjust `ENVS=(...)` in the script before launch.)

---

### Step 2: Convert NPZ to RLDS
Use the isolated converter project (recommended):

```bash
uv sync --project utils/convert_npz_to_rlds/rlds_dataset_builder
```

#### Convert one task
```bash
uv run --project utils/convert_npz_to_rlds/rlds_dataset_builder \
  python utils/convert_npz_to_rlds/convert_npz_task_to_rlds.py \
  --task RememberColor3-VLA-v0 \
  --overwrite-dest
```

#### Convert all currently available tasks in `data_mikasa_robo/data_npz`
```bash
for task_dir in data_mikasa_robo/data_npz/*; do
  [ -d "${task_dir}" ] || continue
  task="$(basename "${task_dir}")"
  [[ "${task}" == _* ]] && continue

  uv run --project utils/convert_npz_to_rlds/rlds_dataset_builder \
    python utils/convert_npz_to_rlds/convert_npz_task_to_rlds.py \
    --task "${task}" \
    --overwrite-dest
done
```

Outputs are written to:
- `data_mikasa_robo/data_rlds/<task>/1.0.0/`

With key files:
- `dataset_info.json`
- `features.json`
- `mikasa_dataset-train.tfrecord-*`
- copied source `metadata.json`

---

### Step 3: Convert RLDS to LeRobot v3
Prepare the converter environment:

```bash
uv sync --project utils/convert_rlds_to_lerobot
```

#### Convert one task
```bash
uv run --project utils/convert_rlds_to_lerobot \
  python utils/convert_rlds_to_lerobot/convert_rlds_to_lerobot.py \
  --task RememberColor3-VLA-v0 \
  --overwrite-dest
```

#### Convert all tasks
```bash
uv run --project utils/convert_rlds_to_lerobot \
  python utils/convert_rlds_to_lerobot/convert_rlds_to_lerobot.py \
  --all \
  --overwrite-dest
```

Default output:
- `data_mikasa_robo/data_lerobot/<task>/...`

## Data Schema
### NPZ episode schema (`data_mikasa_robo/data_npz/<task>/train_data_*.npz`)
- `rgb`: `uint8`, shape `[T, 128, 128, 6]` (top + wrist RGB concatenated by channel)
- `proprio`: `float32`, shape `[T, 7]` (`xyz+rpy+gripper`)
- `action`: `float32`, shape `[T, 7]` (`pd_ee_delta_pose`)
- `reward`: `float32`, shape `[T]`
- `success`: `int32`, shape `[T]`
- `done`: `int32`, shape `[T]`
- `language_instruction`: scalar string
- `success_once`: scalar bool
- `episode_length`: scalar int32
- `episode_seed`: scalar int64

### RLDS schema highlights
RLDS builder (`utils/convert_npz_to_rlds/rlds_dataset_builder/mikasa_dataset/...`) maps NPZ into:
- `steps.observation.image` (base/top camera),
- `steps.observation.wrist_image`,
- `steps.observation.proprio`,
- `steps.action`, `steps.reward`, `steps.is_first/is_last/is_terminal`,
- `steps.language_instruction`.

## Reproducibility Notes
1. Use `uv sync --frozen` for lockfile-consistent environments.
2. Keep `control_mode=pd_ee_delta_pose` consistent between collection and replay.
3. Prefer manifest-driven runs from `mikasa_robo_vla_envs.csv` (`Data Source` column).
4. For long-horizon MP tasks, keep `--max-attempts` sufficiently high.
5. Validate each stage before proceeding:
   - NPZ exists and includes `metadata.json`,
   - RLDS has `dataset_info.json`, `features.json`, and tfrecords,
   - LeRobot output has `meta/info.json` and data/video shards.

## Citation
If you use MIKASA-Robo in research, please cite:

```bibtex
@misc{cherepanov2025shaping,
  title={Shaping Memory: How Environment Design Impacts Memory and Reasoning in Visual Reinforcement Learning},
  author={Egor Cherepanov and Mikhail Terekhov and Yaroslav Ilyushin and Ivan Drokin and Nadezhda Chirkova and Mikhail Burtsev},
  year={2025},
  eprint={2502.10550},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2502.10550}
}
```
