<h1 align="center">MIKASA-Robo</h1>

<h3 align="center">Benchmark for robotic tabletop manipulation memory-intensive tasks</h3>

<div align="center">
    <a href="https://arxiv.org/abs/2502.10550">
        <img src="https://img.shields.io/badge/arXiv-2502.10550-b31b1b.svg"/>
    </a>
    <a href="https://sites.google.com/view/memorybenchrobots/">
        <img src="https://img.shields.io/badge/Website-Project_Page-blue.svg"/>
    </a>
</div>

---

<div align="center" style="display: flex; justify-content: center; gap: 10px;">
    <img src="assets/shell-game-touch-v0.gif" width="200" />
    <img src="assets/chain-of-colors-7.gif" width="200" />
    <img src="assets/take-it-back-v0.gif" width="200" />
    <img src="assets/remember-shape-and-color-5x3.gif" width="200" />
</div>
<p align="center"><i>Example tasks from the MIKASA-Robo benchmark</i></p>

## Overview

MIKASA-Robo is a comprehensive benchmark suite for memory-intensive robotic manipulation tasks, part of the MIKASA (Memory-Intensive Skills Assessment Suite for Agents) framework. It features:

- 12 distinct task types with varying difficulty levels
- 32 total tasks covering different memory aspects
- First benchmark specifically designed for testing agent memory in robotic manipulation

## Key Features

- **Diverse Memory Testing**: Covers four fundamental memory types:
  - Object Memory
  - Spatial Memory
  - Sequential Memory
  - Memory Capacity

- **Built on ManiSkill3**: Leverages the powerful [ManiSkill3](https://maniskill.readthedocs.io/en/latest/) framework, providing:
  - GPU parallelization
  - User-friendly interface
  - Customizable environments


## List of Tasks

| Preview | Memory Task | Mode | Brief Description | T | Memory Task Type |
|--------------------------|------------|------|------|---|--|
| <img src="assets/shell-game-touch-v0.gif" width="200"/> | `ShellGame[Mode]-v0` | `Touch`<br>`Push`<br>`Pick` | Memorize the position of the ball after some time being covered by the cups and then interact with the cup the ball is under. | 90 | Object |
| <img src="assets/intercept-medium-v0.gif" width="200"/> | `Intercept[Mode]-v0` | `Slow`<br>`Medium`<br>`Fast` | Memorize the positions of the rolling ball, estimate its velocity through those positions, and then aim the ball at the target. | 90| Spatial |
| <img src="assets/intercept-grab-medium.gif" width="200"/> | `InterceptGrab[Mode]-v0` | `Slow`<br>`Medium`<br>`Fast` | Memorize the positions of the rolling ball, estimate its velocity through those positions, and then catch the ball with the gripper and lift it up. | 90 | Spatial |
| <img src="assets/rotate-lenient-pos-v0.gif" width="200"/> | `RotateLenient[Mode]-v0` | `Pos`<br>`PosNeg` | Memorize the initial position of the peg and rotate it by a given angle. | 90| Spatial |
| <img src="assets/rotate-strict-pos.gif" width="200"/> | `RotateStrict[Mode]-v0` | `Pos`<br>`PosNeg` | Memorize the initial position of the peg and rotate it to a given angle without shifting its center. | 90 | Object |
| <img src="assets/take-it-back-v0.gif" width="200"/> | `TakeItBack-v0` | --- | Memorize the initial position of the cube, move it to the target region, and then return it to its initial position. | 180 | Spatial |
| <img src="assets/remember-color-9-v0.gif" width="200"/> | `RememberColor[Mode]-v0` | `3`/`5`/`9` | Memorize the color of the cube and choose among other colors. | 60 | Object |
| <img src="assets/remember-shape-9-v0.gif" width="200"/> | `RememberShape[Mode]-v0` | `3`/`5`/`9` | Memorize the shape of the cube and choose among other shapes. | 60 | Object |
| <img src="assets/remember-shape-and-color-5x3.gif" width="200"/> | `RememberShapeAndColor[Mode]-v0` | `3×2`/`3×3`<br>`5×3` | Memorize the shape and color of the cube and choose among other shapes and colors. | 60 | Object |
| <img src="assets/bunch-of-colors-7.gif" width="200"/> | `BunchOfColors[Mode]-v0` | `3`/`5`/`7` | Remember the colors of the set of cubes shown simultaneously in the bunch and touch them in any order. | 120 | Capacity |
| <img src="assets/seq-of-colors-7.gif" width="200"/> | `SeqOfColors[Mode]-v0` | `3`/`5`/`7` | Remember the colors of the set of cubes shown sequentially and then select them in any order. | 120 | Capacity |
| <img src="assets/chain-of-colors-7.gif" width="200"/> | `ChainOfColors[Mode]-v0` | `3`/`5`/`7` | Remember the colors of the set of cubes shown sequentially and then select them in the same order. | 120 | Sequential |

**Total: 32 tabletop robotic manipulation memory-intensive tasks in 12 groups**. T - episode timeout.


## Quick Start

### Installation
```bash
git clone git@github.com:CognitiveAISystems/MIKASA-Robo.git
cd MIKASA-Robo
pip install -r requirements.txt
```


## Basic Usage
```python
import mikasa_robo
from mikasa_robo.utils.wrappers import *

num_envs, seed = 512, 123
# Create the environment via gym.make()
# obs_mode="rgb" for modes "RGB", "RGB+joint", "RGB+oracle" etc.
# obs_mode="state" for mode "state"
env = gym.make("RememberColor9-v0", num_envs=num_envs,
                obs_mode="rgb", render_mode="all")

env = StateOnlyTensorToDictWrapper(env) # [always] gen. obs keys

obs, _ = env.reset(seed)
for i in tqdm(range(89)):
    action = torch.from_numpy(env.action_space.sample())
    obs, reward, terminated, truncated, info = env.step(action)
env.close()
```

## Advanced Usage: Debug Wrappers
```python
import mikasa_robo
from mikasa_robo.utils.wrappers import *
from mani_skill.utils.wrappers import RecordEpisode

num_envs, seed = 512, 123
env = gym.make("RememberColor9-v0", num_envs=num_envs,
obs_mode="rgb", render_mode="all")

env = StateOnlyTensorToDictWrapper(env) # [always] gen. obs keys
env = RememberColorInfoWrapper(env) # [debug] show task info
env = RenderStepInfoWrapper(env) # [debug] show env step
env = RenderRewardInfoWrapper(env) # [debug] show total reward
env = DebugRewardWrapper(env) # [debug] show reward info
env = RecordEpisode(env, "./videos/demo_remember-color-9")

obs, _ = env.reset(seed)
for i in tqdm(range(89)):
    action = torch.from_numpy(env.action_space.sample())
    obs, reward, terminated, truncated, info = env.step(action)
env.close()

Video("./videos/demo_remember-color-9/0.mp4", embed=True)
```


## Training
MIKASA-Robo supports multiple training configurations:

### PPO with MLP (State-Based)
```bash
python3 baselines/ppo/ppo_memtasks.py \
    --env_id=RememberColor9-v0 \
    --exp-name=remember-color-9-v0 \
    --num-steps=60 \
    --num_eval_steps=180 \
    --include-state
```

### PPO with MLP (RGB + Joint)
```bash
python3 baselines/ppo/ppo_memtasks.py \
    --env_id=RememberColor9-v0 \
    --exp-name=remember-color-9-v0 \
    --num-steps=60 \
    --num_eval_steps=180 \
    --include-rgb \
    --include-joints
```

### PPO with LSTM (RGB + Joint)
```bash
python3 baselines/ppo/ppo_memtasks_lstm.py \
    --env_id=RememberColor9-v0 \
    --exp-name=remember-color-9-v0 \
    --num-steps=60 \
    --num_eval_steps=180 \
    --include-rgb \
    --include-joints
```

To train with sparse rewards, add `--reward-mode=sparse`.

## MIKASA-Robo Ideology
The agent's memory capabilities can be accessed not only when the environment demands memory, but also when the observations are provided in the correct format. Currently, we have implemented several training modes:

- `state`: In this mode, the agent receives comprehensive, vectorized information about the environment, joints, and TCP pose, along with oracle data that is essential for solving memory-intensive tasks. When trained in this way, the agent addresses the MDP problem and **does not require memory**.

- `RGB+joints`: Here, the agent receives image data from a camera mounted above and from the manipulator's gripper, along with the position and velocity of its joints. This mode provides no additional information, meaning the agent must learn to store and utilize oracle data. It is designed to **test the agent's memory** capabilities.

These training modes are obtained by using correct flags. Thus,
```bash
# To train in `state` mode:
--include-state

# To train in `RGB+joints` mode:
--include-rgb \
--include-joints

# Additionally, for debugging you can add oracle information to the observation:
--include-oracle
```

## Collecting datasets for Offline RL
1. Run training PPO-MLP on MIKASA-Robo tasks in the `state` mode (i.e. in MDP mode with oracle information):
```bash
# For single task:
python3 dataset_collectors/get_dataset_collectors_ckpt.py --env_id=ShellGameTouch-v0

# For all tasks:
python3dataset_collectors/parallel_training_manager.py
```

2. Collect datasets using oracle checkpoints:
```bash
# For single task:
python3 dataset_collectors/get_mikasa_robo_datasets.py --env-id=ShellGameTouch-v0 --path-to-save-data="data" --ckpt-dir="."

# For all tasks:
python3 dataset_collectors/parallel_dataset_collection_manager.py --path-to-save-data="data" --ckpt-dir="."
```

## Citation
If you find our work useful, please cite our paper:
```
@misc{cherepanov2025mikasa,
      title={Memory, Benchmark & Robots: A Benchmark for Solving Complex Tasks with Reinforcement Learning}, 
      author={Egor Cherepanov and Nikita Kachaev and Alexey K. Kovalev and Aleksandr I. Panov},
      year={2025},
      eprint={2502.10550},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.10550}, 
}
```