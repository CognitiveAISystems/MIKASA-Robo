<h1 align="center">MIKASA-Robo-VLA</h1>

<p align="center">
  <b>A memory-intensive robotic manipulation benchmark for Vision-Language-Action research.</b>
</p>

<p align="center">
  <a href="https://sites.google.com/view/memorybenchrobots/">
    <img src="https://img.shields.io/badge/🌐_Project-Page-blue?style=for-the-badge" alt="Project Page">
  </a>
  <a href="https://cognitiveaisystems.github.io/MIKASA-Robo/">
    <img src="https://img.shields.io/badge/📚_Documentation-MIKASA--Robo--VLA-0b7285?style=for-the-badge" alt="Documentation">
  </a>
  <a href="https://arxiv.org/abs/2502.10550">
    <img src="https://img.shields.io/badge/📄_arXiv-2502.10550-b31b1b?style=for-the-badge" alt="arXiv">
  </a>
  <a href="https://huggingface.co/mikasa-robo">
    <img src="https://img.shields.io/badge/🤗_Datasets-Hugging_Face-yellow?style=for-the-badge" alt="Hugging Face Datasets">
  </a>
  <a href="https://pypi.org/project/mikasa-robo-suite/">
    <img src="https://img.shields.io/badge/📦_PyPI-mikasa--robo--suite-3775A9?style=for-the-badge" alt="PyPI">
  </a>
</p>

## What is MIKASA-Robo-VLA?

MIKASA-Robo-VLA extends the MIKASA-Robo memory benchmark to language-conditioned Vision-Language-Action research. It provides tabletop robotic manipulation environments that require an agent to retain and use information across delayed, occluded, temporal, or multi-stage interactions.

The canonical VLA benchmark contains **90 tasks** with natural-language instructions, ManiSkill/Gymnasium environments, and released trajectory datasets for training and evaluation. The benchmark task manifest is [`mikasa_robo_vla_envs.csv`](mikasa_robo_vla_envs.csv).

> [!IMPORTANT]
> The full documentation is available at [cognitiveaisystems.github.io/MIKASA-Robo](https://cognitiveaisystems.github.io/MIKASA-Robo/). This README keeps only the minimum setup, environment, benchmark, and dataset examples.

This README targets the VLA benchmark. The earlier RL-oriented MIKASA-Robo implementation remains available for legacy use; see the [documentation](https://cognitiveaisystems.github.io/MIKASA-Robo/) for compatibility notes.

## Installation

Install from the repository with the locked `uv` environment:

```bash
git clone git@github.com:CognitiveAISystems/MIKASA-Robo.git
cd MIKASA-Robo
git submodule update --init --recursive
uv sync --frozen
```

See the [installation guide](https://cognitiveaisystems.github.io/MIKASA-Robo/installation.html) for system requirements, package-install alternatives, and setup troubleshooting.

## Quick Start

Every benchmark environment should be wrapped with `apply_mikasa_vla_wrappers` immediately after `gym.make` so its observations and task logic match the released VLA data pipeline.

```python
import gymnasium as gym
import torch

import mikasa_robo_suite.vla.memory_envs  # registers VLA env IDs
from mikasa_robo_suite.vla.utils.apply_wrappers import apply_mikasa_vla_wrappers

env = gym.make(
    "RememberColor3-VLA-v0",
    num_envs=1,
    obs_mode="rgb",
    control_mode="pd_ee_delta_pose",
    reward_mode="normalized_dense",
    render_mode="all",
    sim_backend="gpu",
)
env = apply_mikasa_vla_wrappers(env, include_overlays=False)

obs, info = env.reset(seed=42)
for _ in range(env.max_episode_steps):
    action = torch.as_tensor(env.action_space.sample(), device=env.unwrapped.device)
    obs, reward, terminated, truncated, info = env.step(action)
    if torch.as_tensor(terminated | truncated).any():
        break

env.close()
```

For task browsing, wrapper behavior, language instructions, and the observation/action contract, use the [quick start](https://cognitiveaisystems.github.io/MIKASA-Robo/quickstart.html), [environment catalogue](https://cognitiveaisystems.github.io/MIKASA-Robo/vla_environments/index.html), and [observation/action reference](https://cognitiveaisystems.github.io/MIKASA-Robo/observation_space.html).

## Benchmarking

Run the reference checkpoint-free dummy policy first to smoke-test the evaluation pipeline:

```bash
uv run python examples/eval_demo.py \
  --num-episodes 1 --sim-backend gpu \
  --output-dir eval_results/dummy
```

Canonical evaluation is organized by horizon split and uses the benchmark protocol for task selection, seeds, metrics, and result files. See [Benchmarking](https://cognitiveaisystems.github.io/MIKASA-Robo/benchmarking.html) and the [Evaluation Protocol](https://cognitiveaisystems.github.io/MIKASA-Robo/evaluation_protocol.html) before reporting results.

## Datasets

MIKASA-Robo-VLA provides the full 90-task trajectory release on Hugging Face. The data pipeline supports:

- **NPZ** source episodes for local collection and custom preprocessing.
- **RLDS / TFDS** for episodic dataset pipelines.
- **LeRobotDataset v3** for modern PyTorch and VLA fine-tuning workflows.

Download one LeRobotDataset task with `huggingface_hub`:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="mikasa-robo/mikasa-robo-vla-lerobot",
    repo_type="dataset",
    allow_patterns="RememberColor3-VLA-v0/**",
    local_dir="data_mikasa_robo/data_lerobot",
)
```

The [dataset guide](https://cognitiveaisystems.github.io/MIKASA-Robo/datasets.html) covers the public RLDS and LeRobot releases, local collection, dataset fields, and export workflows.

## Useful Links

- [Documentation](https://cognitiveaisystems.github.io/MIKASA-Robo/)
- [Installation](https://cognitiveaisystems.github.io/MIKASA-Robo/installation.html)
- [Quick Start](https://cognitiveaisystems.github.io/MIKASA-Robo/quickstart.html)
- [Environments and Tasks](https://cognitiveaisystems.github.io/MIKASA-Robo/vla_environments/index.html)
- [Benchmarking](https://cognitiveaisystems.github.io/MIKASA-Robo/benchmarking.html)
- [Datasets](https://cognitiveaisystems.github.io/MIKASA-Robo/datasets.html)

## Citation

If you use MIKASA-Robo-VLA in your research, please cite:

```bibtex
@inproceedings{cherepanov2026memory,
  title     = {Memory, Benchmark \& Robots: A Benchmark for Solving Complex Tasks with Reinforcement Learning},
  author    = {Egor Cherepanov and Nikita Kachaev and Alexey Kovalev and Aleksandr I. Panov},
  booktitle = {The Fourteenth International Conference on Learning Representations},
  year      = {2026},
  url       = {https://openreview.net/forum?id=9cLPurIZMj}
}
```
