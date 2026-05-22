import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# If you experience HF cache lock issues in your environment:
os.environ.setdefault("HF_HOME", "/tmp/hf-home")

from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROOT = Path("data_mikasa_robo/data_lerobot/RememberColor3-VLA-v0")
REPO_ID = "RememberColor3-VLA-v0"
EPISODE_ID = 0  # change to the desired episode index
OUTPUT_DIR = Path(tempfile.mkdtemp(prefix="rlds_to_lerobot_preview_"))


def print_section(title: str) -> None:
    line = "=" * 88
    print(f"\n{line}\n{title}\n{line}")


def vec_to_str(x: np.ndarray | torch.Tensor, precision: int = 4) -> str:
    arr = np.asarray(x, dtype=np.float32)
    return np.array2string(arr, precision=precision, separator=", ", suppress_small=False)


def short_path_list(paths: list[Path]) -> str:
    return "\n".join(f"- {p}" for p in paths)


ds = LeRobotDataset(repo_id=REPO_ID, root=ROOT)

print_section("Run Context")
print(f"Repo ID            : {REPO_ID}")
print(f"Dataset root       : {ROOT.resolve()}")
print(f"HF_HOME            : {os.environ.get('HF_HOME')}")
print(f"Python executable  : {Path(os.sys.executable)}")
print(f"Preview output dir : {OUTPUT_DIR}")

print_section("Dataset Summary")
print(f"num_episodes       : {ds.num_episodes}")
print(f"num_frames         : {len(ds)}")
print(f"feature keys       : {list(ds.features.keys())}")
print(f"fps                : {ds.fps}")

if "observation.images.top" in ds.features:
    primary_cam_key = "observation.images.top"
elif "observation.images.front" in ds.features:
    primary_cam_key = "observation.images.front"
else:
    raise KeyError("No primary RGB camera key found (expected observation.images.top/front)")

if "observation.images.wrist" not in ds.features:
    raise KeyError("Expected observation.images.wrist in dataset features")

print(f"primary_cam_key    : {primary_cam_key}")

# Get global frame indices for the selected episode
ep_arr = np.array([int(x) for x in ds.hf_dataset["episode_index"]], dtype=np.int64)
idxs = np.where(ep_arr == EPISODE_ID)[0]
if len(idxs) == 0:
    raise ValueError(f"Episode {EPISODE_ID} not found")
print_section("Episode Summary")
print(f"episode_id         : {EPISODE_ID}")
print(f"num_frames         : {len(idxs)}")
print(f"global_idx range   : [{idxs[0]}, {idxs[-1]}]")


def chw_to_hwc01(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.ndim == 3 and x.shape[0] in (1, 3):
        x = np.transpose(x, (1, 2, 0))
    x = np.clip(x, 0.0, 1.0)
    return x


# Sample beginning / middle / end of the episode
sample_global_idxs = [idxs[0], idxs[len(idxs) // 2], idxs[-1]]

fig, axes = plt.subplots(2, 3, figsize=(12, 7))
rows = []
saved_images: list[Path] = []
for col, gidx in enumerate(sample_global_idxs):
    item = ds[int(gidx)]
    top = chw_to_hwc01(item[primary_cam_key])
    wrist = chw_to_hwc01(item["observation.images.wrist"])

    top_path = OUTPUT_DIR / f"episode_{EPISODE_ID:04d}_sample_{col}_top.png"
    wrist_path = OUTPUT_DIR / f"episode_{EPISODE_ID:04d}_sample_{col}_wrist.png"
    plt.imsave(top_path, top)
    plt.imsave(wrist_path, wrist)
    saved_images.extend([top_path, wrist_path])

    axes[0, col].imshow(top)
    axes[0, col].set_title(f"top | global={gidx} | ep_frame={int(item['frame_index'])}")
    axes[0, col].axis("off")

    axes[1, col].imshow(wrist)
    axes[1, col].set_title(f"wrist | t={float(item['timestamp']):.2f}s")
    axes[1, col].axis("off")

    rows.append(
        {
            "global_idx": int(gidx),
            "episode_idx": int(item["episode_index"]),
            "frame_index": int(item["frame_index"]),
            "timestamp": float(item["timestamp"]),
            "task": item["task"],
            "state": vec_to_str(item["observation.state"]),
            "action": vec_to_str(item["action"]),
            "top_mean": float(top.mean()),
            "wrist_mean": float(wrist.mean()),
        }
    )

plt.tight_layout()
grid_path = OUTPUT_DIR / f"episode_{EPISODE_ID:04d}_preview_grid.png"
fig.savefig(grid_path, dpi=150)
saved_images.append(grid_path)
plt.show()

df = pd.DataFrame(rows)
pd.set_option("display.max_colwidth", 240)
pd.set_option("display.width", 180)

print_section("Sample Frames (Overview)")
print(
    df[
        [
            "global_idx",
            "episode_idx",
            "frame_index",
            "timestamp",
            "top_mean",
            "wrist_mean",
            "task",
        ]
    ].to_string(index=False)
)

print_section("Sample Frames (State / Action)")
print(
    df[
        [
            "global_idx",
            "state",
            "action",
        ]
    ].to_string(index=False)
)

print_section("Saved Preview Files")
print(short_path_list(saved_images))
