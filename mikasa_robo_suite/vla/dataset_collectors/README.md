# NPZ Dataset Collection (MIKASA-Robo-90 VLA)

This guide documents how to collect source `.npz` trajectories used in the MIKASA-Robo-90 VLA pipeline.

If you are looking for conversion steps:
- NPZ -> RLDS: [`utils/convert_npz_to_rlds/README.md`](../../../utils/convert_npz_to_rlds/README.md)
- RLDS -> LeRobot v3: [`utils/convert_rlds_to_lerobot/README.md`](../../../utils/convert_rlds_to_lerobot/README.md)

---

## 1) Choose the collection method per task

The source of truth is [`mikasa_robo_vla_envs.csv`](../../../mikasa_robo_vla_envs.csv):
- use `Data Source = PPO` for oracle-policy collection,
- use `Data Source = MP` for motion-planning collection.

---

## 2) Single-task collection: PPO-oracle tasks

Use [`get_mikasa_robo_datasets.py`](./get_mikasa_robo_datasets.py):

```bash
uv run python mikasa_robo_suite/vla/dataset_collectors/get_mikasa_robo_datasets.py \
  --env-id RememberColor3-VLA-v0 \
  --path-to-save-data data_mikasa_robo \
  --ckpt-dir . \
  --num-train-data 250
```

Required checkpoint layout:
- the collector scans `--ckpt-dir` for oracle checkpoints under paths like `oracle_checkpoints/**/final_success_ckpt.pt`,
- `--env-id` must have a matching checkpoint key; otherwise the script raises an error.

CLI args:
- `--env-id` (str): task ID.
- `--path-to-save-data` (str, default `data_mikasa_robo`): output root.
- `--ckpt-dir` (str, default `.`): root directory with PPO oracle checkpoints.
- `--num-train-data` (int, default `250`): number of successful trajectories to store.

---

## 3) Single-task collection: motion-planning tasks

Use [`get_mikasa_robo_datasets_motion_planning.py`](./get_mikasa_robo_datasets_motion_planning.py):

```bash
uv run python mikasa_robo_suite/vla/dataset_collectors/get_mikasa_robo_datasets_motion_planning.py \
  --env-id TraceShapeHard-VLA-v0 \
  --path-to-save-data data_mikasa_robo \
  --num-train-data 250 \
  --max-attempts 5000 \
  --seed 0
```

Pipeline behavior:
1. motion planner generates raw trajectories,
2. ManiSkill replay converts trajectories into `pd_ee_delta_pose`,
3. successful replay rollouts are exported as per-episode `.npz`.

CLI args:
- `--env-id` (str): task ID.
- `--path-to-save-data` (str, default `data_mikasa_robo`): output root.
- `--num-train-data` (int, default `250`): number of successful trajectories to store.
- `--max-attempts` (int, default `5000`): max planning attempts budget.
- `--seed` (int, default `0`): start seed; attempt `i` uses `seed + i`.

Notes:
- Several legacy conversion flags are still accepted for backward CLI compatibility, but current conversion is integrated in the collector pipeline.
- The script supports resume behavior and avoids overwriting existing episodes when possible.

---

## 4) Parallel mixed PPO+MP collection (recommended for full benchmark)

Use launcher: [`utils/run_parallel_npz_collection.sh`](../../../utils/run_parallel_npz_collection.sh)

Expected env list format:
`<env_id> <max_length> <enabled(TRUE/FALSE)> <method(PPO/MP)>`

Generate list from manifest:

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

Run:

```bash
GPU_LIST=0,1,2 JOBS_PER_GPU=2 NUM_TRAIN_DATA=250 MAX_ATTEMPTS_MP=5000 \
bash utils/run_parallel_npz_collection.sh envs.txt
```

Useful launcher env vars:
- `PATH_TO_SAVE_DATA` (default `data_mikasa_robo`)
- `CKPT_DIR` (default `.`)
- `NUM_TRAIN_DATA` (default `250`)
- `MAX_ATTEMPTS_MP` (default `5000`)
- `START_SEED_MP` (default `0`)
- `SKIP_IF_COMPLETE` (default `1`)
- `RESET_ENV_DATA` (default `0`)
- `GPU_LIST` (default `0,1,2`)
- `JOBS_PER_GPU` (default `4`)

---

## 5) Resume interrupted MP collection

Use helper script:

```bash
bash utils/resume_interrupted_mp_collection.sh
```

Before running, edit `ENVS=(...)` in [`utils/resume_interrupted_mp_collection.sh`](../../../utils/resume_interrupted_mp_collection.sh) with tasks you want to resume.

The resume script calls the MP collector and continues from existing batched data/log state.

---

## 6) Output layout

Default root: `data_mikasa_robo/`

During collection:
- temporary batched files: `data_mikasa_robo/data_npz/_batched/<env_id>/train_data_*.npz`

Final per-episode output:
- `data_mikasa_robo/data_npz/<env_id>/train_data_0.npz`
- `data_mikasa_robo/data_npz/<env_id>/train_data_1.npz`
- ...
- `data_mikasa_robo/data_npz/<env_id>/metadata.json`

When finalization completes, temporary batched files are cleaned up automatically.

---

## 7) Validation checklist

For each collected task:
1. confirm exactly `NUM_TRAIN_DATA` files match `train_data_*.npz`,
2. inspect `metadata.json` for episode length and seed statistics,
3. run a quick conversion dry-run to RLDS on one task before launching full conversion.
