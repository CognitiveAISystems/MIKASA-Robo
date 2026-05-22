# RLDS to LeRobotDataset v3 Converter

This utility converts MIKASA-Robo-VLA RLDS / TFDS task datasets into
LeRobotDataset v3 datasets. It is the second local export step:

```text
data_mikasa_robo/data_rlds/<task>/<version>/  ->  data_mikasa_robo/data_lerobot/<repo-id>/
```

Use it after the NPZ-to-RLDS converter or on compatible RLDS task directories
downloaded into the expected local layout.

## Input and output

The converter reads one task or every task directory under
`data_mikasa_robo/data_rlds`. For one task, `--task` accepts either a Gymnasium
env ID or the normalized dataset folder name:

```text
RememberColor3-VLA-v0
remember_color_3_vla_v0
```

Expected RLDS input layout:

```text
data_mikasa_robo/
  data_rlds/
    remember_color_3_vla_v0/
      1.0.0/
        dataset_info.json
        features.json
        mikasa_dataset-train.tfrecord-*
        metadata.json
        ...
```

By default, the latest semantic-version RLDS subdirectory is selected and the
output repo ID template is `{task}`. The resulting layout is:

```text
data_mikasa_robo/
  data_lerobot/
    remember_color_3_vla_v0/
      data/
      meta/
      videos/
      source_rlds_metadata.json
```

If the installed `lerobot` version requires a namespace in `repo_id`, the
converter falls back from `{task}` to `local/{task}`. Set an explicit
namespace with `--repo-id-template "your_hf_user/{task}"` when needed.

## Environment setup

Run from the MIKASA-Robo repository root with the isolated converter project:

```bash
uv sync --project utils/convert_rlds_to_lerobot
```

Keep `--project utils/convert_rlds_to_lerobot` on converter commands so they
use the LeRobot, TensorFlow, and TFDS dependencies pinned for this utility.

## Convert one task

```bash
uv run --project utils/convert_rlds_to_lerobot \
  python utils/convert_rlds_to_lerobot/convert_rlds_to_lerobot.py \
  --task RememberColor3-VLA-v0 \
  --overwrite-dest
```

The converter reads RLDS episodes from the selected split, infers LeRobot
features from the first step, appends frames episode by episode, finalizes the
LeRobot dataset, and copies RLDS `metadata.json` to
`source_rlds_metadata.json` when it is present.

## Smoke test

Use a small partial conversion before exporting a full task:

```bash
uv run --project utils/convert_rlds_to_lerobot \
  python utils/convert_rlds_to_lerobot/convert_rlds_to_lerobot.py \
  --task RememberColor3-VLA-v0 \
  --max-episodes 2 \
  --overwrite-dest
```

## Convert all RLDS tasks

```bash
uv run --project utils/convert_rlds_to_lerobot \
  python utils/convert_rlds_to_lerobot/convert_rlds_to_lerobot.py \
  --all \
  --overwrite-dest
```

For incremental conversion, replace `--overwrite-dest` with
`--skip-existing`.

## Field mapping

The converter writes the following LeRobot features:

| RLDS step field | LeRobot field |
|---|---|
| `steps.observation.image` | `observation.images.top` |
| `steps.observation.wrist_image` | `observation.images.wrist` when present |
| `steps.observation.proprio` | `observation.state` |
| `steps.action` | `action` |
| `steps.language_instruction` | LeRobot frame `task` text |

Images are converted to `uint8` HWC arrays before they are written. By
default LeRobot stores image streams as videos; pass `--no-videos` when you
need frame storage instead.

## Useful options

| Option | Purpose |
|---|---|
| `--task TASK` | Convert one task env ID or normalized dataset folder. Mutually exclusive with `--all`. |
| `--all` | Convert every RLDS task directory found under `--rlds-root`. |
| `--version VERSION` | Force one RLDS version directory, for example `1.0.0`. Default: latest semver directory. |
| `--split SPLIT` | RLDS split to read. Default: `train`. |
| `--repo-id-template TEMPLATE` | Output LeRobot repo ID template. It must contain `{task}`. |
| `--fps FPS` | FPS metadata and video rate for the LeRobot dataset. Default: `10`. |
| `--robot-type TYPE` | LeRobot `robot_type` metadata. Default: `mikasa_robo`. |
| `--max-episodes N` | Limit converted episodes per task for a smoke test. |
| `--overwrite-dest` | Replace an existing LeRobot output directory. |
| `--skip-existing` | Skip existing outputs during incremental runs. |
| `--no-videos` | Store images without MP4 encoding when supported by the installed LeRobot version. |

## Verification

Check the basic LeRobot metadata after conversion:

```bash
task=remember_color_3_vla_v0
out="data_mikasa_robo/data_lerobot/${task}"

test -f "${out}/meta/info.json"
test -f "${out}/meta/stats.json"
echo "OK: ${task}"
```

To inspect sample frames, states, and actions with the included preview script:

```bash
uv run --project utils/convert_rlds_to_lerobot \
  python utils/convert_rlds_to_lerobot/test.py
```

The preview script currently targets the example RememberColor LeRobot output
path in `data_mikasa_robo/data_lerobot`; edit its `ROOT`, `REPO_ID`, or
`EPISODE_ID` constants when inspecting another dataset.

## Notes

- `--overwrite-dest` and `--skip-existing` are mutually exclusive.
- The converter expects episodic RLDS data readable through
  `tensorflow_datasets.builder_from_directory`.
- Dataset semantics, the public Hugging Face release, and the complete
  NPZ -> RLDS -> LeRobot export workflow are documented in
  `docs/source/datasets.rst`.
