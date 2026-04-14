# RLDS -> LeRobot v3 Converter

Конвертер для MIKASA-Robo, который переводит RLDS-датасеты из
`data_mikasa_robo/data_rlds/<task-id>/...` в формат LeRobotDataset v3 (Parquet + MP4) в
`data_mikasa_robo/data_lerobot/`.

## Вход и выход

- Вход: `data_mikasa_robo/data_rlds/<task-id>/<version>/` (например `1.0.0`) с TFDS/RLDS артефактами.
- Выход: `data_mikasa_robo/data_lerobot/<repo-id-template>` где `{task}` подставляется автоматически.

По умолчанию используется `--repo-id-template "{task}"`, поэтому структура будет:
- `data_mikasa_robo/data_lerobot/<task-id>/meta/...`
- `data_mikasa_robo/data_lerobot/<task-id>/data/...`
- `data_mikasa_robo/data_lerobot/<task-id>/videos/...`

Если ваш `lerobot` требует HF-формат repo_id с namespace, используйте:
- `--repo-id-template "your_hf_user/{task}"`

## Установка зависимостей (отдельно через uv)

```bash
cd /home/jovyan/echerepanov/REPOSITORIES/mikasa_vla/MIKASA-Robo
uv sync --project utils/convert_rlds_to_lerobot
```

Это отдельный uv-проект с собственным `pyproject.toml`, чтобы не смешивать зависимости
с основным окружением и RLDS-конвертером.

## Конвертация одной задачи

```bash
cd /home/jovyan/echerepanov/REPOSITORIES/mikasa_vla/MIKASA-Robo

uv run --project utils/convert_rlds_to_lerobot \
  python utils/convert_rlds_to_lerobot/convert_rlds_to_lerobot.py \
  --task RememberColor3-VLA-v0 \
  --overwrite-dest
```

## Конвертация всех задач из `data_mikasa_robo/data_rlds`

```bash
cd /home/jovyan/echerepanov/REPOSITORIES/mikasa_vla/MIKASA-Robo

uv run --project utils/convert_rlds_to_lerobot \
  python utils/convert_rlds_to_lerobot/convert_rlds_to_lerobot.py \
  --all \
  --overwrite-dest
```

## Полезные параметры

- `--version 1.0.0` : фиксировать версию RLDS директории.
- `--split train` : какой split читать из RLDS.
- `--fps 10` : fps для LeRobot видео.
- `--robot-type mikasa_robo` : метаданные robot_type.
- `--repo-id-template "{task}"` : путь назначения внутри `data_mikasa_robo/data_lerobot`.
- `--repo-id-template "avanturist322/{task}"` : сразу в HF-style namespace.
- `--no-videos` : писать изображения без mp4 кодирования.
- `--max-episodes N` : быстрый smoke-test на части данных.

## Проверка результата

```bash
task=RememberColor3-VLA-v0
test -f "data_mikasa_robo/data_lerobot/${task}/meta/info.json"
test -f "data_mikasa_robo/data_lerobot/${task}/meta/stats.json"
echo "OK: ${task}"
```

## Примечания

- Скрипт маппит поля RLDS так:
  - `steps.observation.image` -> `observation.images.top`
  - `steps.observation.wrist_image` -> `observation.images.wrist` (если есть)
  - `steps.observation.proprio` -> `observation.state`
  - `steps.action` -> `action`
- `language_instruction` используется как `task` при `save_episode(...)`.
- При наличии `data_mikasa_robo/data_rlds/<task>/<version>/metadata.json` копируется в
  `data_mikasa_robo/data_lerobot/<task>/source_rlds_metadata.json`.



## Проверить данные
```bash
uv add matplotlib ipykernel --project utils/convert_rlds_to_lerobot 
uv run --project utils/convert_rlds_to_lerobot   python utils/convert_rlds_to_lerobot/test.py 
```

sources:
https://docs.phospho.ai/learn/lerobot-dataset
https://huggingface.co/docs/lerobot/porting_datasets_v3
https://huggingface.co/docs/lerobot/lerobot-dataset-v3
https://github.com/huggingface/lerobot/issues/846
https://github.com/huggingface/lerobot/blob/main/src/lerobot/datasets/lerobot_dataset.py
