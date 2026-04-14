# Instruction by MIKASA-Robo authors

### Изолированный запуск из корня репозитория (рекомендуется)

```bash
# cd to repo root
cd /home/jovyan/echerepanov/REPOSITORIES/mikasa_vla/MIKASA-Robo

# 1) один раз создать/обновить отдельное окружение конвертера
uv sync --project utils/convert_npz_to_rlds/rlds_dataset_builder

# 2) запуск конвертации в изолированном окружении rlds_dataset_builder
uv run --project utils/convert_npz_to_rlds/rlds_dataset_builder \
  python utils/convert_npz_to_rlds/convert_npz_task_to_rlds.py \
  --task RememberShape3-VLA-v0 \
  --overwrite-dest
```

Проверка, что используется именно специальное окружение `rlds_dataset_builder`:
```bash
uv run --project utils/convert_npz_to_rlds/rlds_dataset_builder \
  python -c "import sys; print(sys.executable)"
# ожидаемый путь:
# .../utils/convert_npz_to_rlds/rlds_dataset_builder/.venv/bin/python
```

Важно:
1. Не запускай из корня `uv run python ...` без `--project`, иначе возьмется корневой проект.
2. Команды с `--project utils/convert_npz_to_rlds/rlds_dataset_builder` используют отдельное окружение конвертера и не ломают корневое `uv`-окружение.

### Конвертация данных (автоматически для любой задачи)
```bash
# cd to repo root

# Пример для одной задачи:
uv run --project utils/convert_npz_to_rlds/rlds_dataset_builder \
  python utils/convert_npz_to_rlds/convert_npz_task_to_rlds.py \
  --task RememberShape3-VLA-v0 \
  --overwrite-dest
```

Важно про параллельный запуск:
1. Теперь каждый запуск конвертера по умолчанию использует отдельный временный `TFDS data_dir`, поэтому несколько `npz -> rlds` процессов можно запускать параллельно без конфликта за `~/tensorflow_datasets/mikasa_dataset/1.0.0`.
2. Если нужен фиксированный каталог TFDS (например для дебага), укажи его явно:
```bash
uv run --project utils/convert_npz_to_rlds/rlds_dataset_builder \
  python utils/convert_npz_to_rlds/convert_npz_task_to_rlds.py \
  --task RememberShape3-VLA-v0 \
  --overwrite-dest \
  --tfds-data-dir /tmp/tfds_debug_remember_shape3
```

Что делает скрипт:
1. Передает имя задачи в `mikasa_dataset_dataset_builder.py` через `MIKASA_TASK_NAME`.
2. Запускает `tfds build --overwrite`.
3. Копирует только финальную версию `1.0.0` в `data_mikasa_robo/data_rlds/<task>/`.
4. Копирует исходный `metadata.json` в `data_mikasa_robo/data_rlds/<task>/1.0.0/metadata.json`.

Очистка старых артефактов от предыдущих конфликтов:
```bash
find data_mikasa_robo/data_rlds -type d -name '1.0.0.incomplete*' -print -exec rm -rf {} +
```

### Конвертация нескольких задач подряд
```bash
cd /home/jovyan/echerepanov/REPOSITORIES/mikasa_vla/MIKASA-Robo

for task in RememberColor3-VLA-v0 BatteriesCheckerEasy-3-VLA-v0; do
  uv run --project utils/convert_npz_to_rlds/rlds_dataset_builder \
    python utils/convert_npz_to_rlds/convert_npz_task_to_rlds.py \
    --task "${task}" \
    --overwrite-dest
done
```

### Конвертация всех директорий из `data_mikasa_robo/data_npz`
```bash
cd /home/jovyan/echerepanov/REPOSITORIES/mikasa_vla/MIKASA-Robo

for task_dir in data_mikasa_robo/data_npz/*; do
  [ -d "${task_dir}" ] || continue
  task="$(basename "${task_dir}")"
  [[ "${task}" == _* ]] && continue
  echo "Converting: ${task}"
  uv run --project utils/convert_npz_to_rlds/rlds_dataset_builder \
    python utils/convert_npz_to_rlds/convert_npz_task_to_rlds.py \
    --task "${task}" \
    --overwrite-dest
done
```

### Инкрементальная конвертация (только недоконвертированные задачи)
```bash
cd /home/jovyan/echerepanov/REPOSITORIES/mikasa_vla/MIKASA-Robo
set -euo pipefail

NPZ_ROOT="data_mikasa_robo/data_npz"
RLDS_ROOT="data_mikasa_robo/data_rlds"

for task_dir in "${NPZ_ROOT}"/*; do
  [ -d "${task_dir}" ] || continue
  task="$(basename "${task_dir}")"
  [[ "${task}" == _* ]] && continue

  out="${RLDS_ROOT}/${task}/1.0.0"

  # Если RLDS полностью собран, пропускаем.
  if [[ -f "${out}/metadata.json" \
     && -f "${out}/dataset_info.json" \
     && -f "${out}/features.json" ]] \
     && compgen -G "${out}/mikasa_dataset-train.tfrecord-*" > /dev/null; then
    # echo "[SKIP] ${task} (already converted)"
    continue
  fi

  # Если выхода нет или он неполный — пересобираем только эту задачу.
  echo "[CONVERT] ${task}"
  # uv run --project utils/convert_npz_to_rlds/rlds_dataset_builder \
  #   python utils/convert_npz_to_rlds/convert_npz_task_to_rlds.py \
  #     --task "${task}" \
  #     --overwrite-dest
done
```

for task in BunchOfColors3-Long-VLA-v0 BunchOfColors3-VLA-v0 BunchOfColors5-Long-VLA-v0 BunchOfColors5-VLA-v0 BunchOfColors7-Long-VLA-v0 BunchOfColors7-VLA-v0 ChainOfColors3-Long-VLA-v0 ChainOfColors3-VLA-v0; do uv run --project utils/convert_npz_to_rlds/rlds_dataset_builder \
    python utils/convert_npz_to_rlds/convert_npz_task_to_rlds.py \
    --task "${task}" \
    --overwrite-dest
done


for task in ChainOfColors5-Long-VLA-v0 ChainOfColors5-VLA-v0 ChainOfColors7-VLA-v0 RememberColor3-Long-VLA-v0 RememberColor5-Long-VLA-v0 RememberColor9-Long-VLA-v0; do CUDA_VISIBLE_DEVICES=1 uv run --project utils/convert_npz_to_rlds/rlds_dataset_builder \
    python utils/convert_npz_to_rlds/convert_npz_task_to_rlds.py \
    --task "${task}" \
    --overwrite-dest
done

for task in RememberShape3-Long-VLA-v0 RememberShape5-Long-VLA-v0 RememberShape9-Long-VLA-v0 RememberShapeAndColor3x2-Long-VLA-v0 RememberShapeAndColor3x3-Long-VLA-v0 RememberShapeAndColor5x3-Long-VLA-v0 SeqOfColors3-Long-VLA-v0; do CUDA_VISIBLE_DEVICES=2 uv run --project utils/convert_npz_to_rlds/rlds_dataset_builder \
    python utils/convert_npz_to_rlds/convert_npz_task_to_rlds.py \
    --task "${task}" \
    --overwrite-dest
done

for task in SeqOfColors3-VLA-v0 SeqOfColors5-Long-VLA-v0 SeqOfColors5-VLA-v0 SeqOfColors7-VLA-v0 ShellGameShuffleColorLampTouch-Long-VLA-v0 ShellGameShuffleTouch-Long-VLA-v0; do CUDA_VISIBLE_DEVICES=2 uv run --project utils/convert_npz_to_rlds/rlds_dataset_builder \
    python utils/convert_npz_to_rlds/convert_npz_task_to_rlds.py \
    --task "${task}" \
    --overwrite-dest
done
