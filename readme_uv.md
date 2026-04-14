# MIKASA-Robo: установка и управление зависимостями через `uv`

Этот репозиторий переведён на `uv`-workflow:
- зависимости и метаданные проекта описаны в `pyproject.toml`
- зафиксированные версии зависимостей находятся в `uv.lock`
- `setup.py` оставлен только как compatibility shim

## 1. Установка `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Проверка:

```bash
uv --version
```

---

## 2. Быстрый старт (рекомендуемый путь)

```bash
git clone git@github.com:CognitiveAISystems/MIKASA-Robo.git
cd MIKASA-Robo

# Опционально: если на сервере ограничения на ~/.cache
export UV_CACHE_DIR=/tmp/uv-cache

# Создать/обновить .venv строго по lock-файлу
uv sync --frozen
```

`uv sync --frozen`:
- создаст `.venv`, если его нет
- установит зависимости в версиях из `uv.lock`
- установит текущий проект

---

## 3. Запуск команд

Без ручной активации окружения:

```bash
uv run python mikasa_robo_suite/vla/dataset_collectors/get_mikasa_robo_datasets.py --help
```

С активацией:

```bash
source .venv/bin/activate
python mikasa_robo_suite/vla/dataset_collectors/get_mikasa_robo_datasets.py --help
```

---

## 4. Обновление/изменение зависимостей

### Добавить зависимость
```bash
uv add <package>
```

### Удалить зависимость
```bash
uv remove <package>
```

### Перерезолвить lock после правок
```bash
uv lock
uv sync
```

### Обновить все зависимости в lock
```bash
uv lock --upgrade
uv sync
```

---

## 5. Полезные команды

```bash
# Проверить, что lock актуален
uv lock --check

# Показать дерево зависимостей
uv tree

# Синхронизировать уже активированное окружение (если не хотите .venv проекта)
uv sync --active
```

---

## 6. Важные замечания для этого проекта

1. Целевой runtime проекта ограничен в `pyproject.toml`:
   - `python >=3.9,<3.12`
   - lock-файл рассчитан на `Linux x86_64`

2. Если на сервере запрещена запись в `~/.cache`, используйте:
   ```bash
   export UV_CACHE_DIR=/tmp/uv-cache
   ```

3. Для повторяемых запусков в CI/на сервере используйте:
   ```bash
   uv sync --frozen
   ```
   Это гарантирует установку ровно по `uv.lock`.

---

## 7. Минимальный workflow для разработчика

```bash
# 1) подтянуть репозиторий
git pull

# 2) синхронизировать окружение
uv sync --frozen

# 3) запускать скрипты через uv run
uv run python <your_script>.py
```

