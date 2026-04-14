#!/usr/bin/env bash
set -euo pipefail

# Resume interrupted MP dataset collection for selected envs.
# The collector now auto-detects resume seed from:
#   1) existing data_npz/_batched/<env>/train_data_*.npz
#   2) logs/**/<env>.log
# and continues without overwriting existing trajectories.
#
# After reaching NUM_TRAIN_DATA, it automatically:
#   - unbatches trajectories into data_npz/<env>/train_data_*.npz
#   - writes data_npz/<env>/metadata.json
#   - removes temporary _batched files for that env.
#
# Usage:
#   bash utils/resume_interrupted_mp_collection.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COLLECTOR="$ROOT_DIR/mikasa_robo_suite/vla/dataset_collectors/get_mikasa_robo_datasets_motion_planning.py"

UV_BIN="${UV_BIN:-uv}"
PY_BIN="${PY_BIN:-python}"
PATH_TO_SAVE_DATA="${PATH_TO_SAVE_DATA:-data_mikasa_robo}"
NUM_TRAIN_DATA="${NUM_TRAIN_DATA:-250}"
MAX_ATTEMPTS_MP="${MAX_ATTEMPTS_MP:-5000}"
START_SEED_MP="${START_SEED_MP:-0}"

ENVS=(
  "BatteriesCheckerHard-6-VLA-v0"
  # "GatherAndRecall9-VLA-v0"
  # "TimedTransferEasy-VLA-v0"
)

if [[ ! -f "$COLLECTOR" ]]; then
  echo "[ERROR] collector not found: $COLLECTOR" >&2
  exit 1
fi

cd "$ROOT_DIR"
for env_id in "${ENVS[@]}"; do
  echo "[RESUME] env=$env_id target=${NUM_TRAIN_DATA}"
  "$UV_BIN" run "$PY_BIN" "$COLLECTOR" \
    --env-id "$env_id" \
    --path-to-save-data "$PATH_TO_SAVE_DATA" \
    --num-train-data "$NUM_TRAIN_DATA" \
    --max-attempts "$MAX_ATTEMPTS_MP" \
    --seed "$START_SEED_MP"
done

echo "[DONE] Resume finished for all envs."
