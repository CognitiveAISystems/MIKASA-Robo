#!/usr/bin/env bash
set -euo pipefail

# Parallel NPZ dataset collection launcher for mixed PPO/MP env lists.
#
# Expected env list format (tab/space separated):
#   <env_id> <timeout> <enabled(TRUE/FALSE)> <method(PPO/MP)>
#
# Example:
#   CUDA_VISIBLE_DEVICES is assigned per-process automatically from GPU_LIST.
#   JOBS_PER_GPU=2 NUM_TRAIN_DATA=250 MAX_ATTEMPTS_MP=5000 \
#   bash run_scripts/run_parallel_npz_collection.sh envs.txt

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -ge 1 ]]; then
  ENV_FILE="$1"
else
  if [[ -f "$ROOT_DIR/envs.txt" ]]; then
    ENV_FILE="$ROOT_DIR/envs.txt"
  # elif [[ -f "$ROOT_DIR/emvs.txt" ]]; then
  #   # Backward-compatibility for old typo.
  #   ENV_FILE="$ROOT_DIR/emvs.txt"
  else
    echo "[ERROR] env list file not found. Pass path explicitly: $0 <envs.txt>" >&2
    exit 1
  fi
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "[ERROR] env file does not exist: $ENV_FILE" >&2
  exit 1
fi

ENV_FILE_ABS="$ENV_FILE"
if [[ "$ENV_FILE_ABS" != /* ]]; then
  ENV_FILE_ABS="$ROOT_DIR/$ENV_FILE"
fi

PATH_TO_SAVE_DATA="${PATH_TO_SAVE_DATA:-data_mikasa_robo}"
CKPT_DIR="${CKPT_DIR:-.}"
NUM_TRAIN_DATA="${NUM_TRAIN_DATA:-250}"
MAX_ATTEMPTS_MP="${MAX_ATTEMPTS_MP:-5000}"
START_SEED_MP="${START_SEED_MP:-0}"
GPU_LIST="${GPU_LIST:-0,1,2}"
JOBS_PER_GPU="${JOBS_PER_GPU:-4}"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-2}"
RESET_ENV_DATA="${RESET_ENV_DATA:-0}"
SKIP_IF_COMPLETE="${SKIP_IF_COMPLETE:-1}"
SHOW_LIVE_PROGRESS="${SHOW_LIVE_PROGRESS:-1}"
PROGRESS_REPORT_SEC="${PROGRESS_REPORT_SEC:-10}"
LOG_TAIL_LINES="${LOG_TAIL_LINES:-120}"
LOG_EPISODE_LENGTHS="${LOG_EPISODE_LENGTHS:-1}"

UV_BIN="${UV_BIN:-uv}"
PY_BIN="${PY_BIN:-python}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="${LOG_ROOT:-$ROOT_DIR/logs/dataset_collection_parallel/$RUN_TS}"
mkdir -p "$LOG_ROOT"

IFS=',' read -r -a GPUS <<<"$GPU_LIST"
if [[ ${#GPUS[@]} -eq 0 ]]; then
  echo "[ERROR] GPU_LIST is empty" >&2
  exit 1
fi

PPO_COLLECTOR="$ROOT_DIR/mikasa_robo_suite/vla/dataset_collectors/get_mikasa_robo_datasets.py"
MP_COLLECTOR="$ROOT_DIR/mikasa_robo_suite/vla/dataset_collectors/get_mikasa_robo_datasets_motion_planning.py"

if [[ ! -f "$PPO_COLLECTOR" || ! -f "$MP_COLLECTOR" ]]; then
  echo "[ERROR] collector scripts not found." >&2
  echo "  PPO: $PPO_COLLECTOR" >&2
  echo "  MP : $MP_COLLECTOR" >&2
  exit 1
fi

existing_unbatched_count() {
  local env_id="$1"
  local unbatched_dir="$ROOT_DIR/$PATH_TO_SAVE_DATA/data_npz/$env_id"
  if [[ ! -d "$unbatched_dir" ]]; then
    echo 0
    return
  fi
  find "$unbatched_dir" -maxdepth 1 -type f -name 'train_data_*.npz' | wc -l | tr -d ' '
}

mapfile -t JOB_LINES < <(
  awk '
    BEGIN { OFS="\t" }
    /^[[:space:]]*#/ { next }
    NF < 4 { next }
    toupper($3) != "TRUE" { next }
    {
      env=$1
      method=toupper($4)
      if (method=="PPO" || method=="MP") {
        print env, method
      }
    }
  ' "$ENV_FILE_ABS"
)

if [[ ${#JOB_LINES[@]} -eq 0 ]]; then
  echo "[ERROR] no enabled jobs found in $ENV_FILE_ABS" >&2
  exit 1
fi

declare -a ENVS=()
declare -a METHODS=()
for line in "${JOB_LINES[@]}"; do
  env_id="${line%%$'\t'*}"
  method="${line##*$'\t'}"
  ENVS+=("$env_id")
  METHODS+=("$method")
done

TOTAL_ENABLED="${#ENVS[@]}"

declare -a FILTERED_ENVS=()
declare -a FILTERED_METHODS=()
declare -a SKIPPED_ENVS=()
if [[ "$SKIP_IF_COMPLETE" == "1" && "$RESET_ENV_DATA" != "1" ]]; then
  for i in "${!ENVS[@]}"; do
    env_id="${ENVS[$i]}"
    method="${METHODS[$i]}"
    existing_count="$(existing_unbatched_count "$env_id")"
    if (( existing_count >= NUM_TRAIN_DATA )); then
      SKIPPED_ENVS+=("$env_id")
      echo "[SKIP] env=$env_id method=$method existing=${existing_count}/${NUM_TRAIN_DATA} (already complete)"
      continue
    fi
    FILTERED_ENVS+=("$env_id")
    FILTERED_METHODS+=("$method")
  done
  ENVS=("${FILTERED_ENVS[@]}")
  METHODS=("${FILTERED_METHODS[@]}")
fi

TOTAL="${#ENVS[@]}"
echo "[INFO] env file      : $ENV_FILE_ABS"
echo "[INFO] total jobs    : $TOTAL_ENABLED"
echo "[INFO] jobs to run   : $TOTAL"
echo "[INFO] gpu list      : ${GPUS[*]}"
echo "[INFO] jobs / gpu    : $JOBS_PER_GPU"
echo "[INFO] npz root      : $PATH_TO_SAVE_DATA"
echo "[INFO] logs dir      : $LOG_ROOT"
echo "[INFO] reset per env : $RESET_ENV_DATA"
echo "[INFO] skip complete : $SKIP_IF_COMPLETE"
echo "[INFO] live progress : $SHOW_LIVE_PROGRESS (every ${PROGRESS_REPORT_SEC}s)"
echo "[INFO] ep length logs : $LOG_EPISODE_LENGTHS (MIKASA_LOG_EPISODE_LENGTHS)"
echo

# Runtime state
declare -a RUN_PIDS=()
declare -a RUN_ENVS=()
declare -a RUN_METHODS=()
declare -a RUN_GPUS=()
declare -a RUN_LOGS=()

declare -a DONE_ENVS=()
declare -a FAIL_ENVS=()

next_job_idx=0
mp_seed_counter=0
last_progress_epoch=0

count_running_on_gpu() {
  local g="$1"
  local c=0
  local i
  for i in "${!RUN_PIDS[@]}"; do
    if [[ "${RUN_GPUS[$i]}" == "$g" ]]; then
      c=$((c + 1))
    fi
  done
  echo "$c"
}

reap_finished() {
  local i=0
  while [[ $i -lt ${#RUN_PIDS[@]} ]]; do
    local pid="${RUN_PIDS[$i]}"
    if kill -0 "$pid" 2>/dev/null; then
      i=$((i + 1))
      continue
    fi

    local env_id="${RUN_ENVS[$i]}"
    local method="${RUN_METHODS[$i]}"
    local gpu_id="${RUN_GPUS[$i]}"
    local log_path="${RUN_LOGS[$i]}"

    local code=0
    wait "$pid" || code=$?
    if [[ $code -eq 0 ]]; then
      DONE_ENVS+=("$env_id")
      echo "[DONE] env=$env_id method=$method gpu=$gpu_id log=$log_path"
    else
      FAIL_ENVS+=("$env_id")
      echo "[FAIL] env=$env_id method=$method gpu=$gpu_id exit=$code log=$log_path"
    fi

    unset 'RUN_PIDS[i]' 'RUN_ENVS[i]' 'RUN_METHODS[i]' 'RUN_GPUS[i]' 'RUN_LOGS[i]'
    RUN_PIDS=("${RUN_PIDS[@]}")
    RUN_ENVS=("${RUN_ENVS[@]}")
    RUN_METHODS=("${RUN_METHODS[@]}")
    RUN_GPUS=("${RUN_GPUS[@]}")
    RUN_LOGS=("${RUN_LOGS[@]}")
  done
}

extract_last_progress_line() {
  local log_path="$1"
  if [[ ! -f "$log_path" ]]; then
    echo "<no log yet>"
    return
  fi

  local progress_line
  progress_line="$(
    tail -n "$LOG_TAIL_LINES" "$log_path" 2>/dev/null \
      | tr '\r' '\n' \
      | sed '/^[[:space:]]*$/d' \
      | tail -n 1
  )"

  if [[ -z "$progress_line" ]]; then
    echo "<waiting for output>"
  else
    echo "$progress_line"
  fi
}

estimate_collected_count() {
  local env_id="$1"
  local method="$2"
  local progress_line="$3"

  if [[ "$progress_line" =~ ([0-9]+)/([0-9]+) ]]; then
    local parsed_done="${BASH_REMATCH[1]}"
    local parsed_total="${BASH_REMATCH[2]}"
    if [[ "$parsed_total" =~ ^[0-9]+$ && "$parsed_total" -gt 0 ]]; then
      if (( parsed_total == NUM_TRAIN_DATA )); then
        echo "$parsed_done"
        return
      fi
    fi
  fi

  local unbatched_dir="$ROOT_DIR/$PATH_TO_SAVE_DATA/data_npz/$env_id"
  local unbatched_count=0
  if [[ -d "$unbatched_dir" ]]; then
    unbatched_count="$(find "$unbatched_dir" -maxdepth 1 -type f -name 'train_data_*.npz' | wc -l | tr -d ' ')"
  fi

  local total="$unbatched_count"
  if [[ "$method" == "PPO" ]]; then
    local batched_dir="$ROOT_DIR/$PATH_TO_SAVE_DATA/data_npz/_batched/$env_id"
    local batched_count=0
    if [[ -d "$batched_dir" ]]; then
      batched_count="$(find "$batched_dir" -maxdepth 1 -type f -name 'train_data_batch_*.npz' | wc -l | tr -d ' ')"
    fi
    # Approximate in-flight PPO progress: 1 batched file ~ up to 10 trajectories.
    total=$((unbatched_count + batched_count * 10))
  fi

  if (( total > NUM_TRAIN_DATA )); then
    total="$NUM_TRAIN_DATA"
  fi
  echo "$total"
}

render_progress_table_with_rich() {
  local queued="$1"
  local done="$2"
  local failed="$3"
  local rows_file
  rows_file="$(mktemp)"
  cat >"$rows_file"

  "$PY_BIN" - "$queued" "$done" "$failed" "$rows_file" <<'PY'
import sys
from datetime import datetime

queued, done, failed, rows_file = sys.argv[1:5]
rows = []
with open(rows_file, "r", encoding="utf-8", errors="replace") as f:
    for raw_line in f:
        line = raw_line.rstrip("\n")
        if not line:
            continue
        parts = line.split("\t", 6)
        if len(parts) < 7:
            parts = parts + [""] * (7 - len(parts))
        rows.append(parts[:7])

try:
    from rich.console import Console
    from rich.progress_bar import ProgressBar
    from rich.table import Table
except Exception:
    print(
        f"[PROGRESS] time={datetime.now().strftime('%H:%M:%S')} "
        f"running={len(rows)} queued={queued} done={done} failed={failed}"
    )
    for env_id, method, gpu_id, pid, collected, target, progress in rows:
        print(
            f"  [RUN] env={env_id} method={method} gpu={gpu_id} pid={pid} "
            f"traj={collected}/{target} | {progress}"
        )
    raise SystemExit(0)

console = Console()
table = Table(
    title=f"Dataset Collection Progress ({datetime.now().strftime('%H:%M:%S')})",
    expand=True,
)
table.add_column("Env", overflow="fold")
table.add_column("Method", justify="center", width=7)
table.add_column("GPU", justify="center", width=5)
table.add_column("PID", justify="right", width=8)
table.add_column("Traj", justify="right", width=11)
table.add_column("Progress", width=30)
table.add_column("Log", overflow="fold")

for env_id, method, gpu_id, pid, collected, target, progress in rows:
    try:
        c = int(collected)
    except Exception:
        c = 0
    try:
        t = int(target)
    except Exception:
        t = 0
    if t <= 0:
        t = 1
    if c < 0:
        c = 0
    if c > t:
        c = t

    table.add_row(
        env_id,
        method,
        gpu_id,
        pid,
        f"{c}/{t}",
        ProgressBar(total=t, completed=c, width=24),
        progress,
    )

console.print(
    f"[bold cyan]Jobs[/bold cyan] running={len(rows)} queued={queued} done={done} failed={failed}"
)
console.print(table)
PY

  rm -f "$rows_file"
}

print_running_progress() {
  if [[ "$SHOW_LIVE_PROGRESS" != "1" ]]; then
    return
  fi
  if [[ ${#RUN_PIDS[@]} -eq 0 ]]; then
    return
  fi

  local queued=$((TOTAL - next_job_idx))
  {
    local i
    for i in "${!RUN_PIDS[@]}"; do
      local pid="${RUN_PIDS[$i]}"
      local env_id="${RUN_ENVS[$i]}"
      local method="${RUN_METHODS[$i]}"
      local gpu_id="${RUN_GPUS[$i]}"
      local log_path="${RUN_LOGS[$i]}"
      local progress_line
      progress_line="$(extract_last_progress_line "$log_path")"
      local collected_count
      collected_count="$(estimate_collected_count "$env_id" "$method" "$progress_line")"
      progress_line="${progress_line//$'\t'/ }"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$env_id" "$method" "$gpu_id" "$pid" "$collected_count" "$NUM_TRAIN_DATA" "$progress_line"
    done
  } | render_progress_table_with_rich "$queued" "${#DONE_ENVS[@]}" "${#FAIL_ENVS[@]}"
}

launch_one() {
  local env_id="$1"
  local method="$2"
  local gpu_id="$3"

  local env_save_dir="$ROOT_DIR/$PATH_TO_SAVE_DATA/data_npz/$env_id"
  local env_batched_dir="$ROOT_DIR/$PATH_TO_SAVE_DATA/data_npz/_batched/$env_id"
  if [[ "$RESET_ENV_DATA" == "1" ]]; then
    rm -rf "$env_save_dir" "$env_batched_dir"
  fi

  local safe_env
  safe_env="$(echo "$env_id" | tr '/:' '__')"
  local log_path="$LOG_ROOT/${safe_env}.log"

  local -a cmd
  if [[ "$method" == "PPO" ]]; then
    cmd=(
      "$UV_BIN" run "$PY_BIN" "$PPO_COLLECTOR"
      --env-id="$env_id"
      --path-to-save-data="$PATH_TO_SAVE_DATA"
      --ckpt-dir="$CKPT_DIR"
      --num-train-data="$NUM_TRAIN_DATA"
    )
  else
    local seed=$((START_SEED_MP + mp_seed_counter))
    mp_seed_counter=$((mp_seed_counter + 1))
    cmd=(
      "$UV_BIN" run "$PY_BIN" "$MP_COLLECTOR"
      --env-id "$env_id"
      --path-to-save-data "$PATH_TO_SAVE_DATA"
      --num-train-data "$NUM_TRAIN_DATA"
      --max-attempts "$MAX_ATTEMPTS_MP"
      --seed "$seed"
    )
  fi

  (
    cd "$ROOT_DIR"
    export CUDA_VISIBLE_DEVICES="$gpu_id"
    export PYTHONUNBUFFERED=1
    export MIKASA_LOG_EPISODE_LENGTHS="$LOG_EPISODE_LENGTHS"
    "${cmd[@]}"
  ) >"$log_path" 2>&1 &

  local pid=$!
  RUN_PIDS+=("$pid")
  RUN_ENVS+=("$env_id")
  RUN_METHODS+=("$method")
  RUN_GPUS+=("$gpu_id")
  RUN_LOGS+=("$log_path")

  echo "[START] env=$env_id method=$method gpu=$gpu_id pid=$pid log=$log_path"
}

while [[ $next_job_idx -lt $TOTAL || ${#RUN_PIDS[@]} -gt 0 ]]; do
  reap_finished

  # Fill free GPU slots.
  while [[ $next_job_idx -lt $TOTAL ]]; do
    picked_gpu=""
    for g in "${GPUS[@]}"; do
      running_on_g=$(count_running_on_gpu "$g")
      if [[ $running_on_g -lt $JOBS_PER_GPU ]]; then
        picked_gpu="$g"
        break
      fi
    done

    if [[ -z "$picked_gpu" ]]; then
      break
    fi

    env_id="${ENVS[$next_job_idx]}"
    method="${METHODS[$next_job_idx]}"
    launch_one "$env_id" "$method" "$picked_gpu"
    next_job_idx=$((next_job_idx + 1))
  done

  if [[ "$SHOW_LIVE_PROGRESS" == "1" && ${#RUN_PIDS[@]} -gt 0 ]]; then
    now_epoch="$(date +%s)"
    if (( now_epoch - last_progress_epoch >= PROGRESS_REPORT_SEC )); then
      print_running_progress
      last_progress_epoch="$now_epoch"
    fi
  fi

  if [[ ${#RUN_PIDS[@]} -gt 0 ]]; then
    sleep "$CHECK_INTERVAL_SEC"
  fi
done

echo
echo "========== Summary =========="
echo "Total jobs enabled : $TOTAL_ENABLED"
echo "Jobs to run        : $TOTAL"
echo "Skipped complete   : ${#SKIPPED_ENVS[@]}"
echo "Completed  : ${#DONE_ENVS[@]}"
echo "Failed     : ${#FAIL_ENVS[@]}"
if [[ ${#SKIPPED_ENVS[@]} -gt 0 ]]; then
  printf 'Skipped envs:\n'
  for e in "${SKIPPED_ENVS[@]}"; do
    echo "  - $e"
  done
fi
if [[ ${#FAIL_ENVS[@]} -gt 0 ]]; then
  printf 'Failed envs:\n'
  for e in "${FAIL_ENVS[@]}"; do
    echo "  - $e"
  done
fi
echo "Logs: $LOG_ROOT"
