### 1. We have updated all 32 MIKASA-Robo tasks to make it better for VLA training
#### Что конкретно меняем?
1. Исходные данные содержали много шагов с последней фазы задачи, где робот долгое время удерживает клешню над объектом. Чтобы не обучаться на таких бесполезных данных, всюду мы **уменьшаем длительности эпизодов**
2. За счет использования no-op wrappers, мы **запрещаем роботам совершать действия до наступления фазы манипуляции**, когда это действительно нужно
3. **Добавляем рандомизацию по memory horizons**. Например, если ранее в `RememberColor3-v0` первая и вторая фазы длились по 5 шагов ровно, то теперь они могут длиться 1, 2, 3, 4, 5 шагов (выбирается случайно при инициализации эпизода)
4. **Увеличиваем расстояние между объектами**, чтобы намерение VLA было более однозначно, а данные менее похожи при разных objective
5. При обучении oracle для сбора данных **заставляем робота двигаться плавнее** за счет обновленной функции вознаграждения, чтобы имитировать teleoperation, но с небольшой стохастичностью

Дополнительно уже на этапе сбора данных:
6. **Меняем proprio state в данных**: теперь это будет xyz(3) + rpy(3) + gripper(1)
7. **Меняем action в данных**: теперь это `pd_ee_delta_pose`
8. **Добавляем  failure recover** в данные


Что в обновлении?
1. Добавил документацию для задач



#### Какая теперь логика?
**Было:** (старая MIKASA-Robo, заточенная под RL)
mikasa_robo_suite/
    ├── dataset_collectors/
    ├── memory_envs/
    └── utils/


**Стало:** (новая MIKASA-Robo, поддерживающая VLA и RL обучение. Здесь rl/ - старая MIKASA-Robo)
mikasa_robo_suite/
    ├── rl/
        ├── dataset_collectors/
        ├── memory_envs/
        └── utils/
    ├── vla/
        ├── dataset_collectors/
        ├── memory_envs/
        └── utils/
            └── motion_planning/ # тут скрипты для сбора траекторий на тяжелых задачах


> **TODO:** Envs to update 
# Данные собраны через RL
1. ✅ `ShellGameTouch-VLA-v0`
    - увеличил кружки
    - сократил эпизод 60 -> 30
    - сделал движения плавнее
    - добавил рандомизацию dT
2. ✅ `ShellGamePush-VLA-v0`
    - увеличил кружки
    - сократил эпизод 60 -> 30
    - сделал движения плавнее
    - добавил рандомизацию dT
3. ✅ `ShellGamePick-VLA-v0`
    - увеличил кружки
    - сократил эпизод 60 -> 30
    - сделал движения плавнее
    - добавил рандомизацию dT

4. ✅ `InterceptSlow-VLA-v0`
    - 90 -> 60
5. ✅ `InterceptMedium-VLA-v0`
    - 90 -> 60
6. ✅ `InterceptFast-VLA-v0`
    - 90 -> 60

7. ✅ `InterceptGrabSlow-VLA-v0`
    - 90 -> 60
8. ✅ `InterceptGrabMedium-VLA-v0`
    - 90 -> 60
9. ✅ `InterceptGrabFast-VLA-v0`
    - 90 -> 60

10. ✅ `RotateLenientPos-VLA-v0`
    - движения плавнее
    - 90 -> 60
11. ✅ `RotateLenientPosNeg-VLA-v0`
    - движения плавнее
    - 90 -> 60
12. ✅ `RotateStrictPos-VLA-v0`
    - движения плавнее
13. ✅ `RotateStrictPosNeg-VLA-v0`
    - движения плавнее

14. ✅ `TakeItBack-VLA-v0`  # движения трясущиеся
    - сократил длину эпизода 180 -> 60
    - сделал более плавное движение

15. ✅ `RememberColor3-VLA-v0`
    - эпизод 60 -> 25 шагов
    - более плавные движения
    - кубики дальше друг от друга
    - рандомизация интервалов
16. ✅ `RememberColor5-VLA-v0`
    - эпизод 60 -> 25 шагов
    - более плавные движения
    - рандомизация интервалов
17. ✅ `RememberColor9-VLA-v0`
    - эпизод 60 -> 25 шагов
    - более плавные движения
    - рандомизация интервалов

18. ✅ `RememberShape3-VLA-v0`
    - эпизод 60 -> 25 шагов
    - более плавные движения
    - рандомизация интервалов
19. ✅ `RememberShape5-VLA-v0`
    - эпизод 60 -> 25 шагов
    - более плавные движения
    - рандомизация интервалов
20. ✅ `RememberShape9-VLA-v0`
    - эпизод 60 -> 25 шагов
    - более плавные движения
    - рандомизация интервалов

21. ✅ `RememberShapeAndColor3x2-VLA-v0`
    - 60 -> 25
    - движения плавнее
    - рандомизация dT
22. ✅ `RememberShapeAndColor3x3-VLA-v0`
    - 60 -> 25
    - движения плавнее
    - рандомизация dT
23. ✅ `RememberShapeAndColor5x3-VLA-v0`
    - 60 -> 25
    - движения плавнее
    - рандомизация dT

24. ✅ `BunchOfColors3-VLA-v0`
    - 120 -> 100
    - добавил рандомизацию dT
    - сделал движения плавнее
25. ✅ `BunchOfColors5-VLA-v0`
    - 120 -> 100
    - добавил рандомизацию dT
    - сделал движения плавнее
26. ✅ `BunchOfColors7-VLA-v0`
    - 120 -> 100
    - добавил рандомизацию dT
    - сделал движения плавнее

27. ✅ `SeqOfColors3-VLA-v0`
    - 120 -> 100
    - добавил рандомизацию dT
    - сделал движения плавнее
28. ✅ `SeqOfColors5-VLA-v0`
    - 120 -> 100
    - добавил рандомизацию dT
    - сделал движения плавнее
29. ✅ `SeqOfColors7-VLA-v0`
    - 120 -> 100
    - добавил рандомизацию dT
    - сделал движения плавнее

30. ✅ `ChainOfColors3-VLA-v0`
    - 120 -> 100
    - добавил рандомизацию dT
    - сделал движения плавнее
31. ✅ `ChainOfColors5-VLA-v0`
    - 120 -> 100
    - добавил рандомизацию dT
    - сделал движения плавнее
32. ✅ `ChainOfColors7-VLA-v0`
    - 120 -> 100
    - добавил рандомизацию dT
    - сделал движения плавнее

33. ✅ `ShellGameShuffleTouch-VLA-v0`
    - Исходный shellgame, где кружки крутятся

34. ✅ `ShellGameShuffleColorLampTouch`
    - shell game shufle touch, но используется три шарика и нужно выбрать шарик нужного цвета

35. ✅ `ShellGameColorLampTouch`
    - shell game touch, но используется три шарика и нужно выбрать шарик нужного цвета

36. ✅ `FindImposterColor3-VLA-v0`
37. ✅ `FindImposterColor5-VLA-v0`
38. ✅ `FindImposterColor9-VLA-v0`
39. ✅ `FindImposterShape3-VLA-v0`
40. ✅ `FindImposterShape5-VLA-v0`
41. ✅ `FindImposterShape9-VLA-v0`
42. ✅ `FindImposterShapeAndColor3x2-VLA-v0`
43. ✅ `FindImposterShapeAndColor3x3-VLA-v0`
44. ✅ `FindImposterShapeAndColor5x3-VLA-v0`


# Данные собраны через motion planning
45. ✅ `BatteriesCheckerEasy-3-VLA-v0`
46. ✅ `BatteriesCheckerEasy-6-VLA-v0`

47. ✅ `BatteriesCheckerHard-3-VLA-v0`

## работает неплохо на небольшом числе батареек, на большем - не очень, лучше использовать их для проверки
python mikasa_robo_suite/vla/utils/motion_planning/motion_planning_batteries_checker_easy.py --env-id BatteriesCheckerEasy-3-VLA-v0
python mikasa_robo_suite/vla/utils/motion_planning/motion_planning_batteries_checker_hard.py --env-id BatteriesCheckerHard-3-VLA-v0


48. ✅ `BlinkCountButtonPressEasy-VLA-v0`
49. ✅ `BlinkCountButtonPressMedium-VLA-v0`
50. ✅ `BlinkCountButtonPressHard-VLA-v0`
python mikasa_robo_suite/vla/utils/motion_planning/motion_planning_blink_count_button_press.py --env-id BlinkCountButtonPressEasy-VLA-v0
python mikasa_robo_suite/vla/utils/motion_planning/motion_planning_blink_count_button_press.py --env-id BlinkCountButtonPressMedium-VLA-v0
python mikasa_robo_suite/vla/utils/motion_planning/motion_planning_blink_count_button_press.py --env-id BlinkCountButtonPressHard-VLA-v0



uv run mikasa_robo_suite/vla/dataset_collectors/get_dataset_collectors_ckpt.py --env_id BunchOfColors3-VLA-v0




> **TODO:** Что мне нужно сделать?
1. ⚪️ Обновить все среды
2. ⚪️ Откалибровать функции вознаграждения
3. ⚪️ Обучить oracle checkpoints
3.1. ⚪️ Документацию обновить
3.2. ⚪️ Добавить враппер, убирающий reward_dict_info
4. ⚪️ Собрать .npz даные
5. ⚪️ Сконвертировать данные в RLDS
6. ⚪️ Выложить на huggingface




> **TODO New Tasks**
0. ✅ Shufflegame
1. ✅ ShellGameShuffle with multiple balls (ShellGameShuffleColorLampTouch)
2. Battaries Checker
    0. На столе находится белая лампочка, подключенная к коробке с цилиндрическим вырезом. Этот вырез - разъем для батарейки.
    1. Также на столе находится палетка размера 5 x 3 слота. Слот - это цилиндрический вырез. В слотах находятся зеленые цилиндрики (батарейки). Высота палетки и разъема для батарейки равна 1/2 высоты батарейки. Также перед палеткой находится кнопка.
    2. Из 15 батареек рабочие только 3. Агент должен сделать следующее:
    3. Взять батарейку из палетки, поместить ее в разъем для батарейки. Если лампочка загорелась, значит батарейка рабочая. Если нет - значит батарейка не рабочая. После проверки агент должен достать батарейку из разъема для батарейки и вернуть на ее место в палетку. Как только агент вернул батарейку на место, он должен нажать на кнопку перед палеткой. Далее агент должен проделать эту процедуру со всеми оставшимися батарейками.
    4. Трудность заключается в том, что когда агент положил после проверки очередную батарейку назад в палетку и нажал на кнопку, перед ним видна палетка со всеми\ заполненными ячейками с батарейками, и агент не знает, какие он уже проверил, если у него нет памяти. Агент должен за ограниченное число шагов в эпизоде найти все рабочие батарейки.
3. ✅ Лампочка моргает N раз, и затем агент должен N раз нажать на кнопку
4. ✅ Запомнить объекты и коснуться того, которого не было в начале (NegationMemory)



1. bulb lamp: https://sketchfab.com/3d-models/low-poly-light-bulb-a7d27c2224d94c86a04083de8f9df7db

collect data:
python3 mikasa_robo_suite/vla/dataset_collectors/get_mikasa_robo_datasets.py --env-id=ShellGameTouch-VLA-v0 --path-to-save-data=data_mikasa_robo --ckpt-dir=. --num-train-data=250
python3 mikasa_robo_suite/vla/dataset_collectors/parallel_dataset_collection_manager.py   --path-to-save-data=data_mikasa_robo --ckpt-dir=. --num-train-data=250 --max-parallel-processes=16

### Motion-planning collector (official replay-based pipeline)

`get_mikasa_robo_datasets_motion_planning.py` теперь работает в 2 шага:
1. Planner генерирует raw trajectory в `pd_joint_pos`.
2. `mani_skill.trajectory.replay_trajectory` конвертирует trajectory в `pd_ee_delta_pose`.

После этого коллектор делает детерминированный rollout в целевой среде и сохраняет `.npz`
в совместимом schema:
`rgb`, `proprio`, `action`, `reward`, `success`, `done`, `language_instruction`,
`success_once`, `episode_length`, `episode_seed`.

Важно:
- Ручная конвертация `joint -> pd_ee_delta_pose` удалена (без `scale/calibration/split/FK` логики).
- Сохраняются только успешные replay-эпизоды.
- Старые calibration/scale CLI-флаги оставлены только для обратной совместимости:
  если задать не-дефолтные значения, коллектор завершится ошибкой
  `удалено, используется ManiSkill replay`.

Примеры запуска:
```bash
uv run python mikasa_robo_suite/vla/dataset_collectors/get_mikasa_robo_datasets_motion_planning.py \
  --env-id BatteriesCheckerEasy-3-VLA-v0 \
  --path-to-save-data data_mikasa_robo \
  --num-train-data 250 \
  --max-attempts 20000 \
  --seed 123

uv run python mikasa_robo_suite/vla/dataset_collectors/get_mikasa_robo_datasets_motion_planning.py \
  --env-id BlinkCountButtonPressEasy-VLA-v0 \
  --path-to-save-data data_mikasa_robo \
  --num-train-data 250 \
  --max-attempts 20000 \
  --seed 123
```




### Новые long-horizon среды:
> rc, rs, rsac (long)
uv run python mikasa_robo_suite/vla/utils/motion_planning/motion_planning_remember_long.py --env-id RememberColor9-Long-VLA-v0 --seed 123

> soc, boc, coc (standard & long)
uv run python mikasa_robo_suite/vla/utils/motion_planning/motion_planning_memory_capacity_colors.py --env-id ChainOfColors3-VLA-v0 --seed 123

> ShellGameShuffleTouch-Long-VLA-v0, ShellGameShuffleColorLampTouch-Long-VLA-v0
uv run python mikasa_robo_suite/vla/utils/motion_planning/motion_planning_shell_game_shuffle.py --env-id ShellGameShuffleColorLampTouch-Long-VLA-v0 --seed 123 

> BlinkCountButtonPress{Easy/Medium/Hard}-Long-VLA-v0
uv run python mikasa_robo_suite/vla/utils/motion_planning/motion_planning_blink_count_button_press_long.py --env-id BlinkCountButtonPressHard-Long-VLA-v0


`BlinkCountButtonPressEasy-VLA-v0`
> сбор траекторий .npz
CUDA_VISIBLE_DEVICES=1 uv run python mikasa_robo_suite/vla/dataset_collectors/get_mikasa_robo_datasets_motion_planning.py \
  --env-id BlinkCountButtonPressHard-Long-VLA-v0 \
  --path-to-save-data data_mikasa_robo \
  --num-train-data 250 \
  --max-attempts 5000 \
  --seed 123

> последовательный сбор
for env in \
  RememberColor3-Long-VLA-v0 RememberColor5-Long-VLA-v0 RememberColor9-Long-VLA-v0 \
  RememberShape3-Long-VLA-v0 RememberShape5-Long-VLA-v0 RememberShape9-Long-VLA-v0 \
  RememberShapeAndColor3x2-Long-VLA-v0 RememberShapeAndColor3x3-Long-VLA-v0 RememberShapeAndColor5x3-Long-VLA-v0 \
  BunchOfColors3-VLA-v0 BunchOfColors5-VLA-v0 BunchOfColors7-VLA-v0 \
  SeqOfColors3-VLA-v0 SeqOfColors5-VLA-v0 SeqOfColors7-VLA-v0 \
  ChainOfColors3-VLA-v0 ChainOfColors5-VLA-v0 ChainOfColors7-VLA-v0 \
  BunchOfColors3-Long-VLA-v0 BunchOfColors5-Long-VLA-v0 BunchOfColors7-Long-VLA-v0 \
  SeqOfColors3-Long-VLA-v0 SeqOfColors5-Long-VLA-v0 SeqOfColors7-Long-VLA-v0 \
  ChainOfColors3-Long-VLA-v0 ChainOfColors5-Long-VLA-v0 ChainOfColors7-Long-VLA-v0 \
  ShellGameShuffleTouch-Long-VLA-v0 ShellGameShuffleColorLampTouch-Long-VLA-v0
do
  CUDA_VISIBLE_DEVICES=0 uv run python mikasa_robo_suite/vla/dataset_collectors/get_mikasa_robo_datasets_motion_planning.py \
    --env-id "$env" \
    --path-to-save-data data_mikasa_robo \
    --num-train-data 250 \
    --max-attempts 5000 \
    --seed 123
done

> параллельный сбор
```bash
cat > /tmp/run_mp_collect_auto.sh <<'BASH'
#!/usr/bin/env bash
set -euo pipefail

OUT_ROOT="data_mikasa_robo_mp"
NUM_TRAIN=250
MAX_ATTEMPTS=5000
BASE_SEED=123
SEED_MODE="fixed"   # fixed | per_env

CPU_BUDGET=224
CPU_PER_JOB=8
CHECK_SEC=15

GPU_MEM_FRAC_MAX=0.25
GPU_UTIL_MAX=100
MAX_JOBS_PER_GPU=24

ENVS=(
  RememberColor3-Long-VLA-v0
  RememberColor5-Long-VLA-v0
  RememberColor9-Long-VLA-v0
  RememberShape3-Long-VLA-v0
  RememberShape5-Long-VLA-v0
  RememberShape9-Long-VLA-v0
  RememberShapeAndColor3x2-Long-VLA-v0
  RememberShapeAndColor3x3-Long-VLA-v0
  RememberShapeAndColor5x3-Long-VLA-v0

  BunchOfColors3-VLA-v0
  BunchOfColors5-VLA-v0
  BunchOfColors7-VLA-v0
  SeqOfColors3-VLA-v0
  SeqOfColors5-VLA-v0
  SeqOfColors7-VLA-v0
  ChainOfColors3-VLA-v0
  ChainOfColors5-VLA-v0
  ChainOfColors7-VLA-v0

  BunchOfColors3-Long-VLA-v0
  BunchOfColors5-Long-VLA-v0
  BunchOfColors7-Long-VLA-v0
  SeqOfColors3-Long-VLA-v0
  SeqOfColors5-Long-VLA-v0
  SeqOfColors7-Long-VLA-v0
  ChainOfColors3-Long-VLA-v0
  ChainOfColors5-Long-VLA-v0
  ChainOfColors7-Long-VLA-v0

  ShellGameShuffleTouch-Long-VLA-v0
  ShellGameShuffleColorLampTouch-Long-VLA-v0
)

mkdir -p "${OUT_ROOT}" logs/mp_collect

VISIBLE_CPU=$(nproc)
if (( CPU_BUDGET > VISIBLE_CPU )); then
  CPU_BUDGET=$VISIBLE_CPU
fi

MAX_JOBS_CPU=$(( CPU_BUDGET / CPU_PER_JOB ))
if (( MAX_JOBS_CPU < 1 )); then
  MAX_JOBS_CPU=1
fi

echo "CPU visible: ${VISIBLE_CPU}"
echo "CPU budget: ${CPU_BUDGET}"
echo "CPU per job: ${CPU_PER_JOB}"
echo "MAX_JOBS_CPU: ${MAX_JOBS_CPU}"
echo "GPU policy: mem<=${GPU_MEM_FRAC_MAX}, util<=${GPU_UTIL_MAX}, max_jobs_per_gpu=${MAX_JOBS_PER_GPU}"

PIDS=()
PID_ENVS=()
PID_GPUS=()
PID_LOGS=()

env_cursor=0
env_count=${#ENVS[@]}

running_jobs() {
  local c=0
  local i
  for i in "${!PIDS[@]}"; do
    if [[ -n "${PIDS[$i]:-}" ]]; then
      ((++c))
    fi
  done
  echo "$c"
}

jobs_on_gpu() {
  local g="$1"
  local c=0
  local i
  for i in "${!PIDS[@]}"; do
    if [[ -n "${PIDS[$i]:-}" ]] && [[ "${PID_GPUS[$i]:-}" == "$g" ]]; then
      ((++c))
    fi
  done
  echo "$c"
}

free_gpus() {
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits \
  | awk -F',' -v m="${GPU_MEM_FRAC_MAX}" -v u="${GPU_UTIL_MAX}" '
      {
        gsub(/ /, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $3); gsub(/ /, "", $4);
        if ($3 > 0 && ($2 / $3) <= m && $4 <= u) print $1;
      }'
}

pick_gpu() {
  local best_gpu=""
  local best_count=999999
  local g c
  local candidates=()

  mapfile -t candidates < <(free_gpus || true)
  if (( ${#candidates[@]} == 0 )); then
    return 1
  fi

  for g in "${candidates[@]}"; do
    c=$(jobs_on_gpu "$g")
    if (( c < MAX_JOBS_PER_GPU )) && (( c < best_count )); then
      best_gpu="$g"
      best_count="$c"
    fi
  done

  [[ -n "$best_gpu" ]] || return 1
  echo "$best_gpu"
}

reap_finished_jobs() {
  local i pid env gpu log
  for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]:-}"
    [[ -z "$pid" ]] && continue

    if ! kill -0 "$pid" 2>/dev/null; then
      env="${PID_ENVS[$i]:-unknown}"
      gpu="${PID_GPUS[$i]:-unknown}"
      log="${PID_LOGS[$i]:-}"

      if wait "$pid"; then
        echo "[DONE] env=${env} gpu=${gpu} log=${log}"
      else
        echo "[FAIL] env=${env} gpu=${gpu} log=${log}"
      fi

      unset 'PIDS[i]' 'PID_ENVS[i]' 'PID_GPUS[i]' 'PID_LOGS[i]'
    fi
  done
}

cleanup() {
  echo "Stopping running jobs..."
  local i pid
  for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]:-}"
    if [[ -n "$pid" ]]; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  wait || true
}
trap cleanup INT TERM

while true; do
  reap_finished_jobs
  running=$(running_jobs)

  if (( env_cursor >= env_count && running == 0 )); then
    break
  fi

  while (( env_cursor < env_count )); do
    running=$(running_jobs)
    if (( running >= MAX_JOBS_CPU )); then
      break
    fi

    gpu="$(pick_gpu || true)"
    if [[ -z "$gpu" ]]; then
      break
    fi

    env="${ENVS[$env_cursor]}"
    case "${SEED_MODE}" in
      fixed)   seed="${BASE_SEED}" ;;
      per_env) seed=$(( BASE_SEED + env_cursor * 100000 )) ;;
      *) echo "Unknown SEED_MODE='${SEED_MODE}'" >&2; exit 1 ;;
    esac

    log="logs/mp_collect/${env}.log"
    (
      export CUDA_VISIBLE_DEVICES="${gpu}"
      export OMP_NUM_THREADS="${CPU_PER_JOB}"
      export MKL_NUM_THREADS="${CPU_PER_JOB}"
      export OPENBLAS_NUM_THREADS="${CPU_PER_JOB}"
      export NUMEXPR_NUM_THREADS="${CPU_PER_JOB}"

      exec nice -n 10 ionice -c2 -n7 \
        uv run python mikasa_robo_suite/vla/dataset_collectors/get_mikasa_robo_datasets_motion_planning.py \
          --env-id "${env}" \
          --path-to-save-data "${OUT_ROOT}" \
          --num-train-data "${NUM_TRAIN}" \
          --max-attempts "${MAX_ATTEMPTS}" \
          --seed "${seed}"
    ) >"${log}" 2>&1 &

    pid=$!
    PIDS+=("$pid")
    PID_ENVS+=("$env")
    PID_GPUS+=("$gpu")
    PID_LOGS+=("$log")

    echo "[START] env=${env} gpu=${gpu} pid=${pid} seed=${seed} log=${log}"
    env_cursor=$((env_cursor + 1))
  done

  running=$(running_jobs)
  queued=$(( env_count - env_cursor ))
  echo "[STATUS] running=${running} queued=${queued}"
  sleep "${CHECK_SEC}"
done

echo "All done."
BASH

chmod +x /tmp/run_mp_collect_auto.sh
bash /tmp/run_mp_collect_auto.sh

```



### New envs
🤡 Новая среда `mikasa_robo_suite/vla/memory_envs/trace_shape_vla.py`:
> TraceShapeEasy-VLA-v0, TraceShapeMedium-VLA-v0, TraceShapeHard-VLA-v0
> использует RenderTraceShapeDebugWrapper для дебага
uv run python mikasa_robo_suite/vla/utils/motion_planning/motion_planning_trace_shape.py --env-id TraceShapeHard-VLA-v0 --seed 42

🤡 Новая среда `mikasa_robo_suite/vla/memory_envs/trace_shape_vla.py`:
> TraceShapeSeqEasy-VLA-v0, TraceShapeSeqMedium-VLA-v0, TraceShapeSeqHard-VLA-v0
> использует RenderTraceShapeDebugWrapper для дебага
uv run python mikasa_robo_suite/vla/utils/motion_planning/motion_planning_trace_shape_seq.py --env-id TraceShapeSeqHard-VLA-v0 --seed 42

**Сбор данных:**
uv run python mikasa_robo_suite/vla/dataset_collectors/get_mikasa_robo_datasets_motion_planning.py \
  --env-id TraceShapeSeqHard-VLA-v0 \
  --path-to-save-data data_mikasa_robo \
  --num-train-data 250 \
  --max-attempts 5000


> Сбор данных .npz параллельно:
SKIP_IF_COMPLETE=1 GPU_LIST=2 JOBS_PER_GPU=1 CKPT_DIR=. bash utils/run_parallel_npz_collection.sh envs5.txt

!!!! не собрались .npz данные с ShellGamePick-VLA-v0!!!


🤡 Новая среда `mikasa_robo_suite/vla/memory_envs/timed_transfer_vla.py`:
> TimedTransferEasy-VLA-v0, TimedTransferMedium-VLA-v0, TimedTransfeHard-VLA-v0, TimedTransferEasy-Long-VLA-v0, TimedTransferMedium-Long-VLA-v0, TimedTransferHard-Long-VLA-v0
> использует враппер RenderTimedTransferInfoWrapper
uv run python mikasa_robo_suite/vla/utils/motion_planning/motion_planning_timed_transfer.py --env-id TimedTransferEasy-VLA-v0
uv run python mikasa_robo_suite/vla/utils/motion_planning/motion_planning_timed_transfer.py --env-id TimedTransferMedium-Long-VLA-v0


🤡 Новая среда `mikasa_robo_suite/vla/memory_envs/gather_and_recall_vla.py`:
> GatherAndRecall1-VLA-v0, GatherAndRecall3-VLA-v0, GatherAndRecall5-VLA-v0, GatherAndRecall7-VLA-v0, GatherAndRecall9-VLA-v0
uv run python mikasa_robo_suite/vla/utils/motion_planning/motion_planning_gather_and_recall.py --env-id GatherAndRecall3-VLA-v0


🤡 СОВСЕМ НЕ УДАЕТСЯ ОБУЧИТЬ PPO(state) на ShellGamePick-VLA. Далее будем использовать Motion Planner:
uv run python mikasa_robo_suite/vla/utils/motion_planning/motion_planning_shell_game_pick.py \
  --env-id ShellGamePick-VLA-v0 --seed 42 --save-video 1


! если сбор данные остановился
bash utils/resume_interrupted_mp_collection.sh.
НАХЕР ЭТУ СРЕДУ!!!! ВОЗЬМУ ДРУГУЮ