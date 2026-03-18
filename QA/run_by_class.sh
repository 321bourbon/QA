#!/usr/bin/env bash
set -u

PROJECT_ROOT="/home/liuxinyi/projects/QA"
PYTHON_BIN="/home/liuxinyi/.conda/envs/vlm/bin/python"

VLM="internvl"
MODE="api"

CLASSES=(
  "breakfast_box"
  "juice_bottle"
  "pushpins"
  "screw_bag"
  "splicing_connectors"
)

BASE_DIR="$PROJECT_ROOT/test_results/${VLM}_by_class_$(date +%Y%m%d_%H%M%S)"

TRAIN_TIMEOUT=$((6 * 3600))
TEST_TIMEOUT=$((12 * 3600))

SLEEP_BEFORE_TRAIN=60
SLEEP_BETWEEN_TRAIN_TEST=180
SLEEP_AFTER_TEST=180
SLEEP_AFTER_CLASS=300

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "$BASE_DIR"

cd "$PROJECT_ROOT" || exit 1

MAIN_LOG="$PROJECT_ROOT/logs/run_by_class_$(date +%Y%m%d_%H%M%S).log"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$MAIN_LOG"
}

wait_for_gpu_settle() {
  local rounds="${1:-6}"
  local interval="${2:-10}"

  log "等待 GPU 空闲..."

  local stable_count=0

  while true; do
    local busy=0

    while read -r util; do
      util="${util// /}"
      if [[ -n "$util" ]] && (( util > 5 )); then
        busy=1
        break
      fi
    done < <(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null)

    if (( busy == 0 )); then
      stable_count=$((stable_count + 1))
      log "GPU 空闲检测 $stable_count/$rounds"
    else
      stable_count=0
      log "GPU 仍在工作"
    fi

    if (( stable_count >= rounds )); then
      break
    fi

    sleep "$interval"
  done

  log "GPU 已稳定"
}

run_step() {
  local class_name="$1"
  local step_name="$2"
  local timeout_sec="$3"
  local result_dir="$4"
  local step_log="$5"

  export LOGICQA_VLM="$VLM"
  export LOGICQA_MODE="$MODE"
  export LOGICQA_CLASSES="$class_name"
  export LOGICQA_RESULTS_DIR="$result_dir"
  export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

  log "开始 $step_name | class=$class_name"
  log "python: $PYTHON_BIN"
  log "PYTHONPATH: $PYTHONPATH"
  log "结果目录: $result_dir"
  log "日志文件: $step_log"

  if timeout --signal=TERM --kill-after=120s "$timeout_sec" \
    env PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" -u "$PROJECT_ROOT/pipeline/${step_name}.py" >> "$step_log" 2>&1
  then
    log "$step_name 完成 | class=$class_name"
    return 0
  else
    local rc=$?
    log "$step_name 失败或超时 | class=$class_name | rc=$rc"
    return $rc
  fi
}

for class_name in "${CLASSES[@]}"; do

  CLASS_DIR="$BASE_DIR/$class_name"
  mkdir -p "$CLASS_DIR"

  RUN_TS="$(date +%Y%m%d_%H%M%S)"
  TRAIN_LOG="$PROJECT_ROOT/logs/${class_name}_train_${RUN_TS}.log"
  TEST_LOG="$PROJECT_ROOT/logs/${class_name}_test_${RUN_TS}.log"

  log "=================================================="
  log "开始类别: $class_name"
  log "结果目录: $CLASS_DIR"

  log "等待 ${SLEEP_BEFORE_TRAIN}s"
  sleep "$SLEEP_BEFORE_TRAIN"

  wait_for_gpu_settle

  run_step "$class_name" "train" "$TRAIN_TIMEOUT" "$CLASS_DIR" "$TRAIN_LOG"
  train_rc=$?

  log "train 结束等待 ${SLEEP_BETWEEN_TRAIN_TEST}s"
  sleep "$SLEEP_BETWEEN_TRAIN_TEST"

  wait_for_gpu_settle

  if [[ $train_rc -eq 0 ]]; then
      run_step "$class_name" "test" "$TEST_TIMEOUT" "$CLASS_DIR" "$TEST_LOG"
      test_rc=$?
  else
      log "train 失败，跳过 test"
      test_rc=999
  fi

  log "test 后等待 ${SLEEP_AFTER_TEST}s"
  sleep "$SLEEP_AFTER_TEST"

  wait_for_gpu_settle

  log "类别完成: $class_name | train=$train_rc test=$test_rc"

  log "类别间等待 ${SLEEP_AFTER_CLASS}s"
  sleep "$SLEEP_AFTER_CLASS"

done

log "所有类别运行结束"