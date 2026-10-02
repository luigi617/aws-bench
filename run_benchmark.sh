#!/usr/bin/env bash
# Full benchmark cycle: for each scenario, run setup → benchmark → metrics → cleanup.
# Usage: ./run_benchmark.sh [--skip-setup] [--skip-cleanup] [--scenario PATTERN]
set -euo pipefail

ACCOUNT_CONFIG="./accounts.yaml"
JOB_CONFIG="./job-config.yaml"
DATASET="aws-bench-quickstart"
ENV_NAME="aws-bench-env"

# Dataset is downloaded here by aws-bench; adjust if your cache differs
DATASET_TASKS_DIR="/tmp/aws-bench-datasets/tasks"

# Parse flags
SKIP_SETUP=false
SKIP_CLEANUP=false
SCENARIO_FILTER="*"  # fnmatch glob; default = all

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-setup)   SKIP_SETUP=true ;;
    --skip-cleanup) SKIP_CLEANUP=true ;;
    --scenario)     SCENARIO_FILTER="$2"; shift ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
  shift
done

JOBS_DIR=$(grep '^jobs_dir:' "$JOB_CONFIG" | awk '{print $2}')
BASE_JOB_NAME=$(grep '^job_name:' "$JOB_CONFIG" | awk '{print $2}')

# Collect scenario directories matching the filter
SCENARIOS=()
for scenario_dir in "$DATASET_TASKS_DIR"/*/; do
  scenario=$(basename "$scenario_dir")
  # shellcheck disable=SC2254
  case "$scenario" in
    $SCENARIO_FILTER) SCENARIOS+=("$scenario") ;;
  esac
done

if [[ ${#SCENARIOS[@]} -eq 0 ]]; then
  echo "No scenarios matched filter: $SCENARIO_FILTER"
  exit 1
fi

echo "==> Running ${#SCENARIOS[@]} scenario(s): ${SCENARIOS[*]}"
echo

FAILED=()

for scenario in "${SCENARIOS[@]}"; do
  JOB_NAME="${BASE_JOB_NAME}-${scenario}"
  RUN_DIR="${JOBS_DIR}/${JOB_NAME}"
  SCENARIO_PATH="${DATASET_TASKS_DIR}/${scenario}"

  echo "========================================"
  echo "  Scenario: $scenario"
  echo "  Job:      $JOB_NAME"
  echo "========================================"

  # 1. Environment setup
  if [[ "$SKIP_SETUP" == false ]]; then
    echo "--> env setup"
    uv run aws-bench --account-config "$ACCOUNT_CONFIG" \
      env setup \
      --env-name "$ENV_NAME" \
      --dataset "$DATASET" \
      --include-scenarios "$scenario"
  fi

  # 2. Run benchmark (scoped to this scenario's tasks via --path)
  # -c applies agent, concurrency, timeout, retry settings from job-config.yaml;
  # --path and --job-name override the dataset and job name for this scenario.
  echo "--> run"
  uv run aws-bench --account-config "$ACCOUNT_CONFIG" run \
    -c "$JOB_CONFIG" \
    --path "$SCENARIO_PATH" \
    --job-name "$JOB_NAME" \
    --jobs-dir "$JOBS_DIR" \
    --yes \
    || { echo "!! Benchmark failed for $scenario"; FAILED+=("$scenario"); }

  # 3. Compute metrics
  if [[ -d "$RUN_DIR" ]]; then
    echo "--> compute metrics"
    uv run python scripts/compute_metrics.py "$RUN_DIR" \
      && echo "    Metrics: $RUN_DIR/metrics.json" \
      || echo "    Warning: metrics computation failed for $scenario"
  fi

  # 4. Environment cleanup
  if [[ "$SKIP_CLEANUP" == false ]]; then
    echo "--> env cleanup"
    uv run aws-bench --account-config "$ACCOUNT_CONFIG" \
      env cleanup \
      --env-name "$ENV_NAME" \
      --dataset "$DATASET" \
      --include-scenarios "$scenario" \
      --yes
  fi

  echo
done

# Summary
echo "========================================"
echo "==> All scenarios complete."
echo "    Results in: $JOBS_DIR/"
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "    FAILED: ${FAILED[*]}"
  exit 1
fi
