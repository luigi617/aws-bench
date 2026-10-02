#!/usr/bin/env bash
# Full benchmark cycle: for each scenario, run setup → benchmark → metrics → cleanup.
# Usage:
#   ./run_benchmark.sh                          # run all categories
#   ./run_benchmark.sh ec2-multiregion          # run one category
#   ./run_benchmark.sh ec2-multiregion serverless-apps  # run multiple
#   ./run_benchmark.sh --list                   # show available categories
#   ./run_benchmark.sh --skip-setup ec2-multiregion
#   ./run_benchmark.sh --skip-cleanup ec2-multiregion
set -euo pipefail

ACCOUNT_CONFIG="./accounts.yaml"
JOB_CONFIG="./job-config.yaml"
DATASET="aws-bench-quickstart"
DATASET_GIT_URL="https://github.com/aws-bench/aws-bench-datasets.git"
ENV_NAME="aws-bench-env"

# Local clone of the dataset repo — tasks/<scenario>/<task>/ is used for --path.
# On first run the repo is cloned; subsequent runs do a git pull.
DATASET_REPO_DIR="${HOME}/.aws-bench/datasets-repo"
DATASET_TASKS_DIR="${DATASET_REPO_DIR}/tasks"

# Parse flags and positional category names
SKIP_SETUP=false
SKIP_CLEANUP=false
LIST_ONLY=false
SELECTED_CATEGORIES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-setup)   SKIP_SETUP=true ;;
    --skip-cleanup) SKIP_CLEANUP=true ;;
    --list)         LIST_ONLY=true ;;
    --help|-h)
      sed -n '2,8p' "$0" | sed 's/^# //'
      exit 0 ;;
    --*)
      echo "Unknown flag: $1"; exit 1 ;;
    *)
      SELECTED_CATEGORIES+=("$1") ;;
  esac
  shift
done

# --- Ensure dataset repo is available ---
if [[ ! -d "$DATASET_REPO_DIR/.git" ]]; then
  echo "==> Cloning dataset repo to $DATASET_REPO_DIR ..."
  git clone --depth=1 "$DATASET_GIT_URL" "$DATASET_REPO_DIR"
else
  echo "==> Updating dataset repo ..."
  git -C "$DATASET_REPO_DIR" pull --quiet --ff-only
fi

JOBS_DIR=$(grep '^jobs_dir:' "$JOB_CONFIG" | awk '{print $2}')
BASE_JOB_NAME=$(grep '^job_name:' "$JOB_CONFIG" | awk '{print $2}')
ACCOUNT_ID=$(grep 'PRIMARY:' "$ACCOUNT_CONFIG" | head -1 | awk '{print $2}' | tr -d '"')

# Temporary per-scenario accounts.yaml (framework disallows >1 scenario per account)
SCENARIO_ACCOUNT_CONFIG=$(mktemp /tmp/aws-bench-accounts-XXXXXX.yaml)
trap 'rm -f "$SCENARIO_ACCOUNT_CONFIG"' EXIT
RUNNER_ROLE=$(grep 'runner_role:' "$ACCOUNT_CONFIG" | awk '{print $2}')
CFN_ROLE=$(grep 'cfn_role:' "$ACCOUNT_CONFIG" | awk '{print $2}')

# All available categories (from the cloned repo)
ALL_CATEGORIES=()
for d in "$DATASET_TASKS_DIR"/*/; do
  ALL_CATEGORIES+=("$(basename "$d")")
done

if [[ "$LIST_ONLY" == true ]]; then
  echo "Available benchmark categories:"
  for c in "${ALL_CATEGORIES[@]}"; do
    task_count=$(ls "$DATASET_TASKS_DIR/$c" | wc -l | tr -d ' ')
    echo "  $c  ($task_count tasks)"
  done
  exit 0
fi

# Validate and build the run list
SCENARIOS=()
if [[ ${#SELECTED_CATEGORIES[@]} -eq 0 ]]; then
  SCENARIOS=("${ALL_CATEGORIES[@]}")
else
  for cat in "${SELECTED_CATEGORIES[@]}"; do
    if [[ -d "$DATASET_TASKS_DIR/$cat" ]]; then
      SCENARIOS+=("$cat")
    else
      echo "Unknown category: '$cat'"
      echo "Run './run_benchmark.sh --list' to see available categories."
      exit 1
    fi
  done
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

  # Write a single-scenario accounts.yaml (framework disallows >1 scenario per account)
  cat > "$SCENARIO_ACCOUNT_CONFIG" <<EOF
schema_version: "1.0"
mode: preexisting
name: aws-bench-env
runner_role: $RUNNER_ROLE
cfn_role: $CFN_ROLE
accounts:
  $scenario:
    PRIMARY: "$ACCOUNT_ID"
EOF

  # 1. Environment setup
  if [[ "$SKIP_SETUP" == false ]]; then
    echo "--> env setup"
    uv run aws-bench --account-config "$SCENARIO_ACCOUNT_CONFIG" \
      env setup \
      --env-name "$ENV_NAME" \
      --dataset "$DATASET" \
      --include-scenarios "$scenario"
  fi

  # 2. Run benchmark (scoped to this scenario's tasks via --path)
  # -c applies agent, concurrency, timeout, retry settings from job-config.yaml;
  # --path and --job-name override the dataset and job name for this scenario.
  echo "--> run"
  uv run aws-bench --account-config "$SCENARIO_ACCOUNT_CONFIG" run \
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
    uv run aws-bench --account-config "$SCENARIO_ACCOUNT_CONFIG" \
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
