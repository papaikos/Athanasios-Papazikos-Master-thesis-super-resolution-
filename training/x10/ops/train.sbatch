#!/bin/bash -l
#SBATCH --job-name=d3x10-swinir-x10
#SBATCH --account=aristotle-users
#SBATCH --partition=ampere
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --signal=B:TERM@600
#SBATCH --output=slurm/train-%j.log
# Submit from $PROJECT_DIR. Account and partition are the AUTH Aristotle values used in the thesis.
# One allocation of the chained 100k-update run (pattern of the D2 run):
# verified resume -> train for <=18,900 s -> verified pause -> exactly one successor.
set -euo pipefail
umask 077
source "$PROJECT_DIR/ops/env.sh"
exec 9>"$project/training.lock"
flock -n 9 || { echo 'Another writer owns this experiment'; exit 75; }
test ! -f "$project/STOP"
test -f "$project/inputs/data-audit.json"
cd "$project/source"
"$python_bin" "$project/ops/check_env.py"
start_step=$("$python_bin" "$project/ops/after_job.py" --read-step)
finalization_flags=()
if [ "$start_step" -lt 100000 ]; then finalization_flags+=(--defer-finalization); fi
stopping=0
runner=''
trap 'stopping=1; if [ -n "$runner" ]; then kill -TERM "$runner" 2>/dev/null || true; fi' TERM INT
set +e
srun --ntasks=1 --cpu-bind=none "$python_bin" -m research.train \
  --data-root "$data" --index "$project/inputs/data-index.json" \
  --run-dir "$base/run" --durable-root "$project/results" \
  --control "$project/inputs/run-control.json" \
  --model-source "$project/source/vendor/network_swinir.py" \
  --audit "$project/inputs/data-audit.json" \
  --config "$project/source/research/configs/d3x10_swinir_100k.yaml" \
  --workers 12 --job-max-runtime-seconds 18900 "${finalization_flags[@]}" &
runner=$!
wait "$runner"
return_code=$?
if [ "$stopping" = 1 ]; then wait "$runner"; return_code=$?; fi
set -e
if [ "$return_code" -ne 0 ]; then
  echo "Trainer stopped with exit code $return_code; no next job submitted"
  exit "$return_code"
fi
"$python_bin" "$project/ops/after_job.py" --start-step "$start_step" --parent-job "$SLURM_JOB_ID"
