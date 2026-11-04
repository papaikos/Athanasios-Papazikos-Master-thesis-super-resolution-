#!/bin/bash -l
#SBATCH --job-name=maritime-swinir-x4
#SBATCH --account=aristotle-users
#SBATCH --partition=ampere
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --signal=B:TERM@600
#SBATCH --output=slurm/train-%j.log
# Submit from $PROJECT_DIR (ops/submit_initial.py does this). Account and partition are
# the AUTH Aristotle values used in the thesis; change them for another cluster.

set -euo pipefail
umask 077
source "$PROJECT_DIR/ops/env.sh"
exec 9>"$project/training.lock"
flock -n 9 || { echo 'Another writer owns this experiment'; exit 75; }
test ! -f "$project/STOP"
test -f "$project/inputs/data-ready.json"
cd "$project/source"
"$python_bin" "$project/ops/preflight_inputs.py"
if [ ! -f "$project/inputs/tests-pass.json" ]; then
  "$python_bin" -m pytest -q tests
  "$python_bin" -c 'import json,pathlib,sys; pathlib.Path(sys.argv[1]).write_text(json.dumps({"status":"PASS","source":"pinned-source-and-operational-manifests"}))' "$project/inputs/tests-pass.json"
fi
if [ ! -f "$project/inputs/data-audit.json" ]; then
  srun --ntasks=1 "$python_bin" -m research.data_audit \
    --data-root "$base/datasets" --verify-shards \
    --output "$project/inputs/data-audit.json" \
    --source-map "$project/inputs/data-membership.csv"
fi
start_step=$("$python_bin" "$project/ops/after_job.py" --read-step)
finalization_flags=()
if [ "$start_step" -lt 100000 ]; then
  finalization_flags+=(--defer-finalization)
fi
stopping=0
runner=''
trap 'stopping=1; if [ -n "$runner" ]; then kill -TERM "$runner" 2>/dev/null || true; fi' TERM INT
set +e
srun --ntasks=1 --cpu-bind=cores "$python_bin" -m research.train \
  --storage-backend filesystem \
  --data-root "$base/datasets" \
  --run-dir "$base/run" --durable-root "$project/results" \
  --control "$project/inputs/run-control.json" \
  --source-root "$project/science" \
  --model-source "$project/source/vendor/network_swinir.py" \
  --audit "$project/inputs/data-audit.json" \
  --config "$project/source/research/configs/${dataset}_swinir_100k.yaml" \
  --checkpoint-retention best-latest \
  --job-max-runtime-seconds 18900 "${finalization_flags[@]}" &
runner=$!
wait "$runner"
return_code=$?
if [ "$stopping" = 1 ]; then
  wait "$runner"
  return_code=$?
fi
set -e
if [ "$return_code" -ne 0 ]; then
  echo "Trainer stopped with exit code $return_code; no next job submitted"
  exit "$return_code"
fi
"$python_bin" "$project/ops/after_job.py" --start-step "$start_step" --parent-job "$SLURM_JOB_ID"
