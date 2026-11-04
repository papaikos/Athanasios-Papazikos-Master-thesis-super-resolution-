#!/bin/bash -l
#SBATCH --job-name=d3x10-bench
#SBATCH --account=aristotle-users
#SBATCH --partition=ampere
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/benchmark-%j.log
# Submit from $PROJECT_DIR. Account and partition are the AUTH Aristotle values used in the thesis.
# Throughput options + the full test suite on the real GPU; writes nothing into the run.
set -euo pipefail
source "$PROJECT_DIR/ops/env.sh"
cd "$project/source"
"$python_bin" "$project/ops/check_env.py"
"$python_bin" -m pytest -q "$project/tests"
srun --ntasks=1 "$python_bin" -m research.train --data-root "$data" --index "$project/inputs/data-index.json" \
  --run-dir "$base/benchmark" --durable-root "$base/benchmark-durable" --control "$project/inputs/run-control.json" \
  --model-source "$project/source/vendor/network_swinir.py" --audit "$project/inputs/data-audit.json" \
  --config "$project/source/research/configs/d3x10_swinir_100k.yaml" --workers 12 \
  --benchmark "$project/inputs/benchmark.json"
