#!/bin/bash -l
#SBATCH --job-name=d3x10-audit
#SBATCH --account=aristotle-users
#SBATCH --partition=batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/audit-%j.log
# Submit from $PROJECT_DIR. Account and partition are the AUTH Aristotle values used in the thesis.
set -euo pipefail
source "$PROJECT_DIR/ops/env.sh"
cd "$project"
python3 ops/audit_data.py --data-root "$data" --control inputs/run-control.json \
  --audit-out inputs/data-audit.json --index-out inputs/data-index.json
