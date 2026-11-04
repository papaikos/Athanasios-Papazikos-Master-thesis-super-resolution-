# Sourced by every job: paths and pinned runtime. Set these before submitting (see .env.example):
#   PROJECT_DIR  run folder: a copy of training/x10 plus inputs/, results/ and slurm/
#   SCRATCH_DIR  fast scratch: venv/, run/, caches
#   DATA_DIR     the D3x10 dataset (Maritime_D3x10_1m10m)
project=${PROJECT_DIR:?set PROJECT_DIR}
base=${SCRATCH_DIR:?set SCRATCH_DIR}
data=${DATA_DIR:?set DATA_DIR}
export PROJECT_DIR SCRATCH_DIR DATA_DIR
module load gcc/12.2.0 python/3.10.8-cidwh6y
export PYTHONPATH="$project/source:$project/ops"
export TORCH_HOME="$base/torch-cache" MPLCONFIGDIR="$base/matplotlib-cache" XDG_CACHE_HOME="$base/cache"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
python_bin="$base/venv/bin/python"
