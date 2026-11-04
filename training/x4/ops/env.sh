# Sourced by every x4 job. Set these three variables before submitting (see .env.example):
#   PROJECT_DIR       run folder: a copy of training/x4 plus inputs/, results/ and slurm/
#   SCRATCH_DIR       fast scratch: datasets/ (staged D1|D2|D3 + control/), venv/, run/, caches
#   MARITIME_DATASET  d1, d2 or d3
project=${PROJECT_DIR:?set PROJECT_DIR}
base=${SCRATCH_DIR:?set SCRATCH_DIR}
dataset=${MARITIME_DATASET:?set MARITIME_DATASET to d1, d2 or d3}
export PROJECT_DIR SCRATCH_DIR MARITIME_DATASET
module load gcc/12.2.0 python/3.10.8-cidwh6y
# science/ is the dataset package of this repository (copy or link dataset/ there).
export PYTHONPATH="$project/source:$project/science"
export TORCH_HOME="$base/torch-cache"
export MPLCONFIGDIR="$base/matplotlib-cache"
export XDG_CACHE_HOME="$base/cache"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
python_bin="$base/venv/bin/python"
