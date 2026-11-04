#!/bin/bash -l
# Build the pinned runtime on the login node (same versions as the D2 run) and the
# verified AlexNet cache that LPIPS needs. Safe to rerun; never touches the data.
set -euo pipefail
base=${SCRATCH_DIR:?set SCRATCH_DIR}
module load gcc/12.2.0 python/3.10.8-cidwh6y
mkdir -p "$base"
if [ ! -x "$base/venv/bin/python" ]; then python -m venv "$base/venv"; fi
py="$base/venv/bin/python"
"$py" -m pip install -q --upgrade pip
# Exactly the D2 run's package set (pip freeze), torch from the official cu124 index.
"$py" -m pip install -q -r "$(dirname "$0")/requirements-lock-d2.txt" --extra-index-url https://download.pytorch.org/whl/cu124
mkdir -p "$base/torch-cache/hub/checkpoints"
alex="$base/torch-cache/hub/checkpoints/alexnet-owt-7be5be79.pth"
if [ ! -f "$alex" ]; then curl -sSfL -o "$alex.tmp" https://download.pytorch.org/models/alexnet-owt-7be5be79.pth && mv "$alex.tmp" "$alex"; fi
echo "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02  $alex" | sha256sum -c -
"$py" -m pip freeze > "$base/pip-freeze.txt"
echo "ENV_READY $("$py" -c 'import torch; print(torch.__version__, torch.version.cuda)')"
