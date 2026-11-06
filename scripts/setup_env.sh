#!/usr/bin/env bash
# Set up LIME from scratch: a conda environment with everything installed, and the weights.
#
#   bash scripts/setup_env.sh                  # environment "lime", weights in weights/lime_v1
#   bash scripts/setup_env.sh --name lime2     # another environment name
#   bash scripts/setup_env.sh --no-weights     # skip the 8.9 GB download
#
# The environment comes from environment.yml: Python 3.11, PyTorch 2.8 built for CUDA 12.8,
# the versions in requirements.txt, and this checkout installed in editable mode. To use
# another CUDA, change it in environment.yml first. Running this again reuses the
# environment and resumes an interrupted download.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
NAME="lime"
WEIGHTS=true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --name) NAME="${2:?--name needs a value}"; shift 2 ;;
        --no-weights) WEIGHTS=false; shift ;;
        -h|--help) sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown option: $1 (see --help)" >&2; exit 1 ;;
    esac
done

# In a script, `conda` isn't the shell function that `conda init` sets up, so find the program.
CONDA="${CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "$CONDA" || ! -x "$CONDA" ]]; then
    echo "conda isn't installed, or isn't on your PATH." >&2
    echo "Install Miniforge (https://conda-forge.org/download), open a new shell and run this again." >&2
    exit 1
fi

env_prefix() { "$CONDA" env list | awk -v name="$NAME" '$1 == name { print $NF }'; }

PREFIX="$(env_prefix)"
if [[ -n "$PREFIX" ]]; then
    echo "The conda environment $NAME already exists, so this uses it."
    echo "To build it again, run 'conda env remove -n $NAME' first."
else
    echo "Creating the conda environment $NAME. It downloads about 4 GB and takes a few minutes."
    "$CONDA" env create -n "$NAME" -f "$ROOT/environment.yml"
    PREFIX="$(env_prefix)"
fi

echo "Checking the environment..."
"$PREFIX/bin/python" - "$ROOT" "$NAME" <<'EOF'
import sys
from pathlib import Path

import torch
import transformers
import viser  # noqa: F401  the demo's

import lime_camera

root, name = Path(sys.argv[1]), sys.argv[2]
source = Path(lime_camera.__file__).resolve().parents[1]
if source != root:
    sys.exit(f"lime_camera in {name} comes from {source}, not from this checkout. Fix it with:\n"
             f"    conda activate {name} && pip install -e {root} --no-deps")
print(f"  Python {sys.version.split()[0]}, PyTorch {torch.__version__}, "
      f"transformers {transformers.__version__}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
else:
    print("  PyTorch can't see a GPU, so LIME will run on the CPU, at about 18 s per prediction.\n"
          "  If this machine has an NVIDIA GPU, its driver may be too old for CUDA 12.8: see\n"
          "  environment.yml.")
EOF

if $WEIGHTS; then
    # With the environment first on PATH, the download uses its `hf` command.
    PATH="$PREFIX/bin:$PATH" bash "$ROOT/scripts/download_weights.sh"
fi

echo
echo "Done. To run LIME:"
echo "    conda activate $NAME"
echo "    cd $ROOT"
if [[ ! -f "$ROOT/weights/lime_v1/config.json" ]]; then
    echo "    bash scripts/download_weights.sh      # the weights, 8.9 GB"
fi
echo "    python scripts/predict.py --image examples/living_room.jpg --intent \"Find my bike.\" --seed 0"
echo "    bash scripts/run_demo.sh"
