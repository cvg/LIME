#!/usr/bin/env bash
# Start the interactive LIME demo: the examples, with a menu to switch between them.
#
#   conda activate lime
#   bash scripts/run_demo.sh                     # serves on port 8080
#   bash scripts/run_demo.sh --port 8090         # any demo/viewer.py option works here
#   bash scripts/run_demo.sh --image room.jpg    # your own image instead of the examples
#
# Once the model has loaded, it prints how to open the demo: in a browser on this
# machine, or from another computer through an SSH tunnel.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

# Where this environment's lime_camera comes from, if it has everything the demo needs.
# Run from demo/, so the import sees what demo/viewer.py will, not the current directory.
source="$(cd "$ROOT/demo" && python -c "
import pathlib, lime_camera, viser, trimesh, lz4, matplotlib, scipy
print(pathlib.Path(lime_camera.__file__).resolve().parents[1])" 2>/dev/null)" || {
    echo "This environment can't run the demo. Activate the one scripts/setup_env.sh made:" >&2
    echo "    conda activate lime" >&2
    exit 1
}
if [[ "$source" != "$ROOT" ]]; then
    echo "This environment's lime_camera comes from $source, not from $ROOT." >&2
    echo "Activate the environment scripts/setup_env.sh made (conda activate lime), or run:" >&2
    echo "    pip install -e \"$ROOT\" --no-deps" >&2
    exit 1
fi

custom_checkpoint=false
for arg in "$@"; do
    [[ "$arg" == --checkpoint* ]] && custom_checkpoint=true
done
if ! $custom_checkpoint && [[ ! -f "$ROOT/weights/lime_v1/config.json" ]]; then
    echo "There are no weights in $ROOT/weights/lime_v1. Download them first:" >&2
    echo "    bash $ROOT/scripts/download_weights.sh" >&2
    exit 1
fi

echo "Loading the model, which takes a few seconds..."
exec python "$ROOT/demo/viewer.py" "$@"
