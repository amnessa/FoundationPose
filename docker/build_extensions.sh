#!/usr/bin/env bash
# Run INSIDE the container, once, to build the repo's native extensions.
# (The repo's build_all.sh also builds kaolin from /kaolin, which this image omits.)
set -euo pipefail

PROJ_ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"

cd "${PROJ_ROOT}/mycpp"
rm -rf build && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE="$(which python)"
make -j"$(nproc)"

echo
echo "mycpp built. Running check_env.py:"
cd "${PROJ_ROOT}"
python check_env.py
