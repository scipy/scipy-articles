#!/usr/bin/env bash
set -euo pipefail

for backend in numpy; do
    echo
    echo "=============================="
    echo " Benchmarking: $backend"
    echo "=============================="

    BACKEND="$backend" pixi run -e "$backend" python scripts/benchmark_kurtosis.py
done