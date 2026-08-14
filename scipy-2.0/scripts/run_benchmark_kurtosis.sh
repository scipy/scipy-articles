#!/usr/bin/env bash
set -euo pipefail

rm -f scripts/benchmark_kurtosis_results.jsonl

for backend in NumPy; do
    echo
    echo "=============================="
    echo " Benchmarking: $backend"
    echo "=============================="

    pixi run -e "${backend,,}" python scripts/benchmark_kurtosis.py $backend
done

pixi run -e numpy python scripts/benchmark_kurtosis.py plot