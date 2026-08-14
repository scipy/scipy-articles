#!/usr/bin/env bash
set -euo pipefail

rm -f scripts/benchmark_kurtosis_results.jsonl

for backend in NumPy PyTorch-CPU JAX-CPU JAX-CPU-JIT; do
    echo
    echo "=============================="
    echo " Benchmarking: $backend"
    echo "=============================="

    env="${backend,,}"
    env="${env%-jit}"

    pixi run -e "$env" python scripts/benchmark_kurtosis.py $backend
done

pixi run -e numpy python scripts/benchmark_kurtosis.py plot