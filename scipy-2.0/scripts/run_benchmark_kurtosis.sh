#!/usr/bin/env bash
set -euo pipefail

rm -f scripts/benchmark_kurtosis_results.jsonl

for backend in NumPy PyTorch-CPU PyTorch-GPU JAX-CPU JAX-CPU-JIT JAX-GPU JAX-GPU-JIT CuPy; do
    echo
    echo "=============================="
    echo " Benchmarking: $backend"
    echo "=============================="

    env="${backend,,}"
    env="${env%-jit}"

    pixi run -e "$env" python scripts/benchmark_kurtosis.py $backend
done

pixi run -e numpy python scripts/benchmark_kurtosis.py plot