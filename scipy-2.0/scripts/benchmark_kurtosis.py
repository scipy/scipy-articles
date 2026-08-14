import os
os.environ["SCIPY_ARRAY_API"] = "1"
import timeit
import sys
import json

from scipy.stats import kurtosis
import numpy as np
backend = sys.argv[1]

if backend == "plot":
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(layout="constrained")

    with open("scripts/benchmark_kurtosis_results.jsonl") as f:
        for line in f:
            data = json.loads(line)
            ax.loglog(data["ns"], data["times"], label=data["backend"])
    ax.set_xlabel("n")
    ax.set_ylabel("time (s)")
    ax.legend()
    plt.show()

rng = np.random.default_rng(738274923759827)
REPEATS = 10
ns = np.logspace(2, 7, 6, dtype=int)
times = []

if "PyTorch" in backend:
    import torch
if "JAX" in backend:
    import jax.numpy as jnp
if backend == "JAX-CPU-JIT":
    from jax import jit
    kurtosis = jit(kurtosis)

for n in ns:
    data = rng.gamma(5, 0.5, size=n)

    if backend == "NumPy":
        kurtosis(data)
        times.append(timeit.timeit(lambda: kurtosis(data), number=REPEATS))
    elif backend == "PyTorch-CPU":
        data_torch = torch.asarray(data)
        kurtosis(data_torch)
        times.append(timeit.timeit(lambda: kurtosis(data_torch), number=REPEATS))
    elif "JAX-CPU" in backend:
        data_jax = jnp.asarray(data)
        kurtosis(data_jax).block_until_ready()
        times.append(timeit.timeit(lambda: kurtosis(data_jax).block_until_ready(), number=REPEATS))

with open("scripts/benchmark_kurtosis_results.jsonl", "a") as f:
    f.write(json.dumps({"backend": backend, "ns": ns.tolist(), "times": times}) + "\n")