import os
os.environ["SCIPY_ARRAY_API"] = "1"
from timeit import repeat
import sys
import json

from scipy.stats import kurtosis
import numpy as np
backend = sys.argv[1]

def time(func):
    times = repeat(func, number=1, repeat=10)
    return np.min(times)

if backend == "plot":
    import matplotlib.pyplot as plt
    plt.style.use("scripts/scipy.mplstyle")

    fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")

    with open("scripts/benchmark_kurtosis_results.jsonl") as f:
        for line in f:
            data = json.loads(line)
            ax.loglog(data["ns"], data["times"], label=data["backend"])
    ax.set_xlabel("Dataset size")
    ax.set_ylabel("Time (s)")
    fig.legend(ncols=3, loc="outside lower center")
    plt.show()

else:
    rng = np.random.default_rng(738274923759827)
    REPEATS = 10
    ns = np.logspace(2, 7, 6, dtype=int)
    times = []

    if "PyTorch" in backend:
        import torch
    if "JAX" in backend:
        import jax.numpy as jnp
        import jax
        jax.config.update("jax_enable_x64", True)
        if "JIT" in backend:
            from jax import jit
            kurtosis = jit(kurtosis)
    if "CuPy" in backend:
        import cupy as cp

    for n in ns:
        data = rng.gamma(5, 0.5, size=n)

        if backend == "NumPy":
            kurtosis(data)
            times.append(time(lambda: kurtosis(data)))
        elif "PyTorch-CPU" in backend:
            data_torch = torch.asarray(data)
            kurtosis(data_torch)
            times.append(time(lambda: kurtosis(data_torch)))
        elif "PyTorch-GPU" in backend:
            data_torch = torch.asarray(data, device="cuda")
            kurtosis(data_torch)
            torch.cuda.synchronize()
            def run():
                kurtosis(data_torch)
                torch.cuda.synchronize()
            times.append(time(run))
        elif "JAX-CPU" in backend:
            data_jax = jnp.asarray(data, device=jax.devices("cpu")[0])
            kurtosis(data_jax).block_until_ready()
            times.append(time(lambda: kurtosis(data_jax).block_until_ready()))
        elif "JAX-GPU" in backend:
            data_jax = jnp.asarray(data, device=jax.devices("gpu")[0])
            kurtosis(data_jax).block_until_ready()
            times.append(time(lambda: kurtosis(data_jax).block_until_ready()))
        elif "CuPy" in backend:
            data_cupy = cp.asarray(data)
            kurtosis(data_cupy)
            cp.cuda.Stream.null.synchronize()
            def run():
                kurtosis(data_cupy)
                cp.cuda.Stream.null.synchronize()
            times.append(time(run))
    with open("scripts/benchmark_kurtosis_results.jsonl", "a") as f:
        f.write(json.dumps({"backend": backend, "ns": ns.tolist(), "times": times}) + "\n")