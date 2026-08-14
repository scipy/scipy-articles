import os

os.environ["SCIPY_ARRAY_API"] = "1"
from timeit import repeat
import argparse
import json

from scipy.stats import kurtosis
from scipy._lib._array_api import xp_assert_close
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--backend", type=str, default="NumPy", help="Backend to benchmark")
args = parser.parse_args()

backend = args.backend


def time(func):
    times = repeat(func, number=1, repeat=20)
    return np.min(times)


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
    numpy_result = kurtosis(data)

    if backend == "NumPy":
        kurtosis(data)
        times.append(time(lambda: kurtosis(data)))
    elif "PyTorch-CPU" in backend:
        data_torch = torch.asarray(data)
        xp_assert_close(kurtosis(data_torch), torch.asarray(numpy_result))
        times.append(time(lambda: kurtosis(data_torch)))
    elif "PyTorch-GPU" in backend:
        data_torch = torch.asarray(data, device="cuda")
        xp_assert_close(
            kurtosis(data_torch), torch.asarray(numpy_result, device="cuda")
        )
        torch.cuda.synchronize()

        def run():
            kurtosis(data_torch)
            torch.cuda.synchronize()

        times.append(time(run))
    elif "JAX-CPU" in backend:
        data_jax = jnp.asarray(data, device=jax.devices("cpu")[0])
        xp_assert_close(
            kurtosis(data_jax).block_until_ready(), jnp.asarray(numpy_result)
        )
        times.append(time(lambda: kurtosis(data_jax).block_until_ready()))
    elif "JAX-GPU" in backend:
        data_jax = jnp.asarray(data, device=jax.devices("gpu")[0])
        xp_assert_close(
            kurtosis(data_jax).block_until_ready(),
            jnp.asarray(numpy_result, device=jax.devices("gpu")[0]),
        )
        times.append(time(lambda: kurtosis(data_jax).block_until_ready()))
    elif "CuPy" in backend:
        data_cupy = cp.asarray(data)
        xp_assert_close(kurtosis(data_cupy), cp.asarray(numpy_result))
        cp.cuda.Stream.null.synchronize()

        def run():
            kurtosis(data_cupy)
            cp.cuda.Stream.null.synchronize()

        times.append(time(run))

with open("scripts/benchmark_timings.jsonl", "a") as f:
    f.write(json.dumps({"backend": backend, "ns": ns.tolist(), "times": times}) + "\n")
