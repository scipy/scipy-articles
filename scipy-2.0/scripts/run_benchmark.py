import os

os.environ["SCIPY_ARRAY_API"] = "1"
from timeit import repeat
import argparse
import json

from scipy.stats import gmean
from scipy._lib._array_api import xp_assert_close
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("function", type=str, help="Function to benchmark")
parser.add_argument("log_n_start", type=float, help="Starting size of the input data")
parser.add_argument("log_n_end", type=float, help="Ending size of the input data")
parser.add_argument("n_points", type=int, help="Number of points between n_start and n_end")
parser.add_argument("--backend", type=str, default="NumPy", help="Backend to benchmark")
parser.add_argument("--repeats", type=int, default=30, help="Number of repeats for timing")
args = parser.parse_args()

backend = args.backend
repeats = args.repeats


def time(func):
    times = repeat(func, number=1, repeat=repeats)
    return gmean(times)

if args.function == "skew":
    from scipy.stats import skew
    def func(data):
        return skew(data, axis=-1)

    def data_generator(n):
        rng = np.random.default_rng(738274923759827)
        return np.stack([
            rng.gamma(5, 0.5, size=n),
            rng.normal(size=n),
            rng.beta(0.5, 0.5, size=n)
        ])
elif args.function == "welch":
    if backend == "PyTorch-GPU":
        exit()
    from scipy.signal import welch
    def func(data):
        return welch(data, nperseg=256)[0]

    def data_generator(n):
        rng = np.random.default_rng(738274923759827)
        fs = 10e3
        N = 1e5
        amp = 2*np.sqrt(2)
        freq = 1234.0
        noise_power = 0.001 * fs / 2
        time = np.arange(n) / fs
        x = amp*np.sin(2*np.pi*freq*time)
        x += rng.normal(scale=np.sqrt(noise_power), size=time.shape)
        return x
else:
    raise ValueError(f"Unknown function: {args.function}")


REPEATS = 10
ns = np.logspace(args.log_n_start, args.log_n_end, args.n_points, dtype=int)
times = []

if "PyTorch" in backend:
    import torch
if "JAX" in backend:
    import jax.numpy as jnp
    import jax

    jax.config.update("jax_enable_x64", True)
    if "JIT" in backend:
        from jax import jit

        func = jit(func)
if "CuPy" in backend:
    import cupy as cp

for n in ns:
    data = data_generator(n)
    numpy_result = func(data)

    if backend == "NumPy":
        func(data)
        times.append(time(lambda: func(data)))
    elif "PyTorch-CPU" in backend:
        data_torch = torch.asarray(data)
        xp_assert_close(func(data_torch), torch.asarray(numpy_result))
        times.append(time(lambda: func(data_torch)))
    elif "PyTorch-GPU" in backend:
        data_torch = torch.asarray(data, device="cuda")
        xp_assert_close(
            func(data_torch), torch.asarray(numpy_result, device="cuda")
        )
        torch.cuda.synchronize()

        def run():
            func(data_torch)
            torch.cuda.synchronize()

        times.append(time(run))
    elif "JAX-CPU" in backend:
        data_jax = jnp.asarray(data, device=jax.devices("cpu")[0])
        xp_assert_close(
            func(data_jax).block_until_ready(), jnp.asarray(numpy_result)
        )
        times.append(time(lambda: func(data_jax).block_until_ready()))
    elif "JAX-GPU" in backend:
        data_jax = jnp.asarray(data, device=jax.devices("gpu")[0])
        xp_assert_close(
            func(data_jax).block_until_ready(),
            jnp.asarray(numpy_result, device=jax.devices("gpu")[0]),
        )
        times.append(time(lambda: func(data_jax).block_until_ready()))
    elif "CuPy" in backend:
        data_cupy = cp.asarray(data)
        xp_assert_close(func(data_cupy), cp.asarray(numpy_result))
        cp.cuda.Stream.null.synchronize()

        def run():
            func(data_cupy)
            cp.cuda.Stream.null.synchronize()

        times.append(time(run))

with open("scripts/benchmark_timings.jsonl", "a") as f:
    f.write(json.dumps({"backend": backend, "ns": ns.tolist(), "times": times}) + "\n")
