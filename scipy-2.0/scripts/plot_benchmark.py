import json
import argparse

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("scripts/scipy.mplstyle")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--timing",
    choices=["relative", "absolute"],
    default="relative",
    help="Report timings as relative speed-up to NumPy or absolute (wall-clock) values",
)
args = parser.parse_args()

line_styles = {
    "NumPy": {"color": "black", "linestyle": "-", "marker": None, "linewidth": 2},
    "PyTorch-CPU": {"color": "tab:orange", "linestyle": "--", "marker": None},
    "PyTorch-GPU": {"color": "tab:orange", "linestyle": "-", "marker": None},
    "JAX-CPU": {"color": "tab:green", "linestyle": "--", "marker": None},
    "JAX-CPU-JIT": {
        "color": "tab:green",
        "linestyle": "--",
        "marker": "o",
        "markevery": 0.1,
    },
    "JAX-GPU": {"color": "tab:green", "linestyle": "-", "marker": None},
    "JAX-GPU-JIT": {
        "color": "tab:green",
        "linestyle": "-",
        "marker": "o",
        "markevery": 0.1,
    },
    "CuPy": {"color": "tab:red", "linestyle": "-", "marker": None},
}

if args.timing == "absolute":
    fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")

    with open("scripts/benchmark_timings.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            ax.loglog(
                data["ns"],
                data["times"],
                label=data["backend"],
                **line_styles[data["backend"]]
            )
    ax.set_xlabel("Dataset size")
    ax.set_ylabel("Time (s)")
    fig.legend(ncols=3, loc="outside lower center")
else:
    fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")

    with open("scripts/benchmark_timings.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            if data["backend"] == "NumPy":
                numpy_times = data["times"]
            relative_times = np.array(numpy_times) / np.array(data["times"])
            ax.loglog(
                data["ns"],
                relative_times,
                label=data["backend"],
                **line_styles[data["backend"]]
            )
    ax.set_xlabel("Dataset size")
    ax.set_ylabel("Speed-up relative to NumPy ($>1$ is faster)")
    fig.legend(ncols=3, loc="outside lower center")
plt.show()
