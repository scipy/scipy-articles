import json
import argparse

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("scripts/scipy.mplstyle")

parser = argparse.ArgumentParser()
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

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), layout="constrained")

with open("scripts/benchmark_timings.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        ax1.loglog(
            data["ns"],
            data["times"],
            label=data["backend"],
            **line_styles[data["backend"]]
        )
        if data["backend"] == "NumPy":
            numpy_times = data["times"]
        relative_times = np.array(numpy_times) / np.array(data["times"])
        ax2.loglog(
            data["ns"],
            relative_times,
            **line_styles[data["backend"]]
        )
ax1.set_xlabel("Problem size $n$")
ax1.set_ylabel("Time (s)")
ax2.set_xlabel("Problem size $n$")
ax2.set_ylabel("Speed-up relative to NumPy ($>1$ is faster)")
fig.legend(ncols=4, loc="outside lower center")
plt.show()
