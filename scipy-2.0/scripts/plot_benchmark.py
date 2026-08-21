"""Plot the results of the benchmarking."""

import json
import argparse

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("scripts/scipy.mplstyle")

parser = argparse.ArgumentParser()
args = parser.parse_args()

line_styles = {
    "NumPy": {"color": "black", "linestyle": "-", "marker": None, "linewidth": 2},
    "PyTorch-CPU": {"color": "tab:orange", "linestyle": "-", "marker": None},
    "PyTorch-GPU": {"color": "tab:orange", "linestyle": "--", "marker": None},
    "JAX-CPU": {"color": "tab:green", "linestyle": "-", "marker": None},
    "JAX-CPU-JIT": {
        "color": "tab:green",
        "linestyle": "-",
        "marker": "o",
        "markevery": 0.1,
    },
    "JAX-GPU": {"color": "tab:green", "linestyle": "--", "marker": None},
    "JAX-GPU-JIT": {
        "color": "tab:green",
        "linestyle": "--",
        "marker": "o",
        "markevery": 0.1,
    },
    "CuPy": {"color": "tab:red", "linestyle": "--", "marker": None},
}

funcs = {
    "skew": ["a)", r"\texttt{scipy.stats.skew}"],
    "welch": ["b)", r"\texttt{scipy.signal.welch}"],
    "Rotation.mean": ["c)", r"\texttt{scipy.spatial.transform.Rotation.mean}"],
}

fig = plt.figure(figsize=(5, 6), layout="constrained")
subfigs = fig.subfigures(3, 1)

for (func, title), (i, subfig) in zip(funcs.items(), enumerate(subfigs)):
    axl, axr = subfig.subplots(1, 2)
    subfig.suptitle(title[1])
    subfig.text(
        0.1,
        0.97,
        rf"\textbf{{ {title[0]} }}",
        ha="left",
        va="top",
        fontsize="medium",
    )
    with open(f"scripts/{func}_benchmark_timings.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            axl.loglog(
                data["ns"],
                data["times"],
                label=data["backend"] if i == 0 else None,
                **line_styles[data["backend"]],
            )
            if data["backend"] == "NumPy":
                numpy_times = data["times"]
            relative_times = np.array(numpy_times) / np.array(data["times"])
            axr.loglog(data["ns"], relative_times, **line_styles[data["backend"]])
    axl.set_ylabel("Time (s)")
    axr.set_ylabel("Speed-up relative\nto NumPy")
axl.set_xlabel("Problem size $n$")
axr.set_xlabel("Problem size $n$")
fig.legend(ncols=4, loc="outside lower center")
plt.show()
