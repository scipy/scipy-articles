import json

import matplotlib.pyplot as plt

plt.style.use("scripts/scipy.mplstyle")

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
plt.show()