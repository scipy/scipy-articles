"""Plot the results of the benchmarking."""

import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import mpl_tectonic

mpl_tectonic.enable()
plt.style.use("scripts/scipy.mplstyle")

parser = argparse.ArgumentParser(description="Plot the results of the benchmarking.")
parser.add_argument(
    "--save",
    action="store_true",
    help="Save the plot to the figures directory."
)
parser.add_argument(
    "--paper-data",
    action="store_true",
    help="Use the data from the paper, rather than the latest benchmark results.",
)
parser.add_argument(
    "--include-eager",
    action="store_true",
    help="Include eager JAX results in the plot.",
)
args = parser.parse_args()

# Colour blind friendly palette from https://www.nature.com/articles/nmeth.1618
line_styles = {
    "NumPy": {"color": "black", "linestyle": "-", "marker": None, "alpha": 0.5},
    "PyTorch-CPU": {"color": "#56B4E9", "linestyle": "-", "marker": None},
    "PyTorch-GPU": {"color": "#56B4E9", "linestyle": "--", "marker": None},
    "JAX-CPU": {"color": "#D55E00", "linestyle": "-x", "marker": None},
    "JAX-CPU-JIT": {
        "color": "#D55E00",
        "linestyle": "-",
    },
    "JAX-GPU": {"color": "#D55E00", "linestyle": "--x", "marker": None},
    "JAX-GPU-JIT": {
        "color": "#D55E00",
        "linestyle": "--",
    },
    "CuPy": {"color": "#009E73", "linestyle": "--", "marker": None},
}

funcs = {
    "skew": r"\textbf{a)}\; \texttt{scipy.stats.skew}",
    "welch": r"\textbf{b)}\; \texttt{scipy.signal.welch}",
    "Rotation.mean": r"\textbf{c)}\; \texttt{scipy.spatial.transform.Rotation.mean}",
}


fig = plt.figure(figsize=(5, 5), constrained_layout=True)

gs = fig.add_gridspec(2, 4)

ax1 = fig.add_subplot(gs[0, 0:2])
ax2 = fig.add_subplot(gs[0, 2:4])
ax3 = fig.add_subplot(gs[1, 1:3])

if args.paper_data:
    data_dir = Path(__file__).parent / "data_paper"
else:
    data_dir = Path(__file__).parent / "data"

for (func, title), (i, ax) in zip(funcs.items(), enumerate([ax1, ax2, ax3])):
    ax.set_title(title)
    with open(data_dir / f"{func}_benchmark_timings.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            if not args.include_eager and data["backend"] in ["JAX-CPU", "JAX-GPU"]:
                continue
            if data["backend"] == "NumPy":
                numpy_times = data["times"]
                continue
            relative_times = np.array(numpy_times) / np.array(data["times"])
            ax.loglog(numpy_times, relative_times,
                      label=data["backend"] if i == 0 else None,
                      **line_styles[data["backend"]])
    ax.hlines(1, *ax.get_xlim(), color="black", alpha=0.5, linestyle=":")
    ax.set_ylabel("Speed-up vs. NumPy")
    ax.minorticks_off()
    ax.set_xlabel("NumPy execution time (s)")
    ax2 = ax.twiny()
    ax2.set_xscale("log")
    ax2.set_xlim(ax.get_xlim())
    ticks = [numpy_times[0], numpy_times[-1]]
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(rf"$n \approx 10^{{{n}}}$"for n in np.round(np.log10(data["ns"])).astype(np.int64)[[0, -1]])
    ax2.minorticks_off()

fig.legend(ncols=4, loc="outside lower center")
if args.save:
    plt.savefig(Path(__file__).parent.parent / "src" / "figures" / "benchmark.pdf")
plt.show()
