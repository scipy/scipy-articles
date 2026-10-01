"""Plot the results of the benchmarking."""

import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
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
    "NumPy": {"color": "black", "linestyle": "-", "marker": None, "linewidth": 2},
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
    "skew": ["a)", r"\texttt{scipy.stats.skew}"],
    "welch": ["b)", r"\texttt{scipy.signal.welch}"],
    "Rotation.mean": ["c)", r"\texttt{scipy.spatial.transform.Rotation.mean}"],
}


class LogLocatorWithOne(LogLocator):
    def tick_values(self, vmin, vmax):
        ticks = super().tick_values(vmin, vmax)

        if vmin <= 1 <= vmax:
            ticks = np.unique(np.append(ticks, 1.0))

        return ticks


fig = plt.figure(figsize=(5, 6), layout="constrained")
subfigs = fig.subfigures(3, 1)

if args.paper_data:
    data_dir = Path(__file__).parent / "data_paper"
else:
    data_dir = Path(__file__).parent / "data"

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
    with open(data_dir / f"{func}_benchmark_timings.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            if not args.include_eager and data["backend"] in ["JAX-CPU", "JAX-GPU"]:
                continue
            axl.loglog(
                data["ns"],
                data["times"],
                label=data["backend"] if i == 0 else None,
                **line_styles[data["backend"]],
            )
            if data["backend"] == "NumPy":
                numpy_times = data["times"]
                continue
            relative_times = np.array(numpy_times) / np.array(data["times"])
            axr.loglog(data["ns"], relative_times, **line_styles[data["backend"]])
    axl.set_ylabel("Time (s)")
    axr.set_ylabel("Speed-up relative\nto NumPy")
    # we always want to show the 1x tick in the speed-up plot
    axr.yaxis.set_major_locator(LogLocatorWithOne(base=10))
    for ax in (axl, axr):
        ax.minorticks_off()
axl.set_xlabel("Problem size $n$")
axr.set_xlabel("Problem size $n$")
fig.legend(ncols=4, loc="outside lower center")
if args.save:
    plt.savefig(Path(__file__).parent.parent / "src" / "figures" / "benchmark.pdf")
plt.show()
