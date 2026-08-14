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

for n in ns:
    data = rng.gamma(5, 0.5, size=n)

    if backend == "NumPy":
        times.append(timeit.timeit(lambda: kurtosis(data), number=REPEATS))

with open("scripts/benchmark_kurtosis_results.jsonl", "a") as f:
    f.write(json.dumps({"backend": backend, "ns": ns.tolist(), "times": times}) + "\n")