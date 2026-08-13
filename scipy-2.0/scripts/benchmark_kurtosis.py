import os
os.environ["SCIPY_ARRAY_API"] = "1"
import timeit

from scipy.stats import kurtosis
import numpy as np

rng = np.random.default_rng(738274923759827)
REPEATS = 10
ns = np.logspace(2, 7, 6, dtype=int)
times = []

for n in ns:
    data = rng.gamma(5, 0.5, size=n)

    if os.environ["BACKEND"] == "numpy":
        times.append(timeit.timeit(lambda: kurtosis(data), number=REPEATS))
