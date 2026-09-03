import os
from string import ascii_uppercase

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc

plt.style.use(os.path.join(os.path.dirname(__file__), "scipy.mplstyle"))

fig, axs = plt.subplots(1, 5, constrained_layout=True, figsize=(5.48, 1.4))

n = 128
rng = np.random.default_rng(88594742687253747048352356358197231230)

methods = [
    ("IID uniform", lambda: rng.random((n, 2))),
    ("Sobol'", lambda: qmc.Sobol(d=2, rng=rng).random(n)),
    ("Halton", lambda: qmc.Halton(d=2, rng=rng).random(n)),
    ("Latin HC", lambda: qmc.LatinHypercube(d=2, rng=rng).random(n)),
    ("Poisson disk", lambda: qmc.PoissonDisk(d=2, radius=0.1, rng=rng,
                                             ncandidates=1000, hypersphere="surface").fill_space()),
]

for (ax, (title, sampler), letter) in zip(axs, methods, ascii_uppercase):
    sample = sampler()
    print(sample.shape[0])
    discrepancy = qmc.discrepancy(sample)
    ax.set_title(f"{letter}) {title}")
    ax.plot(*sample.T, 'k.')
    ax.text(0.1, -0.15, rf"$\mathrm{{CD}}^2=\num[output-exponent-marker = e]{{{discrepancy}}}$")
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()