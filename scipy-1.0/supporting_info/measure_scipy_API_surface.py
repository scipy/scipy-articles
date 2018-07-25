"""
This script imports all the public submodules and estimates the number
of public objects in each of them.  This can be used to give a number
for the total API surface of SciPy, e.g. "2600 functions and classes".

"""

import scipy
from scipy import (
    cluster,
    constants,
    fftpack,
    integrate,
    interpolate,
    io,
    linalg,
    misc,
    ndimage,
    odr,
    optimize,
    signal,
    sparse,
    spatial,
    special,
    stats,
)

from scipy.cluster import vq, hierarchy
from scipy.io import arff, harwell_boeing, idl, netcdf, matlab, wavfile
from scipy.linalg import blas, lapack, interpolative
from scipy.signal import windows
from scipy.sparse import linalg as sls
from scipy.sparse import csgraph
from scipy.spatial import distance, transform
from scipy.stats import distributions, mstats


modules = [
    scipy.cluster,
    scipy.cluster.vq,
    scipy.cluster.hierarchy,
    scipy.constants,
    scipy.fftpack,
    scipy.integrate,
    scipy.interpolate,
    scipy.io,
    scipy.io.arff,
    scipy.io.harwell_boeing,
    scipy.io.idl,
    scipy.io.matlab,
    scipy.io.netcdf,
    scipy.io.wavfile,
    scipy.linalg,
    scipy.linalg.blas,
    scipy.linalg.cython_blas,
    scipy.linalg.lapack,
    scipy.linalg.cython_lapack,
    scipy.linalg.interpolative,
    scipy.misc,
    scipy.ndimage,
    scipy.odr,
    scipy.optimize,
    scipy.signal,
    scipy.signal.windows,
    scipy.sparse,
    scipy.sparse.linalg,
    scipy.sparse.csgraph,
    scipy.spatial,
    scipy.spatial.distance,
    scipy.spatial.transform,
    scipy.special,
    scipy.stats,
    scipy.stats.distributions,
    scipy.stats.mstats,
]

api_surface = dict()
exclude = set(['np', 'numpy', 'division', 'absolute_import', 'print_function'])
for module in modules:
    all_objects = set(dir(module))
    private = set((s for s in all_objects if s.startswith('_')))
    api_surface[module] = len(all_objects - private - exclude)

print(sum(api_surface.values()))