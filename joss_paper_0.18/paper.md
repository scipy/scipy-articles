---
title: 'SciPy 0.18: Open source toolbox for science and engineering'
tags:
  - science
  - engineering
  - mathematics
authors:
 - name: Evgeni Burovski
   orcid: 0000-0001-8149-0483
   affiliation: National Research University Higher School of Economics
 - name: Andrew Nelson
   orcid: 0000-0002-4548-3558
   affiliation: Australian Nuclear Science and Technology Organisation
date: XXX
bibliography: paper.bib
---

# Summary

SciPy (pronounced "Sigh Pie") is an open source library for mathematics, science,
and engineering. It includes modules for statistics, optimization and root finding,
fitting,  evaluation of special functions, interpolation, integration, ODE solvers,
Fourier transforms, handling of sparse matrices, linear algebra with dense and
sparse matrices, hierarchial clustering, signal and N-dimensional image processing,
spatial algorithms and data structures, graph algorithms, and reading and
writing data files in a variety of common data formats.

The @SciPy library depends on @NumPy, which provides convenient and fast N-dimensional
array manipulation. The SciPy library is built to work with NumPy arrays, and
provides many user-friendly and efficient numerical methods such as routines
for numerical integration and optimization. 

The latest stable release of SciPy is [0.18.1](@SciPy0181), which improves on the feature
release [0.18.0](SciPy0180). Together they contain numerous bug-fixes, improved test coverage,
better documentation, and many new features. 

Highlights of this release include:

* A new ODE solver for two-point boundary value problems, `scipy.optimize.solve_bvp`.
* A new class, `scipy.interpolate.CubicSpline`, for cubic spline interpolation of data.
* N-dimensional tensor product polynomials, `scipy.interpolate.NdPPoly`.
* Spherical Voronoi diagrams, `scipy.spatial.SphericalVoronoi`.
* Support for discrete-time linear systems, `scipy.signal.dlti`.

For the complete description of the contents of the 0.18 series, see the
[release notes](https://github.com/scipy/scipy/releases/tag/v0.18.0).

# Acknowledgements

We gratefully acknowledge contributions of more than four hundred people whose work
over more than fifteen years went into previous releases of SciPy and made the
current release possible.


# References
