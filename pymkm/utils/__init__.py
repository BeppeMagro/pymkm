# This file marks this directory as a Python package
"""
Utility submodule for pyMKM.

This package contains helper tools used throughout pyMKM for interpolation,
geometric modeling of particle interactions, and multiprocessing optimization.

Modules
-------

- :mod:`geometry_tools`: 
  Provides geometric calculations such as intersection areas and radius sampling
  for dose distribution modeling. See :class:`~pymkm.utils.geometry_tools.GeometryTools`.

- :mod:`interpolation`: 
  Contains a general-purpose :class:`~pymkm.utils.interpolation.Interpolator` 
  that supports log-log and non-monotonic interpolation for LET/energy curves.

- :mod:`parallel`: 
  Defines the :func:`~pymkm.utils.parallel.optimal_worker_count` utility 
  to determine optimal multiprocessing configurations based on workload and CPU count.

- :mod:`integration`:
  Provides the shared :func:`~pymkm.utils.integration.integrate_1d` utility
  used for impact-parameter integrations.
"""

