"""
Microdosimetric table generation for MKM, SMK, OSMK, and MCF-MKM.

This subpackage provides the high-level :class:`MKTable` interface used to
compute, inspect, plot, serialize, and export model-specific microdosimetric
quantities from stopping-power/LET data.

Supported table modes
---------------------

- **MKM (classic)**: saturation-corrected dose-averaged domain specific energy.
- **SMK (stochastic)**: domain, saturation-corrected domain, and nucleus
  dose-averaged specific energies.
- **OSMK 2023**: oxygen-dependent scaling of the SMK geometry/parameters when
  enabled in :class:`MKTableParameters`.
- **MCF-MKM**: dose-averaged non-Poisson correction quantities ``c_bar`` and
  ``z_bar_c`` computed from impact-parameter-dependent specific energies.

For MCF-MKM, ``mcf_nucleus_mode="scaled"`` is the default and follows the
original single-lineal-energy scaling between domain and nucleus. The
``"integrated"`` mode is also available as an alternative calculation in which
nuclear specific energy is integrated explicitly on the same impact-parameter
grid.

Each computed ion entry is stored in ``MKTable.table`` together with the model
parameters and stopping-power metadata. The data columns depend on the active
model, for example:

.. code-block:: text

    MKM:     energy, let, z_bar_star_domain
    SMK:     energy, let, z_bar_star_domain, z_bar_domain, z_bar_nucleus
    MCF-MKM: energy, let, c_bar, z_bar_c

Modules
-------

- :mod:`core`: defines :class:`~pymkm.mktable.core.MKTableParameters` and
  :class:`~pymkm.mktable.core.MKTable`.
- :mod:`compute`: numerical engine for per-ion table generation.
- :mod:`plot`: plotting utilities for model-specific table quantities.

Example
-------

.. code-block:: python

    from pymkm import MKTable, MKTableParameters, StoppingPowerTableSet

    ions = [6]  # carbon
    sp = StoppingPowerTableSet.from_default_source("fluka_2020_0").filter_by_ions(ions)

    params = MKTableParameters(
        domain_radius=0.32,
        nucleus_radius=3.9,
        beta0=0.0615,
    )

    table = MKTable(parameters=params, sp_table_set=sp)
    table.compute(ions=ions, parallel=True)
    table.plot(ions=ions, x="energy", y="z_bar_star_domain")
"""

from .core import MKTable, MKTableParameters
from . import compute  # noqa
from . import plot  # noqa

__all__ = ["MKTable", "MKTableParameters"]

