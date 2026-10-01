"""
Survival-fraction table generation for MKM, SMK, OSMK, and MCF-MKM.

This subpackage computes and visualizes survival-fraction (SF) curves from an
associated :class:`~pymkm.mktable.core.MKTable` and biological LQ parameters.
The selected survival model is normally inferred from the MKTable configuration
and can also be supplied explicitly to :meth:`SFTable.compute`.

Models supported
----------------

- **MKM (classic)**: linear-quadratic survival using the saturation-corrected
  MKM microdosimetric quantity.
- **SMK (stochastic)**: stochastic MKM survival using domain and nucleus
  dose-averaged specific energies.
- **OSMK 2021/2023**: oxygen-modified SMK survival with the corresponding
  hypoxia parameterization.
- **MCF-MKM**: linear-quadratic survival reconstructed from the precomputed
  MCF quantities ``c_bar`` and ``z_bar_c``.

MCF-MKM and OSMK are mutually exclusive in the current implementation; oxygen
corrections are applied only to stochastic/SMK calculations.

Modules
-------

- :mod:`core`: defines :class:`~pymkm.sftable.core.SFTableParameters` and
  :class:`~pymkm.sftable.core.SFTable`.
- :mod:`compute`: survival-curve computation for the supported model modes.
- :mod:`plot`: visualization of stored survival curves.

Example
-------

.. code-block:: python

    from pymkm import SFTable, SFTableParameters

    sf_params = SFTableParameters(
        mktable=mktable,
        alpha0=0.1,
        beta0=0.05,
    )
    sf_table = SFTable(sf_params)
    sf_table.compute(ion="C", model="classic")
    sf_table.plot()
"""

from .core import SFTable, SFTableParameters
from . import compute  # noqa
from . import plot  # noqa

__all__ = ["SFTable", "SFTableParameters"]
