"""
pyMKM: microdosimetric modeling toolkit for ion-beam radiotherapy.

pyMKM provides tools to compute microdosimetric and radiobiological quantities
from ion-specific stopping-power/LET data and linear-quadratic (LQ) survival
models. It supports:

- radial dose modeling using Scholz-Kraft, Elsaesser-Scholz, Friedrich, and
  Kiefer-Chatterjee track-structure formalisms;
- single-event and dose-averaged specific-energy calculations;
- the classical Microdosimetric Kinetic Model (MKM);
- the stochastic MKM (SMK) and oxygen-modified SMK formulations (OSMK 2021/2023);
- the Mayo Clinic Florida MKM (MCF-MKM), including its impact-parameter-dependent
  non-Poisson correction;
- bundled stopping-power datasets (MSTAR, Geant4, and FLUKA);
- microdosimetric-table and survival-fraction generation, visualization, and export.

Main subpackages
----------------

- :mod:`pymkm.io`: stopping-power data loading, parsing, and table collections.
- :mod:`pymkm.data`: bundled stopping-power datasets and element metadata.
- :mod:`pymkm.physics`: track-structure and specific-energy calculations.
- :mod:`pymkm.biology`: MKM/SMK, OSMK, and MCF-MKM biological model utilities.
- :mod:`pymkm.mktable`: generation of MKM, SMK, OSMK, and MCF-MKM tables.
- :mod:`pymkm.sftable`: survival-fraction calculations from microdosimetric inputs.
- :mod:`pymkm.utils`: numerical integration, interpolation, geometry, and parallelism.

pyMKM is intended for research applications in particle therapy,
microdosimetry, and radiobiological modeling.
"""


from .io import StoppingPowerTableSet
from .physics import ParticleTrack, SpecificEnergy
from .mktable import MKTable, MKTableParameters
from .sftable import SFTable, SFTableParameters

__all__ = [
    "StoppingPowerTableSet", 
    "ParticleTrack",
    "SpecificEnergy",
    "MKTable",
    "MKTableParameters",
    "SFTable",
    "SFTableParameters"
    ]